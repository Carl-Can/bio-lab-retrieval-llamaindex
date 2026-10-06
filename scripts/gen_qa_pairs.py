"""
用本地 LLM（vLLM）从语料生成 grounded QA 候选：每篇文档生成一个「真实用户会问」的问题，
金标 = 该文档的 doc_id。要求改写、不照抄标题，便于后续用难度探针筛选。

用法（在仓库根目录下）：
    uv run python scripts/gen_qa_pairs.py --per-doc 1 --out evals/qa_pairs_auto.json
    uv run python scripts/gen_qa_pairs.py --limit 20          # 抽查

前置：索引已建；LLM_PROVIDER=vllm 且 vLLM 在跑（或 openai 有额度）。
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config  # noqa: E402
from core.index_manager import IndexManager  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROMPT = (
    "你是体外诊断（IVD）/分子诊断领域的研发人员。下面是一篇知识库文档的节选。\n"
    "请只依据这段内容，提出一个该领域从业者会真实问出的、具体的问题。要求：\n"
    "1) 用同义改写，不要照抄标题或原文的连续短语；\n"
    "2) 问题必须能由这段内容回答，不要问内容里没有的信息；\n"
    "3) 一句话，中文，不要加「问题：」等前缀，不要引号；\n"
    "4) 只输出这个问题本身。\n\n"
    "文档节选：\n{context}\n\n问题："
)

_CLEAN = re.compile(r'^[\s"“”\'’：:、\-*\d.)）]+')


def clean_question(text: str) -> str:
    q = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    q = _CLEAN.sub("", q).strip().rstrip("？?").strip()
    q = q.replace("问题：", "").replace("问题:", "").strip()
    return q + "？" if q else ""


def representative_text(nodes, max_chars: int = 1600) -> str:
    """取该文档靠前的若干 chunk 拼成上下文。"""
    texts, total = [], 0
    for n in nodes:
        t = n.get_content().strip()
        if not t:
            continue
        texts.append(t)
        total += len(t)
        if total >= max_chars:
            break
    return "\n\n".join(texts)[:max_chars]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(ROOT, "evals", "qa_pairs_auto.json"))
    parser.add_argument("--per-doc", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 篇文档（抽查）")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    Config.setup_logging()
    index_manager = IndexManager()
    if index_manager.index is None:
        raise SystemExit("索引为空：请先重建索引。")

    # 按 doc_id 聚合 docstore 节点
    by_doc = {}
    for node in index_manager.index.docstore.docs.values():
        did = node.metadata.get("doc_id")
        if did:
            by_doc.setdefault(did, []).append(node)

    doc_ids = sorted(by_doc)
    if args.limit:
        doc_ids = doc_ids[: args.limit]
    print(f"待生成文档数: {len(doc_ids)}")

    llm = index_manager.llm

    def gen(doc_id):
        ctx = representative_text(by_doc[doc_id])
        if len(ctx) < 80:
            return None
        try:
            resp = llm.complete(PROMPT.format(context=ctx))
            q = clean_question(str(resp))
        except Exception as e:  # noqa: BLE001
            print(f"  生成失败 {doc_id}: {e}")
            return None
        if not (8 <= len(q) <= 100):
            return None
        topic = by_doc[doc_id][0].metadata.get("topic", "")
        return {"query": q, "expected_doc_ids": [doc_id], "topic": topic, "source": "llm"}

    pairs, seen = [], set()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(gen, d): d for d in doc_ids}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r and r["query"] not in seen:
                seen.add(r["query"])
                pairs.append(r)
            if i % 20 == 0:
                print(f"  {i}/{len(doc_ids)} 完成，有效 {len(pairs)}")

    pairs.sort(key=lambda x: x["expected_doc_ids"][0])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"_comment": "LLM 从语料生成的 grounded QA 候选（金标=来源文档）",
                   "version": 1, "pairs": pairs}, f, ensure_ascii=False, indent=2)
    print(f"生成 {len(pairs)} 条 -> {args.out}")


if __name__ == "__main__":
    main()
