"""
检索/链路回归评测 + 门禁。

- 支持两条链路：
    检索链路：`--rerank none`（只看检索质量）
    生产链路：`--rerank bge-onnx --fetch-k 20`（hybrid 召回 -> 重排 -> top_k）
- 用统一的 `build_retriever` / `build_reranker`，确定性（hybrid 用精确余弦 + BM25；
  bge-onnx 为 ONNX int8，结果稳定）。
- 同时报告整体与困难子集（difficulty != trivial）的 hit@1/3/5、MRR。
- `--check-baseline` 对比已提交的 baseline，任一指标跌破（超出 tolerance）即退出非 0。

用法：
    # 检索基线
    uv run python scripts/eval_retrieval.py --rerank none --top-k 5 --out evals/retrieval_baseline.json
    # 生产链路（含重排）基线
    uv run python scripts/eval_retrieval.py --rerank bge-onnx --fetch-k 20 \
        --top-k 5 --out evals/pipeline_baseline.json
    # 门禁
    uv run python scripts/eval_retrieval.py --rerank bge-onnx --fetch-k 20 --top-k 5 \
        --check-baseline evals/pipeline_baseline.json --tolerance 0.02
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config  # noqa: E402
from core.index_manager import IndexManager  # noqa: E402
from core.retrievers import build_retriever  # noqa: E402
from core.rerankers import build_reranker  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 参与门禁对比的指标（越大越好）
GATED = ("hit@1", "hit@3", "hit@5", "mrr")


def load_pairs(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("pairs", [])


def rank_of_gold(doc_ids, expected):
    for i, d in enumerate(doc_ids, 1):
        if d in expected:
            return i
    return 0


def eval_pairs(retriever, reranker, top_k, pairs):
    rows = []
    for item in pairs:
        expected = set(item["expected_doc_ids"])
        t0 = time.perf_counter()
        nodes = retriever.retrieve(item["query"])
        if reranker is not None:
            nodes = reranker.rerank_nodes(item["query"], nodes, top_k)
        else:
            nodes = nodes[:top_k]
        ms = (time.perf_counter() - t0) * 1000
        doc_ids = [n.node.metadata.get("doc_id") for n in nodes]
        rank = rank_of_gold(doc_ids, expected)
        rows.append({
            "query": item["query"],
            "rank": rank,
            "hit@1": 1 if rank == 1 else 0,
            "hit@3": 1 if 0 < rank <= 3 else 0,
            "hit@5": 1 if 0 < rank <= 5 else 0,
            "mrr": (1.0 / rank) if rank else 0.0,
            "latency_ms": round(ms, 1),
        })
    return rows


def aggregate(rows):
    n = len(rows) or 1
    agg = {k: round(sum(r[k] for r in rows) / n, 4) for k in GATED}
    agg["avg_latency_ms"] = round(sum(r["latency_ms"] for r in rows) / n, 1)
    agg["n"] = len(rows)
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", default=os.path.join(ROOT, "evals", "qa_pairs_all.json"))
    parser.add_argument("--annotated", default=os.path.join(ROOT, "evals", "qa_pairs_all_annotated.json"))
    parser.add_argument("--out", default="")
    parser.add_argument("--check-baseline", default="")
    parser.add_argument("--tolerance", type=float, default=0.02, help="允许的下降幅度")
    parser.add_argument("--retrieval", default="hybrid", choices=["dense", "exact", "hybrid"])
    parser.add_argument("--rerank", default="none",
                        choices=["none", "bge", "bge-onnx", "jev", "jev-gate", "cascade"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fetch-k", type=int, default=0, help="候选数，默认=top_k")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    Config.setup_logging()
    index_manager = IndexManager()
    if index_manager.index is None:
        raise SystemExit("索引为空：请先重建索引。")

    pairs = load_pairs(args.qa)
    if args.limit:
        pairs = pairs[: args.limit]

    # 困难子集（difficulty != trivial）
    hard_q = set()
    if os.path.exists(args.annotated):
        hard_q = {p["query"] for p in load_pairs(args.annotated)
                  if p.get("difficulty") and p["difficulty"] != "trivial"}
    hard_pairs = [p for p in pairs if p["query"] in hard_q]

    fetch_k = args.fetch_k or args.top_k
    retriever = build_retriever(index_manager.index, mode=args.retrieval, fetch_k=fetch_k)
    reranker = build_reranker(args.rerank) if args.rerank != "none" else None

    overall = aggregate(eval_pairs(retriever, reranker, args.top_k, pairs))
    hard = aggregate(eval_pairs(retriever, reranker, args.top_k, hard_pairs)) if hard_pairs else {"n": 0}

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"retrieval": args.retrieval, "rerank": args.rerank, "top_k": args.top_k,
                   "fetch_k": fetch_k, "qa": os.path.basename(args.qa)},
        "overall": overall,
        "hard": hard,
    }

    print(f"检索={args.retrieval} 重排={args.rerank} top_k={args.top_k} fetch_k={fetch_k}  "
          f"整体 n={overall['n']}  困难 n={hard.get('n')}")
    print(f"  overall: hit@1={overall['hit@1']} hit@3={overall['hit@3']} "
          f"hit@5={overall['hit@5']} MRR={overall['mrr']}")
    if hard.get("n"):
        print(f"  hard:    hit@1={hard['hit@1']} hit@3={hard['hit@3']} "
              f"hit@5={hard['hit@5']} MRR={hard['mrr']}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"基线已写入: {args.out}")

    if args.check_baseline:
        with open(args.check_baseline, "r", encoding="utf-8") as f:
            base = json.load(f)
        regressions = []
        for section in ("overall", "hard"):
            cur, ref = report.get(section, {}), base.get(section, {})
            if not ref.get("n") or not cur.get("n"):
                continue
            for k in GATED:
                if k in ref and k in cur and cur[k] < ref[k] - args.tolerance:
                    regressions.append(f"{section}.{k}: {cur[k]} < {ref[k]} - {args.tolerance}")
        if regressions:
            print("\n检索回归门禁未通过：")
            for r in regressions:
                print("  -", r)
            sys.exit(1)
        print(f"\n检索回归门禁通过（tolerance={args.tolerance}）")


if __name__ == "__main__":
    main()
