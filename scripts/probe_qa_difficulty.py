"""
评测集难度探针：判断每条 query 的 gold 在稠密候选池里的位置，用于筛选「困难」问句。

- pool_rank == 1        -> trivial（稠密序就能命中，重排/hybrid 无从发挥）
- 2 <= pool_rank <= K   -> hard（有发挥空间）
- 未命中                -> miss（候选池都召不回，需要 hybrid/改写）

用法（在仓库根目录下）：
    uv run python scripts/probe_qa_difficulty.py --qa evals/qa_pairs_hard.json
    uv run python scripts/probe_qa_difficulty.py --qa evals/qa_pairs_hard.json --keep-hard evals/qa_pairs_hard.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config  # noqa: E402
from core.index_manager import IndexManager  # noqa: E402
from core.retrievers import build_retriever  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_pairs(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("pairs", data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", default=os.path.join(ROOT, "evals", "qa_pairs_hard.json"))
    parser.add_argument("--fetch-k", type=int, default=20)
    parser.add_argument("--retrieval", default="exact", choices=["dense", "exact", "hybrid"])
    parser.add_argument("--topic", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--keep-hard", default="", help="把 hard 问句写回该文件")
    args = parser.parse_args()

    Config.setup_logging()
    index_manager = IndexManager()
    if index_manager.index is None:
        raise SystemExit("索引为空：请先重建索引。")

    pairs = load_pairs(args.qa)
    retriever = build_retriever(
        index_manager.index, mode=args.retrieval, topic=args.topic or None, fetch_k=args.fetch_k
    )

    trivial, hard, miss = [], [], []
    rows = []
    for item in pairs:
        query = item["query"]
        expected = set(item["expected_doc_ids"])
        nodes = retriever.retrieve(query)
        doc_ids = [n.node.metadata.get("doc_id") for n in nodes]

        rank = 0
        for i, d in enumerate(doc_ids, 1):
            if d in expected:
                rank = i
                break
        label = "trivial" if rank == 1 else ("hard" if rank > 1 else "miss")
        row = {
            **item,
            "pool_rank": rank,
            "difficulty": label,
            "top3_retrieved": doc_ids[:3],
        }
        rows.append(row)
        {"trivial": trivial, "hard": hard, "miss": miss}[label].append(row)
        print(f"[{label:>7}] rank={rank or 'MISS':>4} | {query[:40]}")
        if label != "trivial":
            print(f"           gold: {list(expected)}")
            print(f"           top3: {doc_ids[:3]}")

    n = len(pairs) or 1
    print("\n" + "=" * 70)
    print(f"总计 {len(pairs)} 条 | trivial {len(trivial)} ({len(trivial)/n:.0%}) | "
          f"hard {len(hard)} ({len(hard)/n:.0%}) | miss {len(miss)} ({len(miss)/n:.0%})")
    print("=" * 70)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"pairs": rows}, f, ensure_ascii=False, indent=2)
        print(f"标注结果 -> {args.out}")

    if args.keep_hard:
        keep = [{"query": r["query"], "expected_doc_ids": r["expected_doc_ids"],
                 "topic": r.get("topic"), "difficulty": r["difficulty"]}
                for r in (hard + miss)]
        with open(args.keep_hard, "w", encoding="utf-8") as f:
            json.dump({"pairs": keep}, f, ensure_ascii=False, indent=2)
        print(f"困难集({len(keep)} 条) -> {args.keep_hard}")


if __name__ == "__main__":
    main()
