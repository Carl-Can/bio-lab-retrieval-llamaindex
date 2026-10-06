"""
从真实查询日志（logs/queries.jsonl）挖候选评测 query。

输出唯一 query + 出现次数，供人工标注金标后并入评测集。没有金标时无法自动评测，
所以这里只做「去重 + 频次 + 是否已存在」，并可选调用 LLM 猜测来源文档辅助标注。

用法：
    uv run python scripts/mine_queries.py
    uv run python scripts/mine_queries.py --guess          # 用 LLM 猜来源文档（辅助，需人工确认）
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=Config.QUERY_LOG)
    parser.add_argument("--qa", default=os.path.join(ROOT, "evals", "qa_pairs_all.json"))
    parser.add_argument("--out", default=os.path.join(ROOT, "evals", "mined_queries.json"))
    parser.add_argument("--min-count", type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists(args.log):
        raise SystemExit(f"没有查询日志: {args.log}（先跑一段时间服务再看）")

    counter = Counter()
    with open(args.log, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                q = json.loads(line).get("q", "").strip()
            except json.JSONDecodeError:
                continue
            if q:
                counter[q] += 1

    existing = set()
    if os.path.exists(args.qa):
        with open(args.qa, "r", encoding="utf-8") as f:
            existing = {p["query"].strip() for p in json.load(f).get("pairs", [])}

    mined = [
        {"query": q, "count": c, "in_eval_set": q in existing}
        for q, c in counter.most_common() if c >= args.min_count
    ]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"_comment": "真实日志挖出的候选 query（需人工补金标 expected_doc_ids）",
                   "version": 1, "pairs": mined}, f, ensure_ascii=False, indent=2)
    new = sum(1 for m in mined if not m["in_eval_set"])
    print(f"日志 {sum(counter.values())} 条 -> 唯一 {len(mined)}，其中 {new} 条不在现有评测集")
    print(f"输出 -> {args.out}")


if __name__ == "__main__":
    main()
