"""
合并多个 QA 文件为一个评测集（按 query 去重，保留 source 标签）。

用法：
    uv run python scripts/build_eval_set.py \
      --inputs evals/qa_pairs_hard.json evals/qa_pairs.json evals/qa_pairs_auto.json \
      --out evals/qa_pairs_all.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path, tag):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for p in data.get("pairs", data):
        p.setdefault("source", tag)
        yield p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=[
        os.path.join(ROOT, "evals", "qa_pairs_hard.json"),
        os.path.join(ROOT, "evals", "qa_pairs.json"),
        os.path.join(ROOT, "evals", "qa_pairs_auto.json"),
    ])
    parser.add_argument("--out", default=os.path.join(ROOT, "evals", "qa_pairs_all.json"))
    args = parser.parse_args()

    merged, seen = [], set()
    for path in args.inputs:
        tag = os.path.splitext(os.path.basename(path))[0].replace("qa_pairs_", "") or "hand"
        n0 = len(merged)
        for p in load(path, tag):
            q = p["query"].strip()
            if not q or q in seen:
                continue
            seen.add(q)
            p["query"] = q
            merged.append(p)
        print(f"{os.path.basename(path):>24}: +{len(merged)-n0}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"_comment": "合并评测集：hard(手写改写) + hand(手写通用) + auto(LLM 生成)",
                   "version": 1, "count": len(merged), "pairs": merged},
                  f, ensure_ascii=False, indent=2)
    print(f"合并 {len(merged)} 条 -> {args.out}")


if __name__ == "__main__":
    main()
