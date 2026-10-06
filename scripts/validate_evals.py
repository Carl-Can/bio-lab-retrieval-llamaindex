"""
评测集校验（CI 门禁用，不需要模型/GPU）。

检查：
  - schema：query 非空字符串；expected_doc_ids 为非空字符串列表
  - query 无重复
  - 每条金标 doc_id 都真实存在于 docs/（按与建库一致的规则）
  - 总数不低于 --min-count

用法：
    uv run python scripts/validate_evals.py --qa evals/qa_pairs_all.json --min-count 150
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config  # noqa: E402
from core import ingestion  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", default=os.path.join(ROOT, "evals", "qa_pairs_all.json"))
    parser.add_argument("--min-count", type=int, default=150)
    parser.add_argument("--docs-root", default=None)
    args = parser.parse_args()

    docs_root = args.docs_root or Config.DOCS_FOLDER
    valid_ids = set(ingestion.list_doc_ids(docs_root))

    with open(args.qa, "r", encoding="utf-8") as f:
        pairs = json.load(f).get("pairs", [])

    errors = []
    seen = set()
    for i, p in enumerate(pairs):
        q = p.get("query")
        golds = p.get("expected_doc_ids")
        if not isinstance(q, str) or not q.strip():
            errors.append(f"#{i} query 非法: {q!r}")
            continue
        if q in seen:
            errors.append(f"#{i} query 重复: {q!r}")
        seen.add(q)
        if not isinstance(golds, list) or not golds or not all(isinstance(g, str) and g for g in golds):
            errors.append(f"#{i} expected_doc_ids 非法: {golds!r}")
            continue
        for g in golds:
            if g not in valid_ids:
                errors.append(f"#{i} 金标不存在于 docs/: {g!r}")

    if len(pairs) < args.min_count:
        errors.append(f"数量不足: {len(pairs)} < {args.min_count}")

    if errors:
        print(f"校验失败（{len(errors)} 个问题）：")
        for e in errors[:40]:
            print("  -", e)
        sys.exit(1)

    print(f"校验通过：{len(pairs)} 条，金标全部命中 docs/（{len(valid_ids)} 个 doc_id）")


if __name__ == "__main__":
    main()
