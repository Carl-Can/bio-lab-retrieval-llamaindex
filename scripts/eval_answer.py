"""
答案质量评测：用 LLM-as-judge 对生成的回答打分（忠实性 / 相关性）。

用法（在仓库根目录下）：
    uv run python scripts/eval_answer.py
    uv run python scripts/eval_answer.py --top-k 5 --limit 10

前置：索引已重建；OPENAI_API_KEY 可用（judge 用 Settings.llm）。
输出：evals/answer_baseline.json
"""
import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator  # noqa: E402

from config.config import Config  # noqa: E402
from core.index_manager import IndexManager  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_qa_pairs(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("pairs", data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", default=os.path.join(ROOT, "evals", "qa_pairs.json"))
    parser.add_argument("--out", default=os.path.join(ROOT, "evals", "answer_baseline.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    Config.setup_logging()
    if not Config.OPENAI_API_KEY:
        raise SystemExit("缺少 OPENAI_API_KEY，无法运行 LLM-as-judge。")

    index_manager = IndexManager()
    if index_manager.index is None:
        raise SystemExit("索引为空：请先重建索引。")

    pairs = load_qa_pairs(args.qa)
    if args.limit:
        pairs = pairs[: args.limit]

    engine = index_manager.get_query_engine(similarity_top_k=args.top_k)
    faithfulness = FaithfulnessEvaluator()
    relevancy = RelevancyEvaluator()

    per_query = []
    for item in pairs:
        query = item["query"]
        response = engine.query(query)
        f = faithfulness.evaluate_response(query=query, response=response)
        r = relevancy.evaluate_response(query=query, response=response)
        per_query.append({
            "query": query,
            "response": str(response)[:500],
            "faithful": bool(f.passing),
            "relevant": bool(r.passing),
            "faithfulness_feedback": f.feedback,
            "relevancy_feedback": r.feedback,
        })
        print(f"[{'OK ' if f.passing else 'NG '}|{'OK ' if r.passing else 'NG '}] {query}")

    n = len(per_query) or 1
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pipeline": "dense_vector_only + gpt synthesis",
        "top_k": args.top_k,
        "num_queries": len(per_query),
        "faithfulness_pass_rate": round(sum(q["faithful"] for q in per_query) / n, 4),
        "relevancy_pass_rate": round(sum(q["relevant"] for q in per_query) / n, 4),
        "per_query": per_query,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n忠实性通过率: {report['faithfulness_pass_rate']}")
    print(f"相关性通过率: {report['relevancy_pass_rate']}")
    print(f"结果已写入: {args.out}")


if __name__ == "__main__":
    main()
