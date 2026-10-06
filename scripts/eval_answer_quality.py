"""
答案质量评测：生成答案并用 LLM-as-judge 打分（忠实性 + 相关性）。

- 忠实性 FaithfulnessEvaluator：答案是否由检索到的 context 支持（幻觉检测）
- 相关性 RelevancyEvaluator：答案是否切题
- judge 走当前 Settings.llm（本项目默认本地 vLLM）

用法：
    uv run python scripts/eval_answer_quality.py --limit 50
    uv run python scripts/eval_answer_quality.py --only-hard --limit 40
    uv run python scripts/eval_answer_quality.py --min-faithfulness 0.8   # CI 门禁（不达标退出非 0）
输出：evals/answer_quality.json
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_pairs(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("pairs", [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", default=os.path.join(ROOT, "evals", "qa_pairs_all.json"))
    parser.add_argument("--hard", default=os.path.join(ROOT, "evals", "qa_pairs_all_annotated.json"))
    parser.add_argument("--out", default=os.path.join(ROOT, "evals", "answer_quality.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-hard", action="store_true", help="只评困难子集（pool_rank != 1）")
    parser.add_argument("--min-faithfulness", type=float, default=None, help="门禁：低于则退出非 0")
    parser.add_argument("--min-relevancy", type=float, default=None)
    args = parser.parse_args()

    Config.setup_logging()
    index_manager = IndexManager()
    if index_manager.index is None:
        raise SystemExit("索引为空：请先重建索引。")

    pairs = load_pairs(args.qa)
    if args.only_hard and os.path.exists(args.hard):
        hard_q = {p["query"] for p in load_pairs(args.hard) if p.get("difficulty") != "trivial"}
        pairs = [p for p in pairs if p["query"] in hard_q]
    if args.limit:
        pairs = pairs[: args.limit]

    from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

    engine = index_manager.get_query_engine(similarity_top_k=args.top_k)
    faithfulness = FaithfulnessEvaluator()
    relevancy = RelevancyEvaluator()

    rows = []
    errors = 0
    for i, item in enumerate(pairs, 1):
        q = item["query"]
        try:
            t0 = time.perf_counter()
            response = engine.query(q)
            gen_ms = (time.perf_counter() - t0) * 1000
            f = faithfulness.evaluate_response(query=q, response=response)
            r = relevancy.evaluate_response(query=q, response=response)
        except Exception as e:  # noqa: BLE001 - 单条失败不中断长跑
            errors += 1
            print(f"[{i:02d}/{len(pairs):02d}] ERROR {q[:40]}: {str(e)[:80]}")
            continue
        rows.append({
            "query": q,
            "topic": item.get("topic"),
            "source": item.get("source"),
            "answer": str(response)[:500],
            "faithful": bool(f.passing),
            "relevant": bool(r.passing),
            "gen_latency_ms": round(gen_ms, 1),
        })
        print(f"[{i:02d}/{len(pairs):02d}] faith={'Y' if f.passing else 'N'} "
              f"rel={'Y' if r.passing else 'N'} | {q[:40]}")

    n = len(rows) or 1
    summary = {
        "num_queries": len(rows),
        "errors": errors,
        "faithfulness_pass_rate": round(sum(x["faithful"] for x in rows) / n, 4),
        "relevancy_pass_rate": round(sum(x["relevant"] for x in rows) / n, 4),
        "avg_gen_latency_ms": round(sum(x["gen_latency_ms"] for x in rows) / n, 1),
    }
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"llm_provider": Config.LLM_PROVIDER, "top_k": args.top_k,
                   "only_hard": args.only_hard},
        "summary": summary,
        "details": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("summary:", json.dumps(summary, ensure_ascii=False))
    print(f"结果已写入: {args.out}")

    failed = (
        (args.min_faithfulness is not None and summary["faithfulness_pass_rate"] < args.min_faithfulness)
        or (args.min_relevancy is not None and summary["relevancy_pass_rate"] < args.min_relevancy)
    )
    if failed:
        print("门禁未通过")
        sys.exit(1)


if __name__ == "__main__":
    main()
