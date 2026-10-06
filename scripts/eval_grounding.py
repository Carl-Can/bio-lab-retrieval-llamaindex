"""
Grounding 核验评测：对生成的答案做逐条陈述的 grounding 检查。

对比两条路线：
- Jev guard（MiniCPM Yes/No 逐条批验，毫秒级）
- LLM-as-judge（FaithfulnessEvaluator，走本地 vLLM，秒级）作为参照

用法（在仓库根目录下）：
    uv run python scripts/eval_grounding.py --limit 12
    uv run python scripts/eval_grounding.py --limit 12 --no-judge
输出：evals/grounding_eval.json
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
from core.grounding import JevGroundingGuard  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_pairs(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("pairs", [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", default=os.path.join(ROOT, "evals", "qa_pairs_hard.json"))
    parser.add_argument("--out", default=os.path.join(ROOT, "evals", "grounding_eval.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM-as-judge（只跑 Jev）")
    args = parser.parse_args()

    Config.setup_logging()
    index_manager = IndexManager()
    if index_manager.index is None:
        raise SystemExit("索引为空：请先重建索引。")

    pairs = load_pairs(args.qa)
    if args.limit:
        pairs = pairs[: args.limit]

    engine = index_manager.get_query_engine(similarity_top_k=args.top_k)
    guard = JevGroundingGuard()

    judge = None
    if not args.no_judge:
        from llama_index.core.evaluation import FaithfulnessEvaluator

        judge = FaithfulnessEvaluator()

    rows = []
    for i, item in enumerate(pairs, 1):
        q = item["query"]
        response = engine.query(q)
        contexts = [n.node.get_content() for n in getattr(response, "source_nodes", [])]

        t0 = time.perf_counter()
        g = guard.verify(str(response), contexts, q)
        jev_ms = (time.perf_counter() - t0) * 1000

        row = {
            "query": q,
            "answer": str(response)[:600],
            "jev": {**g, "latency_ms": round(jev_ms, 1)},
        }

        if judge is not None:
            t0 = time.perf_counter()
            jr = judge.evaluate_response(query=q, response=response)
            judge_ms = (time.perf_counter() - t0) * 1000
            row["judge"] = {
                "passing": bool(jr.passing),
                "score": float(jr.score) if jr.score is not None else None,
                "feedback": jr.feedback,
                "latency_ms": round(judge_ms, 1),
            }

        rows.append(row)
        j = row.get("judge")
        print(f"[{i:02d}/{len(pairs):02d}] jev={'PASS' if g['passed'] else 'FLAG'}"
              f" faith={g['faithfulness']:.2f} flagged={len(g['flagged'])}"
              + (f" | judge={'PASS' if j['passing'] else 'FAIL'}" if j else "")
              + f" | {q[:34]}")

    n = len(rows) or 1
    summary = {
        "num_queries": len(rows),
        "jev": {
            "pass_rate": round(sum(r["jev"]["passed"] for r in rows) / n, 4),
            "avg_faithfulness": round(sum(r["jev"]["faithfulness"] for r in rows) / n, 4),
            "avg_claims": round(sum(r["jev"]["num_claims"] for r in rows) / n, 2),
            "avg_flagged": round(sum(len(r["jev"]["flagged"]) for r in rows) / n, 2),
            "avg_latency_ms": round(sum(r["jev"]["latency_ms"] for r in rows) / n, 1),
        },
    }
    if judge is not None:
        judged = [r for r in rows if "judge" in r]
        m = len(judged) or 1
        agree = sum(1 for r in judged if r["jev"]["passed"] == r["judge"]["passing"]) / m
        summary["judge"] = {
            "pass_rate": round(sum(r["judge"]["passing"] for r in judged) / m, 4),
            "avg_latency_ms": round(sum(r["judge"]["latency_ms"] for r in judged) / m, 1),
        }
        summary["agreement"] = round(agree, 4)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "llm_provider": Config.LLM_PROVIDER,
            "grounding_threshold": Config.GROUNDING_THRESHOLD,
            "jev_model": Config.JEV_MODEL,
        },
        "summary": summary,
        "details": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("summary:", json.dumps(summary, ensure_ascii=False))
    print(f"结果已写入: {args.out}")


if __name__ == "__main__":
    main()
