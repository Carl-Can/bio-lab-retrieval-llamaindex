"""
重排 A/B：在同一稠密候选池上对比 none / bge / jev 三种重排策略。

用法（在仓库根目录下）：
    uv run python scripts/eval_rerank_ab.py                 # fetch_k=20, top_k=5
    uv run python scripts/eval_rerank_ab.py --strategies none,bge,jev
    uv run python scripts/eval_rerank_ab.py --limit 10

指标：hit@1/3/5、MRR（文档级）、重排延迟、送入上下文的近似 token 数、噪声剔除比。
输出：evals/rerank_ab.json
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
from core.rerankers import build_reranker  # noqa: E402
from core.retrievers import build_retriever  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_qa_pairs(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("pairs", [])


def approx_tokens(text: str) -> int:
    return max(1, int(len(text or "") * 0.75))


def rank_of_gold(doc_ids, expected):
    for i, d in enumerate(doc_ids, 1):
        if d in expected:
            return i
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", default=os.path.join(ROOT, "evals", "qa_pairs_hard.json"))
    parser.add_argument("--out", default=os.path.join(ROOT, "evals", "rerank_ab.json"))
    parser.add_argument("--fetch-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--strategies", default="none,bge,jev")
    parser.add_argument("--retrieval", default="exact", choices=["dense", "exact", "hybrid"])
    parser.add_argument("--topic", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    Config.setup_logging()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    index_manager = IndexManager()
    if index_manager.index is None:
        raise SystemExit("索引为空：请先重建索引。")

    pairs = load_qa_pairs(args.qa)
    if args.limit:
        pairs = pairs[: args.limit]

    retriever = build_retriever(
        index_manager.index, mode=args.retrieval, topic=args.topic or None, fetch_k=args.fetch_k
    )
    rerankers = {}
    for s in strategies:
        if s == "none":
            rerankers[s] = None
        else:
            rerankers[s] = build_reranker(s)
            print(f"重排器已就绪: {s}")

    results = {s: [] for s in strategies}

    for idx, item in enumerate(pairs, 1):
        query = item["query"]
        expected = set(item["expected_doc_ids"])

        t0 = time.perf_counter()
        pool = retriever.retrieve(query)
        retrieve_ms = (time.perf_counter() - t0) * 1000
        pool = pool[: Config.RERANK_MAX_CANDIDATES]  # 与生产一致：候选上限
        pool_doc_ids = [n.node.metadata.get("doc_id") for n in pool]
        pool_rank = rank_of_gold(pool_doc_ids, expected)

        for s in strategies:
            reranker = rerankers[s]
            kept_by_threshold = len(pool)
            if reranker is None:
                ranked = pool[: args.top_k]
                rerank_ms = 0.0
            elif hasattr(reranker, "rank_with_stats"):
                t0 = time.perf_counter()
                ranked, kept_by_threshold = reranker.rank_with_stats(query, pool, args.top_k)
                rerank_ms = (time.perf_counter() - t0) * 1000
            else:
                t0 = time.perf_counter()
                ranked = reranker.rerank_nodes(query, pool, args.top_k)
                rerank_ms = (time.perf_counter() - t0) * 1000

            doc_ids = [n.node.metadata.get("doc_id") for n in ranked]
            rank = rank_of_gold(doc_ids, expected)
            results[s].append({
                "query": query,
                "expected_doc_ids": list(expected),
                "retrieved_doc_ids": doc_ids,
                "rank": rank,
                "hit@1": 1 if rank == 1 else 0,
                "hit@3": 1 if 0 < rank <= 3 else 0,
                "hit@5": 1 if 0 < rank <= 5 else 0,
                "mrr": (1.0 / rank) if rank else 0.0,
                "docs": len(ranked),
                "kept_by_threshold": kept_by_threshold,
                "pool_rank": pool_rank,
                "context_tokens": sum(approx_tokens(n.node.get_content()) for n in ranked),
                "rerank_ms": round(rerank_ms, 1),
                "retrieve_ms": round(retrieve_ms, 1),
            })

        marks = " ".join(f"{s}:r={results[s][-1]['rank'] or 'MISS'}" for s in strategies)
        print(f"[{idx:02d}/{len(pairs):02d}] pool={len(pool)} | {marks} | {query[:34]}")

    n = len(pairs) or 1
    summary = {}
    for s in strategies:
        rows = results[s]
        summary[s] = {
            "hit@1": round(sum(r["hit@1"] for r in rows) / n, 4),
            "hit@3": round(sum(r["hit@3"] for r in rows) / n, 4),
            "hit@5": round(sum(r["hit@5"] for r in rows) / n, 4),
            "mrr": round(sum(r["mrr"] for r in rows) / n, 4),
            "avg_docs": round(sum(r["docs"] for r in rows) / n, 2),
            "avg_context_tokens": round(sum(r["context_tokens"] for r in rows) / n, 1),
            "avg_rerank_ms": round(sum(r["rerank_ms"] for r in rows) / n, 1),
            "avg_total_ms": round(sum(r["rerank_ms"] + r["retrieve_ms"] for r in rows) / n, 1),
        }

    baseline_tokens = summary.get("none", {}).get("avg_context_tokens")
    if baseline_tokens:
        for s in strategies:
            summary[s]["context_token_reduction_pct"] = round(
                (1 - summary[s]["avg_context_tokens"] / baseline_tokens) * 100, 1
            )
    noise = 1.0 - summary.get("jev", {}).get("avg_docs", args.top_k) / args.fetch_k
    if "jev" in summary:
        summary["jev"]["noise_filtered_pct"] = round(noise * 100, 1)

    # 困难子集：稠密候选池里 gold 不在第 1 位的 query（重排才有发挥空间）
    hard_idx = [i for i, r in enumerate(results[strategies[0]]) if r["pool_rank"] != 1]
    hard = {}
    for s in strategies:
        rows = [results[s][i] for i in hard_idx]
        m = len(rows) or 1
        hard[s] = {
            "hit@1": round(sum(r["hit@1"] for r in rows) / m, 4),
            "hit@5": round(sum(r["hit@5"] for r in rows) / m, 4),
            "mrr": round(sum(r["mrr"] for r in rows) / m, 4),
            "avg_rerank_ms": round(sum(r["rerank_ms"] for r in rows) / m, 1),
        }

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "candidate_pool": f"{args.retrieval} retrieval (fetch_k={args.fetch_k})",
        "retrieval": args.retrieval,
        "topic_filter": args.topic or None,
        "fetch_k": args.fetch_k,
        "top_k": args.top_k,
        "num_queries": len(pairs),
        "summary": summary,
        "hard_subset": {
            "definition": "稠密候选池中 gold 不在第 1 位的 query",
            "n": len(hard_idx),
            "metrics": hard,
        },
        "details": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    header = (f"{'strategy':<10}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}"
              f"{'MRR':>8}{'docs':>7}{'tokens':>9}{'rerank_ms':>11}")
    print(header)
    print("-" * 78)
    for s in strategies:
        m = summary[s]
        print(f"{s:<10}{m['hit@1']:>8.3f}{m['hit@3']:>8.3f}{m['hit@5']:>8.3f}{m['mrr']:>8.3f}"
              f"{m['avg_docs']:>7.2f}{m['avg_context_tokens']:>9.0f}{m['avg_rerank_ms']:>11.1f}")
    print("=" * 78)
    print(f"困难子集 (pool_rank != 1, n={len(hard_idx)})")
    print(f"{'strategy':<10}{'hit@1':>8}{'hit@5':>8}{'MRR':>8}{'rerank_ms':>11}")
    print("-" * 78)
    for s in strategies:
        m = hard[s]
        print(f"{s:<10}{m['hit@1']:>8.3f}{m['hit@5']:>8.3f}{m['mrr']:>8.3f}{m['avg_rerank_ms']:>11.1f}")
    print("=" * 78)
    print(f"结果已写入: {args.out}")


if __name__ == "__main__":
    main()
