"""
检索层：稠密 / BM25 / 混合（RRF）+ 元数据过滤。

说明：
- llama-index-retrievers-bm25 与 core 0.11.23 不兼容（KwargPackComponent ImportError），
  这里用 rank_bm25 + jieba 自实现。
- Chroma 0.4 的 HNSW 检索在不同进程间对边界 query 结果会漂移（embedding 相同、top-k 不同），
  评测不可复现。ExactDenseRetriever 直接从 Chroma 读出向量做精确余弦，保证可复现。
"""
import logging
import re
from typing import Dict, List, Optional

import jieba
import numpy as np

from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters

from config.config import Config

logger = logging.getLogger("bio_lab_retrieval")

_PUNCT = re.compile(r"[\s，。、；：？！“”‘’（）《》【】,.;:?!()\[\]{}\-_/\\|<>@#$%^&*+=~`'\"]+")


def _as_str(query) -> str:
    """llama-index query engine 会传 QueryBundle，这里统一成 str。"""
    return getattr(query, "query_str", query) or ""


def tokenize(text: str) -> List[str]:
    """中英文混合分词：jieba 切中文，过滤标点，英文/数字小写。"""
    if not text:
        return []
    tokens = []
    for chunk in jieba.cut(text):
        chunk = chunk.strip()
        if not chunk or _PUNCT.fullmatch(chunk):
            continue
        tokens.append(chunk.lower())
    return tokens


class ExactDenseRetriever:
    """精确余弦检索（向量从 Chroma 一次性读出，numpy 计算，结果可复现）。"""

    def __init__(self, index, *, top_k: Optional[int] = None, topic: Optional[str] = None):
        self.top_k = top_k or Config.RETRIEVAL_FETCH_K
        self.node_by_id = dict(index.docstore.docs) if index.docstore else {}

        collection = index.storage_context.vector_store._collection
        data = collection.get(include=["embeddings", "metadatas"])
        ids = data["ids"]
        metas = data["metadatas"] or [{} for _ in ids]
        emb = np.asarray(data["embeddings"], dtype=np.float32)

        if topic:
            keep = [i for i, m in enumerate(metas) if m.get("topic") == topic]
            ids = [ids[i] for i in keep]
            emb = emb[keep] if len(keep) else emb[:0]

        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        self.emb = emb / norm
        self.ids = ids
        self._embed = index.storage_context.vector_store  # keep ref for embed_model
        from llama_index.core.settings import Settings

        self._embed_model = Settings.embed_model

    def retrieve(self, query: str) -> List[NodeWithScore]:
        if len(self.ids) == 0:
            return []
        query = _as_str(query)
        q = np.asarray(self._embed_model.get_query_embedding(query), dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        sims = self.emb @ q
        k = min(self.top_k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        out = []
        for i in idx:
            node = self.node_by_id.get(self.ids[i])
            if node is not None:
                out.append(NodeWithScore(node=node, score=float(sims[i])))
        return out


class BM25Index:
    """基于 docstore 节点的内存 BM25（中英混合）。"""

    def __init__(self, nodes: List):
        from rank_bm25 import BM25Okapi

        self.nodes = nodes
        self.corpus = [tokenize(n.get_content()) for n in nodes]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    def retrieve(self, query: str, top_k: int) -> List[NodeWithScore]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(_as_str(query)))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:
                continue
            out.append(NodeWithScore(node=self.nodes[i], score=float(scores[i])))
        return out


class HybridRetriever:
    """稠密 + BM25，按 RRF（Reciprocal Rank Fusion）融合。"""

    def __init__(
        self,
        index,
        *,
        similarity_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
        rrf_k: int = 60,
        topic: Optional[str] = None,
    ):
        self.dense = ExactDenseRetriever(index, top_k=similarity_top_k, topic=topic)
        self.bm25_top_k = bm25_top_k or Config.BM25_TOP_K
        self.rrf_k = rrf_k

        nodes = list(index.docstore.docs.values()) if index.docstore else []
        if topic:
            nodes = [n for n in nodes if n.metadata.get("topic") == topic]
        self.bm25 = BM25Index(nodes)

    def retrieve(self, query: str) -> List[NodeWithScore]:
        query = _as_str(query)
        dense_nodes = self.dense.retrieve(query)
        bm25_nodes = self.bm25.retrieve(query, self.bm25_top_k)

        fused: Dict[str, float] = {}
        pool: Dict[str, NodeWithScore] = {}
        for rank, n in enumerate(dense_nodes):
            fused[n.node_id] = fused.get(n.node_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            pool[n.node_id] = n
        for rank, n in enumerate(bm25_nodes):
            fused[n.node_id] = fused.get(n.node_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            pool.setdefault(n.node_id, n)

        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [NodeWithScore(node=pool[nid].node, score=score) for nid, score in ordered]


def build_retriever(
    index,
    *,
    mode: Optional[str] = None,
    topic: Optional[str] = None,
    fetch_k: Optional[int] = None,
):
    """mode: dense(chroma) | exact(精确余弦) | hybrid。"""
    mode = (mode or Config.RETRIEVAL_MODE or "exact").lower()
    fetch_k = fetch_k or Config.RETRIEVAL_FETCH_K

    if mode == "hybrid":
        return HybridRetriever(index, similarity_top_k=fetch_k, bm25_top_k=fetch_k, topic=topic)
    if mode == "exact":
        return ExactDenseRetriever(index, top_k=fetch_k, topic=topic)
    if mode == "dense":
        filters = (
            MetadataFilters(filters=[MetadataFilter(key="topic", value=topic)]) if topic else None
        )
        return index.as_retriever(similarity_top_k=fetch_k, filters=filters)
    raise ValueError(f"未知检索模式: {mode}（可选 dense / exact / hybrid）")
