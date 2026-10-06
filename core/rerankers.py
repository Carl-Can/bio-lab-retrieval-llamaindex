"""
可插拔重排层。

- BgeReranker     : BAAI/bge-reranker-v2-m3 Cross-Encoder（进程内，GPU/CPU）
- OnnxBgeReranker : 同款 bge-reranker 的 ONNX int8 量化版（CPU，AVX2，比 PyTorch CPU 快）
- JevReranker     : Jev 模式——1B 模型单次前向取 Yes/No logits，做相关性打分 + 硬噪声门控
                    （参考 JEV_IN_RAG_APPLICATION_GUIDE.md Case 1；模型 cyankiwi/MiniCPM5-1B-AWQ-INT4）

都提供 rerank_nodes(query, nodes, top_k) -> List[NodeWithScore]，可直接放进
query engine 的 node_postprocessors，也可在 A/B 脚本里单独调用。
"""
import glob
import logging
import os
from functools import lru_cache
from typing import List, Optional

from llama_index.core.schema import NodeWithScore

from config.config import Config

logger = logging.getLogger("bio_lab_rerank")


def _resolve_hf_path(model_name: str) -> str:
    """把 HF repo id / 缓存目录 / 快照目录统一解析成本地快照路径（避免联网）。"""
    if os.path.isdir(model_name):
        snapshots = glob.glob(os.path.join(model_name, "snapshots", "*"))
        if snapshots:  # 传的是 HF 缓存目录 models--xxx
            return snapshots[0]
        return model_name
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    safe = "models--" + model_name.replace("/", "--")
    snaps = glob.glob(os.path.join(hub, safe, "snapshots", "*"))
    return snaps[0] if snaps else model_name


def _resolve_jev_path(model_name: str) -> str:
    if Config.JEV_LOCAL_PATH and os.path.isdir(Config.JEV_LOCAL_PATH):
        return Config.JEV_LOCAL_PATH
    return _resolve_hf_path(model_name)


@lru_cache(maxsize=1)
def _load_jev():
    """懒加载 Jev 决策模型（进程内复用）。"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    path = _resolve_jev_path(Config.JEV_MODEL)
    logger.info("加载 Jev 决策模型: %s (device=%s)", path, Config.JEV_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map=Config.JEV_DEVICE,
        dtype=torch.float16 if str(Config.JEV_DEVICE).startswith("cuda") else torch.float32,
        trust_remote_code=True,
    ).eval()
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[-1]
    no_id = tokenizer.encode("No", add_special_tokens=False)[-1]
    return model, tokenizer, yes_id, no_id


@lru_cache(maxsize=1)
def _load_bge():
    from sentence_transformers import CrossEncoder

    logger.info("加载 BGE 重排模型: %s (device=%s)", Config.RERANK_MODEL, Config.RERANK_DEVICE)
    return CrossEncoder(Config.RERANK_MODEL, device=Config.RERANK_DEVICE)


@lru_cache(maxsize=1)
def _load_onnx(model_name: str):
    """加载 ONNX int8 重排模型（CPU）。"""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    path = _resolve_hf_path(model_name)
    onnx_path = os.path.join(path, "model.onnx")
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"未找到 ONNX 模型: {onnx_path}")
    logger.info("加载 ONNX int8 重排模型: %s", onnx_path)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(path)
    return session, tokenizer


class BaseReranker:
    name = "base"

    def rerank_nodes(self, query: str, nodes: List[NodeWithScore], top_k: int) -> List[NodeWithScore]:
        raise NotImplementedError


class BgeReranker(BaseReranker):
    name = "bge"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or Config.RERANK_MODEL

    def rerank_nodes(self, query: str, nodes: List[NodeWithScore], top_k: int) -> List[NodeWithScore]:
        if not nodes:
            return []
        model = _load_bge()
        pairs = [[query, n.node.get_content()] for n in nodes]
        scores = model.predict(pairs)
        ranked = sorted(zip(nodes, scores), key=lambda x: float(x[1]), reverse=True)
        return [NodeWithScore(node=n.node, score=float(s)) for n, s in ranked[:top_k]]


class OnnxBgeReranker(BaseReranker):
    """bge-reranker-v2-m3 的 ONNX int8 版（CPU/AVX2）。

    逐条打分：CPU cross-encoder 是内存带宽瓶颈，批量 padding 会撑大 attention
    矩阵、反而更慢（Bio_rag_haystack 的实测结论）。
    """

    name = "bge-onnx"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or Config.RERANK_ONNX_MODEL

    def rerank_nodes(self, query: str, nodes: List[NodeWithScore], top_k: int) -> List[NodeWithScore]:
        if not nodes:
            return []
        import numpy as np

        session, tokenizer = _load_onnx(self.model_name)
        input_names = {i.name for i in session.get_inputs()}
        scores = []
        for n in nodes:
            enc = tokenizer(
                [[query, n.node.get_content()]],
                padding=True,
                truncation=True,
                max_length=Config.RERANK_MAX_LENGTH,
                return_tensors="np",
            )
            feed = {k: v.astype(np.int64) for k, v in enc.items() if k in input_names}
            logit = float(np.asarray(session.run(None, feed)[0]).reshape(-1)[0])
            scores.append(1.0 / (1.0 + np.exp(-logit)))
        ranked = sorted(zip(nodes, scores), key=lambda x: float(x[1]), reverse=True)
        return [NodeWithScore(node=n.node, score=float(s)) for n, s in ranked[:top_k]]


class JevReranker(BaseReranker):
    """Jev 决策模型（MiniCPM-1B，Yes/No logits）。

    定位：**粗筛/噪声门控（Noise Gate / ContextBudgetFilter）**，不是 Cross-Encoder 的
    替代品。它擅长宏观相关性粗分与硬截断（砍掉明显噪声、压上下文预算），但微观分辨
    与保序弱于 bge-reranker。实测（困难集）hit@1/ MRR 明显低于 bge，故：
      - 不要作为默认精排；
      - 推荐作为 bge 的前置预过滤（见 CascadeReranker），或 k=30~50 时的上下文预算控制。
    """

    name = "jev"

    def __init__(self, model_name: Optional[str] = None, threshold: Optional[float] = None):
        self.model_name = model_name or Config.JEV_MODEL
        self.threshold = Config.JEV_THRESHOLD if threshold is None else threshold

    def score_prompts(self, prompts: List[str]) -> List[float]:
        """对任意 Yes/No prompt 批量取 Yes 概率（单次前向，无生成）。"""
        import torch

        if not prompts:
            return []
        model, tokenizer, yes_id, no_id = _load_jev()
        scores: List[float] = []
        batch_size = max(1, Config.JEV_BATCH_SIZE)
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=Config.JEV_MAX_LENGTH,
                return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                logits = model(**inputs).logits[:, -1, :]
                pair = torch.stack([logits[:, yes_id], logits[:, no_id]], dim=-1)
                probs = torch.softmax(pair, dim=-1)[:, 0]
            scores.extend(probs.float().cpu().tolist())
        return scores

    def score(self, query: str, texts: List[str]) -> List[float]:
        if not texts:
            return []
        prompts = [
            "任务：判断参考文档是否提供了回答用户问题所需的核心事实依据。\n"
            f"用户问题：{query}\n"
            f"参考文档：\n{t.strip()}\n\n"
            "判断：如果参考文档能回答该问题，回答 Yes；如果无关或不能回答，回答 No。\n"
            "答案(Yes/No)："
            for t in texts
        ]
        return self.score_prompts(prompts)

    def rerank_nodes(self, query: str, nodes: List[NodeWithScore], top_k: int) -> List[NodeWithScore]:
        kept, _ = self.rank_with_stats(query, nodes, top_k)
        return kept

    def rank_with_stats(self, query: str, nodes: List[NodeWithScore], top_k: int):
        """返回 (top_k 节点, 阈值过滤后保留数)。保留数用于衡量噪声剔除。"""
        if not nodes:
            return [], 0
        scores = self.score(query, [n.node.get_content() for n in nodes])
        scored = [NodeWithScore(node=n.node, score=float(s)) for n, s in zip(nodes, scores)]
        scored.sort(key=lambda x: x.score, reverse=True)

        # 相对自适应阈值 + 绝对阈值（指南 Case 1）
        max_s = scored[0].score if scored else 0.0
        effective = max(self.threshold, max_s * 0.45)
        kept = [n for n in scored if n.score >= effective]
        if not kept:
            kept = scored[:1]
        return kept[:top_k], len(kept)


class JevNoiseGate(BaseReranker):
    """高召回布尔噪声门控（Jev Noul，**不重排**）。

    与 `JevReranker` 的区别：
      - 只做「明显不相关」的硬截断，保留的候选**沿用输入原序**（如 RRF 序），
        不采用 Jev 的分数排序，从而不引入它的排序误差；
      - 阈值取得低（高召回），并有 `min_keep` 兜底，尽量避免把 gold 砍掉。

    适配：作为 bge 精排的前置预过滤；或 k 很大时单独用于上下文预算控制。
    """

    name = "jev-gate"

    def __init__(self, threshold: Optional[float] = None, relative: Optional[float] = None,
                 min_keep: Optional[int] = None):
        self._scorer = JevReranker()
        self.threshold = Config.JEV_GATE_THRESHOLD if threshold is None else threshold
        self.relative = Config.JEV_GATE_RELATIVE if relative is None else relative
        self.min_keep = Config.JEV_GATE_MIN_KEEP if min_keep is None else min_keep

    def gate(self, query: str, nodes: List[NodeWithScore]):
        """返回 (保序保留的节点, 保留数)。"""
        if not nodes:
            return [], 0
        scores = self._scorer.score(query, [n.node.get_content() for n in nodes])
        max_s = max(scores) if scores else 0.0
        cutoff = max(self.threshold, max_s * self.relative)
        kept = [n for n, s in zip(nodes, scores) if s >= cutoff]
        if len(kept) < self.min_keep:  # 兜底：至少保留原序前 min_keep 个
            kept = nodes[: self.min_keep]
        return kept, len(kept)

    def rerank_nodes(self, query: str, nodes: List[NodeWithScore], top_k: int) -> List[NodeWithScore]:
        kept, _ = self.rank_with_stats(query, nodes, top_k)
        return kept

    def rank_with_stats(self, query: str, nodes: List[NodeWithScore], top_k: int):
        kept, count = self.gate(query, nodes)
        return kept[:top_k], count


class CascadeReranker(BaseReranker):
    """两阶段级联：Jev 粗筛/噪声门控 -> bge 精排。

    目的：先用便宜的 Jev 把候选从 20 砍到 gate_top_k（省上下文/时间），
    再由 bge 在小候选集上保序出最终 Top-N。

    注意：级联的召回上限 = Jev 门控保留 gold 的能力。若 Jev 把 gold 砍掉，
    bge 再强也救不回来，因此必须用评测验证，不能假设「精度不打折」。
    """

    name = "cascade"

    def __init__(self, gate: Optional[BaseReranker] = None, fine: Optional[BaseReranker] = None,
                 gate_top_k: Optional[int] = None):
        self.gate = gate or build_reranker(Config.RERANK_CASCADE_GATE)
        self.fine = fine or build_reranker(Config.RERANK_CASCADE_FINE)
        self.gate_top_k = gate_top_k or Config.RERANK_CASCADE_GATE_TOP_K

    def rerank_nodes(self, query: str, nodes: List[NodeWithScore], top_k: int) -> List[NodeWithScore]:
        kept, _ = self.rank_with_stats(query, nodes, top_k)
        return kept

    def rank_with_stats(self, query: str, nodes: List[NodeWithScore], top_k: int):
        """返回 (top_k 节点, 门控后保留数)。"""
        if not nodes:
            return [], 0
        if hasattr(self.gate, "gate"):
            # 高召回门控：保留全部幸存者（不按 gate_top_k 截断），再交精排
            gated, gate_kept = self.gate.gate(query, nodes)
        elif hasattr(self.gate, "rank_with_stats"):
            gated, gate_kept = self.gate.rank_with_stats(query, nodes, self.gate_top_k)
        else:
            gated = self.gate.rerank_nodes(query, nodes, self.gate_top_k)
            gate_kept = len(gated)
        return self.fine.rerank_nodes(query, gated, top_k), gate_kept


_RERANKERS = {
    "bge": BgeReranker,
    "bge-onnx": OnnxBgeReranker,
    "onnx": OnnxBgeReranker,
    "jev": JevReranker,
    "jev-gate": JevNoiseGate,
    "gate": JevNoiseGate,
    "cascade": CascadeReranker,
}


def build_reranker(name: Optional[str] = None) -> Optional[BaseReranker]:
    name = (name or Config.RERANK_PROVIDER or "none").lower()
    if name in ("none", "", "off", "false"):
        return None
    if name not in _RERANKERS:
        raise ValueError(f"未知的重排器: {name}（可选 bge / bge-onnx / jev / jev-gate / cascade / none）")
    return _RERANKERS[name]()


class RerankerPostprocessor:
    """把 BaseReranker 适配成 llama_index 的 node_postprocessor。"""

    def __new__(cls, reranker: BaseReranker, top_n: Optional[int] = None):
        from llama_index.core.postprocessor.types import BaseNodePostprocessor
        from llama_index.core.schema import NodeWithScore

        top_n = top_n or Config.RERANK_TOP_N

        class _Adapter(BaseNodePostprocessor):
            def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle=None):
                query = query_bundle.query_str if query_bundle else ""
                candidates = nodes[: Config.RERANK_MAX_CANDIDATES]
                return reranker.rerank_nodes(query, candidates, top_n)

        return _Adapter()


class TopKPostprocessor:
    """不重排时，把检索候选截断到 top_k（避免把 fetch_k 个节点全喂给 LLM）。"""

    def __new__(cls, top_k: int):
        from llama_index.core.postprocessor.types import BaseNodePostprocessor
        from llama_index.core.schema import NodeWithScore

        class _Adapter(BaseNodePostprocessor):
            def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle=None):
                return nodes[:top_k]

        return _Adapter()


def build_node_postprocessor(name: Optional[str] = None, top_n: Optional[int] = None):
    reranker = build_reranker(name)
    if reranker is None:
        return None
    return RerankerPostprocessor(reranker, top_n)
