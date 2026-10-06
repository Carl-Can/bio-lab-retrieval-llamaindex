"""
生成后 grounding 核验（幻觉检测）。

把答案拆成逐条陈述（claim），用 Jev 的 Yes/No 原语**批量**判断每条是否被检索到的
context 支持；也可反向查「context 与答案是否矛盾」。

设计要点（对应 JEV 指南 Case 4/6）：
- 逐条一个 prompt、按 batch 并行前向（不是把多条拼进一个 prompt，否则只能拿到聚合判断）；
- 这是 guard，不是正确性保证：阈值需按语料校准；
- 1B 模型对生物医学细节/双重否定的 NLI 可靠性有限，务必用评测对照更强 verifier。
"""
import logging
import re
from typing import Dict, List, Optional

from config.config import Config

logger = logging.getLogger("bio_lab_grounding")

# 中文/英文句末 + 换行 + 列表项编号
_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*|\n+")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)、]|第[一二三四五六七八九十]+[、.)]?)\s*")


def split_claims(answer: str, min_len: int = 8) -> List[str]:
    """把答案拆成逐条陈述。"""
    claims = []
    for raw in _SPLIT.split(answer or ""):
        if not raw:
            continue
        text = _BULLET.sub("", raw).strip()
        if len(text) >= min_len:
            claims.append(text)
    return claims


class JevGroundingGuard:
    """用 Jev（MiniCPM Yes/No logits）做 grounding / 幻觉核验。

    ⚠️ 实测结论（本语料）：**不可用**。MiniCPM5-1B 的 NLI 太弱——
      - 短、单事实的前提能分开（支持 0.73 vs 矛盾 0.28），
      - 但真实检索块（多事实、100~300 token）分不开：矛盾/无关/未提及 与「支持」
        分数重叠（实测 pos_min 0.52 < neg_max 0.59，无阈值可切）；
      - 端到端 8 条：Jev pass_rate=0.0（全判为幻觉），9B judge pass_rate=1.0，**一致率 0**。
    保留此类是为了记录这套原语形状；生产 grounding 请用 `LlmGroundingJudge`。
    """

    def __init__(self, threshold: Optional[float] = None):
        # 延迟导入，避免无 GPU 时 import 就拉模型
        from core.rerankers import JevReranker

        self.scorer = JevReranker()
        self.threshold = Config.GROUNDING_THRESHOLD if threshold is None else threshold

    @staticmethod
    def _entail_prompt(context: str, claim: str) -> str:
        # claim-first 变体：实测区分度最好（优于「前提在前」「NLI 蕴含」等写法）
        return (
            f"陈述：{claim}\n"
            f"资料：{context}\n"
            "判断：资料是否明确支持该陈述？支持回答 Yes，不支持回答 No。\n"
            "答案(Yes/No)："
        )

    def verify(self, answer: str, contexts: List[str], question: Optional[str] = None) -> Dict:
        """返回 {passed, faithfulness, num_claims, flagged, claims:[{claim,score,supported}]}。"""
        claims = split_claims(answer)[: Config.GROUNDING_MAX_CLAIMS]
        if not claims:
            return {"passed": True, "faithfulness": 1.0, "num_claims": 0, "flagged": [], "claims": []}

        context = "\n\n".join(c for c in contexts if c)[: Config.GROUNDING_MAX_CONTEXT_CHARS]
        prompts = [self._entail_prompt(context, c) for c in claims]
        scores = self.scorer.score_prompts(prompts)

        per_claim = [
            {"claim": c, "score": round(float(s), 4), "supported": float(s) >= self.threshold}
            for c, s in zip(claims, scores)
        ]
        flagged = [c["claim"] for c in per_claim if not c["supported"]]
        faithfulness = sum(c["score"] for c in per_claim) / len(per_claim)
        return {
            "passed": not flagged,
            "faithfulness": round(faithfulness, 4),
            "num_claims": len(per_claim),
            "flagged": flagged,
            "claims": per_claim,
        }


class LlmGroundingJudge:
    """用 LLM-as-judge（`FaithfulnessEvaluator`，走当前 Settings.llm）做 grounding。

    实测（Qwen3.5-9B via vLLM）：8 条全部判为忠实，约 0.8s/次——比 1B 可靠得多，
    且本地 vLLM 下延迟可接受。作为 `GROUNDING_PROVIDER=llm`（默认）。
    """

    def __init__(self):
        from llama_index.core.evaluation import FaithfulnessEvaluator

        self.evaluator = FaithfulnessEvaluator()

    def verify(self, answer: str, contexts: List[str], question: Optional[str] = None) -> Dict:
        result = self.evaluator.evaluate(query=question or "", response=answer, contexts=contexts)
        passed = bool(result.passing)
        return {
            "passed": passed,
            "faithfulness": round(float(result.score or 0.0), 4),
            "num_claims": 0,
            "flagged": [] if passed else ["<LLM 判定存在未经支持的陈述>"],
            "claims": [],
            "feedback": result.feedback,
        }


def build_grounding_guard():
    """按 Config.GROUNDING_PROVIDER 构建：llm（默认，可靠） | jev（实验，实测不可用）。"""
    provider = (Config.GROUNDING_PROVIDER or "llm").lower()
    if provider == "jev":
        return JevGroundingGuard()
    return LlmGroundingJudge()
