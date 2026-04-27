"""RAGAS-algorithm metrics implemented via LLM prompts (no ragas library required)."""
from __future__ import annotations

import json
import logging
from statistics import mean
from typing import TYPE_CHECKING

from app.models.schemas import EvalResult, EvalSample, ScoreItem, SystemResponse
from app.services.llm_client import OpenAICompatibleClient

if TYPE_CHECKING:
    from app.core.config import AppConfig

logger = logging.getLogger(__name__)

_SYS = "你是严格的中文 RAG 评测裁判。只输出 JSON，不要输出任何解释。"


class JudgeCallError(RuntimeError):
    """Raised when the LLM judge cannot produce usable scoring data."""


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _si(v: float, reason: str) -> ScoreItem:
    v = _clamp(v)
    return ScoreItem(raw_score=round(v, 4), normalized_score=round(v, 4), reason=reason)


class RagasEvaluator:
    """
    Implements RAGAS-style metrics via LLM prompts.

    Per-sample LLM call budget (~8 calls):
      faithfulness, correctness, relevance, completeness,
      context_precision, context_recall, context_relevance, evidence_coverage
    """

    def __init__(
        self,
        config: "AppConfig",
        llm: OpenAICompatibleClient,
        max_contexts: int = 5,
        context_max_chars: int = 500,
    ):
        self.config = config
        self.llm = llm
        self.max_contexts = max_contexts
        self.context_max_chars = context_max_chars

    def evaluate_one(self, item: tuple) -> EvalResult:
        sample, response = item
        reference = response.reference_answer or (sample.reference_answer if sample else "")
        expected_evidence = sample.expected_evidence if sample else ""
        answer = response.answer or ""
        question = response.question or ""
        contexts = response.retrieved_contexts + response.citations

        if not response.success or not answer.strip():
            return self._empty_result(response, "答案为空或请求失败。")

        from app.services.evaluator import is_answer_only
        answer_only = is_answer_only(response)
        if answer_only:
            contexts = []  # 排除 [""] 这类伪空列表，避免下游误用
        try:
            scores: dict[str, ScoreItem] = {}
            scores["correctness"] = self._correctness(question, answer, reference)
            scores["relevance"] = self._relevance(question, answer)
            scores["completeness"] = self._completeness(answer, reference)

            if not answer_only:
                scores["faithfulness"] = self._faithfulness(answer, contexts)
                faith = scores["faithfulness"].normalized_score
                scores["hallucination_risk"] = ScoreItem(
                    raw_score=round(1 - faith, 4),
                    normalized_score=round(1 - faith, 4),
                    reason="幻觉风险 = 1 − 忠实性得分。",
                )
                scores["context_precision"] = self._context_precision(question, reference, contexts)
                scores["context_recall"] = self._context_recall(reference, contexts)
                scores["context_relevance"] = self._context_relevance(question, contexts)
                scores["hit_rate"] = ScoreItem(
                    raw_score=scores["context_relevance"].raw_score,
                    normalized_score=scores["context_relevance"].normalized_score,
                    reason="上下文对问题的整体相关性（与上下文相关性一致）。",
                )
                scores["evidence_coverage"] = self._evidence_coverage(expected_evidence or reference, contexts)
        except JudgeCallError as exc:
            logger.warning("RAGAS 评分失败 question_id=%s error=%s", response.question_id, exc)
            return self._empty_result(response, f"RAGAS 评估器失败：{exc}", str(exc))

        agg = [
            (1 - scores["hallucination_risk"].normalized_score) if k == "hallucination_risk"
            else scores[k].normalized_score
            for k in scores
        ]
        normalized = mean(agg)
        labels = self._failure_labels(response, scores)

        judge_reason = "[仅答案模式] RAGAS 算法评分（仅答案侧 3 指标）" if answer_only else "RAGAS 算法评分（LLM 逐步推理）"
        return EvalResult(
            question_id=response.question_id,
            response_id=response.response_id,
            scores=scores,
            normalized_score=round(normalized, 4),
            judge_reason=judge_reason,
            judge_model=self.llm.model,
            score_version=self.config.score_version,
            failure_labels=labels,
        )

    # ── private helpers ──────────────────────────────────────────────────────

    def _call(self, prompt: str) -> dict:
        try:
            data = self.llm.chat_json(_SYS, prompt, temperature=0.0)
        except Exception as exc:
            raise JudgeCallError(str(exc)) from exc
        if not isinstance(data, dict):
            raise JudgeCallError("LLM 裁判返回的 JSON 不是对象。")
        return data

    def _faithfulness(self, answer: str, contexts: list[str]) -> ScoreItem:
        """Decompose answer into statements; verify each against contexts."""
        if not contexts:
            return _si(0.5, "无检索上下文，忠实性无法验证。")
        ctx = "\n---\n".join(c[:self.context_max_chars] for c in contexts[:self.max_contexts])
        data = self._call(f"""
将答案分解为独立的原子陈述，判断每条陈述是否可从检索上下文直接推断。
答案：{answer}
检索上下文：{ctx}
输出 JSON：{{"statements":["陈述1","陈述2"],"supported":[true,false],"reason":"简短说明"}}
""")
        stmts = data.get("statements", [])
        supported = data.get("supported", [])
        if not stmts:
            return _si(0.5, "无法解析答案陈述，使用默认值。")
        score = _clamp(sum(1 for s in supported if s) / len(stmts))
        return _si(score, data.get("reason", ""))

    def _correctness(self, question: str, answer: str, reference: str) -> ScoreItem:
        """TP/FP/FN counting, then F1."""
        if not reference:
            return _si(0.5, "无参考答案，无法评估正确性。")
        data = self._call(f"""
比较系统答案与参考答案，统计：
- tp：正确匹配的陈述数
- fp：系统答案中错误或多余的陈述数
- fn：参考答案中被遗漏的要点数
问题：{question}
参考答案：{reference}
系统答案：{answer}
输出 JSON：{{"tp":3,"fp":1,"fn":1,"reason":"简短说明"}}
""")
        tp = float(data.get("tp", 0))
        fp = float(data.get("fp", 0))
        fn = float(data.get("fn", 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = _clamp(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return _si(f1, data.get("reason", ""))

    def _relevance(self, question: str, answer: str) -> ScoreItem:
        data = self._call(f"""
评估回答对问题的直接相关性，打分 0-1（1=完全切题，0=完全不相关）。
问题：{question}
答案：{answer}
输出 JSON：{{"score":0.85,"reason":"简短说明"}}
""")
        return _si(float(data.get("score", 0.5)), data.get("reason", ""))

    def _completeness(self, answer: str, reference: str) -> ScoreItem:
        if not reference:
            return _si(0.5, "无参考答案。")
        data = self._call(f"""
评估系统答案覆盖参考答案关键信息点的完整程度，打分 0-1（1=全部覆盖，0=完全遗漏）。
参考答案：{reference}
系统答案：{answer}
输出 JSON：{{"score":0.8,"reason":"哪些要点被遗漏"}}
""")
        return _si(float(data.get("score", 0.5)), data.get("reason", ""))

    def _context_precision(self, question: str, reference: str, contexts: list[str]) -> ScoreItem:
        """Single LLM call to evaluate all contexts at once."""
        ctx_list = json.dumps([c[:self.context_max_chars] for c in contexts[:self.max_contexts]], ensure_ascii=False)
        data = self._call(f"""
对每个检索片段，判断它是否有助于回答问题（参考标准答案判断）。
问题：{question}
标准答案：{reference}
检索片段列表（按顺序）：{ctx_list}
输出 JSON：{{"results":[true,false,true],"reason":"简短说明"}}
""")
        results = data.get("results", [])
        if not results:
            return _si(0.5, "无法解析结果，使用默认值。")
        score = _clamp(sum(1 for r in results if r) / len(results))
        return _si(score, data.get("reason", ""))

    def _context_recall(self, reference: str, contexts: list[str]) -> ScoreItem:
        """Decompose reference into statements; verify each is attributable to contexts."""
        if not reference or not contexts:
            return _si(0.0, "缺少参考答案或上下文。")
        ctx = "\n---\n".join(c[:self.context_max_chars] for c in contexts[:self.max_contexts])
        data = self._call(f"""
将参考答案分解为独立的事实陈述，判断每条陈述是否可从检索上下文归因。
参考答案：{reference}
检索上下文：{ctx}
输出 JSON：{{"statements":["陈述1"],"attributed":[true],"reason":"简短说明"}}
""")
        stmts = data.get("statements", [])
        attributed = data.get("attributed", [])
        if not stmts:
            return _si(0.5, "无法解析参考答案陈述。")
        score = _clamp(sum(1 for a in attributed if a) / len(stmts))
        return _si(score, data.get("reason", ""))

    def _context_relevance(self, question: str, contexts: list[str]) -> ScoreItem:
        ctx_list = "\n".join(f"{i+1}. {c[:self.context_max_chars]}" for i, c in enumerate(contexts[:self.max_contexts]))
        data = self._call(f"""
评估检索上下文对回答以下问题的整体相关性，打分 0-1。
问题：{question}
检索片段：
{ctx_list}
输出 JSON：{{"score":0.8,"reason":"简短说明"}}
""")
        return _si(float(data.get("score", 0.5)), data.get("reason", ""))

    def _evidence_coverage(self, expected: str, contexts: list[str]) -> ScoreItem:
        if not expected or not contexts:
            return _si(0.0, "缺少期望证据或上下文。")
        ctx = "\n---\n".join(c[:self.context_max_chars] for c in contexts[:self.max_contexts])
        data = self._call(f"""
判断期望证据在检索上下文中的覆盖程度，打分 0-1。
期望证据：{expected}
检索上下文：{ctx}
输出 JSON：{{"score":0.9,"reason":"简短说明"}}
""")
        return _si(float(data.get("score", 0.0)), data.get("reason", ""))

    def _failure_labels(self, response: SystemResponse, scores: dict) -> list[str]:
        if not response.success:
            return ["cannot_judge"]
        labels = []
        if "correctness" in scores and scores["correctness"].normalized_score < 0.4:
            labels.append("wrong_answer")
        if "completeness" in scores and scores["completeness"].normalized_score < 0.4:
            labels.append("incomplete_answer")
        if "faithfulness" in scores and scores["faithfulness"].normalized_score < 0.4:
            labels.append("unsupported_answer")
        if response.retrieved_contexts and "hit_rate" in scores and scores["hit_rate"].normalized_score < 0.5:
            labels.append("retrieval_issue")
        if response.retrieved_contexts and "evidence_coverage" in scores and scores["evidence_coverage"].normalized_score < 0.4:
            labels.append("missing_evidence")
        return labels

    def _empty_result(self, response: SystemResponse, reason: str, evaluator_error: str = "") -> EvalResult:
        scores = {
            name: ScoreItem(raw_score=0.0, normalized_score=0.0, reason=reason)
            for name in [
                "correctness", "relevance", "faithfulness", "completeness", "hallucination_risk",
                "hit_rate", "context_relevance", "context_precision", "context_recall", "evidence_coverage",
            ]
        }
        return EvalResult(
            question_id=response.question_id,
            response_id=response.response_id,
            scores=scores,
            normalized_score=0.0,
            judge_reason=reason,
            judge_model=self.llm.model,
            score_version=self.config.score_version,
            failure_labels=["cannot_judge"],
            evaluation_status="judge_failed" if evaluator_error else "input_failed",
            evaluator_error=evaluator_error,
        )
