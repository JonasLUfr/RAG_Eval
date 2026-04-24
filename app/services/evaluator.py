from __future__ import annotations

import json
import logging
import re
from statistics import mean

from app.core.config import AppConfig
from app.models import EvalResult, EvalSample, ScoreItem, SystemResponse
from app.models.schemas import FAILURE_LABELS
from app.services.executor import BatchExecutor, BatchProgress
from app.services.llm_client import OpenAICompatibleClient


logger = logging.getLogger(__name__)


ANSWER_METRICS = [
    "correctness",
    "relevance",
    "faithfulness",
    "completeness",
    "hallucination_risk",
]
RETRIEVAL_METRICS = [
    "hit_rate",
    "context_relevance",
    "context_precision",
    "context_recall",
    "evidence_coverage",
]


METRIC_USER_INFO: dict[str, tuple[str, str]] = {
    "correctness":        ("正确性",       "答案内容与标准答案的一致程度"),
    "relevance":          ("相关性",       "回答是否直接切中问题要点"),
    "faithfulness":       ("忠实性",       "答案是否有检索内容支撑，有没有捏造内容"),
    "completeness":       ("完整性",       "标准答案的关键信息是否都被覆盖"),
    "hallucination_risk": ("幻觉风险 ↓",   "答案中无依据内容的比例（越低越好）"),
    "hit_rate":           ("检索命中率",   "检索结果是否包含了回答所需的核心信息"),
    "context_relevance":  ("上下文相关性", "召回文档与问题的匹配程度"),
    "context_precision":  ("上下文精确率", "召回文档中真正有用的比例"),
    "context_recall":     ("上下文召回率", "标准答案的支撑内容有多少被成功检索到"),
    "evidence_coverage":  ("证据覆盖率",   "期望证据在检索结果中的覆盖程度"),
}

SCORING_DEFINITIONS = {
    "correctness": "答案与参考答案/预期范围的一致性，0-1 越高越正确。",
    "relevance": "答案是否直接回应问题，0-1 越高越相关。",
    "faithfulness": "答案是否能被检索上下文或引用支持，0-1 越高越可信。",
    "completeness": "答案覆盖参考答案关键点的充分程度，0-1 越高越完整。",
    "hallucination_risk": "幻觉风险，0-1 越高风险越大，汇总时会反向计分。",
    "hit_rate": "检索上下文/引用是否命中预期证据，命中为 1，否则 0。",
    "context_relevance": "检索上下文与问题/参考答案的相关性。",
    "context_precision": "返回上下文中有用片段的比例估计。",
    "context_recall": "预期证据被返回上下文覆盖的比例估计。",
    "evidence_coverage": "答案关键主张是否有证据覆盖。",
}


class EvaluationEngine:
    def __init__(
        self,
        config: AppConfig,
        use_llm_judge: bool = False,
        eval_mode: str = "rule",
        embedding_model: str | None = None,
        ragas_api_base: str | None = None,
        ragas_api_key: str | None = None,
        ragas_model: str | None = None,
    ):
        self.config = config
        self.use_llm_judge = use_llm_judge
        # eval_mode takes precedence; use_llm_judge kept for backward compat
        if eval_mode in ("embedding", "ragas"):
            self._mode = eval_mode
        elif use_llm_judge:
            self._mode = "llm_judge"
        else:
            self._mode = "rule"
        self._embedding_model_name = embedding_model
        self._sub_evaluator = None
        self.llm = OpenAICompatibleClient(
            config,
            model=ragas_model or config.judge_model,
            api_base=ragas_api_base or config.llm_api_base,
            api_key=ragas_api_key or config.llm_api_key,
        )
        # 嵌入模式强制单线程：避免多线程同时加载 PyTorch DLL 触发 Windows SAC 拦截
        _workers = 1 if self._mode == "embedding" else config.max_workers
        self.executor = BatchExecutor[tuple[EvalSample | None, SystemResponse], EvalResult](
            max_workers=_workers,
            retry_times=config.retry_times,
        )

    def evaluate_batch(
        self,
        responses: list[SystemResponse],
        samples_by_id: dict[str, EvalSample] | None = None,
        progress_callback: callable | None = None,
    ) -> list[EvalResult]:
        items = [(samples_by_id.get(r.question_id) if samples_by_id else None, r) for r in responses]
        return self.executor.run(items, self.evaluate_one, progress_callback)

    def evaluate_one(self, item: tuple[EvalSample | None, SystemResponse]) -> EvalResult:
        if self._mode == "embedding":
            if self._sub_evaluator is None:
                from app.services.embedding_evaluator import EmbeddingEvaluator
                self._sub_evaluator = EmbeddingEvaluator(
                    self.config,
                    self._embedding_model_name or "paraphrase-multilingual-MiniLM-L12-v2",
                )
            return self._sub_evaluator.evaluate_one(item)

        if self._mode == "ragas":
            if self._sub_evaluator is None:
                from app.services.ragas_evaluator import RagasEvaluator
                self._sub_evaluator = RagasEvaluator(self.config, self.llm)
            return self._sub_evaluator.evaluate_one(item)

        sample, response = item
        if self._mode == "llm_judge" and self.llm.is_configured:
            try:
                return self._evaluate_with_llm(sample, response)
            except Exception as exc:  # noqa: BLE001
                logger.exception("LLM 评分失败，降级为规则评分 question_id=%s error=%s", response.question_id, exc)
        return self._evaluate_with_rules(sample, response)

    def _evaluate_with_llm(self, sample: EvalSample | None, response: SystemResponse) -> EvalResult:
        system_prompt = "你是严格的中文 RAG 评测裁判。只输出 JSON，不要输出解释。"
        user_prompt = f"""
请对系统回答进行评分。所有分数范围为 0 到 1。
评分定义：
{json.dumps(SCORING_DEFINITIONS, ensure_ascii=False, indent=2)}

问题：{response.question}
参考答案：{response.reference_answer or (sample.reference_answer if sample else "")}
预期证据：{sample.expected_evidence if sample else ""}
系统答案：{response.answer}
检索上下文：{response.retrieved_contexts}
引用：{response.citations}

输出 JSON 格式：
{{
  "scores": {{"correctness": 0.0, "relevance": 0.0, "faithfulness": 0.0,
              "completeness": 0.0, "hallucination_risk": 0.0,
              "hit_rate": 0.0, "context_relevance": 0.0, "context_precision": 0.0,
              "context_recall": 0.0, "evidence_coverage": 0.0}},
  "judge_reason": "中文理由",
  "failure_labels": ["wrong_answer"]
}}
"""
        data = self.llm.chat_json(system_prompt, user_prompt, temperature=0.0)
        scores = {
            name: ScoreItem(
                raw_score=float(value),
                normalized_score=self._clamp(float(value)),
                reason=data.get("judge_reason", ""),
            )
            for name, value in data.get("scores", {}).items()
        }
        return self._build_result(response, scores, data.get("judge_reason", ""), data.get("failure_labels", []))

    def _evaluate_with_rules(self, sample: EvalSample | None, response: SystemResponse) -> EvalResult:
        reference = response.reference_answer or (sample.reference_answer if sample else "")
        expected_evidence = sample.expected_evidence if sample else ""
        contexts_text = "\n".join(response.retrieved_contexts + response.citations)
        answer = response.answer or ""

        overlap_ref = self._overlap(answer, reference) if reference else 0.5
        overlap_question = self._overlap(answer, response.question)
        overlap_context = self._overlap(answer, contexts_text) if contexts_text else 0.5
        evidence_overlap = self._overlap(expected_evidence, contexts_text) if expected_evidence and contexts_text else 0.0
        answer_has_evidence = bool(contexts_text and overlap_context >= 0.18)
        success_score = 1.0 if response.success and answer.strip() else 0.0

        scores = {
            "correctness": self._score(overlap_ref * success_score, "与参考答案的字符/词片段重合度估计。"),
            "relevance": self._score(max(overlap_question, overlap_ref) * success_score, "回答与问题及参考答案的相关性估计。"),
            "faithfulness": self._score(overlap_context if contexts_text else overlap_ref * 0.8, "有上下文时检查答案是否被上下文支持。"),
            "completeness": self._score(min(1.0, overlap_ref * 1.15) * success_score, "参考答案关键内容覆盖估计。"),
            "hallucination_risk": self._score(1 - (overlap_context if contexts_text else overlap_ref), "支持证据越少，幻觉风险越高。"),
            "hit_rate": self._score(1.0 if evidence_overlap >= 0.12 or answer_has_evidence else 0.0, "预期证据或答案内容是否被检索上下文命中。"),
            "context_relevance": self._score(max(self._overlap(response.question, contexts_text), evidence_overlap) if contexts_text else 0.0, "上下文与问题/预期证据的相关性。"),
            "context_precision": self._score(self._context_precision(response.retrieved_contexts, response.question, reference), "有用上下文比例估计。"),
            "context_recall": self._score(evidence_overlap if expected_evidence else overlap_context, "预期证据覆盖估计。"),
            "evidence_coverage": self._score(max(evidence_overlap, overlap_context if contexts_text else 0.0), "答案证据覆盖估计。"),
        }
        labels = self._infer_failure_labels(response, scores)
        reason = "规则评分：基于参考答案、问题、系统答案、检索上下文和预期证据的重合度估计。"
        return self._build_result(response, scores, reason, labels)

    def _build_result(
        self,
        response: SystemResponse,
        scores: dict[str, ScoreItem],
        reason: str,
        failure_labels: list[str],
    ) -> EvalResult:
        aggregate_inputs = []
        for name, item in scores.items():
            value = 1 - item.normalized_score if name == "hallucination_risk" else item.normalized_score
            aggregate_inputs.append(value)
        normalized = mean(aggregate_inputs) if aggregate_inputs else 0.0
        valid_labels = [x for x in failure_labels if x in FAILURE_LABELS]
        return EvalResult(
            question_id=response.question_id,
            response_id=response.response_id,
            scores=scores,
            normalized_score=round(normalized, 4),
            judge_reason=reason,
            judge_model=self.config.judge_model if self.use_llm_judge and self.llm.is_configured else "rule-based-mvp",
            score_version=self.config.score_version,
            failure_labels=valid_labels,
        )

    def _infer_failure_labels(self, response: SystemResponse, scores: dict[str, ScoreItem]) -> list[str]:
        labels: list[str] = []
        if not response.success:
            return ["cannot_judge"]
        if scores["correctness"].normalized_score < 0.35:
            labels.append("wrong_answer")
        if scores["completeness"].normalized_score < 0.45:
            labels.append("incomplete_answer")
        if scores["faithfulness"].normalized_score < 0.35:
            labels.append("unsupported_answer")
        if response.retrieved_contexts and scores["hit_rate"].normalized_score < 0.5:
            labels.append("retrieval_issue")
        if response.retrieved_contexts and scores["evidence_coverage"].normalized_score < 0.35:
            labels.append("missing_evidence")
        return labels

    @staticmethod
    def _score(value: float, reason: str) -> ScoreItem:
        value = EvaluationEngine._clamp(value)
        return ScoreItem(raw_score=round(value, 4), normalized_score=round(value, 4), reason=reason)

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _tokens(text: str) -> set[str]:
        text = re.sub(r"\s+", "", str(text).lower())
        if not text:
            return set()
        chars = set(text)
        bigrams = {text[i : i + 2] for i in range(max(0, len(text) - 1))}
        return chars | bigrams

    @classmethod
    def _overlap(cls, left: str, right: str) -> float:
        a = cls._tokens(left)
        b = cls._tokens(right)
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @classmethod
    def _context_precision(cls, contexts: list[str], question: str, reference: str) -> float:
        if not contexts:
            return 0.0
        useful = 0
        target = f"{question}\n{reference}"
        for context in contexts:
            if cls._overlap(context, target) >= 0.08:
                useful += 1
        return useful / len(contexts)

