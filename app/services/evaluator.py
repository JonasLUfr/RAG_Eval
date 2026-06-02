from __future__ import annotations

import json
import logging
import re
from statistics import mean
from typing import Any

from app.core.config import AppConfig
from app.models import EvalResult, EvalSample, ScoreItem, SystemResponse
from app.models.schemas import FAILURE_LABELS
from app.services.executor import BatchExecutor
from app.services.llm_client import OpenAICompatibleClient
from app.services.metric_metadata import filter_scores_for_available_annotations, recompute_normalized_score
from app.services.rank_metrics import compute_rank_metrics, compute_strict_rank_metrics


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


# 仅答案模式：被测系统未返回 retrieved_contexts/citations 时，
# 只评估证据无关的 3 个答案侧指标，避免把"无证据"误判成"低忠实性/高幻觉"。
ANSWER_ONLY_METRICS = ["correctness", "relevance", "completeness"]


def is_answer_only(response: SystemResponse) -> bool:
    """被测系统未提供任何检索证据（既无上下文也无引用）。

    判定为「真实非空」的标准：列表中至少存在一个去除首尾空白后非空的字符串。
    这样可正确识别 importer 产生的 [""]、[" "] 等"空内容包装"——
    它们在语义上等价于无证据，但 list 本身 truthy，简单的 `not list` 会漏判。
    """
    def _has_real_content(items) -> bool:
        if not items:
            return False
        return any(str(x).strip() for x in items)

    return not (_has_real_content(response.retrieved_contexts) or _has_real_content(response.citations))


METRIC_USER_INFO: dict[str, tuple[str, str]] = {
    "correctness": ("正确性", "回答与参考答案是否一致"),
    "relevance": ("相关性", "回答是否直接回答了问题"),
    "faithfulness": ("忠实性", "回答是否被检索上下文支持"),
    "completeness": ("完整性", "关键要点是否覆盖完整"),
    "hallucination_risk": ("幻觉风险", "无依据内容风险，越低越好"),
    "hit_rate": ("检索命中率", "检索结果是否命中关键证据"),
    "context_relevance": ("上下文相关性", "检索上下文与问题是否相关"),
    "context_precision": ("上下文精确率", "检索内容里有用内容比例"),
    "context_recall": ("上下文召回率", "该找回的证据找回了多少"),
    "evidence_coverage": ("证据覆盖率", "回答主张是否有证据覆盖"),
}


METRIC_DETAILED_INFO: dict[str, dict[str, Any]] = {
    "correctness": {
        "label": "正确性",
        "high_is_good": True,
        "meaning": "最终答案与参考答案/事实标准的一致程度。",
        "low_risk": "容易出现结论错误、数值错误、实体错误。",
        "improve": "优先检查知识源准确性、字段映射、答案后处理规则。",
    },
    "relevance": {
        "label": "相关性",
        "high_is_good": True,
        "meaning": "回答是否围绕用户问题，不偏题、不答非所问。",
        "low_risk": "回答冗长但不切题，用户感知质量下降。",
        "improve": "优化检索 query、回答模板和问题重写策略。",
    },
    "faithfulness": {
        "label": "忠实性",
        "high_is_good": True,
        "meaning": "回答中的主张是否可以被检索上下文或引用支持。",
        "low_risk": "可能在“看起来合理”但证据不足的情况下编造结论。",
        "improve": "增加证据约束、答案生成时强制引用支持片段。",
    },
    "completeness": {
        "label": "完整性",
        "high_is_good": True,
        "meaning": "回答是否覆盖问题所需的关键要点。",
        "low_risk": "只回答部分信息，遗漏条件、范围或限制。",
        "improve": "加补全检查（must-have slots）和多步骤回答模板。",
    },
    "hallucination_risk": {
        "label": "幻觉风险",
        "high_is_good": False,
        "meaning": "回答中无证据支撑或臆测内容的风险，越低越好。",
        "low_risk": "高值表示风险高，容易触发不可信回答。",
        "improve": "引入拒答策略、证据门控、低置信度回退。",
    },
    "hit_rate": {
        "label": "检索命中率",
        "high_is_good": True,
        "meaning": "是否检索到预期证据或关键上下文。",
        "low_risk": "没命中核心证据，后续生成再强也难答对。",
        "improve": "调整召回策略、索引粒度、query 构造和过滤规则。",
    },
    "context_relevance": {
        "label": "上下文相关性",
        "high_is_good": True,
        "meaning": "召回内容与问题本身的相关程度。",
        "low_risk": "检索结果噪声高，模型容易被干扰。",
        "improve": "加强重排（rerank）与语义过滤。",
    },
    "context_precision": {
        "label": "上下文精确率",
        "high_is_good": True,
        "meaning": "召回内容中有价值信息占比。",
        "low_risk": "冗余内容多，增加回答偏航概率和成本。",
        "improve": "缩短 chunk、优化 top-k、加强去噪。",
    },
    "context_recall": {
        "label": "上下文召回率",
        "high_is_good": True,
        "meaning": "应召回的关键证据是否被覆盖。",
        "low_risk": "关键证据缺失，导致“答不全/答错”。",
        "improve": "扩大召回范围、增加多路检索、补充索引覆盖。",
    },
    "evidence_coverage": {
        "label": "证据覆盖率",
        "high_is_good": True,
        "meaning": "答案关键主张是否都能在证据中找到对应支撑。",
        "low_risk": "主张与证据脱节，可信度不足。",
        "improve": "答案生成时逐条主张对齐证据并校验。",
    },
}


METRIC_MODE_EXPLANATIONS: dict[str, dict[str, str]] = {
    "rule": {
        "correctness": "Jaccard(答案, 参考答案)，token = 单字符 ∪ 相邻 bigram。",
        "relevance": "max(Jaccard(答案,问题), Jaccard(答案,参考答案))。",
        "faithfulness": "Jaccard(答案, 检索上下文+引用拼接)。",
        "completeness": "Jaccard(答案, 参考答案) × 1.15（略放宽）。",
        "hallucination_risk": "1 − Jaccard(答案, 上下文)，越大风险越高。",
        "hit_rate": "Jaccard(预期证据, 上下文) ≥ 0.12 → 1.0，否则 0。",
        "context_relevance": "Jaccard(问题, 上下文)。",
        "context_precision": "上下文中与「问题+参考答案」Jaccard ≥ 0.08 的片段比例。",
        "context_recall": "Jaccard(预期证据, 上下文)；缺证据时退回到答案-上下文 overlap。",
        "evidence_coverage": "max(证据 overlap, 答案-上下文 overlap)。",
    },
    "llm_judge": {
        "correctness": "单次 LLM 调用，10 个分数一并返回（同一 prompt）。",
        "relevance": "同上，由裁判 LLM 在同一 JSON 中给出。",
        "faithfulness": "同上。",
        "completeness": "同上。",
        "hallucination_risk": "同上（裁判直接给风险分，越低越好）。",
        "hit_rate": "同上。",
        "context_relevance": "同上。",
        "context_precision": "同上。",
        "context_recall": "同上。",
        "evidence_coverage": "同上。",
    },
    "ragas": {
        "correctness": "独立 LLM 调用：判定答案与参考答案语义一致性。",
        "relevance": "独立 LLM 调用：答案是否对问题作答。",
        "faithfulness": "独立 LLM 调用：答案主张是否被上下文支持。",
        "completeness": "独立 LLM 调用：是否覆盖参考答案的关键要点。",
        "hallucination_risk": "独立 LLM 调用：识别无证据/编造的部分。",
        "hit_rate": "独立 LLM 调用：检索是否命中预期证据。",
        "context_relevance": "独立 LLM 调用：上下文与问题的相关性。",
        "context_precision": "复用 context_relevance 输出（不另发请求）。",
        "context_recall": "独立 LLM 调用：参考答案要点在上下文中的覆盖。",
        "evidence_coverage": "复用 context_recall 输出（不另发请求）。",
    },
    "embedding": {
        "correctness": "cosine(embed(答案), embed(参考答案))。",
        "relevance": "max(cosine(答案,问题), cosine(答案,参考)×0.7)。",
        "faithfulness": "cosine(答案, 上下文拼接)。",
        "completeness": "复用 correctness 的 cosine（语义覆盖近似）。",
        "hallucination_risk": "1 − cosine(答案, 上下文)。",
        "hit_rate": "上下文-问题 cosine 均值或证据 cosine > 0.4 → 1.0。",
        "context_relevance": "每段上下文与问题 cosine 的均值。",
        "context_precision": "上下文-问题 cosine > 0.4 的片段比例。",
        "context_recall": "max(cosine(参考, 每段上下文))。",
        "evidence_coverage": "max(cosine(预期证据, 每段上下文))；缺证据时退回 recall。",
    },
}


METRIC_COMBINATION_GUIDE: list[dict[str, str]] = [
    {
        "pattern": "正确性低 + 忠实性低",
        "meaning": "答案既不正确也缺证据，通常是检索与生成都存在问题。",
        "direction": "先修检索命中与上下文质量，再收紧生成约束。",
    },
    {
        "pattern": "正确性低 + 忠实性高",
        "meaning": "回答有引用但结论错，常见于证据理解错误或推理链错误。",
        "direction": "优化推理提示词、数值计算链、结构化后处理校验。",
    },
    {
        "pattern": "相关性高 + 完整性低",
        "meaning": "答到了点上，但覆盖不全。",
        "direction": "增加答案骨架模板和必答要点检查。",
    },
    {
        "pattern": "检索命中低 + 召回低",
        "meaning": "关键证据根本没被拉回来。",
        "direction": "优先优化检索召回（query、索引、top-k、多路召回）。",
    },
    {
        "pattern": "检索相关性高 + 精确率低",
        "meaning": "方向对了，但噪声太多。",
        "direction": "加强重排、减少噪声 chunk、调小返回窗口。",
    },
    {
        "pattern": "幻觉风险高 + 证据覆盖低",
        "meaning": "回答与证据脱节，存在明显编造风险。",
        "direction": "加入证据门控与拒答策略，强制主张-证据对齐。",
    },
]


SCORING_DEFINITIONS = {
    "correctness": "答案与参考答案/预期范围的一致性，0-1 越高越正确。",
    "relevance": "答案是否直接回应问题，0-1 越高越相关。",
    "faithfulness": "答案是否能被检索上下文或引用支持，0-1 越高越可信。",
    "completeness": "答案覆盖关键要点的充分程度，0-1 越高越完整。",
    "hallucination_risk": "幻觉风险，0-1 越高风险越大（汇总时反向计分）。",
    "hit_rate": "检索上下文/引用是否命中预期证据，命中越高分越高。",
    "context_relevance": "检索上下文与问题/参考答案的相关程度。",
    "context_precision": "返回上下文中有用片段的比例估计。",
    "context_recall": "预期证据被检索覆盖的比例估计。",
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
        eval_max_contexts: int | None = None,
        eval_context_max_chars: int | None = None,
    ):
        self.config = config
        self.use_llm_judge = use_llm_judge
        if eval_mode in ("embedding", "ragas", "llm_judge"):
            self._mode = eval_mode
        elif use_llm_judge:
            self._mode = "llm_judge"
        else:
            self._mode = "rule"
        self._embedding_model_name = embedding_model
        self._eval_max_contexts = eval_max_contexts if eval_max_contexts is not None else config.eval_max_contexts
        self._eval_context_max_chars = eval_context_max_chars if eval_context_max_chars is not None else config.eval_context_max_chars
        self._sub_evaluator = None
        self.llm = OpenAICompatibleClient(
            config,
            model=ragas_model or config.judge_model,
            api_base=ragas_api_base or config.llm_api_base,
            api_key=ragas_api_key or config.llm_api_key,
        )
        workers = 1 if self._mode == "embedding" else config.max_workers
        self.executor = BatchExecutor[tuple[EvalSample | None, SystemResponse], EvalResult](
            max_workers=workers,
            retry_times=config.retry_times,
        )

    def evaluate_batch(
        self,
        responses: list[SystemResponse],
        samples_by_id: dict[str, EvalSample] | None = None,
        progress_callback: callable | None = None,
    ) -> list[EvalResult]:
        import time
        items = [(samples_by_id.get(r.question_id) if samples_by_id else None, r) for r in responses]
        logger.info("eval_batch_start mode=%s samples=%d", self._mode, len(items))
        t0 = time.monotonic()
        try:
            results = self.executor.run(items, self.evaluate_one, progress_callback)
        except Exception:
            logger.exception("eval_batch_failed mode=%s samples=%d", self._mode, len(items))
            raise

        # 统一注入 rank-aware 检索指标（MRR / Recall@k / Precision@k）。
        # 与具体 evaluator 解耦，对 4 种模式（rule / embedding / ragas / llm_judge）一视同仁。
        # 仅当样本具备 expected_evidence 且至少一条上下文（retrieved_contexts 或 citations）时才注入。
        # contexts 拼接顺序与 ragas_evaluator / 规则评分保持一致：retrieved_contexts 在前，citations 在后。
        responses_by_id = {r.response_id: r for r in responses}
        for result in results:
            response = responses_by_id.get(result.response_id)
            sample = samples_by_id.get(result.question_id) if samples_by_id else None
            if response is None or sample is None:
                continue
            evidence = sample.expected_evidence or ""
            contexts = [str(c) for c in (response.retrieved_contexts or []) + (response.citations or [])]
            if evidence.strip() and contexts:
                rank_scores = compute_rank_metrics(contexts, evidence)
                if rank_scores:
                    result.scores.update(rank_scores)
            strict_rank_items = [str(c) for c in (response.citations or response.retrieved_contexts or [])]
            strict_scores = compute_strict_rank_metrics(strict_rank_items, sample.relevant_context_ids)
            if strict_scores:
                result.scores.update(strict_scores)
            filtered_scores, skipped_metrics = filter_scores_for_available_annotations(result.scores, sample)
            if skipped_metrics:
                result.scores = filtered_scores
                result.normalized_score = recompute_normalized_score(filtered_scores)
                skipped_text = ", ".join(skipped_metrics)
                result.judge_reason = (
                    f"{result.judge_reason} "
                    f"[标注依赖不足：{skipped_text} 暂无法准确评分，未计入综合分。]"
                ).strip()
                if not filtered_scores:
                    result.evaluation_status = "insufficient_annotations"

        elapsed = time.monotonic() - t0
        failed = sum(1 for r in results if not r.scores)
        logger.info(
            "eval_batch_done mode=%s samples=%d failed=%d elapsed_sec=%.1f",
            self._mode, len(results), failed, elapsed,
        )
        return results

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

                self._sub_evaluator = RagasEvaluator(
                    self.config,
                    self.llm,
                    max_contexts=self._eval_max_contexts,
                    context_max_chars=self._eval_context_max_chars,
                )
            return self._sub_evaluator.evaluate_one(item)

        sample, response = item
        if self._mode == "llm_judge" and self.llm.is_configured:
            try:
                return self._evaluate_with_llm(sample, response)
            except Exception as exc:  # noqa: BLE001
                logger.exception("LLM 评分失败，降级规则评分 question_id=%s error=%s", response.question_id, exc)
        return self._evaluate_with_rules(sample, response)

    def _evaluate_with_llm(self, sample: EvalSample | None, response: SystemResponse) -> EvalResult:
        if is_answer_only(response):
            return self._evaluate_with_llm_answer_only(sample, response)
        n = self._eval_max_contexts
        c = self._eval_context_max_chars
        trimmed_contexts = [ctx[:c] for ctx in response.retrieved_contexts[:n]]
        trimmed_citations = [cit[:c] for cit in response.citations[:n]]
        system_prompt = "你是严格的中文或英文的 RAG 系统评测专家裁判。只输出 JSON，不要输出解释。"
        user_prompt = f"""
请对系统回答进行评分，所有分数范围为 0 到 1。
评分定义：
{json.dumps(SCORING_DEFINITIONS, ensure_ascii=False, indent=2)}

问题：{response.question}
参考答案：{response.reference_answer or (sample.reference_answer if sample else "")}
预期证据：{sample.expected_evidence if sample else ""}
系统答案：{response.answer}
检索上下文：{trimmed_contexts}
引用：{trimmed_citations}

输出 JSON：
{{
  "scores": {{
    "correctness": 0.0, "relevance": 0.0, "faithfulness": 0.0, "completeness": 0.0,
    "hallucination_risk": 0.0, "hit_rate": 0.0, "context_relevance": 0.0,
    "context_precision": 0.0, "context_recall": 0.0, "evidence_coverage": 0.0
  }},
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

    def _evaluate_with_llm_answer_only(self, sample: EvalSample | None, response: SystemResponse) -> EvalResult:
        """仅答案模式：被测系统不返回检索上下文，只评估 3 个证据无关的指标。"""
        reference = response.reference_answer or (sample.reference_answer if sample else "")
        system_prompt = "你是严格的 RAG 系统评测专家裁判。只输出 JSON，不要输出解释。"
        user_prompt = f"""
被测系统未提供检索上下文，请仅基于问题、参考答案和系统答案，对以下 3 个指标打分（0-1）：
- correctness：与参考答案的事实一致性
- relevance：是否直接回应问题
- completeness：是否覆盖参考答案的关键要点

问题：{response.question}
参考答案：{reference}
系统答案：{response.answer}

输出 JSON：
{{
  "scores": {{"correctness": 0.0, "relevance": 0.0, "completeness": 0.0}},
  "judge_reason": "中文理由",
  "failure_labels": ["wrong_answer"]
}}
"""
        data = self.llm.chat_json(system_prompt, user_prompt, temperature=0.0)
        raw = data.get("scores", {}) or {}
        scores = {
            name: ScoreItem(
                raw_score=float(raw.get(name, 0.0)),
                normalized_score=self._clamp(float(raw.get(name, 0.0))),
                reason=data.get("judge_reason", ""),
            )
            for name in ANSWER_ONLY_METRICS
            if name in raw
        }
        reason = "[仅答案模式] " + data.get("judge_reason", "未提供检索上下文，仅评估答案与参考答案对齐。")
        return self._build_result(response, scores, reason, data.get("failure_labels", []))

    def _evaluate_with_rules(self, sample: EvalSample | None, response: SystemResponse) -> EvalResult:
        reference = response.reference_answer or (sample.reference_answer if sample else "")
        expected_evidence = sample.expected_evidence if sample else ""
        contexts_text = "\n".join(response.retrieved_contexts + response.citations)
        answer = response.answer or ""

        overlap_ref = self._overlap(answer, reference) if reference else 0.5
        overlap_question = self._overlap(answer, response.question)
        success_score = 1.0 if response.success and answer.strip() else 0.0

        if is_answer_only(response):
            scores = {
                "correctness": self._score(overlap_ref * success_score, "与参考答案重合度估计。"),
                "relevance": self._score(max(overlap_question, overlap_ref) * success_score, "回答与问题相关性估计。"),
                "completeness": self._score(min(1.0, overlap_ref * 1.15) * success_score, "关键要点覆盖程度估计。"),
            }
            labels = self._infer_failure_labels(response, scores)
            reason = "[仅答案模式] 规则评分：被测系统未提供检索上下文，仅评估答案与参考答案的对齐程度。"
            return self._build_result(response, scores, reason, labels)

        overlap_context = self._overlap(answer, contexts_text)
        evidence_overlap = self._overlap(expected_evidence, contexts_text) if expected_evidence else 0.0
        answer_has_evidence = overlap_context >= 0.18

        scores = {
            "correctness": self._score(overlap_ref * success_score, "与参考答案重合度估计。"),
            "relevance": self._score(max(overlap_question, overlap_ref) * success_score, "回答与问题相关性估计。"),
            "faithfulness": self._score(overlap_context if contexts_text else overlap_ref * 0.8, "是否被上下文支持。"),
            "completeness": self._score(min(1.0, overlap_ref * 1.15) * success_score, "关键要点覆盖程度估计。"),
            "hallucination_risk": self._score(1 - (overlap_context if contexts_text else overlap_ref), "证据不足时风险更高。"),
            "hit_rate": self._score(1.0 if evidence_overlap >= 0.12 or answer_has_evidence else 0.0, "证据命中估计。"),
            "context_relevance": self._score(max(self._overlap(response.question, contexts_text), evidence_overlap) if contexts_text else 0.0, "上下文相关性。"),
            "context_precision": self._score(self._context_precision(response.retrieved_contexts, response.question, reference), "上下文精确率估计。"),
            "context_recall": self._score(evidence_overlap if expected_evidence else overlap_context, "上下文召回率估计。"),
            "evidence_coverage": self._score(max(evidence_overlap, overlap_context if contexts_text else 0.0), "证据覆盖估计。"),
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
            score_value = 1 - item.normalized_score if name == "hallucination_risk" else item.normalized_score
            aggregate_inputs.append(score_value)
        normalized = mean(aggregate_inputs) if aggregate_inputs else 0.0
        valid_labels = [x for x in failure_labels if x in FAILURE_LABELS]
        return EvalResult(
            question_id=response.question_id,
            response_id=response.response_id,
            scores=scores,
            normalized_score=round(normalized, 4),
            judge_reason=reason,
            judge_model=self.config.judge_model if self._mode == "llm_judge" and self.llm.is_configured else "rule-based-mvp",
            score_version=self.config.score_version,
            failure_labels=valid_labels,
        )

    def _infer_failure_labels(self, response: SystemResponse, scores: dict[str, ScoreItem]) -> list[str]:
        labels: list[str] = []
        if not response.success:
            return ["cannot_judge"]
        if "correctness" in scores and scores["correctness"].normalized_score < 0.35:
            labels.append("wrong_answer")
        if "completeness" in scores and scores["completeness"].normalized_score < 0.45:
            labels.append("incomplete_answer")
        if "faithfulness" in scores and scores["faithfulness"].normalized_score < 0.35:
            labels.append("unsupported_answer")
        if response.retrieved_contexts and "hit_rate" in scores and scores["hit_rate"].normalized_score < 0.5:
            labels.append("retrieval_issue")
        if response.retrieved_contexts and "evidence_coverage" in scores and scores["evidence_coverage"].normalized_score < 0.35:
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
