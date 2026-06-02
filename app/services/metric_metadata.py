from __future__ import annotations

from statistics import mean

from app.models import EvalSample, ScoreItem


GOLD_LABEL_STATUS_OPTIONS = [
    "unverified",
    "reference_verified",
    "evidence_verified",
    "full_relevance_verified",
]

DEPENDENCY_LABELS = {
    "no_gold_needed": "无需人工金标",
    "needs_reference_answer": "需要参考答案",
    "needs_expected_evidence": "需要预期证据",
    "needs_full_relevance_labels": "需要完整相关标注",
}

DEPENDENCY_DESCRIPTIONS = {
    "no_gold_needed": "依赖问题、回答、真实检索上下文或系统运行数据，不要求人工标准答案。",
    "needs_reference_answer": "需要人工审核过的 reference_answer，否则只能作为弱评估。",
    "needs_expected_evidence": "需要人工确认 expected_evidence，否则只是弱证据匹配。",
    "needs_full_relevance_labels": "需要人工标注 relevant_context_ids / 全部相关 chunk，才能作为严格 IR 指标。",
}

METRIC_DEPENDENCIES = {
    "success_rate": "no_gold_needed",
    "latency_ms": "no_gold_needed",
    "token_usage": "no_gold_needed",
    "estimated_cost": "no_gold_needed",
    "relevance": "no_gold_needed",
    "faithfulness": "no_gold_needed",
    "hallucination_risk": "no_gold_needed",
    "context_relevance": "no_gold_needed",
    "correctness": "needs_reference_answer",
    "completeness": "needs_reference_answer",
    "context_recall": "needs_reference_answer",
    "hit_rate": "needs_expected_evidence",
    "evidence_coverage": "needs_expected_evidence",
    "mrr": "needs_expected_evidence",
    "recall_at_1": "needs_expected_evidence",
    "recall_at_3": "needs_expected_evidence",
    "recall_at_5": "needs_expected_evidence",
    "precision_at_1": "needs_expected_evidence",
    "precision_at_3": "needs_expected_evidence",
    "precision_at_5": "needs_expected_evidence",
    "strict_mrr": "needs_full_relevance_labels",
    "strict_recall_at_1": "needs_full_relevance_labels",
    "strict_recall_at_3": "needs_full_relevance_labels",
    "strict_recall_at_5": "needs_full_relevance_labels",
    "strict_precision_at_1": "needs_full_relevance_labels",
    "strict_precision_at_3": "needs_full_relevance_labels",
    "strict_precision_at_5": "needs_full_relevance_labels",
}

METRIC_QUALITY_NOTES = {
    "mrr": "弱标注排序指标：基于 expected_evidence 文本近似命中，不等同严格 MRR。",
    "recall_at_1": "弱标注排序指标：单条 expected_evidence 下实际是 Evidence Hit@1。",
    "recall_at_3": "弱标注排序指标：单条 expected_evidence 下实际是 Evidence Hit@3。",
    "recall_at_5": "弱标注排序指标：单条 expected_evidence 下实际是 Evidence Hit@5。",
    "precision_at_1": "弱标注排序指标：只判断与 expected_evidence 相似，不代表完整相关性标注。",
    "precision_at_3": "弱标注排序指标：只判断与 expected_evidence 相似，不代表完整相关性标注。",
    "precision_at_5": "弱标注排序指标：只判断与 expected_evidence 相似，不代表完整相关性标注。",
    "strict_mrr": "严格排序指标：基于人工 relevant_context_ids 与返回 ID 匹配。",
    "strict_recall_at_1": "严格 Recall@1：基于人工相关 chunk/doc ID 集合。",
    "strict_recall_at_3": "严格 Recall@3：基于人工相关 chunk/doc ID 集合。",
    "strict_recall_at_5": "严格 Recall@5：基于人工相关 chunk/doc ID 集合。",
    "strict_precision_at_1": "严格 Precision@1：基于人工相关 chunk/doc ID 集合。",
    "strict_precision_at_3": "严格 Precision@3：基于人工相关 chunk/doc ID 集合。",
    "strict_precision_at_5": "严格 Precision@5：基于人工相关 chunk/doc ID 集合。",
}

AGGREGATE_EXCLUDED_METRICS = {
    "mrr",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "precision_at_1",
    "precision_at_3",
    "precision_at_5",
    "strict_mrr",
    "strict_recall_at_1",
    "strict_recall_at_3",
    "strict_recall_at_5",
    "strict_precision_at_1",
    "strict_precision_at_3",
    "strict_precision_at_5",
}


def metric_dependency(metric_name: str) -> str:
    return METRIC_DEPENDENCIES.get(metric_name, "no_gold_needed")


def metric_dependency_label(metric_name: str) -> str:
    return DEPENDENCY_LABELS[metric_dependency(metric_name)]


def metric_quality_note(metric_name: str) -> str:
    dep = metric_dependency(metric_name)
    return METRIC_QUALITY_NOTES.get(metric_name, DEPENDENCY_DESCRIPTIONS[dep])


def strict_rank_metric_names() -> list[str]:
    return [
        "strict_mrr",
        "strict_recall_at_1",
        "strict_recall_at_3",
        "strict_recall_at_5",
        "strict_precision_at_1",
        "strict_precision_at_3",
        "strict_precision_at_5",
    ]


def sample_annotation_status(sample: EvalSample | None) -> str:
    if sample is None:
        return "unverified"
    if sample.gold_label_status in GOLD_LABEL_STATUS_OPTIONS and sample.gold_label_status != "unverified":
        return sample.gold_label_status
    if sample.relevant_context_ids:
        return "full_relevance_verified"
    if sample.expected_evidence:
        return "evidence_verified"
    if sample.reference_answer:
        return "reference_verified"
    return "unverified"


def sample_has_dependency(sample: EvalSample | None, dependency: str) -> bool:
    if dependency == "no_gold_needed":
        return True
    if sample is None:
        return False
    if dependency == "needs_reference_answer":
        return bool(str(sample.reference_answer).strip())
    if dependency == "needs_expected_evidence":
        return bool(str(sample.expected_evidence).strip())
    if dependency == "needs_full_relevance_labels":
        return bool(sample.relevant_context_ids)
    return False


def metric_is_high_confidence(metric_name: str, sample: EvalSample | None) -> bool:
    return sample_has_dependency(sample, metric_dependency(metric_name))


def filter_scores_for_available_annotations(
    scores: dict[str, ScoreItem],
    sample: EvalSample | None,
) -> tuple[dict[str, ScoreItem], list[str]]:
    filtered: dict[str, ScoreItem] = {}
    skipped: list[str] = []
    for name, item in scores.items():
        if sample_has_dependency(sample, metric_dependency(name)):
            filtered[name] = item
        else:
            skipped.append(name)
    return filtered, skipped


def recompute_normalized_score(scores: dict[str, ScoreItem]) -> float:
    aggregate_inputs = []
    for name, item in scores.items():
        if name in AGGREGATE_EXCLUDED_METRICS:
            continue
        score_value = 1 - item.normalized_score if name == "hallucination_risk" else item.normalized_score
        aggregate_inputs.append(score_value)
    return round(mean(aggregate_inputs), 4) if aggregate_inputs else 0.0


def annotation_status_summary(samples: list[EvalSample]) -> dict[str, int]:
    counts = {status: 0 for status in GOLD_LABEL_STATUS_OPTIONS}
    for sample in samples:
        counts[sample_annotation_status(sample)] = counts.get(sample_annotation_status(sample), 0) + 1
    return counts
