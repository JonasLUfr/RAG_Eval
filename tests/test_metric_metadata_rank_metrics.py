from app.core.config import AppConfig
from app.models import EvalSample, ScoreItem, SystemResponse
from app.services.evaluator import EvaluationEngine
from app.services.metric_metadata import (
    filter_scores_for_available_annotations,
    metric_is_high_confidence,
    recompute_normalized_score,
    sample_annotation_status,
)
from app.services.rank_metrics import compute_rank_metrics, compute_strict_rank_metrics


def test_metric_confidence_requires_reference_answer():
    sample = EvalSample(question="q", reference_answer="")

    assert metric_is_high_confidence("relevance", sample) is True
    assert metric_is_high_confidence("correctness", sample) is False
    assert metric_is_high_confidence("completeness", sample) is False


def test_weak_rank_metrics_require_expected_evidence():
    assert compute_rank_metrics(["ctx"], "") == {}


def test_expected_evidence_without_relevant_ids_only_creates_weak_rank_metrics():
    sample = EvalSample(
        question_id="q1",
        question="q",
        reference_answer="answer",
        expected_evidence="gold evidence",
    )
    response = SystemResponse(
        response_id="r1",
        question_id="q1",
        question="q",
        answer="answer",
        retrieved_contexts=["gold evidence"],
    )

    result = EvaluationEngine(AppConfig()).evaluate_batch(
        [response],
        {"q1": sample},
    )[0]

    assert "mrr" in result.scores
    assert "strict_mrr" not in result.scores


def test_strict_rank_metrics_use_relevant_context_ids():
    scores = compute_strict_rank_metrics(
        returned_items=["chunk-a", "chunk-b", "chunk-c"],
        relevant_context_ids=["chunk-b", "chunk-c"],
    )

    assert scores["strict_mrr"].normalized_score == 0.5
    assert scores["strict_recall_at_1"].normalized_score == 0.0
    assert scores["strict_recall_at_3"].normalized_score == 1.0
    assert scores["strict_precision_at_3"].normalized_score == 0.6667


def test_sample_annotation_status_infers_full_relevance_labels():
    sample = EvalSample(question="q", relevant_context_ids=["chunk-1"])

    assert sample_annotation_status(sample) == "full_relevance_verified"


def test_filter_scores_excludes_reference_metrics_without_reference_answer():
    sample = EvalSample(question="q", reference_answer="")
    scores = {
        "correctness": ScoreItem(raw_score=1, normalized_score=1, reason="needs gold"),
        "relevance": ScoreItem(raw_score=0.8, normalized_score=0.8, reason="question answer only"),
        "completeness": ScoreItem(raw_score=1, normalized_score=1, reason="needs gold"),
    }

    filtered, skipped = filter_scores_for_available_annotations(scores, sample)

    assert list(filtered) == ["relevance"]
    assert skipped == ["correctness", "completeness"]
    assert recompute_normalized_score(filtered) == 0.8


def test_filter_scores_excludes_evidence_metrics_without_expected_evidence():
    sample = EvalSample(question="q", reference_answer="answer")
    scores = {
        "relevance": ScoreItem(raw_score=0.7, normalized_score=0.7, reason="ok"),
        "hit_rate": ScoreItem(raw_score=1, normalized_score=1, reason="needs expected evidence"),
        "evidence_coverage": ScoreItem(raw_score=1, normalized_score=1, reason="needs expected evidence"),
    }

    filtered, skipped = filter_scores_for_available_annotations(scores, sample)

    assert list(filtered) == ["relevance"]
    assert skipped == ["hit_rate", "evidence_coverage"]


def test_evaluation_engine_aggregates_only_annotated_metrics():
    sample = EvalSample(question_id="q1", question="battery warranty", reference_answer="")
    response = SystemResponse(
        response_id="r1",
        question_id="q1",
        question="battery warranty",
        answer="battery warranty",
    )

    result = EvaluationEngine(AppConfig()).evaluate_batch([response], {"q1": sample})[0]

    assert "relevance" in result.scores
    assert "correctness" not in result.scores
    assert "completeness" not in result.scores
    assert result.normalized_score == result.scores["relevance"].normalized_score
    assert "暂无法准确评分" in result.judge_reason
