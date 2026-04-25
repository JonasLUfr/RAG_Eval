import pytest

from app.core.config import AppConfig
from app.models import EvalSample, SystemResponse
from app.services.ragas_evaluator import RagasEvaluator


class FakeLLM:
    model = "fake-judge"

    def __init__(self, mode="ok"):
        self.mode = mode

    def chat_json(self, _system_prompt, user_prompt, temperature=0.0):
        if self.mode == "raise":
            raise RuntimeError("judge unavailable")
        if self.mode == "invalid":
            raise ValueError("invalid json")
        if "原子陈述" in user_prompt:
            return {"statements": ["回答有依据"], "supported": [True], "reason": "supported"}
        if "tp" in user_prompt and "fp" in user_prompt:
            return {"tp": 1, "fp": 0, "fn": 0, "reason": "match"}
        if "直接相关性" in user_prompt:
            return {"score": 1.0, "reason": "relevant"}
        if "完整程度" in user_prompt:
            return {"score": 1.0, "reason": "complete"}
        if "检索片段列表" in user_prompt:
            return {"results": [True], "reason": "useful"}
        if "参考答案分解" in user_prompt:
            return {"statements": ["事实"], "attributed": [True], "reason": "covered"}
        if "整体相关性" in user_prompt:
            return {"score": 1.0, "reason": "ctx relevant"}
        if "期望证据" in user_prompt:
            return {"score": 1.0, "reason": "evidence covered"}
        pytest.fail(f"unexpected prompt: {user_prompt[:80]}")


def _item():
    sample = EvalSample(
        question_id="q1",
        question="订单状态是什么？",
        reference_answer="订单已发货。",
        expected_evidence="状态字段为已发货",
    )
    response = SystemResponse(
        response_id="resp1",
        question_id="q1",
        question=sample.question,
        reference_answer=sample.reference_answer,
        answer="订单已发货。",
        retrieved_contexts=["状态字段为已发货"],
    )
    return sample, response


def test_ragas_scores_with_fake_llm():
    result = RagasEvaluator(AppConfig(), FakeLLM()).evaluate_one(_item())

    assert result.normalized_score > 0.8
    assert result.failure_labels == []
    assert result.evaluation_status == "scored"


def test_ragas_judge_exception_is_cannot_judge():
    result = RagasEvaluator(AppConfig(), FakeLLM("raise")).evaluate_one(_item())

    assert result.normalized_score == 0.0
    assert result.failure_labels == ["cannot_judge"]
    assert result.evaluation_status == "judge_failed"
    assert "judge unavailable" in result.evaluator_error


def test_ragas_invalid_json_is_cannot_judge():
    result = RagasEvaluator(AppConfig(), FakeLLM("invalid")).evaluate_one(_item())

    assert result.normalized_score == 0.0
    assert result.failure_labels == ["cannot_judge"]
    assert result.evaluation_status == "judge_failed"
    assert "invalid json" in result.evaluator_error
