from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


FAILURE_LABELS = [
    "wrong_answer",
    "incomplete_answer",
    "unsupported_answer",
    "missing_evidence",
    "retrieval_issue",
    "should_abstain_but_answered",
    "ambiguous_question_failure",
    "cannot_judge",
]


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class ProjectContext:
    project_id: str = field(default_factory=lambda: new_id("proj"))
    name: str = "示例项目"
    project_background: str = ""
    system_description: str = ""
    evaluation_goals: str = ""
    business_rules: str = ""
    question_type_instructions: str = ""
    uploaded_assets: list[dict[str, Any]] = field(default_factory=list)
    # 兼容高级用户和旧数据：这些字段不再作为主流程必填。
    schema_text: str = ""
    metadata_json: str = "{}"
    sample_rows: str = ""
    few_shot_examples: str = ""
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectContext":
        if isinstance(data.get("uploaded_assets"), str):
            try:
                data["uploaded_assets"] = json.loads(data["uploaded_assets"])
            except json.JSONDecodeError:
                data["uploaded_assets"] = []
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EvalSample:
    question_id: str = field(default_factory=lambda: new_id("q"))
    question: str = ""
    question_type: str = "事实核对"
    difficulty: str = "中"
    expected_scope: str = ""
    reference_answer: str = ""
    expected_evidence: str = ""
    tags: list[str] = field(default_factory=list)
    source_context_refs: list[str] = field(default_factory=list)
    generation_method: str = "manual"
    review_status: str = "待审核"
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalSample":
        if isinstance(data.get("tags"), str):
            data["tags"] = [x.strip() for x in data["tags"].split(",") if x.strip()]
        if isinstance(data.get("source_context_refs"), str):
            data["source_context_refs"] = [
                x.strip() for x in data["source_context_refs"].split(",") if x.strip()
            ]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SystemResponse:
    response_id: str = field(default_factory=lambda: new_id("resp"))
    question_id: str = ""
    question: str = ""
    reference_answer: str = ""
    answer: str = ""
    retrieved_contexts: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    latency_ms: float | None = None
    token_usage: int | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str = ""
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SystemResponse":
        for key in ("retrieved_contexts", "citations"):
            value = data.get(key)
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    data[key] = parsed if isinstance(parsed, list) else [str(parsed)]
                except json.JSONDecodeError:
                    data[key] = [x.strip() for x in value.split("\n") if x.strip()]
        if isinstance(data.get("raw_response"), str):
            try:
                data["raw_response"] = json.loads(data["raw_response"])
            except json.JSONDecodeError:
                data["raw_response"] = {"text": data["raw_response"]}
        if data.get("latency_ms") not in (None, ""):
            try:
                data["latency_ms"] = float(data["latency_ms"])
            except (TypeError, ValueError):
                data["latency_ms"] = None
        if data.get("token_usage") not in (None, ""):
            try:
                data["token_usage"] = int(float(data["token_usage"]))
            except (TypeError, ValueError):
                data["token_usage"] = None
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ScoreItem:
    raw_score: float = 0.0
    normalized_score: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalResult:
    result_id: str = field(default_factory=lambda: new_id("eval"))
    question_id: str = ""
    response_id: str = ""
    scores: dict[str, ScoreItem] = field(default_factory=dict)
    normalized_score: float = 0.0
    judge_reason: str = ""
    judge_model: str = ""
    score_version: str = "mvp-v1"
    failure_labels: list[str] = field(default_factory=list)
    evaluation_status: str = "scored"
    evaluator_error: str = ""
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["scores"] = {
            k: v.to_dict() if isinstance(v, ScoreItem) else v for k, v in self.scores.items()
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalResult":
        scores = data.get("scores", {})
        data["scores"] = {
            k: v if isinstance(v, ScoreItem) else ScoreItem(**v) for k, v in scores.items()
        }
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ExperimentRun:
    run_id: str = field(default_factory=lambda: new_id("run"))
    project_id: str = ""
    name: str = "未命名实验"
    mode: str = "historical_import"
    config: dict[str, Any] = field(default_factory=dict)
    aggregate: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentRun":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
