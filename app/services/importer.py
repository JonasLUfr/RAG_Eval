from __future__ import annotations

import json
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from app.models import EvalSample, SystemResponse


DEFAULT_IMPORT_MAPPING = {
    "question_id": "question_id",
    "question": "question",
    "reference_answer": "reference_answer",
    "answer": "answer",
    "retrieved_contexts": "retrieved_contexts",
    "citations": "citations",
    "latency_ms": "latency_ms",
    "token_usage": "token_usage",
}


def read_uploaded_table(file_name: str, content: bytes) -> pd.DataFrame:
    suffix = Path(file_name).suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(BytesIO(content))
    if suffix == ".json":
        payload = json.loads(content.decode("utf-8"))
        return pd.json_normalize(payload)
    return pd.read_csv(StringIO(content.decode("utf-8-sig")))


def dataframe_to_responses(
    df: pd.DataFrame,
    mapping: dict[str, str] | None = None,
) -> list[SystemResponse]:
    mapping = mapping or DEFAULT_IMPORT_MAPPING
    responses: list[SystemResponse] = []
    for _, row in df.fillna("").iterrows():
        data = row.to_dict()
        item = {internal: data.get(col, "") for internal, col in mapping.items() if col}
        if not item.get("question_id"):
            item.pop("question_id", None)
        responses.append(SystemResponse.from_dict(item))
    return responses


def dataframe_to_samples(
    df: pd.DataFrame,
    mapping: dict[str, str] | None = None,
) -> list[EvalSample]:
    mapping = mapping or {
        "question_id": "question_id",
        "question": "question",
        "question_type": "question_type",
        "difficulty": "difficulty",
        "expected_scope": "expected_scope",
        "reference_answer": "reference_answer",
        "expected_evidence": "expected_evidence",
        "gold_contexts": "gold_contexts",
        "relevant_context_ids": "relevant_context_ids",
        "gold_label_status": "gold_label_status",
        "tags": "tags",
    }
    samples: list[EvalSample] = []
    for _, row in df.fillna("").iterrows():
        data: dict[str, Any] = row.to_dict()
        item = {internal: data.get(col, "") for internal, col in mapping.items() if col}
        if not item.get("question_id"):
            item.pop("question_id", None)
        item["generation_method"] = "import"
        item["review_status"] = item.get("review_status") or "待审核"
        samples.append(EvalSample.from_dict(item))
    return samples
