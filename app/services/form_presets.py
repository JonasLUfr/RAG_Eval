"""轻量表单预设：跨会话记住「最近一次」填写的实验配置，避免重复输入。

存储位置：app/data/form_presets.json，按表单名分桶。API Key 等敏感字段**不存**。
"""
from __future__ import annotations

import json
from typing import Any

from app.core.config import ROOT_DIR

_PRESETS_PATH = ROOT_DIR / "data" / "form_presets.json"

_SENSITIVE_HEADER_KEYWORDS = ("authorization", "api-key", "api_key", "x-api-key", "token", "secret")


def _load_all() -> dict[str, dict]:
    if not _PRESETS_PATH.exists():
        return {}
    try:
        data = json.loads(_PRESETS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def load_preset(name: str) -> dict[str, Any]:
    return _load_all().get(name, {})


def save_preset(name: str, data: dict[str, Any]) -> None:
    _PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    bucket = _load_all()
    bucket[name] = data
    _PRESETS_PATH.write_text(
        json.dumps(bucket, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def redact_headers_json(headers_json: str) -> str:
    """Authorization / API Key 这类字段把值清空再保存。键名保留，方便用户回填。"""
    try:
        data = json.loads(headers_json) if headers_json.strip() else {}
    except json.JSONDecodeError:
        return headers_json
    if not isinstance(data, dict):
        return headers_json
    redacted: dict[str, Any] = {}
    for key, value in data.items():
        if any(s in str(key).lower() for s in _SENSITIVE_HEADER_KEYWORDS):
            redacted[key] = ""
        else:
            redacted[key] = value
    return json.dumps(redacted, ensure_ascii=False, indent=2)
