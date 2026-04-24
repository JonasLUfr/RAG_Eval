from __future__ import annotations

import json
from typing import Any


def dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def loads(text: str | None, default: Any = None) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default

