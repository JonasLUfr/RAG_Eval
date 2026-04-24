from __future__ import annotations

import json
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import pandas as pd


MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024
MAX_TEXT_CHARS = 20_000
MAX_TABLE_ROWS = 100


class SourceFileTooLargeError(ValueError):
    pass


def parse_source_file(file_name: str, content: bytes) -> dict[str, Any]:
    """把用户上传的小型项目材料抽样成可发送给 LLM 的文本摘要。"""
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise SourceFileTooLargeError(f"{file_name} 超过 5MB，MVP 暂不支持过大材料。")

    suffix = Path(file_name).suffix.lower()
    if suffix in [".csv"]:
        return _parse_csv(file_name, content)
    if suffix in [".xlsx", ".xls"]:
        return _parse_excel(file_name, content)
    if suffix in [".json"]:
        return _parse_json(file_name, content)
    if suffix in [".md", ".txt", ".sql"]:
        return _parse_text(file_name, content, suffix)
    raise ValueError(f"{file_name} 的格式暂不支持。")


def build_materials_prompt(uploaded_assets: list[dict[str, Any]]) -> str:
    if not uploaded_assets:
        return "用户未上传项目材料。"
    blocks = []
    for asset in uploaded_assets:
        blocks.append(
            f"""文件名：{asset.get("file_name", "")}
类型：{asset.get("file_type", "")}
大小：{asset.get("size_bytes", 0)} bytes
摘要/抽样内容：
{asset.get("excerpt", "")}
"""
        )
    return "\n---\n".join(blocks)


def _parse_csv(file_name: str, content: bytes) -> dict[str, Any]:
    df = pd.read_csv(StringIO(content.decode("utf-8-sig")))
    sample = df.head(MAX_TABLE_ROWS)
    return {
        "file_name": file_name,
        "file_type": "csv",
        "size_bytes": len(content),
        "row_count": int(len(df)),
        "columns": list(map(str, df.columns)),
        "excerpt": sample.to_csv(index=False)[:MAX_TEXT_CHARS],
    }


def _parse_excel(file_name: str, content: bytes) -> dict[str, Any]:
    sheets = pd.read_excel(BytesIO(content), sheet_name=None)
    excerpts = []
    meta = {}
    for sheet_name, df in sheets.items():
        sample = df.head(MAX_TABLE_ROWS)
        meta[sheet_name] = {"rows": int(len(df)), "columns": list(map(str, df.columns))}
        excerpts.append(f"Sheet: {sheet_name}\n{sample.to_csv(index=False)}")
    return {
        "file_name": file_name,
        "file_type": "excel",
        "size_bytes": len(content),
        "sheets": meta,
        "excerpt": "\n\n".join(excerpts)[:MAX_TEXT_CHARS],
    }


def _parse_json(file_name: str, content: bytes) -> dict[str, Any]:
    text = content.decode("utf-8-sig")
    data = json.loads(text)
    compact = json.dumps(data, ensure_ascii=False, indent=2)
    return {
        "file_name": file_name,
        "file_type": "json",
        "size_bytes": len(content),
        "excerpt": compact[:MAX_TEXT_CHARS],
    }


def _parse_text(file_name: str, content: bytes, suffix: str) -> dict[str, Any]:
    text = content.decode("utf-8-sig", errors="replace")
    return {
        "file_name": file_name,
        "file_type": suffix.lstrip("."),
        "size_bytes": len(content),
        "excerpt": text[:MAX_TEXT_CHARS],
    }
