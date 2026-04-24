from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AppConfig:
    app_name: str = "RAG 评测工作台 MVP"
    db_path: Path = ROOT_DIR / "data" / "rag_eval.sqlite3"
    export_dir: Path = ROOT_DIR / "exports"
    log_dir: Path = ROOT_DIR / "data" / "logs"
    default_max_generated_questions: int = 200
    default_max_eval_questions: int = 300
    max_workers: int = 5
    request_timeout_seconds: int = 30
    retry_times: int = 2
    llm_api_base: str = ""
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    score_version: str = "mvp-v1"


def load_config() -> AppConfig:
    """集中读取环境变量，避免配置散落在 UI 和服务层。"""
    cfg = AppConfig(
        db_path=Path(os.getenv("RAG_EVAL_DB_PATH", str(AppConfig.db_path))),
        export_dir=Path(os.getenv("RAG_EVAL_EXPORT_DIR", str(AppConfig.export_dir))),
        log_dir=Path(os.getenv("RAG_EVAL_LOG_DIR", str(AppConfig.log_dir))),
        default_max_generated_questions=int(os.getenv("RAG_EVAL_MAX_GENERATED", "200")),
        default_max_eval_questions=int(os.getenv("RAG_EVAL_MAX_EVAL", "300")),
        max_workers=int(os.getenv("RAG_EVAL_MAX_WORKERS", "5")),
        request_timeout_seconds=int(os.getenv("RAG_EVAL_TIMEOUT", "30")),
        retry_times=int(os.getenv("RAG_EVAL_RETRY", "2")),
        llm_api_base=os.getenv("LLM_API_BASE", ""),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        judge_model=os.getenv("JUDGE_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini")),
    )
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.export_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    return cfg

