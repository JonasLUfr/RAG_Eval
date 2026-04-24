from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from app.core.config import AppConfig


def setup_logging(config: AppConfig) -> None:
    """初始化文件日志，记录批量调用、导入、评分和导出错误。"""
    log_file = config.log_dir / "app.log"
    handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        root.addHandler(handler)

