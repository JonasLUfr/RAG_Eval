from __future__ import annotations

import logging
import os
import signal
import sys
from logging.handlers import RotatingFileHandler

from app.core.config import AppConfig


_INSTALLED = False  # 模块多次 import 时避免重复装钩子


def setup_logging(config: AppConfig) -> None:
    """初始化文件 + stderr 日志，并安装崩溃诊断钩子。

    诊断目的：进程死掉前，app.log 末尾应能直接看出死因
    - 正常退出：=== shutdown signal=... ===
    - 代码异常：=== UNCAUGHT EXCEPTION === + traceback
    - 都没有：高度怀疑被 SIGKILL（OOM / kernel），需查 dmesg / journalctl
    """
    global _INSTALLED

    log_file = config.log_dir / "app.log"
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
    )

    file_handler = RotatingFileHandler(
        log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        root.addHandler(file_handler)
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
        for h in root.handlers
    ):
        root.addHandler(stderr_handler)

    if _INSTALLED:
        return
    _INSTALLED = True

    logger = logging.getLogger("app.lifecycle")
    logger.info(
        "=== STARTUP pid=%s py=%s cwd=%s log_dir=%s ===",
        os.getpid(),
        sys.version.split()[0],
        os.getcwd(),
        config.log_dir,
    )

    def _excepthook(exc_type, exc_value, exc_tb):
        # 标记非常显眼，便于 grep；先打标记再打 traceback，避免日志被截断时漏掉关键字
        logger.critical("=== UNCAUGHT EXCEPTION ===", exc_info=(exc_type, exc_value, exc_tb))
        # 调用原 hook 让 stderr 也保留默认 traceback 输出（systemd journald 会抓到）
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook

    def _signal_handler(signum, _frame):
        name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        logger.warning("=== shutdown signal=%s pid=%s ===", name, os.getpid())
        # 正常退出，让 streamlit 自己清理；不要 os._exit 否则跳过 finally
        sys.exit(0)

    for sig_name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _signal_handler)
            except (ValueError, OSError):
                # 非主线程 / 平台不支持时会抛，跳过即可
                pass
