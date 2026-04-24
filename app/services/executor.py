from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar


T = TypeVar("T")
R = TypeVar("R")
logger = logging.getLogger(__name__)


@dataclass
class BatchProgress:
    total: int = 0
    completed: int = 0
    failed: int = 0

    @property
    def ratio(self) -> float:
        return self.completed / self.total if self.total else 0.0


class BatchExecutor(Generic[T, R]):
    """批量执行器：统一处理并发、重试、进度和错误日志。"""

    def __init__(self, max_workers: int = 5, retry_times: int = 2, retry_sleep: float = 0.8):
        self.max_workers = max_workers
        self.retry_times = retry_times
        self.retry_sleep = retry_sleep

    def _run_one(self, item: T, fn: Callable[[T], R]) -> R:
        last_error: Exception | None = None
        for attempt in range(self.retry_times + 1):
            try:
                return fn(item)
            except Exception as exc:  # noqa: BLE001 - 批量任务需要捕获并记录单条失败
                last_error = exc
                logger.warning("批量任务失败，准备重试 attempt=%s error=%s", attempt + 1, exc)
                time.sleep(self.retry_sleep * (attempt + 1))
        assert last_error is not None
        raise last_error

    def run(
        self,
        items: Iterable[T],
        fn: Callable[[T], R],
        progress_callback: Callable[[BatchProgress], None] | None = None,
    ) -> list[R]:
        item_list = list(items)
        progress = BatchProgress(total=len(item_list))
        results: list[R] = []
        if progress_callback:
            progress_callback(progress)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(self._run_one, item, fn) for item in item_list]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    logger.exception("批量任务最终失败: %s", exc)
                    progress.failed += 1
                finally:
                    progress.completed += 1
                    if progress_callback:
                        progress_callback(progress)
        return results

