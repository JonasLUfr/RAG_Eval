from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from app.core.config import AppConfig
from app.models import EvalSample, SystemResponse
from app.services.executor import BatchExecutor


logger = logging.getLogger(__name__)


@dataclass
class ConnectorConfig:
    endpoint: str
    method: str = "POST"
    headers_json: str = "{}"
    request_mapping: dict[str, str] = field(default_factory=lambda: {"question": "question"})
    response_mapping: dict[str, str] = field(default_factory=lambda: {"answer": "answer"})


class ExternalAPIConnector:
    """黑盒 RAG / 生成 API 连接器；只做请求映射和响应解析，不实现检索。"""

    def __init__(self, config: AppConfig, connector_config: ConnectorConfig):
        self.config = config
        self.connector_config = connector_config
        self.executor = BatchExecutor[EvalSample, SystemResponse](
            max_workers=config.max_workers,
            retry_times=config.retry_times,
        )

    def validate(self) -> tuple[bool, str]:
        if not self.connector_config.endpoint.startswith(("http://", "https://")):
            return False, "API 地址必须以 http:// 或 https:// 开头"
        try:
            json.loads(self.connector_config.headers_json or "{}")
        except json.JSONDecodeError:
            return False, "请求头必须是合法 JSON"
        return True, "连接器配置格式有效"

    def run_batch(
        self,
        samples: list[EvalSample],
        progress_callback: callable | None = None,
    ) -> list[SystemResponse]:
        return self.executor.run(samples, self.run_one, progress_callback)

    def run_one(self, sample: EvalSample) -> SystemResponse:
        headers = json.loads(self.connector_config.headers_json or "{}")
        payload = self._build_payload(sample)
        started = time.perf_counter()
        attempts = self.config.retry_times + 1
        last_error: Exception | None = None
        last_attempt = 0
        for attempt in range(1, attempts + 1):
            last_attempt = attempt
            try:
                response = self._send_request(headers, payload)
                response.raise_for_status()
                raw = response.json()
                latency_ms = (time.perf_counter() - started) * 1000
                parsed = self._parse_response(sample, raw, latency_ms)
                parsed.raw_response = {**parsed.raw_response, "_attempts": attempt}
                return parsed
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= attempts or not self._should_retry(exc):
                    break
                logger.warning(
                    "外部 API 调用失败，准备重试 question_id=%s attempt=%s/%s error=%s",
                    sample.question_id,
                    attempt,
                    attempts,
                    exc,
                )
                time.sleep(0.8 * attempt)
        assert last_error is not None
        logger.error("外部 API 调用最终失败 question_id=%s error=%s", sample.question_id, last_error)
        latency_ms = (time.perf_counter() - started) * 1000
        return SystemResponse(
            question_id=sample.question_id,
            question=sample.question,
            reference_answer=sample.reference_answer,
            answer="",
            success=False,
            error=str(last_error),
            latency_ms=latency_ms,
            raw_response={"_attempts": last_attempt, "_error": str(last_error)},
        )

    def _send_request(self, headers: dict[str, Any], payload: dict[str, Any]) -> requests.Response:
        if self.connector_config.method.upper() == "GET":
            return requests.get(
                self.connector_config.endpoint,
                headers=headers,
                params=payload,
                timeout=self.config.request_timeout_seconds,
            )
        return requests.post(
            self.connector_config.endpoint,
            headers=headers,
            json=payload,
            timeout=self.config.request_timeout_seconds,
        )

    @staticmethod
    def _should_retry(exc: Exception) -> bool:
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            return exc.response.status_code == 429 or exc.response.status_code >= 500
        return False

    def _build_payload(self, sample: EvalSample) -> dict[str, Any]:
        source = sample.to_dict()
        payload: dict[str, Any] = {}
        for internal_field, outbound_field in self.connector_config.request_mapping.items():
            payload[outbound_field] = source.get(internal_field, "")
        return payload

    def _parse_response(self, sample: EvalSample, raw: dict[str, Any], latency_ms: float) -> SystemResponse:
        mapping = self.connector_config.response_mapping
        answer = self._get_path(raw, mapping.get("answer", "answer"), "")
        contexts = self._as_list(self._get_path(raw, mapping.get("retrieved_contexts", ""), []))
        citations = self._as_list(self._get_path(raw, mapping.get("citations", ""), []))
        mapped_latency = self._get_path(raw, mapping.get("latency_ms", ""), None)
        token_usage = self._get_path(raw, mapping.get("token_usage", ""), None)
        return SystemResponse(
            question_id=sample.question_id,
            question=sample.question,
            reference_answer=sample.reference_answer,
            answer=str(answer or ""),
            retrieved_contexts=contexts,
            citations=citations,
            latency_ms=float(mapped_latency or latency_ms),
            token_usage=int(token_usage) if token_usage not in (None, "") else None,
            raw_response=raw,
            success=True,
        )

    @staticmethod
    def _get_path(data: dict[str, Any], path: str, default: Any) -> Any:
        if not path:
            return default
        cur: Any = data
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    @staticmethod
    def _as_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(x) for x in value]
        return [str(value)]
