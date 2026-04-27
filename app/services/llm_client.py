from __future__ import annotations

import json
import logging
from typing import Any

import requests

from app.core.config import AppConfig


logger = logging.getLogger(__name__)


class OpenAICompatibleClient:
    """OpenAI-compatible Chat Completions 客户端；未配置时由上层走本地模拟逻辑。"""

    def __init__(
        self,
        config: AppConfig,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        proxy_url: str | None = None,
    ):
        self.config = config
        self.api_base = api_base if api_base is not None else config.llm_api_base
        self.api_key = api_key if api_key is not None else config.llm_api_key
        self.model = model or config.llm_model
        self.proxy_url = proxy_url or ""

    @property
    def is_configured(self) -> bool:
        return bool(self.api_base and self.api_key)

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        timeout: int | None = None,
    ) -> Any:
        text = self.chat(system_prompt, user_prompt, temperature, timeout)
        return self._extract_json(text)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        timeout: int | None = None,
    ) -> str:
        if not self.is_configured:
            raise RuntimeError("未配置 LLM_API_BASE / LLM_API_KEY")
        base = self.api_base.rstrip("/")
        url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout or self.config.request_timeout_seconds,
                proxies=self._proxies(),
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("llm_call_failed model=%s url=%s err=%s", self.model, url, exc)
            raise
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _proxies(self) -> dict[str, str] | None:
        if not self.proxy_url:
            return None
        return {"http": self.proxy_url, "https": self.proxy_url}

    @staticmethod
    def _extract_json(text: str) -> Any:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.removeprefix("json").strip()
        start = min([idx for idx in [cleaned.find("["), cleaned.find("{")] if idx >= 0], default=0)
        cleaned = cleaned[start:]
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("LLM 返回不是合法 JSON: %s", text[:500])
            raise
