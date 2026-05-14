import requests

from app.core.config import AppConfig
from app.models import EvalSample
from app.services.connectors import ConnectorConfig, ExternalAPIConnector


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {"answer": "ok", "retrieved_contexts": ["ctx"], "token_usage": 3}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def json(self):
        return self._payload


def _connector(retry_times=1):
    config = AppConfig(request_timeout_seconds=1, retry_times=retry_times)
    return ExternalAPIConnector(config, ConnectorConfig(endpoint="https://example.test/query"))


def _sample():
    return EvalSample(question_id="q1", question="问题", reference_answer="参考")


def test_api_retries_500_then_succeeds(monkeypatch):
    calls = []

    def fake_post(*_args, **_kwargs):
        calls.append(1)
        return FakeResponse(500) if len(calls) == 1 else FakeResponse(200)

    monkeypatch.setattr(requests, "post", fake_post)

    response = _connector(retry_times=1).run_one(_sample())

    assert response.success is True
    assert response.answer == "ok"
    assert response.raw_response["_attempts"] == 2
    assert len(calls) == 2


def test_api_does_not_retry_401(monkeypatch):
    calls = []

    def fake_post(*_args, **_kwargs):
        calls.append(1)
        return FakeResponse(401)

    monkeypatch.setattr(requests, "post", fake_post)

    response = _connector(retry_times=2).run_one(_sample())

    assert response.success is False
    assert response.raw_response["_attempts"] == 1
    assert len(calls) == 1


def test_api_timeout_retries_then_fails(monkeypatch):
    calls = []

    def fake_post(*_args, **_kwargs):
        calls.append(1)
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "post", fake_post)

    response = _connector(retry_times=2).run_one(_sample())

    assert response.success is False
    assert "slow" in response.error
    assert response.raw_response["_attempts"] == 3
    assert len(calls) == 3


def test_api_accepts_usage_dict_for_token_usage(monkeypatch):
    def fake_post(*_args, **_kwargs):
        return FakeResponse(200, {"answer": "ok", "usage": {"total_tokens": 17}})

    monkeypatch.setattr(requests, "post", fake_post)
    connector = ExternalAPIConnector(
        AppConfig(request_timeout_seconds=1, retry_times=0),
        ConnectorConfig(
            endpoint="https://example.test/query",
            response_mapping={"answer": "answer", "token_usage": "usage"},
        ),
    )

    response = connector.run_one(_sample())

    assert response.success is True
    assert response.token_usage == 17
