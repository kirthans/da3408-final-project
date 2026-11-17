"""HTTP metrics use bounded labels and account for rejected requests."""

from unittest.mock import Mock

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from elo.api import app as api


def sample(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0


def test_request_metrics(monkeypatch):
    monkeypatch.setenv("ELO_METRICS", "0")
    monkeypatch.setattr(api, "EngineAnalyzer", Mock())
    health = sample("elo_requests_total", endpoint="/health", status="200")
    bad = sample("elo_requests_total", endpoint="/v1/predict", status="400")
    errors = sample("elo_parse_errors_total", reason="invalid_pgn")
    latency = sample("elo_request_latency_seconds_count", endpoint="/v1/predict")
    with TestClient(api.app) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/v1/predict", json={"pgn": "garbage"}).status_code == 400
        assert client.get("/arbitrary-user-path-12345").status_code == 404
    assert sample("elo_requests_total", endpoint="/health", status="200") == health + 1
    assert sample("elo_requests_total", endpoint="/v1/predict", status="400") == bad + 1
    assert sample("elo_parse_errors_total", reason="invalid_pgn") == errors + 1
    assert sample("elo_request_latency_seconds_count", endpoint="/v1/predict") == latency + 1
    assert sample("elo_requests_total", endpoint="other", status="404") >= 1
    assert (
        REGISTRY.get_sample_value(
            "elo_requests_total", {"endpoint": "/arbitrary-user-path-12345", "status": "404"}
        )
        is None
    )
    assert sample("elo_inflight_requests") == 0


def test_terminal_model_error_counted_once(monkeypatch):
    import pytest

    monkeypatch.setenv("MODEL_SERVER_URL", "http://model-server:5001")
    monkeypatch.setattr(api.requests, "request", Mock(side_effect=api.requests.ConnectionError))
    monkeypatch.setattr(api.time, "sleep", lambda _: None)
    before = sample("elo_model_server_errors_total")
    with pytest.raises(api.EloError):
        api.remote("http://model-server:5001/ping", retry=True)
    assert sample("elo_model_server_errors_total") == before + 1
    with pytest.raises(api.EloError):
        api.remote("http://trainer:8001/status", code="trainer_unavailable")
    assert sample("elo_model_server_errors_total") == before + 1
