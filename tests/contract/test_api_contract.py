"""AC4 validates malformed requests with engine and HTTP boundaries mocked."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from elo.api import app as api
from elo.common.errors import EngineError, EngineTimeoutError

PGN = (
    '[TimeControl "600+0"]\n\n'
    + " ".join(f"{i}. Nf3 Nf6 {i+1}. Ng1 Ng8" for i in range(1, 21, 2))
    + " *"
)


@pytest.fixture
def client(monkeypatch):
    class Engine:
        nodes = 5000

        def __init__(self, **kwargs):
            assert kwargs["nodes"] == 5000

        def close(self):
            pass

        def analyse_game(self, board, moves):
            return dict(
                evals=[0] * (len(moves) + 1),
                best_moves=[None] * (len(moves) + 1),
                analysis_source="stockfish",
            )

    monkeypatch.setattr(api, "EngineAnalyzer", Engine)
    monkeypatch.setenv("API_KEY", "test-key")
    predictions = [
        dict(
            side=s,
            elo=1500,
            lo=1200,
            hi=1800,
            influence=json.dumps(
                [
                    dict(
                        ply=1 if s == "w" else 2,
                        side=s,
                        san="Nf3" if s == "w" else "Nf6",
                        delta_elo=10,
                    )
                ]
            ),
        )
        for s in ("w", "b")
    ]

    def request(method, url, **kwargs):
        if "/invocations" in url:
            frame = kwargs["json"]["dataframe_split"]
            assert not any("elo" in c.lower() or c in ("White", "Black") for c in frame["columns"])
            return SimpleNamespace(
                status_code=200, ok=True, json=lambda: {"predictions": predictions}
            )
        if "/alias?" in url:
            return SimpleNamespace(
                status_code=200,
                ok=True,
                json=lambda: {
                    "model_version": dict(
                        version="1",
                        run_id="abc",
                        aliases=["champion"],
                        creation_timestamp=1,
                        tags=[dict(key="test_mae", value="180"), dict(key="coverage", value="0.8")],
                    )
                },
            )
        return SimpleNamespace(status_code=200, ok=True, json=lambda: {})

    monkeypatch.setattr(api.requests, "request", request)
    with TestClient(api.app) as client:
        yield client


BAD = [
    ({}, "validation_error"),
    ({"pgn": 123}, "validation_error"),
    ({"pgn": "garbage"}, "invalid_pgn"),
    ({"pgn": "1. e4 e5 2. Bh6 *"}, "invalid_pgn"),
    ({"pgn": '[Event "Empty"]\n\n*'}, "invalid_pgn"),
    ({"pgn": "x" * 1048577}, "payload_too_large"),
    ({"pgn": PGN, "game_index": -1}, "validation_error"),
    ({"pgn": PGN, "game_index": 100}, "game_index_out_of_range"),
    ({"pgn": PGN, "top_k": 0}, "validation_error"),
    ({"pgn": PGN, "top_k": 1000}, "validation_error"),
    ({"pgn": PGN.replace("600+0", "60+0")}, "unsupported_time_control"),
    ({"pgn": PGN.replace("600+0", "-")}, "unsupported_time_control"),
    ({"pgn": "1. e4 e5 *"}, "game_too_short"),
    ({"pgn": PGN, "tc_class": "unknown"}, "validation_error"),
]


@pytest.mark.parametrize("body,code", BAD)
def test_bad_inputs(client, body, code):
    response = client.post("/v1/predict", json=body)
    assert 400 <= response.status_code < 500
    assert response.json()["error"]["code"] == code
    assert set(response.json()["error"]) == {"code", "message", "request_id"}
    assert response.headers["X-Request-ID"] == response.json()["error"]["request_id"]


def test_non_json_and_empty(client):
    for raw in (b"", b"not json"):
        response = client.post("/v1/predict", content=raw)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize("key", [None, "wrong"])
def test_unauthorized(client, key):
    response = client.post("/v1/pipeline/retrain", headers={"X-API-Key": key} if key else {})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_happy_path(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == dict(status="ready", model_server=True, engine=True)
    assert client.get("/v1/model").json()["metrics"] == dict(test_mae=180, coverage=0.8)
    response = client.post("/v1/predict", json={"pgn": PGN}, headers={"X-Request-ID": "my-id"})
    assert response.status_code == 200, response.text
    data = response.json()
    assert set(data) == {
        "request_id",
        "game",
        "white",
        "black",
        "interval_level",
        "pgn_ratings",
        "moves",
        "influential_moves",
        "analysis",
        "model",
        "warnings",
    }
    assert data["request_id"] == "my-id"
    assert data["white"]["elo"] == 1500
    assert len(data["moves"]) == 40
    assert data["analysis"]["nodes"] == 5000
    assert data["interval_level"] == 0.8
    otb = client.post("/v1/predict", json={"pgn": PGN.split("\n\n")[1]}).json()
    assert otb["warnings"] and otb["game"]["tc_class"] == "classical"
    assert client.post("/v1/games/parse", json={"pgn": PGN}).json()["games"][0]["plies"] == 40
    uploaded = client.post("/v1/games/parse", files={"file": ("game.pgn", PGN)})
    assert uploaded.status_code == 200


def test_retry(client, monkeypatch):
    original = api.requests.request
    failures = []

    def request(method, url, **kwargs):
        if "/invocations" in url:
            failures.append(1)
            if len(failures) <= 2:
                return SimpleNamespace(status_code=500, ok=False)
        return original(method, url, **kwargs)

    monkeypatch.setattr(api.requests, "request", request)
    monkeypatch.setattr(api.time, "sleep", lambda _: None)
    assert client.post("/v1/predict", json={"pgn": PGN}).status_code == 200
    assert len(failures) == 3
    calls = Mock(return_value=SimpleNamespace(status_code=500, ok=False))
    monkeypatch.setattr(api.requests, "request", calls)
    response = client.post("/v1/predict", json={"pgn": PGN})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"
    assert calls.call_count == 3


@pytest.mark.parametrize(
    "error,status", [(EngineError("failed"), 503), (EngineTimeoutError("slow"), 504)]
)
def test_engine_errors(client, monkeypatch, error, status):
    engine = api.app.state.engines.queue[0]
    monkeypatch.setattr(engine, "analyse_game", Mock(side_effect=error))
    assert client.post("/v1/predict", json={"pgn": PGN}).status_code == status


def test_boundaries():
    values = [0, 28, 29, 178, 179, 478, 479, 1498, 1499]
    assert [api.tc_class(v, 0) for v in values] == [
        "ultrabullet",
        "ultrabullet",
        "bullet",
        "bullet",
        "blitz",
        "blitz",
        "rapid",
        "rapid",
        "classical",
    ]
    assert api.tc_class(99, 2) == "blitz"
    assert api.tc_class("-") == "correspondence"


def test_no_retry_on_4xx_and_network_retry(client, monkeypatch):
    original = api.requests.request
    monkeypatch.setattr(api.time, "sleep", lambda _: None)
    rejected = Mock(return_value=SimpleNamespace(status_code=400, ok=False))
    monkeypatch.setattr(api.requests, "request", rejected)
    assert client.post("/v1/predict", json={"pgn": PGN}).status_code == 503
    assert rejected.call_count == 1
    attempts = []

    def transient(method, url, **kwargs):
        if "/invocations" in url:
            attempts.append(1)
            if len(attempts) < 3:
                raise api.requests.ConnectionError("unreachable")
        return original(method, url, **kwargs)

    monkeypatch.setattr(api.requests, "request", transient)
    assert client.post("/v1/predict", json={"pgn": PGN}).status_code == 200
    assert len(attempts) == 3


def test_trainer_unavailable(client, monkeypatch):
    monkeypatch.setattr(api.requests, "request", Mock(side_effect=api.requests.ConnectionError))
    for path in ("/v1/pipeline/dag", "/v1/pipeline/runs"):
        response = client.get(path)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "trainer_unavailable"
    response = client.post("/v1/pipeline/retrain", headers={"X-API-Key": "test-key"}, json={})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "trainer_unavailable"


def test_retrain_forwards_validated_key_only_to_trainer(client, monkeypatch):
    upstream = Mock(
        return_value=SimpleNamespace(
            status_code=202, ok=True, json=lambda: {"run_id": "contract-run"}
        )
    )
    monkeypatch.setattr(api.requests, "request", upstream)
    response = client.post(
        "/v1/pipeline/retrain", headers={"X-API-Key": "test-key"}, json={"reason": "fix1"}
    )
    assert response.status_code == 202
    assert response.json() == {"run_id": "contract-run"}
    args, kwargs = upstream.call_args
    assert args == ("POST", "http://trainer:8001/retrain")
    assert kwargs["headers"] == {"X-API-Key": "test-key"}
    assert kwargs["json"] == {"reason": "fix1"}
    upstream.reset_mock()
    assert client.post("/v1/pipeline/retrain", headers={"X-API-Key": "wrong"}).status_code == 401
    upstream.assert_not_called()
    client.get("/v1/pipeline/runs")
    assert upstream.call_args.kwargs["headers"] is None
