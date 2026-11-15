"""Check shared configuration, error contracts and request-aware logging."""

import re
from pathlib import Path

import pytest

from elo.common.config import get_env, load_params
from elo.common.errors import (
    DataValidationError,
    EloError,
    EngineError,
    EngineTimeoutError,
    InvalidPgnError,
    ModelUnavailableError,
    PayloadTooLargeError,
    PipelineBusyError,
)
from elo.common.logging import get_logger, request_id


def test_logger_format_and_request_context(capsys: pytest.CaptureFixture[str]) -> None:
    """Logs preserve the course fields and pick up the current request ID."""
    logger = get_logger("elo.test.common")
    assert get_logger("elo.test.common") is logger
    assert len(logger.handlers) == 1
    logger.info("default context")
    token = request_id.set("request-123")
    try:
        logger.info("active context")
    finally:
        request_id.reset(token)
    logger.info("restored context")
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 3
    assert re.match(r"\d{4}-\d{2}-\d{2} .* - elo.test.common - INFO - ", lines[0])
    assert lines[0].endswith("default context [request_id=-]")
    assert lines[1].endswith("active context [request_id=request-123]")
    assert lines[2].endswith("restored context [request_id=-]")


@pytest.mark.parametrize(
    ("error_type", "status", "code"),
    [
        (EloError, 500, "elo_error"),
        (InvalidPgnError, 400, "invalid_pgn"),
        (PayloadTooLargeError, 413, "payload_too_large"),
        (EngineError, 503, "engine_error"),
        (EngineTimeoutError, 504, "engine_timeout"),
        (ModelUnavailableError, 503, "model_unavailable"),
        (PipelineBusyError, 409, "pipeline_busy"),
        (DataValidationError, 422, "data_validation"),
    ],
)
def test_errors(error_type: type[EloError], status: int, code: str) -> None:
    """Each domain error preserves its message and stable HTTP contract."""
    error = error_type("useful message")
    assert isinstance(error, EloError)
    assert str(error) == "useful message"
    assert error.http_status == status
    assert error.code == code


def test_load_project_params() -> None:
    """The checked-in parameters have the required structure and valid values."""
    params = load_params(Path(__file__).resolve().parents[2] / "params.yaml")
    assert {"data", "sample", "engine", "seed"} <= params.keys()
    train_months = params["data"]["train_months"]
    replay_months = params["data"]["replay_months"]
    for months in (train_months, replay_months):
        assert isinstance(months, list)
        for month in months:
            assert isinstance(month, str)
            assert re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", month)
    assert set(train_months).isdisjoint(replay_months)
    for value in (
        params["data"]["max_games_per_month"],
        params["sample"]["train_per_month"],
        params["sample"]["replay_per_month"],
        params["engine"]["nodes"],
        params["seed"],
    ):
        assert type(value) is int
        assert value > 0


def test_load_params_rejects_non_mapping(tmp_path: Path) -> None:
    """An invalid parameter document fails at the configuration boundary."""
    path = tmp_path / "params.yaml"
    path.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_params(path)


def test_environment_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment settings support defaults and explicit required values."""
    monkeypatch.delenv("ELO_TEST_SETTING", raising=False)
    assert get_env("ELO_TEST_SETTING", "fallback") == "fallback"
    with pytest.raises(ValueError, match="ELO_TEST_SETTING"):
        get_env("ELO_TEST_SETTING")
    monkeypatch.setenv("ELO_TEST_SETTING", "configured")
    assert get_env("ELO_TEST_SETTING") == "configured"
