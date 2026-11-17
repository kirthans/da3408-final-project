"""PSI binning, replay persistence and gating guards."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from elo.monitoring import drift
from elo.monitoring.drift import psi
from elo.training.controller import authorize, gate, set_months


def test_psi_numeric_missing_and_shift():
    reference = dict(type="numeric", edges=[0, 1], proportions=[0.25, 0.25, 0.25, 0.25])
    assert psi(reference, [-1, 1, 2, np.nan]) == pytest.approx(0)
    assert psi(reference, [2] * 4) > 0.1
    assert psi(reference, [-1, 1, 2, 2]) > 0.1


def test_psi_boolean_categorical_and_legacy(monkeypatch):
    reference = dict(
        type="categorical", categories=[False, True, "__missing__"], proportions=[0.25, 0.5, 0.25]
    )
    assert psi(reference, [False, True, True, None]) == pytest.approx(0)
    monkeypatch.setattr(drift, "FEATURE_COLUMNS", ["has_clock"])
    frame = pd.DataFrame({"has_clock": [True, False]})
    legacy = {"has_clock": dict(type="numeric", edges=[0], proportions=[0.5, 0.5, 0])}
    drift._warned.clear()
    with pytest.warns(UserWarning, match="legacy"):
        assert drift.psi_report(legacy, frame) == {}


@pytest.mark.parametrize(
    "challenger,champion,coverage,passed",
    [
        (180, 182, 0.8, True),
        (181, 182, 0.75, True),
        (181, 182, 0.85, True),
        (181.1, 182, 0.8, False),
        (180, 182, 0.749, False),
        (180, 182, 0.851, False),
        (float("nan"), 182, 0.8, False),
    ],
)
def test_same_holdout_gate(challenger, champion, coverage, passed):
    assert (
        gate(challenger, champion, coverage, dict(min_improvement=1, coverage_band=[0.75, 0.85]))
        == passed
    )


def test_params_only_month_lines(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from pathlib import Path

    original = (
        'train:\n  fit_months: ["2019-09"]\n'
        '  holdout_month: "2019-12"\nmonitor: {interval_s: 900}\n'
    )
    Path("params.yaml").write_text(original)
    set_months(["2019-09", "2019-12"], "2020-01")
    assert Path("params.yaml").read_text().splitlines()[3:] == original.splitlines()[3:]


def test_auth_bearer_and_forwarded_key(monkeypatch):
    from starlette.requests import Request

    monkeypatch.setenv("API_KEY", "secret")
    for headers in [[], [(b"authorization", b"Bearer wrong")]]:
        with pytest.raises(HTTPException) as error:
            authorize(Request({"type": "http", "headers": headers}))
        assert error.value.status_code == 401
    for headers in [[(b"authorization", b"Bearer secret")], [(b"x-api-key", b"secret")]]:
        authorize(Request({"type": "http", "headers": headers}))


def test_clock_persistence(tmp_path, monkeypatch):
    from elo.monitoring.monitor import Monitor

    monkeypatch.setenv("MONITOR_STATE", str(tmp_path / "monitor.json"))
    monitor = Monitor()
    assert monitor.status()["current_month"] is None
    monitor.clock("advance")
    assert Monitor().status()["revealed"] == ["2020-01"]
    monitor.clock("play")
    assert json.loads((tmp_path / "monitor.json").read_text())["playing"]
    monitor.clock("reset")
    assert Monitor().status()["revealed"] == []


def test_monitor_reports_null_when_previous_champion_is_absent(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from elo.monitoring import monitor

    monkeypatch.setenv("MONITOR_STATE", str(tmp_path / "monitor.json"))

    def bundle(alias):
        if alias == "previous_champion":
            raise ValueError("Alias absent")
        return SimpleNamespace(version="4")

    monkeypatch.setattr(monitor, "load_bundle", bundle)
    clock = monitor.Monitor()
    clock.versions = {"champion_version": "4", "previous_champion_version": "7"}
    clock.evaluate()
    assert clock.status()["champion_version"] == "4"
    assert clock.status()["previous_champion_version"] is None


def test_controller_restart_busy_and_month_guard(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from elo.training import controller

    monkeypatch.setenv("TRAINER_STATE", str(tmp_path))
    monkeypatch.setattr(controller, "load_bundle", lambda _: SimpleNamespace(version="2"))
    first = controller.Controller()
    first.save(dict(run_id="interrupted", trigger="manual", replay_month=None, status="running"))
    restored = controller.Controller()
    assert restored.runs()[0]["status"] == "failed"
    assert restored.runs()[0]["error"] == "trainer restarted"
    monkeypatch.setattr(restored, "monitor", lambda: {"revealed": ["2020-04"]})
    monkeypatch.setattr(controller.threading.Thread, "start", lambda self: None)
    restored.current = "busy"
    with pytest.raises(HTTPException) as error:
        restored.start()
    assert error.value.status_code == 409
    restored.current = None
    # An alert-triggered run is limited to one per month, even after cooldown expires.
    restored.save(
        dict(
            run_id="prior",
            trigger="drift_alert",
            replay_month="2020-04",
            status="rejected",
            ended_at=0,
        )
    )
    assert restored.start("drift_alert", "ModelMAEDegraded") is None
    manual = restored.start("manual")
    assert manual is not None
    restored.current = None
    pending = restored.runs(1)[0]
    pending.update(status="failed", ended_at=0)
    restored.save(pending)
    restored.db.execute("INSERT INTO settings VALUES (?,?)", ("generation", "1"))
    restored.db.commit()
    assert restored.start("drift_alert", "ModelMAEDegraded") is not None
    assert restored.runs(1)[0]["replay_generation"] == 1


@pytest.mark.parametrize("unreachable", [False, True])
def test_auto_clock_waits_for_trainer_and_still_evaluates(tmp_path, monkeypatch, unreachable):
    import requests

    from elo.monitoring import monitor

    monkeypatch.setenv("MONITOR_STATE", str(tmp_path / "clock.json"))
    clock = monitor.Monitor()
    clock.state["playing"] = True
    evaluated = []

    def trainer(*args, **kwargs):
        if unreachable:
            raise requests.ConnectionError("offline")
        from types import SimpleNamespace

        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"in_progress": True})

    def evaluate():
        evaluated.append(True)
        clock.stop.set()
        clock.wake.set()

    monkeypatch.setattr(monitor.requests, "get", trainer)
    monkeypatch.setattr(clock, "evaluate", evaluate)
    clock.loop()
    assert evaluated == [True]
    assert clock.state["revealed"] == []


def test_dvc_run_records_reused_stages_and_summarises_upstream(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from elo.training import controller

    monkeypatch.setenv("TRAINER_STATE", str(tmp_path))
    monkeypatch.setattr(controller, "load_bundle", lambda _: SimpleNamespace(version="4"))
    runner = controller.Controller()
    output = [f"Stage 'analyze@{i}' didn't change, skipping\n" for i in range(65)]
    output += [
        "Stage 'aggregate' didn't change, skipping\n",
        "Restored stage 'build_dataset' from run-cache\n",
        "Restored stage 'train_baselines' from run-cache\n",
        "Stage 'train_xgb' didn't change, skipping\n",
        "Running stage 'evaluate':\n",
        "Running stage 'register':\n",
    ]
    monkeypatch.setattr(
        controller.subprocess,
        "Popen",
        lambda *a, **kw: SimpleNamespace(stdout=iter(output), wait=lambda: 0),
    )
    run = dict(run_id="reused", status="running", stages=[])
    runner.command(["dvc", "repro", "register"], run, stream=True)
    stages = runner.runs(1)[0]["stages"]
    assert [(s["name"], s["status"]) for s in stages] == [
        ("upstream", "skipped"),
        ("build_dataset", "restored"),
        ("train_baselines", "restored"),
        ("train_xgb", "skipped"),
        ("evaluate", "completed"),
        ("register", "completed"),
    ]
    assert stages[0]["count"] == stages[0]["skipped"] == 66


def test_command_failure_reports_first_error_line(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from elo.common.errors import EloError
    from elo.training import controller

    monkeypatch.setenv("TRAINER_STATE", str(tmp_path))
    monkeypatch.setattr(controller, "load_bundle", lambda _: SimpleNamespace(version="4"))
    runner = controller.Controller()
    monkeypatch.setattr(
        controller.subprocess,
        "Popen",
        lambda *a, **kw: SimpleNamespace(
            stdout=iter(["Collecting\n", "ERROR: connection refused\n", "Traceback...\n"]),
            wait=lambda: 1,
        ),
    )
    with pytest.raises(EloError, match="^ERROR: connection refused$"):
        runner.command(["dvc", "push"], dict(run_id="push", stages=[]))


@pytest.mark.parametrize("register_status", ["skipped", "restored"])
@pytest.mark.parametrize("reset", [False, True])
def test_reused_registration_does_not_gate_or_mutate_stale_version(
    tmp_path, monkeypatch, register_status, reset
):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from elo.training import controller

    monkeypatch.setenv("TRAINER_STATE", str(tmp_path / "state"))
    monkeypatch.setenv("ELO_DISABLE_PUSH", "1")
    monkeypatch.setattr(controller, "load_bundle", lambda _: SimpleNamespace(version="4"))
    runner = controller.Controller()
    params = controller.load_params()
    monkeypatch.setattr(controller, "load_params", lambda: params)
    monkeypatch.chdir(tmp_path)
    # No registration/metrics reports exist: stale paths must never be read.
    monkeypatch.setattr(controller, "set_months", Mock())
    monkeypatch.setattr(
        controller.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="{}")
    )
    monkeypatch.setattr(
        runner,
        "command",
        lambda args, run, *a, **kw: run["stages"].append(
            dict(name="register", status=register_status)
        ),
    )
    forbidden = Mock(side_effect=AssertionError("Stale model must not be gated or tagged"))
    for name in ("score", "gate", "MlflowClient"):
        monkeypatch.setattr(controller, name, forbidden)
    monkeypatch.setattr(controller.registry, "promote", forbidden)
    monkeypatch.setattr(controller.registry, "reject", forbidden)
    baseline_reset = Mock()
    monkeypatch.setattr(controller.registry, "reset", baseline_reset)
    clock_reset = Mock(return_value=SimpleNamespace(raise_for_status=lambda: None))
    monkeypatch.setattr(controller.requests, "post", clock_reset)
    run = dict(run_id="reused", trigger="manual", alert_name="", reset=reset, stages=[])
    runner.current = run["run_id"]
    runner.execute(run, [])
    assert run["status"] == ("promoted" if reset else "no_change")
    assert run["message"]
    assert "registry_version" not in run
    assert "gate_champion_mae" not in run
    forbidden.assert_not_called()
    if reset:
        baseline_reset.assert_called_once_with("4")
        clock_reset.assert_called_once()
        assert runner.generation() == 1
    else:
        baseline_reset.assert_not_called()
        clock_reset.assert_not_called()
    assert runner.runs(1)[0]["status"] == run["status"]
    assert runner.current is None
