"""Single-run retraining controller with durable history and a same-holdout gate."""

import hashlib
import hmac
import json
import os
import re
import sqlite3
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from fastapi import Depends, FastAPI, HTTPException, Request
from mlflow import MlflowClient
from prometheus_client import Counter, Gauge, start_http_server

from elo.common.config import load_params
from elo.common.errors import EloError
from elo.common.logging import get_logger
from elo.monitoring.champion import load_bundle, score
from elo.training import registry

LOG = get_logger(__name__)
RUNS = Counter("elo_pipeline_runs_total", "Completed pipeline runs", ["status"])
ALERTS = Counter("elo_alerts_received_total", "Alert decisions", ["alertname", "action"])
DURATION = Gauge("elo_stage_duration_seconds", "Last stage duration", ["stage"])
THROUGHPUT = Gauge("elo_stage_throughput_games_per_second", "Last throughput", ["stage"])
BUSY = Gauge("elo_retrain_in_progress", "Retraining active")
VERSION = Gauge("elo_champion_version", "Champion registry version")
LAST = Gauge("elo_last_retrain_timestamp_seconds", "Last completed run")
for _status in ["promoted", "rejected", "failed", "no_change"]:
    RUNS.labels(_status)


def authorize(request: Request):
    """Accept a constant-time API-key or Bearer-key match, otherwise raise 401 (DESIGN D20)."""
    key = os.getenv("API_KEY", "")
    supplied = request.headers.get("X-API-Key", "")
    bearer = request.headers.get("Authorization", "")
    if not key or not (
        hmac.compare_digest(supplied, key) or hmac.compare_digest(bearer, "Bearer " + key)
    ):
        raise HTTPException(401, "Invalid API key")


def gate(challenger_mae, champion_mae, coverage, config):
    """Require an MAE improvement and calibrated coverage before promotion.

    Args:
        challenger_mae: Challenger error on the current untouched test split.
        champion_mae: Serving model error on that same split.
        coverage: Challenger conformal interval coverage on the split.
        config: Minimum improvement and inclusive coverage band.

    Returns:
        Whether both promotion conditions pass (DESIGN §24 A3-A4)."""
    return (
        challenger_mae <= champion_mae - config["min_improvement"]
        and config["coverage_band"][0] <= coverage <= config["coverage_band"][1]
    )


def set_months(fit, holdout):
    """Only replace the two train lines; leave unrelated formatting and settings intact."""
    path = Path("params.yaml")
    text = path.read_text()
    text, n = re.subn(r"(?m)^  fit_months:.*$", "  fit_months: " + json.dumps(fit), text)
    text, m = re.subn(r"(?m)^  holdout_month:.*$", "  holdout_month: " + json.dumps(holdout), text)
    if (n, m) != (1, 1):
        raise EloError("Expected one inline fit_months and holdout_month in params.yaml")
    path.write_text(text)


class Controller:
    """Serialize retraining, persist history and gate serving alias changes (DESIGN D19)."""

    def __init__(self):
        """Restore SQLite history, mark interrupted runs failed and retain a reset baseline."""
        self.root = Path(os.getenv("TRAINER_STATE", "/state"))
        (self.root / "runs").mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root / "trainer.db", check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, body TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
        # A background worker cannot survive a restart; do not leave stale runs marked running.
        for run in self.runs():
            if run["status"] == "running":
                run.update(status="failed", error="trainer restarted", ended_at=time.time())
                self.save(run)
        last = self.runs(1)
        LAST.set(last[0].get("ended_at", 0) if last else 0)
        self.current = None
        self.dag_cache = None
        baseline = self.db.execute("SELECT value FROM settings WHERE key='baseline'").fetchone()
        if not baseline:
            value = os.getenv("BASELINE_VERSION") or load_bundle("champion").version
            self.db.execute("INSERT INTO settings VALUES (?, ?)", ("baseline", value))
            self.db.commit()
        VERSION.set(float(load_bundle("champion").version))
        for path in Path("reports/throughput").glob("*.json"):
            report = json.loads(path.read_text())
            THROUGHPUT.labels("analyze").set(report["games_per_second"])

    def save(self, run):
        """Commit a complete run snapshot under the controller lock."""
        with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?)", (run["run_id"], json.dumps(run))
            )
            self.db.commit()

    def runs(self, limit=10000):
        """Read up to limit persisted runs in reverse insertion order."""
        with self.lock:
            rows = self.db.execute(
                "SELECT body FROM runs ORDER BY rowid DESC LIMIT ?", (limit,)
            ).fetchall()
            return [json.loads(row[0]) for row in rows]

    def monitor(self):
        """Fetch revealed-month state from the monitor, raising on upstream errors."""
        response = requests.get(
            os.getenv("MONITOR_URL", "http://monitor:8002") + "/status", timeout=10
        )
        response.raise_for_status()
        return response.json()

    def status(self):
        """Expose active run, champion version and remaining cooldown seconds."""
        runs = self.runs(1)
        last = runs[0] if runs else None
        cooldown = load_params()["retrain"]["cooldown_s"]
        VERSION.set(float(load_bundle("champion").version))
        return dict(
            in_progress=self.current is not None,
            current_run=self.current,
            champion_version=load_bundle("champion").version,
            last_run=last,
            cooldown_remaining_s=max(0, cooldown - time.time() + (last or {}).get("ended_at", 0)),
        )

    def generation(self):
        """Read the replay generation used to deduplicate alerts across resets."""
        row = self.db.execute("SELECT value FROM settings WHERE key='generation'").fetchone()
        return int(row[0]) if row else 0

    def start(self, trigger="manual", alert_name="", reset=False, reason=""):
        """Reserve the single run slot and launch a durable background run.

        Args:
            trigger: Manual or drift_alert origin.
            alert_name: Alert label retained for provenance.
            reset: Restore baseline training months and replay state after training.
            reason: Human explanation retained in run history.

        Returns:
            A new run ID, or None for a cooldown or duplicate-month drift alert.

        Raises:
            HTTPException: A run already occupies the slot (409)."""
        with self.lock:
            if self.current:
                raise HTTPException(409, "Retrain already running")
            months = self.monitor()["revealed"]
            generation = self.generation()
            if trigger == "drift_alert":
                history = self.runs()
                # Cooldown limits alert storms; generation/month dedup still works after it expires.
                if (
                    history
                    and time.time() - history[0].get("ended_at", time.time())
                    < load_params()["retrain"]["cooldown_s"]
                ) or any(
                    r["trigger"] == trigger
                    and r.get("replay_generation", 0) == generation
                    and r["replay_month"] == (months or [None])[-1]
                    for r in history
                ):
                    return None
            run = dict(
                run_id=uuid.uuid4().hex,
                trigger=trigger,
                alert_name=alert_name,
                reason=reason,
                replay_month=(months or [None])[-1],
                replay_generation=generation,
                status="running",
                stages=[],
                mlflow_run_id=None,
                started_at=time.time(),
                reset=reset,
            )
            self.current = run["run_id"]
            BUSY.set(1)
            self.save(run)
            threading.Thread(target=self.execute, args=(run, months), daemon=True).start()
            return run["run_id"]

    def command(self, args, run, env=None, stream=False):
        """Run a subprocess and persist streamed DVC stage transitions.

        Args:
            args: Executable and argument sequence.
            run: Mutable run snapshot whose stages are updated.
            env: Optional subprocess environment.
            stream: Parse DVC running, restored and skipped stage messages.

        Raises:
            EloError: The subprocess fails; its first ERROR line is retained if present."""
        with (self.root / "runs" / f"{run['run_id']}.log").open("a") as log:
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1,
            )
            stage = None
            first_error = None
            for line in process.stdout:
                log.write(line)
                log.flush()
                if first_error is None and line.lstrip().startswith("ERROR:"):
                    first_error = line.strip()
                if stream:
                    match = re.search(r"Running stage ['\"]([^'\"]+)", line)
                    if match:
                        if stage:
                            self.finish_stage(stage)
                        stage = dict(
                            name=match[1],
                            status="running",
                            started_at=time.time(),
                            duration_s=0,
                            games_per_s=None,
                        )
                        run["stages"].append(stage)
                        self.save(run)
                    reused = re.search(
                        r"Restored stage ['\"]([^'\"]+)['\"] from run-cache|"
                        r"Stage ['\"]([^'\"]+)['\"] didn't change, skipping",
                        line,
                    )
                    if reused:
                        if stage:
                            self.finish_stage(stage)
                            stage = None
                        name = reused[1] or reused[2]
                        # Reuse is history evidence, not fresh execution or a new model candidate.
                        status = "restored" if reused[1] else "skipped"
                        if name in {
                            "build_dataset",
                            "train_baselines",
                            "train_xgb",
                            "evaluate",
                            "register",
                        }:
                            now = time.time()
                            run["stages"].append(
                                dict(
                                    name=name,
                                    status=status,
                                    started_at=now,
                                    ended_at=now,
                                    duration_s=0,
                                    games_per_s=None,
                                )
                            )
                        else:
                            upstream = next(
                                (s for s in run["stages"] if s["name"] == "upstream"), None
                            )
                            if upstream is None:
                                upstream = dict(
                                    name="upstream",
                                    status=status,
                                    count=0,
                                    restored=0,
                                    skipped=0,
                                    duration_s=0,
                                    games_per_s=None,
                                )
                                run["stages"].append(upstream)
                            upstream["count"] += 1
                            upstream[status] += 1
                            if upstream["restored"] and upstream["skipped"]:
                                upstream["status"] = "reused"
                        self.save(run)
            code = process.wait()
            if stage:
                self.finish_stage(stage, "failed" if code else "completed")
                self.save(run)
            if code:
                raise EloError(first_error or f"Command {args} failed ({code}); see run log")

    def begin_stage(self, run, name):
        """Append and persist a running stage, returning its mutable snapshot."""
        stage = dict(
            name=name, status="running", started_at=time.time(), duration_s=0, games_per_s=None
        )
        run["stages"].append(stage)
        self.save(run)
        return stage

    def finish_stage(self, stage, status="completed"):
        """Finalize stage timing and publish duration and training throughput."""
        stage.update(
            status=status, ended_at=time.time(), duration_s=time.time() - stage["started_at"]
        )
        DURATION.labels(stage["name"]).set(stage["duration_s"])
        if stage["name"].startswith("train"):
            rows = sum(
                v["games"]
                for v in json.loads(Path("reports/dataset.json").read_text())["splits"].values()
            )
            stage["games_per_s"] = rows / max(stage["duration_s"], 1e-6)
            THROUGHPUT.labels(stage["name"]).set(stage["games_per_s"])

    def execute(self, run, months):
        """Run cached-upstream retraining and gate the registered challenger.

        Args:
            run: Reserved run snapshot, updated through terminal status.
            months: Revealed replay months captured when the run was accepted.

        Errors become failed history entries; push failures remain pending without
        undoing the gate outcome. The newest month is held out (DESIGN §24 A2-A3)."""
        try:
            step = self.begin_stage(run, "months")
            p = load_params()
            baseline = p["data"]["train_months"]
            # Hold out the latest revealed month; both models face its disjoint test third (A3-A4).
            fit = baseline[:-1] if not months or run["reset"] else baseline + months[:-1]
            holdout = baseline[-1] if not months or run["reset"] else months[-1]
            run.update(fit_months=fit, holdout_month=holdout)
            parent = load_bundle("champion")
            set_months(fit, holdout)
            self.finish_stage(step)
            self.save(run)
            # Upstream stages require Java/Stockfish and must remain cached in this image.
            status = subprocess.run(["dvc", "status", "--json"], capture_output=True, text=True)
            if status.returncode:
                raise EloError("DVC preflight failed: " + status.stderr.strip())
            dirty = json.loads(status.stdout)
            forbidden = [
                s
                for s in dirty
                if s.split("@")[0] in {"download", "ingest_validate", "sample", "analyze"}
            ]
            if forbidden:
                raise EloError(f"Uncached upstream stages require host rebuild: {forbidden}")
            env = dict(
                os.environ,
                ELO_TRIGGER=run["trigger"],
                ELO_ALERT_NAME=run["alert_name"],
                ELO_PARENT_CHAMPION_VERSION=parent.version,
            )
            self.command(["dvc", "repro", "register"], run, env, stream=True)
            step = self.begin_stage(run, "gate")
            reused_register = any(
                stage["name"] == "register" and stage["status"] in {"skipped", "restored"}
                for stage in run["stages"]
            )
            if run["reset"]:
                version = (
                    os.getenv("BASELINE_VERSION")
                    or self.db.execute(
                        "SELECT value FROM settings WHERE key='baseline'"
                    ).fetchone()[0]
                )
                registry.reset(version)
                response = requests.post(
                    os.getenv("MONITOR_URL", "http://monitor:8002") + "/clock/reset",
                    headers={"Authorization": "Bearer " + os.getenv("API_KEY", "")},
                    timeout=30,
                )
                response.raise_for_status()
                # A fresh replay may alert for the same month; retain all earlier run history.
                with self.lock:
                    self.db.execute(
                        "INSERT OR REPLACE INTO settings VALUES (?,?)",
                        ("generation", str(self.generation() + 1)),
                    )
                    self.db.commit()
                outcome = "promoted"
                run["message"] = "Baseline restored; registration report was not gated or tagged."
            elif reused_register:
                outcome = "no_change"
                run["message"] = (
                    "Registration was skipped or restored; no new model to gate. "
                    "Existing versions and tags are unchanged."
                )
                LOG.info(run["message"])
            else:
                registered = json.loads(Path("reports/register.json").read_text())
                metrics = json.loads(Path("reports/metrics.json").read_text())
                report = json.loads(Path("reports/train_xgb.json").read_text())
                run.update(registry_version=registered["version"], mlflow_run_id=report["run_id"])
                frame = pd.read_parquet("data/datasets/test.parquet")
                # Re-score the old champion on this exact holdout, not its original test MAE.
                prediction, _ = score(parent, frame)
                champion_mae = float(np.abs(frame.elo - prediction).mean())
                run.update(
                    gate_champion_mae=champion_mae,
                    gate_challenger_mae=metrics["test_mae"],
                    coverage=metrics["coverage"],
                )
                passed = gate(metrics["test_mae"], champion_mae, metrics["coverage"], p["retrain"])
                (registry.promote if passed else registry.reject)(registered["version"])
                outcome = "promoted" if passed else "rejected"
                tags = dict(
                    gate="passed" if passed else "rejected",
                    gate_champion_mae=champion_mae,
                    gate_challenger_mae=metrics["test_mae"],
                    pending_commit="true",
                    base_git_commit=subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True
                    ).strip(),
                    git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
                    dvc_lock_md5=hashlib.md5(Path("dvc.lock").read_bytes()).hexdigest(),
                )
                client = MlflowClient()
                for key, value in tags.items():
                    client.set_model_version_tag(
                        registry.NAME, registered["version"], key, str(value)
                    )
                    client.set_tag(report["run_id"], key, str(value))
            self.finish_stage(step)
            step = self.begin_stage(run, "push")
            try:
                if os.getenv("ELO_DISABLE_PUSH") == "1":
                    raise EloError("Push disabled for authorized M7 verification (no dvc push)")
                self.command(["dvc", "push"], run)
                run["push"] = "complete"
            # A remote push failure does not undo a completed gate or serving-alias change.
            except Exception as exc:
                run.update(push="pending", push_error=str(exc).splitlines()[0][:500])
            self.finish_stage(step, run["push"])
            run["status"] = outcome
            VERSION.set(float(load_bundle("champion").version))
        except Exception as exc:
            LOG.exception("Retraining failed")
            run.update(status="failed", error=str(exc))
            for stage in run["stages"]:
                if stage["status"] == "running":
                    self.finish_stage(stage, "failed")
                    stage["error"] = str(exc)
        finally:
            with self.lock:
                run["ended_at"] = time.time()
                self.save(run)
                RUNS.labels(run["status"]).inc()
                LAST.set(run["ended_at"])
                BUSY.set(0)
                self.current = None


@asynccontextmanager
async def lifespan(app):
    """Initialize durable controller state and its separate metrics port (DESIGN D16)."""
    app.state.controller = Controller()
    start_http_server(int(os.getenv("METRICS_PORT", "18002")))
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    """Return trainer process liveness."""
    return {"status": "ok"}


@app.get("/status")
def status():
    """Expose the active run and cooldown state (DESIGN §9)."""
    return app.state.controller.status()


@app.get("/runs")
def runs(limit: int = 20):
    """Return persisted run history with the limit clamped to 1..1000 (DESIGN §9)."""
    return app.state.controller.runs(max(1, min(limit, 1000)))


@app.get("/runs/{run_id}")
def run(run_id: str):
    """Look up a persisted run by ID (DESIGN §9).

    Args:
        run_id: Run identifier returned on acceptance.

    Returns:
        The matching run snapshot.

    Raises:
        HTTPException: No matching run exists (404)."""
    result = next((r for r in app.state.controller.runs() if r["run_id"] == run_id), None)
    if result is None:
        raise HTTPException(404, "Unknown run")
    return result


@app.get("/dag")
def dag():
    """Return Mermaid DVC dependencies, refreshing the cache when dvc.yaml changes."""
    controller = app.state.controller
    stamp = Path("dvc.yaml").stat().st_mtime_ns
    if not controller.dag_cache or controller.dag_cache[0] != stamp:
        controller.dag_cache = (stamp, subprocess.check_output(["dvc", "dag", "--md"], text=True))
    return {"mermaid": controller.dag_cache[1]}


@app.post("/retrain", status_code=202, dependencies=[Depends(authorize)])
def retrain(payload: dict):
    """Queue an authenticated manual run and return its ID (DESIGN §9)."""
    return {"run_id": app.state.controller.start(reason=str(payload.get("reason", "")))}


@app.post("/replay/reset", status_code=202, dependencies=[Depends(authorize)])
def reset():
    """Queue baseline restoration and replay reset with the normal single-run guard."""
    return {"run_id": app.state.controller.start(reset=True)}


@app.post("/alerts", dependencies=[Depends(authorize)])
def alerts(payload: dict):
    """Accept only firing MAE alerts, reporting cooldown and busy decisions (DESIGN §24 A1)."""
    decisions = []
    for alert in payload.get("alerts", []):
        name = alert.get("labels", {}).get("alertname", "")
        action = "ignored_resolved"
        if alert.get("status", payload.get("status")) == "firing" and name == "ModelMAEDegraded":
            try:
                run_id = app.state.controller.start("drift_alert", name)
                action = "accepted" if run_id else "ignored_cooldown"
            except HTTPException as exc:
                if exc.status_code != 409:
                    raise
                action = "ignored_busy"
        LOG.info("Alert %s -> %s", name, action)
        ALERTS.labels(name, action).inc()
        decisions.append(dict(alertname=name, action=action))
    return {"alerts": decisions}
