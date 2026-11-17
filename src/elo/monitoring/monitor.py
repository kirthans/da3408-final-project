"""Persistent replay clock and labelled monthly model monitoring."""

import hmac
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from fastapi import Depends, FastAPI, HTTPException, Request
from prometheus_client import Gauge, start_http_server

from elo.common.config import load_params
from elo.common.logging import get_logger
from elo.monitoring.champion import load_bundle, score
from elo.monitoring.drift import psi_report

LOG = get_logger(__name__)
G = {
    name: Gauge("elo_" + name, name)
    for name in [
        "replay_month",
        "replay_months_revealed",
        "reference_mae",
        "drift_detected",
        "drift_injection_enabled",
        "batch_rows",
        "monitor_last_eval_timestamp_seconds",
    ]
}
MAE = Gauge("elo_batch_mae", "Monthly absolute error", ["model", "tc"])
COVER = Gauge("elo_interval_coverage", "Champion coverage", ["tc"])
PSI = Gauge("elo_feature_psi", "Population stability", ["feature"])


def authorize(request: Request):
    key = os.getenv("API_KEY", "")
    if not key or not hmac.compare_digest(
        request.headers.get("Authorization", ""), "Bearer " + key
    ):
        raise HTTPException(401, "Invalid API key")


class Monitor:
    def __init__(self):
        self.params = load_params()
        self.cfg = self.params["monitor"]
        self.path = Path(os.getenv("MONITOR_STATE", "/state/monitor.json"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else dict(revealed=[], playing=False, last_reveal=0)
        )
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.metrics, self.versions = {}, {}
        self.injection = (
            self.cfg["drift_injection"]["enabled"] or os.getenv("DRIFT_INJECTION") == "1"
        )
        G["drift_injection_enabled"].set(self.injection)
        self.clear_quality()
        self.clock_metrics()

    def clear_quality(self):
        MAE.clear()
        COVER.clear()
        PSI.clear()
        for name in ["reference_mae", "batch_rows", "monitor_last_eval_timestamp_seconds"]:
            G[name].set(float("nan"))
        G["drift_detected"].set(0)

    def clock_metrics(self):
        months = self.state["revealed"]
        G["replay_month"].set(int(months[-1].replace("-", "")) if months else 0)
        G["replay_months_revealed"].set(len(months))

    def interval(self):
        return float(os.getenv("REPLAY_INTERVAL_S", self.cfg["interval_s"]))

    def status(self):
        with self.lock:
            return dict(
                **self.state,
                current_month=(self.state["revealed"] or [None])[-1],
                seconds_to_next_reveal=max(
                    0, self.interval() - time.time() + self.state["last_reveal"]
                ),
                **self.versions,
                last_metrics=self.metrics,
                psi_top5=sorted(self.metrics.get("psi", {}).items(), key=lambda x: -x[1])[:5],
                drift_injection=dict(self.cfg["drift_injection"], enabled=self.injection),
            )

    def clock(self, action):
        with self.lock:
            if action == "reset":
                self.state = dict(revealed=[], playing=False, last_reveal=0)
                self.metrics = {}
                self.clear_quality()
            elif action == "advance":
                remaining = [
                    m
                    for m in self.params["data"]["replay_months"]
                    if m not in self.state["revealed"]
                ]
                if remaining:
                    self.state["revealed"].append(remaining[0])
                    self.state["last_reveal"] = time.time()
            else:
                self.state["playing"] = action == "play"
            temp = self.path.with_suffix(".tmp")
            temp.write_text(json.dumps(self.state))
            temp.replace(self.path)
            self.clock_metrics()
            self.wake.set()
        return self.status()

    def evaluate(self):
        with self.lock:
            month = (self.state["revealed"] or [None])[-1]
            champion = load_bundle("champion")
            bundles = {"champion": champion}
            try:
                bundles["previous_champion"] = load_bundle("previous_champion")
            except Exception:
                LOG.info("No previous champion available")
            self.versions = {
                "previous_champion_version": None,
                **{name + "_version": bundle.version for name, bundle in bundles.items()},
            }
            if not month:
                return
            frame = pd.read_parquet(f"data/features/games/{month}.parquet")
            if self.injection and month == self.cfg["drift_injection"]["month"]:
                LOG.warning("DRIFT INJECTION enabled for %s", month)
                columns = [c for c in champion.columns if "cpl" in c]
                frame[columns] *= self.cfg["drift_injection"]["cpl_scale"]
            errors, coverages = {}, {}
            for name, bundle in bundles.items():
                pred, radius = score(bundle, frame)
                err = np.abs(frame.elo.to_numpy() - pred)
                errors[name] = {}
                for tc in ["all", "blitz", "rapid", "classical"]:
                    mask = (
                        np.ones(len(frame), dtype=bool)
                        if tc == "all"
                        else frame.tc_class.eq(tc).to_numpy()
                    )
                    value = float(err[mask].mean()) if mask.any() else float("nan")
                    errors[name][tc] = value
                    MAE.labels(name, tc).set(value)
                    if name == "champion":
                        coverage = (
                            float((err[mask] <= radius[mask]).mean())
                            if mask.any()
                            else float("nan")
                        )
                        coverages[tc] = coverage
                        COVER.labels(tc).set(coverage)
            drift = psi_report(champion.reference, frame)
            for feature, value in drift.items():
                PSI.labels(feature).set(value)
            G["reference_mae"].set(champion.test_mae)
            G["batch_rows"].set(len(frame))
            G["drift_detected"].set(
                errors["champion"]["all"] > self.cfg["mae_ratio"] * champion.test_mae
                or max(drift.values(), default=0) > self.cfg["psi_warn"]
            )
            G["monitor_last_eval_timestamp_seconds"].set(time.time())
            self.metrics = dict(
                month=month,
                mae=errors,
                coverage=coverages,
                reference_mae=champion.test_mae,
                psi=drift,
            )

    def loop(self):
        while not self.stop.is_set():
            try:
                if (
                    self.state["playing"]
                    and time.time() - self.state["last_reveal"] >= self.interval()
                ):
                    status = requests.get(
                        os.getenv("TRAINER_URL", "http://trainer:8001") + "/status", timeout=5
                    )
                    status.raise_for_status()
                    if not status.json()["in_progress"]:
                        self.clock("advance")
            except Exception:
                LOG.exception("Trainer unreachable; replay clock waiting")
            try:
                self.evaluate()
            except Exception:
                LOG.exception("Monitor evaluation failed; waiting")
            self.wake.wait(self.cfg["eval_interval_s"])
            self.wake.clear()


@asynccontextmanager
async def lifespan(app):
    app.state.monitor = Monitor()
    start_http_server(int(os.getenv("METRICS_PORT", "18001")))
    thread = threading.Thread(target=app.state.monitor.loop, daemon=True)
    thread.start()
    yield
    app.state.monitor.stop.set()
    app.state.monitor.wake.set()
    thread.join(timeout=10)


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    return app.state.monitor.status()


@app.post("/clock/{action}", dependencies=[Depends(authorize)])
def clock(action: str):
    if action not in {"advance", "play", "pause", "reset"}:
        raise HTTPException(404)
    return app.state.monitor.clock(action)
