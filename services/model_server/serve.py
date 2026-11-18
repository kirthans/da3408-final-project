"""Restart MLflow serving when the champion alias changes."""

import os
import signal
import subprocess
import time

from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Gauge, start_http_server

from elo.common.logging import get_logger

LOG = get_logger(__name__)
NAME = "elo-estimator"
URI = f"models:/{NAME}@champion"
VERSION = Gauge("elo_model_server_serving_version", "Registry version served")
RELOADS = Counter("elo_model_server_reloads_total", "Restarts caused by alias changes")
CHILD_UP = Gauge("elo_model_server_child_up", "Serving child is running")


def main():
    """Poll champion and restart the MLflow child when its version changes (DESIGN D9, D11)."""
    start_http_server(int(os.getenv("METRICS_PORT", "18003")))
    client = MlflowClient()
    child = None

    def stop(*_):
        """Terminate the child gracefully, then kill it if the ten-second deadline expires."""
        CHILD_UP.set(0)
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()

    def shutdown(*_):
        """Stop the serving child and exit cleanly on a termination signal."""
        stop()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    current = None
    try:
        while True:
            if child is not None and child.poll() is not None:
                raise SystemExit(child.returncode or 1)
            try:
                version = client.get_model_version_by_alias(NAME, "champion").version
            except Exception:
                if child is None:
                    raise
                # A tracking-server outage must not discard an already serving model (DESIGN D9).
                LOG.exception("Alias polling failed; retaining current server")
                time.sleep(float(os.getenv("ALIAS_POLL_S", "60")))
                continue
            # MLflow resolves the alias at startup; a new version needs a fresh child (DESIGN D11).
            if version != current:
                if current is not None:
                    RELOADS.inc()
                stop()
                LOG.info("Champion changed %s → %s; restarting model server", current, version)
                child = subprocess.Popen(
                    [
                        "mlflow",
                        "models",
                        "serve",
                        "-m",
                        URI,
                        "--env-manager",
                        "local",
                        "-h",
                        "0.0.0.0",
                        "-p",
                        "5001",
                        "--workers",
                        "1",
                    ]
                )
                current = version
                VERSION.set(float(version))
                CHILD_UP.set(1)
            # Check child health each second even with a long alias polling interval.
            deadline = time.monotonic() + float(os.getenv("ALIAS_POLL_S", "60"))
            while time.monotonic() < deadline:
                CHILD_UP.set(child.poll() is None)
                if child.poll() is not None:
                    raise SystemExit(child.returncode or 1)
                time.sleep(min(1, max(0, deadline - time.monotonic())))
    finally:
        stop()


if __name__ == "__main__":
    main()
