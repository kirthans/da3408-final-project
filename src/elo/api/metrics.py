"""Single-process API metrics; route labels are templates, never user paths."""

import os
from threading import Lock

from prometheus_client import Counter, Gauge, Histogram, Info, Summary, start_http_server

REQUESTS = Counter("elo_requests_total", "Completed HTTP requests", ["endpoint", "status"])
LATENCY = Histogram(
    "elo_request_latency_seconds",
    "HTTP request time",
    ["endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)
INFLIGHT = Gauge("elo_inflight_requests", "HTTP requests currently running")
PGN_BYTES = Summary("elo_pgn_bytes", "Submitted PGN bytes")
ENGINE = Histogram(
    "elo_engine_analysis_seconds", "Engine analysis time", buckets=(0.1, 0.5, 1, 2, 5, 10)
)
PARSE_ERRORS = Counter("elo_parse_errors_total", "Rejected requests by error code", ["reason"])
RATING = Histogram(
    "elo_predicted_rating",
    "Predicted player Elo",
    ["side", "tc"],
    buckets=tuple(range(600, 3201, 200)),
)
INTERVAL = Histogram(
    "elo_interval_width",
    "Prediction interval width in Elo",
    buckets=(100, 200, 400, 600, 800, 1200),
)
MODEL_ERRORS = Counter("elo_model_server_errors_total", "Failed model-server calls or responses")
BUILD = Info("elo_build", "API build metadata")
_LOCK = Lock()
_SERVER = None


def start_metrics_server():
    """Start once; pytest opts in explicitly to avoid opening a socket in tests."""
    global _SERVER
    if os.getenv("ELO_METRICS", "0" if "PYTEST_CURRENT_TEST" in os.environ else "1") == "0":
        return
    with _LOCK:
        if _SERVER is None:
            BUILD.info({"version": "0.1.0", "git_commit": os.getenv("GIT_COMMIT", "unknown")})
            _SERVER = start_http_server(int(os.getenv("METRICS_PORT", "18000")))
