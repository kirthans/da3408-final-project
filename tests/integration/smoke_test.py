"""Opt-in live stack smoke and sequential 40-move prediction latency."""

import os
import time
from pathlib import Path

import numpy as np
import pytest
import requests

pytestmark = pytest.mark.skipif(
    os.getenv("ELO_SMOKE") != "1", reason="Set ELO_SMOKE=1 for live stack"
)


def test_live_stack():
    base = os.getenv("API_URL", "http://localhost:8000")
    for path in ("/health", "/ready", "/v1/model"):
        response = requests.get(base + path, timeout=10)
        print(path, response.status_code, response.text)
        assert response.status_code == 200
    pgn = Path("tests/fixtures/pgn/serving.pgn").read_text()
    latencies = []
    for _ in range(20):
        started = time.monotonic()
        response = requests.post(base + "/v1/predict", json={"pgn": pgn}, timeout=120)
        latencies.append(time.monotonic() - started)
        assert response.status_code == 200, response.text
        assert response.json()["analysis"]["nodes"] == 5000
    p50, p95 = np.percentile(latencies, [50, 95])
    print(f"20 sequential predictions: p50={p50:.3f}s p95={p95:.3f}s latencies={latencies}")
    assert p95 <= 5, f"AC3 failed: p95={p95:.3f}s"
    invalid = requests.post(base + "/v1/predict", json={"pgn": "garbage"}, timeout=10)
    assert 400 <= invalid.status_code < 500
