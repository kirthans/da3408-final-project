"""Closed-loop prediction load: each client waits for its response before sending again."""

import argparse
import concurrent.futures
import io
import itertools
import json
import math
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import chess.pgn
import requests


def percentile(values, quantile):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def measure(base_url, games, concurrency, duration):
    samples, codes = [], Counter()
    lock, sequence = threading.Lock(), itertools.count()
    started = time.monotonic()
    deadline = started + duration

    def client():
        with requests.Session() as session:
            while time.monotonic() < deadline:
                with lock:
                    game = games[next(sequence) % len(games)]
                before = time.monotonic()
                try:
                    response = session.post(
                        base_url.rstrip("/") + "/v1/predict", json={"pgn": game}, timeout=120
                    )
                    code = str(response.status_code)
                except requests.RequestException:
                    code = "network_error"
                elapsed = time.monotonic() - before
                with lock:
                    samples.append(elapsed)
                    codes[code] += 1
                if code == "network_error":
                    # Retry backoff is excluded from request latency.
                    time.sleep(0.05)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(lambda _: client(), range(concurrency)))
    elapsed = time.monotonic() - started
    errors = sum(n for code, n in codes.items() if not code.startswith("2"))
    return dict(
        concurrency=concurrency,
        duration_s=elapsed,
        requests=len(samples),
        p50_s=percentile(samples, 0.5),
        p95_s=percentile(samples, 0.95),
        p99_s=percentile(samples, 0.99),
        mean_s=sum(samples) / len(samples) if samples else None,
        rps=len(samples) / elapsed,
        errors=errors,
        status_codes=dict(codes),
        ac3_pass=bool(samples) and errors == 0 and percentile(samples, 0.95) <= 5,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    clients = [int(n) for n in args.concurrency.split(",")]
    if args.duration <= 0 or any(n <= 0 for n in clients):
        parser.error("Duration and concurrency must be positive")
    source = io.StringIO(Path("tests/fixtures/pgn/serving.pgn").read_text())
    games = []
    while (game := chess.pgn.read_game(source)) is not None:
        games.append(str(game))
    if not games:
        parser.error("No fixture games available")
    results = []
    print("clients  p50(s)  p95(s)  p99(s)  mean(s)  RPS    errors  status codes", flush=True)
    for concurrency in clients:
        row = measure(args.base_url, games, concurrency, args.duration)
        results.append(row)
        print(
            f"{concurrency:7}  {row['p50_s']:.3f}   {row['p95_s']:.3f}   "
            f"{row['p99_s']:.3f}   {row['mean_s']:.3f}    {row['rps']:.2f}  "
            f"{row['errors']:6}  {row['status_codes']}",
            flush=True,
        )
    output = Path("reports/load_test.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(
        json.dumps(
            dict(
                generated_at=datetime.now(timezone.utc).isoformat(),
                base_url=args.base_url,
                fixture="tests/fixtures/pgn/serving.pgn",
                results=results,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
