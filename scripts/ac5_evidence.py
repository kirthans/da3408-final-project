"""Reconstruct replay-month MAE from aligned Prometheus range samples (read-only)."""

import argparse
import bisect
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

QUERIES = {
    "month": "elo_replay_month",
    "mae": 'elo_batch_mae{tc="all"}',
    "version": "elo_champion_version",
}


def points(series):
    """Sort finite timestamp/value range samples, excluding missing or nonfinite values."""
    return sorted(
        (float(t), float(v)) for t, v in series.get("values", []) if math.isfinite(float(v))
    )


def reconstruct(series, runs):
    """Join identical range timestamps to avoid cross-month lookahead (DESIGN §24 A7).

    Args:
        series: Prometheus month, model MAE and champion-version range results.
        runs: Trainer history with promotion versions and replay generations.

    Returns:
        Last aligned samples per month and next-month verdicts for every drift
        promotion in the current replay generation. Incomplete pairs stay pending."""
    generation = max(
        (
            r.get("replay_generation", 0)
            + int(bool(r.get("reset") and r.get("status") == "promoted"))
            for r in runs
        ),
        default=0,
    )
    reset_at = max(
        (
            r.get("ended_at", r.get("started_at", 0))
            for r in runs
            if r.get("reset") and r.get("status") == "promoted"
        ),
        default=0,
    )
    month_points = {t: int(v) for s in series.get("month", []) for t, v in points(s)}
    versions = {t: str(int(v)) for s in series.get("version", []) for t, v in points(s)}
    values = {}
    for s in series.get("mae", []):
        model = s.get("metric", {}).get("model")
        if model in {"champion", "previous_champion"}:
            for stamp, value in points(s):
                values.setdefault(stamp, {})[model] = value
    timeline = sorted(
        (float(r.get("ended_at", r.get("started_at", 0))), str(r["registry_version"]))
        for r in runs
        if r.get("status") == "promoted" and r.get("registry_version")
    )
    times = [t for t, _ in timeline]
    months = {}
    for stamp, month in sorted(month_points.items()):
        if stamp < reset_at:
            continue
        if month <= 0:
            months.clear()
            continue
        if month % 100 not in range(1, 13):
            continue
        key = f"{month // 100:04}-{month % 100:02}"
        index = bisect.bisect_right(times, stamp) - 1
        # Prefer the aligned version sample; fallback uses only promotions at or before this time.
        version = versions.get(stamp) or (timeline[index][1] if index >= 0 else None)
        # Keep the last clock sample, even if its quality samples are missing.
        months[key] = dict(
            month=key, sampled_at=stamp, champion_version=version, **values.get(stamp, {})
        )
    promotions = []
    for run in sorted(runs, key=lambda r: r.get("started_at", 0)):
        if run.get("replay_generation", 0) != generation:
            continue
        if run.get("trigger") != "drift_alert" or run.get("status") != "promoted":
            continue
        if run.get("reset") or not run.get("replay_month"):
            continue
        year, month = map(int, run["replay_month"].split("-"))
        following = f"{year + (month == 12):04}-{month % 12 + 1:02}"
        evidence = months.get(following, {})
        new, old = evidence.get("champion"), evidence.get("previous_champion")
        expected = str(run.get("registry_version", ""))
        verdict, detail = (
            "pending",
            "Next-month paired MAE and matching champion version not available",
        )
        # AC5 needs a paired next-month sample of the promoted version, never the gate month.
        if new is not None and old is not None and evidence.get("champion_version") == expected:
            verdict = "PASS" if new < old else "FAIL"
            detail = "New champion MAE compared with previous champion on the next replay month"
        promotions.append(
            dict(
                run_id=run.get("run_id"),
                replay_month=run["replay_month"],
                replay_generation=run.get("replay_generation", 0),
                registry_version=expected,
                next_month=following,
                gate_challenger_mae=run.get("gate_challenger_mae"),
                gate_champion_mae=run.get("gate_champion_mae"),
                coverage=run.get("coverage"),
                champion_mae=new,
                previous_champion_mae=old,
                verdict=verdict,
                details=detail,
            )
        )
    verdicts = [p["verdict"] for p in promotions]
    verdict = (
        "FAIL"
        if "FAIL" in verdicts
        else "PASS" if verdicts and all(v == "PASS" for v in verdicts) else "pending"
    )
    return dict(
        months=list(months.values()),
        promotions=promotions,
        verdict=verdict,
        details="Last aligned sample per replay month; pending until every drift promotion "
        "has a next-month pair with its champion version. No promotion means pending.",
    )


def main():
    """Fetch read-only replay evidence and write an AC5 report, marking missing evidence pending."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", type=float, default=12, help="Lookback hours (default: 12)")
    parser.add_argument("--step", type=int, default=15)
    parser.add_argument("--prometheus-url", default="http://localhost:9090")
    parser.add_argument("--trainer-url", default="http://localhost:8001")
    args = parser.parse_args()
    if args.since <= 0 or args.step <= 0:
        parser.error("Window and step must be positive")
    end = time.time()
    try:
        series = {}
        for key, query in QUERIES.items():
            response = requests.get(
                args.prometheus_url.rstrip("/") + "/api/v1/query_range",
                params=dict(query=query, start=end - args.since * 3600, end=end, step=args.step),
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") != "success":
                raise ValueError("Prometheus query failed")
            series[key] = payload["data"]["result"]
        response = requests.get(
            args.trainer_url.rstrip("/") + "/runs", params={"limit": 1000}, timeout=30
        )
        response.raise_for_status()
        report = reconstruct(series, response.json())
    except (requests.RequestException, ValueError, KeyError) as exc:
        report = dict(
            months=[],
            promotions=[],
            verdict="pending",
            details=f"Evidence not available: {type(exc).__name__}",
        )
    report.update(
        generated_at=datetime.now(timezone.utc).isoformat(),
        since_hours=args.since,
        step_s=args.step,
        start=end - args.since * 3600,
        end=end,
    )
    output = Path("reports/ac5.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("month    champion version  champion MAE  previous MAE")
    for row in report["months"]:
        print(
            row["month"],
            row.get("champion_version"),
            row.get("champion"),
            row.get("previous_champion"),
        )
    print("promotion month  next month  new MAE  old MAE  verdict")
    for row in report["promotions"]:
        print(
            row["replay_month"],
            row["next_month"],
            row["champion_mae"],
            row["previous_champion_mae"],
            row["verdict"],
        )
    print(f"AC5: {report['verdict']} — {report['details']}")


if __name__ == "__main__":
    main()
