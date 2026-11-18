"""Generate acceptance evidence solely from existing report files."""

import json
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

COUNTS = ("total", "passed", "failed", "skipped", "errors")


def parse_junit(path):
    """Count testcase outcomes once, supporting testsuite and testsuites roots."""
    files = defaultdict(lambda: dict.fromkeys(COUNTS, 0))
    for case in ET.parse(path).getroot().iter("testcase"):
        name = case.get("file") or case.get("classname", "unknown").replace(".", "/") + ".py"
        file_path = Path(name)
        if file_path.is_absolute():
            # Older pytest reports may retain a different machine's checkout root.
            parts = file_path.parts
            name = (
                Path(*parts[parts.index("tests") :]).as_posix()
                if "tests" in parts
                else file_path.name
            )
        row = files[name]
        row["total"] += 1
        outcome = next(
            (
                k
                for tag, k in [("error", "errors"), ("failure", "failed"), ("skipped", "skipped")]
                if case.find(tag) is not None
            ),
            "passed",
        )
        row[outcome] += 1
    totals = {k: sum(r[k] for r in files.values()) for k in COUNTS}
    suites = ET.parse(path).getroot()
    timestamp = suites.get("timestamp") or next(
        (s.get("timestamp") for s in suites.iter("testsuite") if s.get("timestamp")),
        "not available",
    )
    return dict(totals=totals, files=dict(sorted(files.items())), timestamp=timestamp)


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def test_verdict(files, prefix):
    selected = [r for name, r in files.items() if name.startswith(prefix)]
    if not selected or not sum(r["total"] for r in selected):
        return "not available", "not available"
    counts = {k: sum(r[k] for r in selected) for k in COUNTS}
    verdict = (
        "FAIL"
        if counts["failed"] or counts["errors"]
        else "PASS" if counts["passed"] == counts["total"] else "not available"
    )
    return ", ".join(f"{counts[k]} {k}" for k in COUNTS), verdict


def generate(root=Path(".")):
    reports = root / "reports"
    junit_path = reports / "tests.xml"
    junit = parse_junit(junit_path) if junit_path.exists() else None
    metrics, baselines = load(reports / "metrics.json"), load(reports / "baselines.json")
    loadtest, ac5 = load(reports / "load_test.json"), load(reports / "ac5.json")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "not available"
    lines = [
        "# Test report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Git commit: `{commit}`",
        "",
        "Measurements reflect the report files at generation "
        "time; training may update metrics during the replay.",
        "",
        "## Test results",
        "",
    ]
    if junit:
        lines += [
            f"JUnit suite timestamp: {junit['timestamp']} (may predate this generation).",
            "",
            ", ".join(f"{junit['totals'][k]} {k}" for k in COUNTS),
            "",
            "| File | Total | Passed | Failed | Skipped | Errors |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, counts in junit["files"].items():
            lines.append("| " + name + " | " + " | ".join(str(counts[k]) for k in COUNTS) + " |")
    else:
        lines.append("reports/tests.xml: not available")
    ratio = None
    if metrics and baselines and baselines.get("B1", {}).get("test_mae"):
        ratio = metrics.get("test_mae") / baselines["B1"]["test_mae"]
    coverage = metrics.get("coverage") if metrics else None
    single = next(
        (r for r in (loadtest or {}).get("results", []) if r.get("concurrency") == 1), None
    )
    p95 = single.get("p95_s") if single else None
    contract_value, contract_status = test_verdict(
        (junit or {}).get("files", {}), "tests/contract/"
    )
    unit_value, unit_status = test_verdict((junit or {}).get("files", {}), "tests/unit/")

    def verdict(value, condition):
        return "not available" if value is None else "PASS" if condition else "FAIL"

    rows = [
        (
            "AC1",
            "Test MAE / B1 MAE ≤ 0.75",
            ratio,
            verdict(ratio, ratio is not None and ratio <= 0.75),
            "reports/metrics.json; reports/baselines.json",
        ),
        (
            "AC2",
            "80% interval coverage in [0.75, 0.85]",
            coverage,
            verdict(coverage, coverage is not None and 0.75 <= coverage <= 0.85),
            "reports/metrics.json",
        ),
        (
            "AC3",
            "Prediction p95 ≤ 5 s at concurrency 1; no request errors",
            f"p95={p95} s; errors={single.get('errors')}" if single else None,
            verdict(p95, p95 is not None and p95 <= 5 and single.get("errors") == 0),
            "reports/load_test.json",
        ),
        (
            "AC4",
            "All invalid-input contract tests pass",
            contract_value,
            contract_status,
            "reports/tests.xml",
        ),
        (
            "AC5",
            "Drift promotion improves next-month MAE",
            ac5.get("verdict") if ac5 else None,
            ac5.get("verdict", "not available") if ac5 else "not available",
            "reports/ac5.json",
        ),
        ("AC6", "All unit tests pass", unit_value, unit_status, "reports/tests.xml"),
    ]
    lines += [
        "",
        "## Acceptance criteria",
        "",
        "Missing evidence is **not available**; incomplete replay evidence is **pending**.",
        "",
    ]
    if ac5:
        lines += [ac5.get("details", ""), ""]
    lines += [
        "| AC | Criterion | Measured value | PASS/FAIL | Source file |",
        "|---|---|---|---|---|",
    ]
    for ac, criterion, value, status, source in rows:
        lines.append(
            f"| {ac} | {criterion} | {value if value is not None else 'not available'} "
            f"| {status} | {source} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    output = Path("docs/test_report.md")
    output.parent.mkdir(exist_ok=True)
    output.write_text(generate())
    print(f"Wrote {output}")
