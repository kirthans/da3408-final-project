# Test report

## Test results

118 total, 117 passed, 0 failed, 1 skipped, 0 errors

| File | Total | Passed | Failed | Skipped | Errors |
|---|---:|---:|---:|---:|---:|
| tests/contract/test_api_contract.py | 32 | 32 | 0 | 0 | 0 |
| tests/integration/smoke_test.py | 1 | 0 | 0 | 1 | 0 |
| tests/unit/test_api_metrics.py | 2 | 2 | 0 | 0 | 0 |
| tests/unit/test_checks.py | 2 | 2 | 0 | 0 | 0 |
| tests/unit/test_common.py | 12 | 12 | 0 | 0 | 0 |
| tests/unit/test_engine.py | 3 | 3 | 0 | 0 | 0 |
| tests/unit/test_features.py | 7 | 7 | 0 | 0 | 0 |
| tests/unit/test_ingest.py | 8 | 8 | 0 | 0 | 0 |
| tests/unit/test_models.py | 7 | 7 | 0 | 0 | 0 |
| tests/unit/test_monitoring.py | 22 | 22 | 0 | 0 | 0 |
| tests/unit/test_pgn.py | 4 | 4 | 0 | 0 | 0 |
| tests/unit/test_pyfunc.py | 4 | 4 | 0 | 0 | 0 |
| tests/unit/test_ray_analyze.py | 1 | 1 | 0 | 0 | 0 |
| tests/unit/test_registry.py | 5 | 5 | 0 | 0 | 0 |
| tests/unit/test_reporting.py | 8 | 8 | 0 | 0 | 0 |

## Acceptance criteria

Missing evidence is **not available**; incomplete replay evidence is **pending**.

Last aligned sample per replay month; pending until every drift promotion has a next-month pair with its champion version. No promotion means pending.

| AC | Criterion | Measured value | PASS/FAIL | Source file |
|---|---|---|---|---|
| AC1 | Test MAE / B1 MAE ≤ 0.75 | 0.6509785114783206 | PASS | reports/metrics.json; reports/baselines.json |
| AC2 | 80% interval coverage in [0.75, 0.85] | 0.7926575505350772 | PASS | reports/metrics.json |
| AC3 | Prediction p95 ≤ 5 s at concurrency 1; no request errors | p95=0.6330596023537509 s; errors=0 | PASS | reports/load_test.json |
| AC4 | All invalid-input contract tests pass | 32 total, 32 passed, 0 failed, 0 skipped, 0 errors | PASS | reports/tests.xml |
| AC5 | Drift promotion improves next-month MAE | PASS | PASS | reports/ac5.json |
| AC6 | All unit tests pass | 85 total, 85 passed, 0 failed, 0 skipped, 0 errors | PASS | reports/tests.xml |
