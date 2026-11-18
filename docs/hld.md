# Guess the Elo: High-Level Design

This is the high-level design of the system as built. The approved design is `DESIGN.md` (its sections 20–23 override earlier sections); the alternatives considered for each decision are in `DESIGN.md` §4. Where code and design differ, this document follows the code and says so. The component diagram and sequence diagrams are in `docs/architecture.md`; endpoint schemas and module details are in `docs/lld.md`.

## 1. Problem and goals

**Input.** One PGN text, which may hold several games; the user picks one.

**Output, per side.** A point estimate of the player's *Lichess-equivalent* rating for the game's time-control class, an 80% prediction interval, and the most influential moves (moves whose removal changes the estimate most).

**Users.** Non-technical players (over-the-board or unrated) who want a strength estimate from a single game, plus an explanation.

**Constraints.** One laptop (12 threads, 22 GB RAM, no cloud), the course stack (Spark, Ray, DVC, MLflow, FastAPI, Docker Compose, Prometheus, Grafana), and a drift-triggered retraining loop.

### Acceptance criteria (`DESIGN.md` §1)

| ID | Criterion | Evidence (source) |
|---|---|---|
| AC1 | Test MAE ≤ 0.75 × B1 MAE | `reports/metrics.json`: test MAE 182.72, B1 test MAE 265.93 (`reports/baselines.json`), ratio 0.687, `ac1_pass: true` |
| AC2 | 80% interval coverage in [0.75, 0.85] | `reports/metrics.json`: coverage 0.808 (blitz 0.806, rapid 0.821, classical 0.794), `ac2_pass: true` |
| AC3 | `/v1/predict` p95 ≤ 5 s for a 40-move game | Gate B report: p95 0.66–0.79 s over 20 real predictions (`tests/integration/smoke_test.py`) |
| AC4 | Invalid input returns 4xx, never 5xx | `tests/contract/test_api_contract.py` (table of bad inputs) |
| AC5 | In the 2020 replay, a drift alert triggers a retrain and the promoted model has lower MAE on the next month | Gate C drill: 2020-04 MAE ratio 1.06 fired `ModelMAEDegraded`; gate compared 191.11 (challenger) vs 196.02 (champion) and promoted; on 2020-05 the new champion scored 183.25 vs 186.84 for the old. The full overnight replay was still running when this was written. |
| AC6 | All unit tests pass | `reports/tests.json`: 98 total, 97 passed, 0 failed, 1 skipped |

`make ci` checks AC1 and AC2 automatically (`src/elo/training/checks.py`, `gate` mode).

## 2. Data

| Aspect | As built | Source |
|---|---|---|
| Source | Lichess Open Database (CC0), `lichess_db_standard_rated_YYYY-MM.pgn.zst` from `https://database.lichess.org/standard` | `params.yaml data.base_url` |
| Months | Training reference 2019-09..2019-12; replay ("production") 2020-01..2020-12 | `params.yaml data` (U3) |
| Acquisition | Stream-decompress and keep the first `max_games_per_month = 60000` games of each month; atomic publish, 4 attempts | `src/elo/data/download.py` |
| Filters (Spark, sequential, counts reported) | rated; standard variant; has `%clk`; time-control class in {blitz, rapid, classical} (class = base + 40 × increment, Lichess boundaries); total plies ≥ 20; both Elo in [600, 3200]; not aborted and a decisive or drawn result; then dedupe on `game_id` | `src/elo/data/spark_ingest.py:129` |
| Sampling | Stratified by `tc_class` (`sampleBy`, seeded), 20 000 games per training month, 10 000 per replay month | `params.yaml sample` (D21) |
| Example | 2019-09: 60 000 raw → 35 741 kept (22 507 removed by time control, 1 565 by plies, 187 without clocks) | `reports/validation/2019-09.json` |

**Leakage controls.**
- Spark extracts headers separately from movetext; only `game_id`, context headers, the two Elo labels and *hashed* player names (`sha2(seed + name)`) are kept. Names are never features.
- At serving time `sanitize_headers` removes `White`, `Black`, `WhiteElo`, `BlackElo`, `*RatingDiff` and `Title*` before any feature is built (`src/elo/pgn/__init__.py:17`). The PGN's own ratings are returned separately as `pgn_ratings` for comparison only.
- Splits are by time. The holdout month is split by a seeded SHA-256 hash of `game_id` into thirds, so the two perspective rows of a game never land in different splits (`src/elo/training/dataset.py:65`).

**Splits (baseline model).**

| Split | Months | Rows (2 per game) | Games |
|---|---|---|---|
| train | 2019-09, 2019-10, 2019-11 | 120 000 | 60 000 |
| val | 2019-12, hash bucket 0 | 13 182 | 6 591 |
| cal (conformal) | 2019-12, hash bucket 1 | 13 432 | 6 716 |
| test | 2019-12, hash bucket 2 | 13 386 | 6 693 |

Source: `reports/dataset.json`. Clock dropout removed clocks from 12 000 training games (20%, `train.clock_dropout`) so the model also works on OTB games without clocks.

## 3. Offline pipeline

The pipeline is a DVC DAG (`dvc.yaml`, D12). Five stages are `foreach` over the 16 months.

| Stage | What it does | Technology | Main outputs |
|---|---|---|---|
| `download@m` | Stream the monthly dump, keep 60 000 games | requests + zstandard | `data/raw/m.pgn`, `reports/download/m.json` |
| `ingest_validate@m` | Split records on `\n\n[Event `, extract headers, filter, validate the data contract (21 typed columns, no nulls) | Spark `local[*]` (D1, D3) | `data/interim/games/m.parquet`, `reports/validation/m.json` |
| `sample@m` | Seeded stratified sample | Spark | `data/interim/sample/m.parquet` |
| `analyze@m` | Per-move Stockfish analysis at 5 000 nodes, reusing `%eval` when every move has one; per-move features | Ray Data `map_batches` with an actor pool of 10 actors, one Stockfish process each (D2, D28) | `data/features/moves/m.parquet`, `reports/throughput/analyze_m.json` |
| `aggregate@m` | Per-game, per-side aggregates (two perspective rows per game) with labels | pandas | `data/features/games/m.parquet` |
| `build_dataset` | Train/val/cal/test splits, clock dropout, `reference_stats.json` (training distribution for PSI) | pandas | `data/datasets/`, `reports/dataset.json` |
| `train_baselines` | B0 global mean, B1 mean per tc class, B2 linear regression on (tc class, score, plies) | `mlflow run . -e train_baselines` (D13) | `reports/baselines.json` |
| `train_xgb` | Grid sweep of 6 XGBoost configs, run in parallel as Ray tasks, each a nested MLflow run; best by validation MAE; conformal calibration | Ray tasks, XGBoost, MLflow | `models/xgb/`, `reports/train_xgb.json` |
| `evaluate` | Test metrics, per-class and per-band MAE, no-clock MAE, AC1/AC2 flags, replay-month diagnostics, plots | MLflow | `reports/metrics.json`, `reports/replay_eval.json`, `reports/plots/` |
| `register` | Log the pyfunc with signature and input example, register a version, set alias | MLflow registry | `reports/register.json` |
| `tests` | pytest (not `engine`-marked) to JUnit XML and a summary | pytest | `reports/tests.xml`, `reports/tests.json` |

Measured Stockfish throughput is about 10.3–11.2 games/s per month with 10 actors (`reports/throughput/*.json`; mean 10.84 in `reports/metrics.json`).

## 4. Features and model

**Per-move features** (`src/elo/features/moves.py`, shared by the pipeline and the api to avoid training/serving skew): evaluation before and after (clipped to ±1000 cp, mate = ±1000), centipawn loss, win-probability loss (Lichess formula, percentage points), best-move match, move class (inaccuracy ≥ 5, mistake ≥ 10, blunder ≥ 15 win-% points), legal-move count, in-check, material balance and change, clock, time spent (with increment), clock fraction, time-trouble flag, phase (opening / middlegame / endgame by material).

**Game features** (`src/elo/models/aggregate.py`, 45 columns): for each of *own* and *opponent* side, 18 aggregates (CPL mean/median/p90, win-% loss, Lichess accuracy, best-move rate, inaccuracy/mistake/blunder rates, opening CPL and fraction, mean legal moves, material-change std, time-use mean/std, time-trouble rate, final clock fraction, number of moves), plus `is_white`, the player's score, termination group, tc class, base, increment, plies, `has_clock`, and the two-character ECO group (D29). Each game gives two symmetric rows, one per player (D6).

**Model** (D4). XGBoost regressor with absolute-error objective, `hist` trees, native categorical support, early stopping (100 rounds, up to 3 000 trees), grid `max_depth ∈ {4, 6, 8}` × `learning_rate ∈ {0.05, 0.1}`. The selected config is depth 4, learning rate 0.05, validation MAE 182.51 (`reports/train_xgb.json`). The booster is truncated to the best iteration.

**Uncertainty** (D7). Mondrian split conformal: absolute residuals on the `cal` split, grouped by tc class; the radius per class is the ⌈(n+1)(1−α)⌉-th smallest residual with α = 0.2; classes with fewer than 50 rows use the global quantile. The interval is `[pred − q, pred + q]`. Mean width on test is 592 Elo (`reports/metrics.json`).

**Move influence** (D8). Occlusion: for each move outside the opening phase, drop that ply, re-aggregate and re-predict. Opening moves are not ranked, because they are often memorised theory (user decision, 2026-11-18). The full game and all its variants of the game are scored in one XGBoost batch. Influence = prediction(full game) − prediction(game without the move), for the mover's side.

**Packaging** (D10). All three steps live in one MLflow pyfunc (`EloPyfunc`) whose input is per-move rows. The model artifact bundles the booster, `conformal.json`, `feature_columns.json` and `reference_stats.json`, and the `src/elo` code.

**Results** (champion v4, `reports/metrics.json`): test MAE 182.72, RMSE 228.05, improvement over B0/B1/B2 of 85.0/83.2/79.7 Elo, no-clock MAE 194.37. MAE by band ranges from 138.8 (1500–1800) to 308.7 (≥ 2100), so extremes regress toward the mean.

## 5. Experiment tracking and registry

- **MLflow Projects** (`MLproject`, `python_env.yaml`): DVC stages call `mlflow run . -e <entry> --env-manager=local --experiment-name elo`, so each run records the source git commit (D13).
- **Autolog:** `mlflow.xgboost.autolog(log_models=False)` inside each Ray sweep trial records `validation_0-mae` per boosting step, `stopped_iteration`, the XGBoost params and the feature-importance artifacts (`feature_importance_weight.json` and `.png`).
- **Logged explicitly, in addition to autolog:** every XGBoost trial's `val_mae_iteration` per boosting step (batched `log_batch`), trial params, best params, all test metrics, the flattened `params.yaml`, tags `trigger`, `alert_name`, `train_months`, `parent_champion_version`, `dvc_data_md5` (from `dvc.lock`), and artifacts (plots, gain importance, dataset and metrics reports, `params.yaml`).
- **Registry** (D11): model `elo-estimator` with aliases `champion`, `challenger`, `previous_champion`. `register` sets `champion` if none exists, otherwise `challenger`. Version tags: `run_id`, `test_mae`, `coverage`, `trigger`, and after the gate `gate`, `gate_champion_mae`, `gate_challenger_mae`, `pending_commit`, `base_git_commit`, `git_dirty`, `dvc_lock_md5`.

## 6. Serving

- The browser calls nginx; nginx proxies `/api/` to the api (D9, U1).
- The api parses, sanitises and analyses the game with a pool of 2 Stockfish engines at 5 000 nodes (it refuses to start if `engine.nodes` differs, so serving matches training), builds per-move rows, and calls `POST /invocations` on the model-server with up to 3 attempts on network errors or 5xx (exponential back-off of 0.1 s and 0.2 s).
- The model-server serves `models:/elo-estimator@champion` and restarts itself when the alias moves (polled every 60 s).
- Only blitz, rapid and classical games with at least 10 plies per side are accepted; a missing `TimeControl` header falls back to the client-supplied class (default classical) with a warning.

## 7. Monitoring and alerting

Every service exports Prometheus metrics on a separate port with `start_http_server` (D16). The full catalogue is in `docs/lld.md` §6; the key signals are:

| Area | Metrics |
|---|---|
| Service health | `elo_requests_total`, `elo_request_latency_seconds`, `elo_inflight_requests`, `elo_engine_analysis_seconds`, `elo_parse_errors_total`, `elo_model_server_errors_total`, `up` |
| Model outputs | `elo_predicted_rating`, `elo_interval_width` |
| Model quality and drift (replay) | `elo_batch_mae{model,tc}`, `elo_reference_mae`, `elo_interval_coverage{tc}`, `elo_feature_psi{feature}`, `elo_drift_detected`, `elo_drift_injection_enabled` |
| Pipeline | `elo_pipeline_runs_total{status}`, `elo_stage_duration_seconds`, `elo_stage_throughput_games_per_second`, `elo_retrain_in_progress`, `elo_champion_version`, `elo_model_server_serving_version` |

**Alert rules** (`infra/prometheus/rules.yml`; each PromQL expression carries the prompt that generated it, as the AI-use rule requires):

| Alert | Condition | `for` | Severity | Routed to |
|---|---|---|---|---|
| `ModelMAEDegraded` | `max(elo_batch_mae{model="champion",tc="all"}) > 1.04 × max(elo_reference_mae)` | 2m | critical | trainer webhook |
| `FeatureDriftHigh` | `max(elo_feature_psi) > 0.05` | 2m | warning | log only |
| `CoverageOutOfBand` | `elo_interval_coverage{tc="all"}` outside [0.75, 0.85] | 5m | warning | log only |
| `ApiErrorRateHigh` | 5-minute 5xx ratio > 0.05 | 5m | warning | log only |
| `ApiLatencyP95High` | p95 of `/v1/predict` over 5 min > 5 s | 5m | warning | log only |
| `ServiceDown` | `up == 0` | 1m | critical | log only |

Alertmanager inhibits warnings while a critical alert with the same `job` fires, and inhibits `FeatureDriftHigh` while `ModelMAEDegraded` fires. The thresholds 1.04 and 0.05 were calibrated from measured noise (per-month MAE standard error about 1.5 Elo) in D30, replacing the original 1.15 and 0.25 that would never fire on real 2020 data.

**Grafana.** Two provisioned dashboards (`infra/grafana/dashboards/elo-service.json`, `elo-model.json`).

## 8. Drift → retrain → promote loop

1. **Replay.** The monitor reveals one 2020 month every `interval_s` (900 s, overridable by `REPLAY_INTERVAL_S`) while playing, but never while a retrain is running. Each 30 s it scores the newest month with `@champion` and `@previous_champion` (labels are known for replay months).
2. **Detect.** `ModelMAEDegraded` fires after 2 minutes above threshold; Alertmanager posts it to `trainer /alerts`.
3. **Admit.** The trainer starts a `drift_alert` run unless one is running (409 → `ignored_busy`), the last run ended less than 600 s ago, or a drift run already exists for this month in the current replay generation (`ignored_cooldown`).
4. **Months rule.** With revealed months `r1..rk`: fit = the 4 training months + `r1..r(k−1)`, holdout = `rk` (its val/cal/test thirds). Without revealed months, or on reset: fit = 2019-09..11, holdout = 2019-12.
5. **Rebuild.** `dvc repro register`. Upstream stages (download, ingest, sample, analyze) must already be cached: the trainer image has no Java or Stockfish and refuses to run them.
6. **Gate.** The current champion is scored on the same test rows as the challenger. Promote only if challenger MAE ≤ champion MAE − 1.0 Elo **and** challenger coverage ∈ [0.75, 0.85] (`params.yaml retrain`). If DVC restored or skipped `register`, the run ends as `no_change` and nothing is gated.
7. **Promote or reject.** Promote sets `@previous_champion` to the old champion and `@champion` to the new version; reject removes `@challenger`. The model-server reloads on its next poll.
8. **Provenance.** Tags on the run and version; `dvc push` is attempted and recorded as `complete` or `pending`. The trainer never commits to git (D22); the user runs `make sync` and commits `params.yaml` and `dvc.lock`.
9. **Rollback.** `make rollback` moves `@champion` back to `@previous_champion`. `make replay-reset` restores the baseline version and resets the replay clock.

Judging the challenger on the newest month's held-out third, not on the next month, avoids using data that has not yet "arrived"; the next month is then the out-of-sample check for AC5.

## 9. Security (D20)

- Upload cap: request bodies are streamed and rejected above `MAX_PGN_BYTES` (1 MiB) with 413; nginx also caps bodies at 2 MB.
- PGN is parsed only by python-chess; illegal mainlines are rejected.
- CORS: the api allows only `FRONTEND_ORIGINS` (default `http://localhost:8080`, `https://localhost:8443`), methods GET and POST.
- API key: required for `POST /v1/pipeline/retrain` (`X-API-Key`, constant-time compare), for trainer write endpoints (`X-API-Key` or `Authorization: Bearer`) and for monitor clock control (Bearer). Alertmanager reads the key from a Compose secret file, not from committed config. The key itself lives in the untracked `.env`.
- TLS: nginx serves HTTPS on 8443 with a mkcert certificate when `certs/` is populated (`make certs`); HTTP 8080 remains available.
- Headers: CSP (`default-src 'self'`, no inline scripts), `X-Content-Type-Options`, `Referrer-Policy`, `server_tokens off`.
- Containers run as non-root users; admin services (trainer, monitor, Prometheus, Alertmanager, Grafana) bind to 127.0.0.1.

## 10. Deployment

- `docker compose up` builds five custom images (multi-stage, `python:3.12-slim` or nginx-unprivileged) and starts five stock images (D18).
- Healthchecks gate start-up: `mlflow` healthy → `model-server` healthy → `api`.
- State lives in named volumes; restart policy `unless-stopped`. Memory limits: trainer 6 GB, monitor 1 GB, Prometheus and Grafana 512 MB each.
- The offline pipeline's first full run (Spark and Stockfish stages) runs on the host with `dvc repro`; the trainer container re-runs only the downstream stages.

## 11. Design paradigm (D19)

| Style | Where | Why |
|---|---|---|
| Object-oriented | Stateful components: `PgnParser`, `EngineAnalyzer` (owns a Stockfish process), `AnalyzerActor` (Ray actor), `EloPyfunc` (MLflow model), `Monitor` (clock, state, thread), `Controller` (run lock, SQLite, subprocesses) | They own resources or lifecycle that must be created once and closed |
| Functional | Transforms: `build_move_rows`, `side_features`, `perspective_rows`, `drop_clocks`, `calibrate`/`radii`, `psi`, `gate`, `split_holdout`, Spark `parse_records`/`validate` | Pure functions on DataFrames are easy to test and are shared unchanged between training, serving and monitoring |

D19 named `FeatureBuilder`, `DriftMonitor` and `RetrainController`; the code uses the function `build_move_rows` and the classes `Monitor` and `Controller` instead.

## 12. Out of scope

| Item | Reason |
|---|---|
| Kubernetes, KServe, Minikube (U4) | Single host; Compose is what is graded; RAM is limited. |
| Ray Serve | MLflow serving is required by the rubric; two serving layers add nothing. |
| Airflow | DVC is the mandated orchestrator. |
| Ray Tune | Not taught; parallel Ray tasks run the sweep. |
| GRU challenger (D5 → D24) | Dropped for the deadline. |
| `/v1/feedback`, online labels (D26) | Drift uses labelled replay months only. |
| MinIO / S3 (D15 → D27) | Images no longer published; SSH DVC remote and MLflow-served artifacts instead. |
| KS test, SHAP plot (D26) | PSI only; XGBoost gain importance is logged instead of SHAP. |

**Compose → Kubernetes mapping** (`DESIGN.md` §12), for the question "how would this scale?":

| Compose | Kubernetes |
|---|---|
| `api`, `model-server`, `frontend` | Deployments with readiness and liveness probes (today's healthchecks), HPA on CPU or latency |
| `trainer` retrain run | a Job ("training is a Job, serving is a Deployment") |
| alias move + model-server restart | rolling update of the model-server Deployment |
| `make rollback` | `kubectl rollout undo` (or moving the alias back) |
| named volumes | PersistentVolumeClaims |
| `.env` and Compose secret | Kubernetes Secrets |
| Prometheus / Alertmanager / Grafana | kube-prometheus-stack |
