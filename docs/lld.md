# Guess the Elo: Low-Level Design

Endpoint contracts, module design, algorithms, metrics and configuration, taken from the code. `file:line` references are to the repository root. `src/elo/api/app.py` was being extended during M9 when this was written, so its line numbers may drift by a few lines. JSON examples marked *live* are real `GET` responses from the running stack; long arrays are shortened with `…`.

## 1. Conventions

**api error shape** (all api errors, `src/elo/api/app.py:181`):

```json
{"error": {"code": "invalid_pgn", "message": "Illegal PGN mainline or starting position", "request_id": "68c97118-4b12-4e6c-b552-fe9caeb35c4c"}}
```

Every api response carries `X-Request-ID` (taken from the request header or generated as a UUID4). The request ID is also in every log line (`src/elo/common/logging.py`).

| HTTP | `code` | Raised by |
|---|---|---|
| 400 | `invalid_pgn` | `InvalidPgnError`: unparsable or illegal PGN, no moves, no games, non-UTF-8 upload |
| 401 | `unauthorized` | missing or wrong `X-API-Key` on retrain |
| 404/405 | `validation_error` | unknown route or method (Starlette `HTTPException`; message `Not Found`, verified live) |
| 413 | `payload_too_large` | body over `MAX_PGN_BYTES` (1 048 576), checked while streaming in the middleware |
| 422 | `validation_error` | Pydantic validation (strict types), bad multipart, bad JSON |
| 422 | `game_index_out_of_range`, `unsupported_time_control`, `game_too_short` | `ApiError` in `/v1/predict` and `/v1/games/parse` |
| 500 | `internal_error` | any unhandled exception |
| 503 | `engine_error` | engine pool busy for 5 s, or Stockfish failed to start or crashed |
| 503 | `model_unavailable`, `trainer_unavailable`, `monitor_unavailable` | upstream unreachable, 5xx after retries, or any non-2xx upstream reply |
| 504 | `engine_timeout` | per-game engine budget of 120 s exhausted |

The trainer, monitor and model-server use their frameworks' default error bodies (FastAPI: `{"detail": "..."}`), not the api shape.

## 2. api (FastAPI, port 8000, `src/elo/api/app.py`)

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | none | liveness |
| GET | `/ready` | none | engine pool non-empty and model-server `/ping` OK |
| POST | `/v1/games/parse` | none | list games in a PGN |
| POST | `/v1/predict` | none | estimate both ratings |
| GET | `/v1/model` | none | champion metadata from the registry |
| GET | `/v1/pipeline/dag` | none | proxy of trainer `/dag` |
| GET | `/v1/pipeline/runs` | none | proxy of trainer `/runs` |
| GET | `/v1/pipeline/replay` | none | replay and drift summary from monitor `/status` (added in M9) |
| POST | `/v1/pipeline/retrain` | `X-API-Key` | proxy of trainer `/retrain` |

### GET /health, GET /ready (live)

```json
{"status": "ok"}
{"status": "ready", "model_server": true, "engine": true}
```

`/ready` errors: 503 `engine_error` (no engine), 503 `model_unavailable` (model-server down).

### POST /v1/games/parse

Request, either JSON `{"pgn": "<string>"}` (strict: `pgn` must be a string) or `multipart/form-data` with exactly one part named `file` (UTF-8, BOM allowed). Body ≤ 1 MiB.

```json
{"games": [{"index": 0, "white": "Player A", "black": "Player B", "date": "2020.04.03",
            "event": "Rated Blitz game", "time_control": "300+0", "tc_class": "blitz",
            "plies": 64, "has_clock": true, "result": "1-0"}]}
```

`tc_class` is `null` when there is no `TimeControl` header, and may be `ultrabullet`, `bullet` or `correspondence` (listing does not reject them). Errors: 400, 413, 422 (`validation_error`, `unsupported_time_control` for a malformed header).

### POST /v1/predict (`app.py:294`)

| Field | Type | Default | Limits |
|---|---|---|---|
| `pgn` | string | required | ≤ 1 MiB (UTF-8 bytes) |
| `game_index` | int | 0 | ≥ 0 and < number of games |
| `top_k` | int | 5 | 1–50 |
| `tc_class` | `"blitz"` \| `"rapid"` \| `"classical"` \| null | null | used only if the game has no `TimeControl` header (else ignored); fallback is `classical` with a warning |

Response 200 (shape from `app.py:372`; values illustrative, since a POST could not be run during the live experiment):

```json
{
  "request_id": "…",
  "game": {"index": 0, "white": "Player A", "black": "Player B", "date": "2020.04.03", "event": "Rated Blitz game",
           "time_control": "300+0", "tc_class": "blitz", "plies": 64, "has_clock": true, "result": "1-0"},
  "white": {"elo": 1650, "lo": 1350, "hi": 1950},
  "black": {"elo": 1580, "lo": 1280, "hi": 1880},
  "interval_level": 0.8,
  "pgn_ratings": {"white": 1702, "black": 1611},
  "moves": [{"ply": 1, "side": "w", "san": "e4", "cpl": 0.0, "move_class": "ok", "clock_s": 300.0, "uci": "e2e4"}, "…"],
  "influential_moves": [{"ply": 31, "san": "Qxf7", "side": "b", "delta_elo": -42}, "…"],
  "analysis": {"source": "stockfish", "nodes": 5000, "seconds": 0.62},
  "model": {"name": "elo-estimator", "version": "4", "run_id": "1e611f6712804a3c9a78e73f1c6afb25"},
  "warnings": []
}
```

- `move_class` ∈ {`ok`, `inaccuracy`, `mistake`, `blunder`}; `analysis.source` ∈ {`stockfish`, `pgn_eval`}.
- `influential_moves` merges both sides and keeps the `top_k` largest `|delta_elo|`.
- `pgn_ratings` holds the PGN's own `WhiteElo`/`BlackElo` (or null); they are never model inputs.

Errors: 400 `invalid_pgn`; 413; 422 `validation_error`, `game_index_out_of_range`, `unsupported_time_control` (correspondence, ultrabullet, bullet or malformed), `game_too_short` (< 10 plies for either side); 503 `engine_error`, `model_unavailable`; 504 `engine_timeout`.

### GET /v1/model (live)

```json
{"name": "elo-estimator", "version": "4", "run_id": "1e611f6712804a3c9a78e73f1c6afb25", "aliases": ["champion"],
 "metrics": {"test_mae": 182.72233240254948, "coverage": 0.808456596444046}, "created": "1794871445925"}
```

Read through MLflow REST `GET /api/2.0/mlflow/registered-models/alias?name=elo-estimator&alias=champion`; metrics come from version tags. Error: 503 `model_unavailable`.

### GET /v1/pipeline/dag, GET /v1/pipeline/runs?limit=N

`dag` returns the trainer's `{"mermaid": "```mermaid\nflowchart TD …```"}` unchanged. `runs` returns the trainer's run list (§3) with `limit` clamped to 1–1000 (default 100). Error: 503 `trainer_unavailable`.

### GET /v1/pipeline/replay (added in M9, `app.py:412`)

A read-only projection of monitor `/status`; it never advances the clock. Shape (values from the live monitor status at the time of writing):

```json
{"current_month": "2020-01", "months_revealed": 1, "total_months": 12, "playing": true,
 "seconds_to_next_reveal": 536.1, "champion_version": "4", "previous_champion_version": null,
 "metrics_month": "2020-01", "reference_mae": 182.72,
 "mae": {"champion": 186.46}, "mae_ratio": {"champion": 1.0205}, "coverage": 0.7944,
 "drift_detected": false, "drift_injection": false,
 "psi_top5": [["eco_group", 0.0114], ["tc_inc", 0.0023], ["tc_base", 0.0019], ["tc_class", 0.0018], ["own_best_rate", 0.0013]]}
```

`drift_detected` is recomputed from `params.yaml monitor` (`mae_ratio`, `psi_warn`) and is `null` before the first evaluation. Error: 503 `monitor_unavailable`.

### POST /v1/pipeline/retrain

Header `X-API-Key: <API_KEY>`; body optional JSON object, e.g. `{"reason": "manual demo"}`. Response 202 `{"run_id": "8e80d886ce224b51b49ec059a42021cf"}`. Errors: 401 `unauthorized`, 422 `validation_error` (invalid JSON), 503 `trainer_unavailable`. A busy trainer answers 409, but the api's `remote()` helper maps every non-2xx upstream reply to 503 `trainer_unavailable`, so clients see 503, not 409.

## 3. trainer (FastAPI, 127.0.0.1:8001, `src/elo/training/controller.py`)

| Method | Path | Auth | Request | Response | Errors |
|---|---|---|---|---|---|
| GET | `/health` | none | – | `{"status":"ok"}` | – |
| GET | `/status` | none | – | `{in_progress, current_run, champion_version, last_run, cooldown_remaining_s}` | 500 if MLflow down |
| GET | `/runs` | none | `?limit` (1–1000, default 20) | list of run records, newest first | – |
| GET | `/runs/{run_id}` | none | – | one run record | 404 `Unknown run` |
| GET | `/dag` | none | – | `{"mermaid": "<dvc dag --md>"}`, cached by `dvc.yaml` mtime | 500 |
| POST | `/retrain` | `X-API-Key` or `Authorization: Bearer` | JSON object, optional `reason` | 202 `{"run_id": "…"}` | 401, 409 `Retrain already running`, 422 |
| POST | `/replay/reset` | same | – | 202 `{"run_id": "…"}` | 401, 409 |
| POST | `/alerts` | same (Alertmanager uses Bearer from a secret file) | Alertmanager webhook v4 payload | `{"alerts": [{"alertname": "ModelMAEDegraded", "action": "accepted"}]}` | 401 |

`/alerts` actions: `accepted`, `ignored_busy`, `ignored_cooldown` (cooldown or duplicate month), `ignored_resolved` (resolved, or any alert other than firing `ModelMAEDegraded`).

**Run record** (table `runs(id TEXT PRIMARY KEY, body TEXT)` in `/state/trainer.db`, body is JSON; live example abbreviated):

```json
{"run_id": "8e80d886ce224b51b49ec059a42021cf", "trigger": "manual", "alert_name": "", "reason": "",
 "replay_month": null, "replay_generation": 3, "reset": true, "status": "promoted",
 "fit_months": ["2019-09", "2019-10", "2019-11"], "holdout_month": "2019-12",
 "stages": [
   {"name": "months", "status": "completed", "started_at": 1794873909.39, "ended_at": 1794873909.42, "duration_s": 0.031, "games_per_s": null},
   {"name": "upstream", "status": "skipped", "count": 80, "restored": 0, "skipped": 80, "duration_s": 0, "games_per_s": null},
   {"name": "build_dataset", "status": "skipped", "…": "…"},
   {"name": "gate", "status": "completed", "…": "…"},
   {"name": "push", "status": "pending", "…": "…"}],
 "mlflow_run_id": null, "message": "Baseline restored; registration report was not gated or tagged.",
 "push": "pending", "push_error": "ERROR: unexpected error - [Errno 111] Connect call failed …",
 "started_at": 1794873909.37, "ended_at": 1794873914.79}
```

| Field | Meaning |
|---|---|
| `trigger` | `manual` or `drift_alert` |
| `status` | `running`, `promoted`, `rejected`, `no_change`, `failed` (a `running` record found at start-up becomes `failed`, `error: "trainer restarted"`) |
| `stages[].status` | `running`, `completed`, `failed`, `skipped`, `restored`, `reused`, and for `push`: `complete` or `pending` |
| `upstream` | one aggregated entry for the 80 per-month stages that DVC skipped or restored |
| gated runs add | `registry_version`, `mlflow_run_id`, `gate_champion_mae`, `gate_challenger_mae`, `coverage` |
| failures add | `error` (run and failing stage) |

Table `settings(key, value)` holds `baseline` (the registry version that reset restores, from `BASELINE_VERSION` or the champion at first start) and `generation` (incremented by each replay reset, used for the once-per-month dedupe). Per-run subprocess logs are in `/state/runs/<run_id>.log`.

## 4. monitor (FastAPI, 127.0.0.1:8002, `src/elo/monitoring/monitor.py`)

| Method | Path | Auth | Response |
|---|---|---|---|
| GET | `/health` | none | `{"status":"ok"}` |
| GET | `/status` | none | replay state + latest evaluation (below) |
| POST | `/clock/{action}` | `Authorization: Bearer <API_KEY>` | `action` ∈ `advance`, `play`, `pause`, `reset`; returns `/status`. 401, 404 for other actions |

`GET /status` (live, `psi` shortened):

```json
{"revealed": ["2020-01"], "playing": true, "last_reveal": 1794880793.41, "current_month": "2020-01",
 "seconds_to_next_reveal": 536.13, "previous_champion_version": null, "champion_version": "4",
 "last_metrics": {"month": "2020-01",
   "mae": {"champion": {"all": 186.46, "blitz": 192.14, "rapid": 169.20, "classical": 151.81}},
   "coverage": {"all": 0.7944, "blitz": 0.7939, "rapid": 0.8001, "classical": 0.7825},
   "reference_mae": 182.72, "psi": {"is_white": 0.0, "own_cpl_mean": 0.00033, "…": "…", "eco_group": 0.0114}},
 "psi_top5": [["eco_group", 0.0114], ["tc_inc", 0.0023], "…"],
 "drift_injection": {"enabled": false, "month": "2020-06", "cpl_scale": 1.4}}
```

`last_metrics.mae` gains a `previous_champion` entry once a promotion has happened.

**Replay state** (`/state/monitor.json`, written atomically through a `.tmp` file): `{"revealed": [months…], "playing": bool, "last_reveal": unix_seconds}`. `reset` empties it and clears the quality gauges; `advance` appends the next month of `data.replay_months`.

## 5. model-server (MLflow scoring server, port 5001)

| Method | Path | Response |
|---|---|---|
| GET | `/ping`, `/health` | 200, empty body (live) |
| GET | `/version` | `3.18.0` (live) |
| POST | `/invocations` | `{"predictions": [...]}` |

**`/invocations` input**: one row per ply, `dataframe_split` orientation without an index, plus the `top_k` param:

```json
{"dataframe_split": {
   "columns": ["game_id","ply","side","san","eval_before_cp","eval_after_cp","cpl","wp_loss","is_best",
               "move_class","legal_moves","in_check","material_mover","material_change","clock_s",
               "time_spent_s","clock_frac","time_trouble","phase",
               "tc_base","tc_inc","tc_class","result","termination","eco","n_plies","has_clock"],
   "data": [["3fa1c0d2a9e4b6c7",1,"w","e4",18,23,0,0.0,true,"ok",20,false,0,0,300.0,0.0,1.0,false,"opening",
             300,0,"blitz","1-0","Normal","C20",64,true], "…"]},
 "params": {"top_k": 5}}
```

The first 19 columns are `build_move_rows` output; the last 8 are game metadata repeated on every row. `game_id` is the first 16 hex characters of SHA-256 of the PGN text. Several games may be sent in one call.

**Output**: two rows per game:

```json
{"predictions": [
  {"game_id": "3fa1c0d2a9e4b6c7", "side": "w", "elo": 1650, "lo": 1350, "hi": 1950, "tc_class": "blitz",
   "influence": "[{\"ply\": 23, \"san\": \"Nxe5\", \"side\": \"w\", \"delta_elo\": 37}]"},
  {"game_id": "3fa1c0d2a9e4b6c7", "side": "b", "…": "…"}]}
```

`influence` is a JSON string (the signature allows only flat columns). The supervisor `services/model_server/serve.py:21` polls `@champion` every `ALIAS_POLL_S` and restarts the child server on change; if the child exits, the container exits and Compose restarts it.

## 6. Prometheus metric catalogue

| Metric | Type | Labels | Exporter |
|---|---|---|---|
| `elo_requests_total` | Counter | `endpoint` (route template), `status` | api :18000 |
| `elo_request_latency_seconds` | Histogram (0.01–10 s) | `endpoint` | api |
| `elo_inflight_requests` | Gauge | – | api |
| `elo_pgn_bytes` | Summary | – | api |
| `elo_engine_analysis_seconds` | Histogram (0.1–10 s) | – | api |
| `elo_parse_errors_total` | Counter | `reason` (error code of any 4xx) | api |
| `elo_predicted_rating` | Histogram (600–3200 step 200) | `side`, `tc` | api |
| `elo_interval_width` | Histogram (100–1200) | – | api |
| `elo_model_server_errors_total` | Counter | – | api |
| `elo_build_info` | Info | `version`, `git_commit` | api |
| `elo_model_server_serving_version` | Gauge | – | model-server :18003 |
| `elo_model_server_reloads_total` | Counter | – | model-server |
| `elo_model_server_child_up` | Gauge | – | model-server |
| `elo_replay_month` (YYYYMM), `elo_replay_months_revealed` | Gauge | – | monitor :18001 |
| `elo_batch_mae` | Gauge | `model` (`champion`, `previous_champion`), `tc` (`all`, `blitz`, `rapid`, `classical`) | monitor |
| `elo_interval_coverage` | Gauge | `tc` | monitor |
| `elo_feature_psi` | Gauge | `feature` (45 values) | monitor |
| `elo_reference_mae`, `elo_drift_detected`, `elo_drift_injection_enabled`, `elo_batch_rows`, `elo_monitor_last_eval_timestamp_seconds` | Gauge | – | monitor |
| `elo_pipeline_runs_total` | Counter | `status` | trainer :18002 |
| `elo_alerts_received_total` | Counter | `alertname`, `action` | trainer |
| `elo_stage_duration_seconds`, `elo_stage_throughput_games_per_second` | Gauge | `stage` | trainer |
| `elo_retrain_in_progress`, `elo_champion_version`, `elo_last_retrain_timestamp_seconds` | Gauge | – | trainer |
| MLflow server metrics | – | – | mlflow :5000 `/metrics` |
| `node_*` | – | – | node-exporter :9100 |

All `elo_*` names above were confirmed present in Prometheus (`/api/v1/label/__name__/values`).

## 7. Module design

| Package | Main items | Role |
|---|---|---|
| `elo.pgn` | `PgnParser.parse` `src/elo/pgn/__init__.py:33`, `parse_movetext` `:66`, `sanitize_headers` `:17`, `score_cp` `:11` | python-chess mainline parsing with clocks and `%eval`; leakage stripping |
| `elo.engine` | `EngineAnalyzer` `src/elo/engine/__init__.py:13`, `analyse_game` `:52` | one UCI process; N+1 evaluations at fixed nodes; per-game deadline; restart on error |
| `elo.features` | `build_move_rows` `src/elo/features/moves.py:23`, `_wp` `:18` | pure per-ply features |
| `elo.data` | `download` `src/elo/data/download.py:38`; `parse_records` `spark_ingest.py:72`, `validate` `:129`, `ingest` `:194`; `sample_games` `spark_sample.py:14`; `AnalyzerActor` `ray_analyze.py:38`, `run` `:70` | DVC stage entry points |
| `elo.models` | `side_features` `aggregate.py:55`, `perspective_rows` `:100`, `drop_clocks` `:130`; `quantile` `conformal.py:8`, `calibrate` `:15`, `radii` `:31`; `EloPyfunc` `pyfunc.py:12`; `predict` `scoring.py:8` | aggregation, conformal, pyfunc |
| `elo.training` | `aggregate_month` `dataset.py:39`, `split_holdout` `:65`, `reference_stats` `:73`, `build_dataset` `:104`; `train_baselines` `pipeline.py:62`, `fit_trial` `:84`, `train_xgb` `:125`, `evaluate` `:239`; `register` `registry.py:22`, `promote` `:79`, `reject` `:97`, `reset` `:105`, `rollback` `:114`; `gate` `controller.py:51`, `set_months` `:58`, `Controller` `:69` | dataset, MLproject entries, registry, retrain controller |
| `elo.monitoring` | `psi` `drift.py:11`, `psi_report` `:31`; `load_bundle` `champion.py:16`, `score` `:44`; `Monitor` `monitor.py:48` (`clock` `:100`, `evaluate` `:124`, `loop` `:184`) | drift and replay |
| `elo.api` | `predict` `app.py:294`, `remote` `:101`, `request_context` middleware `:191`; `metrics.py` | HTTP boundary |
| `elo.common` | `EloError` hierarchy `errors.py`, `load_params` `config.py:10`, `get_logger` `logging.py:19` | shared |

## 8. Key algorithms

**Win probability and move class** (`moves.py:18`, `:33`, `:64`). `wp(cp) = 100 / (1 + exp(−0.00368208·cp))`, evaluations clipped to ±1000. `wp_loss = max(0, sign·(wp(before) − wp(after)))` for the mover; `cpl = clip(sign·(before − after), 0, 1000)`. Class: blunder ≥ 15, mistake ≥ 10, inaccuracy ≥ 5. Accuracy per move = `clip(103.1668·exp(−0.04354·wp_loss) − 3.1669, 0, 100)` (`aggregate.py:59`).

**Conformal radius** (`conformal.py:8–33`). Residuals `|y − ŷ|` on the `cal` split. For each tc class with n ≥ 50: `q_c` = the `min(n, ⌈(n+1)(1−α)⌉)`-th smallest residual, α = 0.2; otherwise the global quantile. Interval `[round(ŷ − q_c), round(ŷ + q_c)]`; unseen classes use the global quantile.

**Occlusion influence** (`pyfunc.py:21`). For a game with N plies, build N variants, each with one ply removed and `game_id = "<gid>#<ply>"`. Concatenate the full game and all variants, run `side_features` → `perspective_rows` → one `booster.predict`. For the move at ply p by side s: `delta_elo = pred(gid, s) − pred(gid#p, s)`. Positive means the move pushed the estimate up. Each side keeps its `top_k` largest `|delta_elo|`, rounded.

**PSI** (`drift.py:11`, reference from `dataset.py:73`). Numeric features: bin edges are the unique training deciles; bins `(−∞,e0], …, (e_last, ∞)` plus a missing bin. Categorical and boolean features: one bin per category plus `__missing__`. With proportions floored at 1e-4: `PSI = Σ (a_i − e_i)·ln(a_i / e_i)`. The reference uses all fit-month rows before clock dropout, so synthetic dropout is not reported as drift.

**Monitor drift flag** (`monitor.py:171`). `drift_detected = MAE_champion,all > 1.04 × reference_mae OR max PSI > 0.05`. `reference_mae` is the champion version's `test_mae` tag. With injection enabled and the current month equal to `drift_injection.month`, every model input column containing `cpl` is multiplied by `cpl_scale` before scoring.

**Gate** (`controller.py:51`):

```python
def gate(challenger_mae, champion_mae, coverage, config):
    return (challenger_mae <= champion_mae - config["min_improvement"]
            and config["coverage_band"][0] <= coverage <= config["coverage_band"][1])
```

`challenger_mae` and `coverage` come from the new `reports/metrics.json`; `champion_mae` is the current champion scored on the same `data/datasets/test.parquet` (`controller.py:350`).

**Months rule** (`controller.py:281–288`). With `B = data.train_months` and `R` = revealed months: if `R` is empty or the run is a reset, `fit = B[:-1]`, `holdout = B[-1]`; else `fit = B + R[:-1]`, `holdout = R[-1]`. `set_months` rewrites only the `fit_months:` and `holdout_month:` lines of `params.yaml` and fails unless each matches exactly once.

**Admission** (`controller.py:133`). 409 if a run is active. For `drift_alert` only: skip if the last run ended < `cooldown_s` (600 s) ago, or if a `drift_alert` run already exists with the same `replay_month` and `replay_generation`.

## 9. Configuration (`params.yaml`)

| Section | Keys (value) | Used by |
|---|---|---|
| `data` | `months` (16), `base_url`, `train_months` (2019-09..12), `replay_months` (2020-01..12), `max_games_per_month` (60000) | download, ingest, monitor, trainer |
| `sample` | `train_per_month` (20000), `replay_per_month` (10000) | sample |
| `spark` | `driver_memory` (4g), `shuffle_partitions` (8) | ingest, sample |
| `engine` | `nodes` (5000), `hash_mb` (16), `path`, `game_timeout_s` (120) | analyze, api |
| `ray` | `num_workers` (10), `batch_size` (16) | analyze, train_xgb |
| `seed` | 42 | hashing, splits, sampling, XGBoost |
| `train` | `fit_months`, `holdout_month` (rewritten by the trainer), `clock_dropout` (0.2), `conformal_alpha` (0.2), `experiment` (elo), `xgb.grid`, `n_estimators` (3000), `early_stopping_rounds` (100), `n_jobs_per_trial` (2) | build_dataset, train_xgb |
| `drift` | `{}` (unused) | – |
| `monitor` | `interval_s` (900), `eval_interval_s` (30), `mae_ratio` (1.04), `psi_warn` (0.05), `drift_injection` (`enabled: false`, `month: 2020-06`, `cpl_scale: 1.4`) | monitor, api replay view |
| `retrain` | `min_improvement` (1.0), `cooldown_s` (600), `coverage_band` ([0.75, 0.85]) | trainer |

Environment variables (`docker-compose.yml`): `API_KEY`, `MAX_PGN_BYTES`, `ENGINE_POOL`, `FRONTEND_ORIGINS`, `MODEL_SERVER_URL`, `MLFLOW_TRACKING_URI`, `TRAINER_URL`, `MONITOR_URL`, `ALIAS_POLL_S`, `REPLAY_INTERVAL_S`, `DRIFT_INJECTION`, `BASELINE_VERSION`, `ELO_DISABLE_PUSH`, `ELO_RAY_CPUS`, `API_BASE_URL`, `API_UPSTREAM`.
