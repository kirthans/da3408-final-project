# Guess the Elo

A course AIOps project estimating Lichess-equivalent ratings from PGN games.
M1 provides the package skeleton, shared utilities, an SSH DVC remote and MLflow.

Use the existing Python 3.12 conda environment `da3408-project` and Docker Compose:

```bash
conda activate da3408-project
export DVC_NO_ANALYTICS=true MLFLOW_DISABLE_TELEMETRY=true RAY_USAGE_STATS_ENABLED=0
make setup
cp .env.example .env
make up
make test
make lint
MLFLOW_TRACKING_URI=http://localhost:5000 mlflow run . -e smoke --env-manager=local --experiment-name m1-smoke
```

MLflow: http://localhost:5000. SQLite and artifacts are stored on the `mlflow-data`
named Docker volume, under `/mlflow/mlflow.db` and `/mlflow/artifacts`. Clients
upload and download artifacts through the tracking server with `--serve-artifacts`.
`make down` stops the stack and retains the volume.

`make PYTHON=/path/to/python test` overrides the active environment’s Python interpreter.
`make ci` runs lint, unit tests and `dvc status`.

DVC uses SSH to localhost. The user provisions sshd and a dedicated restricted key;
project setup does not create keys or change SSH configuration. For a fresh clone,
initialize DVC once with `dvc init` if `.dvc/` is absent, then configure the remote:

```bash
# The shared config contains a placeholder; machine-specific settings stay local.
dvc remote add -d storage ssh://localhost/path/to/guess-the-elo
dvc remote modify --local storage url "ssh://localhost/home/<user>/dvc-storage/guess-the-elo"
dvc remote modify --local storage user '<user>'
dvc remote modify --local storage keyfile "$HOME/.ssh/dvc_elo"
```

Secrets and machine-specific settings belong in the ignored `.env` and
`.dvc/config.local` files. Pipeline data and models belong in DVC; small PGN fixtures
use the existing Git LFS patterns. Later milestones implement the data pipeline
and application services. `python_env.yaml` records resolved Python dependencies;
`environment.yml` records the conda environment without build strings.

## Pipeline console

The Pipeline tab shows the current and previous model, the replay clock and quality
numbers, a stage-family diagram, and expandable training history with gate decisions,
durations and throughput. Links open MLflow, both Grafana dashboards and Prometheus
alerts. The manual retrain form uses the months revealed so far; its password field
is cleared after submission and the API key is never stored. The visible tab refreshes
every ten seconds. Help explains drift, rating limits and privacy.

`make loadtest` runs 60 seconds each with 1, 2 and 4 concurrent prediction clients
and writes `reports/load_test.json`. Options can be passed directly to
`python tests/load/load_test.py --help`. `make ac5` reads the last 12 hours of
Prometheus samples and trainer history into `reports/ac5.json`; use
`python scripts/ac5_evidence.py --help` for URL and time-window options.
`make test` runs pytest; to capture evidence use
`python -m pytest --junitxml=reports/tests.xml`. Then `make test-report` generates
`docs/test_report.md` from available evidence, explicitly marking missing files.
