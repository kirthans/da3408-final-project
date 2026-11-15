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

`PYTHON=/path/to/python make test` overrides the default course interpreter.
`make ci` runs lint, unit tests and `dvc status`.

DVC uses SSH to localhost. The user provisions sshd and a dedicated restricted key;
project setup does not create keys or change SSH configuration. For a fresh clone,
initialize DVC once with `dvc init` if `.dvc/` is absent, then configure the remote:

```bash
# The shared config contains a placeholder; machine-specific settings stay local.
dvc remote add -d storage ssh://localhost/path/to/guess-the-elo
dvc remote modify --local storage url ssh://localhost/home/kirthan/dvc-storage/guess-the-elo
dvc remote modify --local storage user kirthan
dvc remote modify --local storage keyfile /home/kirthan/.ssh/dvc_elo
```

Secrets and machine-specific settings belong in the ignored `.env` and
`.dvc/config.local` files. Pipeline data and models belong in DVC; small PGN fixtures
use the existing Git LFS patterns. Later milestones implement the data pipeline
and application services. `python_env.yaml` records resolved Python dependencies;
`environment.yml` records the conda environment without build strings.
