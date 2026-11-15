PYTHON ?= /home/kirthan/miniforge3/envs/da3408-project/bin/python
# MLproject subprocesses must inherit the same interpreter's bin directory.
export PATH := $(dir $(PYTHON)):$(PATH)
# Keep course tooling local, including its usage reporting.
export DVC_NO_ANALYTICS := true
export MLFLOW_DISABLE_TELEMETRY := true
export RAY_USAGE_STATS_ENABLED := 0

.PHONY: setup lint format test up down ci

.PHONY: stockfish
stockfish:
	bash scripts/get_stockfish.sh

setup:
	"$(PYTHON)" -m pip install -e ".[dev]"

lint:
	"$(PYTHON)" -m ruff check src tests
	"$(PYTHON)" -m black --check src tests

format:
	"$(PYTHON)" -m ruff check --fix src tests
	"$(PYTHON)" -m black src tests

test:
	"$(PYTHON)" -m pytest

up:
	docker compose up -d mlflow

down:
	docker compose down

ci: lint test
	"$(PYTHON)" -m dvc status
