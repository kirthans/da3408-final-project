PYTHON ?= python
# MLproject subprocesses must inherit the same interpreter's bin directory.
export PATH := $(dir $(PYTHON)):$(PATH)
# Keep course tooling local, including its usage reporting.
export DVC_NO_ANALYTICS := true
export MLFLOW_DISABLE_TELEMETRY := true
export RAY_USAGE_STATS_ENABLED := 0

.PHONY: setup lint format test up down ci train

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
	@test -f services/trainer/dvc.local || { echo "Missing trainer config: run make trainer-config first (requires .dvc/config.local)."; exit 1; }
	docker compose up -d

.PHONY: monitoring
monitoring:
	docker compose up -d prometheus alertmanager grafana node-exporter

down:
	docker compose down

train:
	"$(PYTHON)" -m dvc repro evaluate

ci: lint test
	"$(PYTHON)" -m dvc status
	"$(PYTHON)" -m elo.training.checks gate

.PHONY: rollback
rollback:
	set -a; . ./.env; set +a; "$(PYTHON)" -m elo.training.registry rollback

.PHONY: promote
promote:
	set -a; . ./.env; set +a; "$(PYTHON)" -m elo.training.registry promote --version "$(VERSION)"

.PHONY: certs
certs:
	@mkdir -p certs
	@# Compose may have created the bind-mount directory as root on first startup.
	@if [ ! -w certs ]; then \
		docker run --rm --user 0 --entrypoint chown -v "$(CURDIR)/certs:/certs" elo-frontend "$$(id -u):$$(id -g)" /certs; \
	fi
	@if command -v mkcert >/dev/null 2>&1; then \
		mkcert -cert-file certs/localhost.pem -key-file certs/localhost-key.pem localhost 127.0.0.1 ::1; \
	else \
		echo 'Using a self-signed certificate; browsers will warn. For trusted TLS: sudo apt install mkcert && mkcert -install'; \
		openssl req -x509 -newkey rsa:2048 -sha256 -nodes -days 365 \
			-keyout certs/localhost-key.pem -out certs/localhost.pem \
			-subj '/CN=localhost' -addext 'subjectAltName=DNS:localhost,IP:127.0.0.1,IP:::1'; \
	fi
	@chmod 644 certs/localhost.pem certs/localhost-key.pem

.PHONY: sync replay-reset
sync:
	"$(PYTHON)" -m dvc push
	@echo 'git add params.yaml dvc.lock && git commit -m "Record retraining run"'

replay-reset:
	@set -a; . ./.env; set +a; "$(PYTHON)" services/trainer/reset.py

.PHONY: replay-play replay-pause replay-status
replay-play replay-pause replay-status:
	@set -a; . ./.env; set +a; "$(PYTHON)" services/monitor/replay.py "$(@:replay-%=%)"

.PHONY: trainer-config
trainer-config:
	"$(PYTHON)" services/trainer/config.py

.PHONY: loadtest test-report ac5
loadtest:
	"$(PYTHON)" tests/load/load_test.py

test-report:
	"$(PYTHON)" scripts/test_report.py

ac5:
	"$(PYTHON)" scripts/ac5_evidence.py
