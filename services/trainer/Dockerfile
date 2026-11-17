# syntax=docker/dockerfile:1
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git openssh-client && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/project
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip pip install '.[dev]' matplotlib
RUN mkdir /state && chmod 777 /state
COPY services/trainer/entrypoint.sh /usr/local/bin/trainer-entrypoint
WORKDIR /repo
ENV USER=trainer LOGNAME=trainer PYTHONPATH=/repo/src HOME=/tmp/trainer-home MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 DVC_NO_ANALYTICS=true \
    MLFLOW_DISABLE_TELEMETRY=true RAY_USAGE_STATS_ENABLED=0 ELO_RAY_CPUS=4
ENTRYPOINT ["sh", "/usr/local/bin/trainer-entrypoint"]
EXPOSE 8001 18002
CMD ["uvicorn", "elo.training.controller:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
