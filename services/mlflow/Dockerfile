FROM python:3.12-slim AS builder
WORKDIR /build
COPY services/mlflow/requirements.txt .
RUN python -m pip wheel --wheel-dir /wheels -r requirements.txt

FROM python:3.12-slim AS runtime
COPY --from=builder /wheels /wheels
COPY services/mlflow/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-index --find-links=/wheels -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && useradd --create-home --uid 10001 mlflow \
    && mkdir -p /mlflow/metrics /mlflow/artifacts \
    && chown -R mlflow:mlflow /mlflow
USER mlflow
WORKDIR /mlflow
EXPOSE 5000
