# syntax=docker/dockerfile:1
FROM python:3.12-slim
RUN --mount=type=cache,target=/root/.cache/pip pip install \
    xgboost-cpu==3.4.1 pandas==3.1.1 pyarrow==26.0.1 numpy==2.5.4 \
    mlflow-skinny==3.18.0 fastapi==0.163.1 uvicorn==0.54.0 requests==2.34.2 \
    prometheus-client==0.26.0 pyyaml==6.0.3
RUN useradd --create-home monitor && mkdir /state && chown monitor /state
WORKDIR /app
COPY src/elo /app/elo
COPY params.yaml /app/params.yaml
ENV PYTHONPATH=/app OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MLFLOW_DISABLE_TELEMETRY=true
USER monitor
EXPOSE 8002 18001
CMD ["uvicorn", "elo.monitoring.monitor:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
