# syntax=docker/dockerfile:1
FROM elo-mlflow:latest AS builder
USER root
RUN --mount=type=cache,target=/root/.cache/pip python -m pip wheel --wheel-dir=/wheels \
    xgboost-cpu==3.4.1 pandas==3.1.1 numpy==2.5.4 prometheus-client==0.26.0 pyarrow==26.0.1

FROM elo-mlflow:latest AS runtime
USER root
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels \
    xgboost-cpu==3.4.1 pandas==3.1.1 numpy==2.5.4 prometheus-client==0.26.0 pyarrow==26.0.1 && rm -rf /wheels
WORKDIR /app
COPY src/elo /app/elo
COPY services/model_server/serve.py /app/serve.py
ENV PYTHONPATH=/app OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
USER mlflow
EXPOSE 5001 18003
CMD ["python", "serve.py"]
