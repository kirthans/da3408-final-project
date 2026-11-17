# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder
WORKDIR /build
RUN --mount=type=cache,target=/root/.cache/pip python -m pip wheel --wheel-dir=/wheels \
    fastapi==0.163.1 uvicorn==0.54.0 pydantic==2.14.3 chess==1.11.2 \
    requests==2.34.2 pyyaml==6.0.3 pandas==3.1.1 numpy==2.5.4 prometheus-client==0.26.0
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
COPY scripts/get_stockfish.sh scripts/get_stockfish.sh
RUN bash scripts/get_stockfish.sh

FROM python:3.12-slim AS runtime
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels \
    fastapi==0.163.1 uvicorn==0.54.0 pydantic==2.14.3 chess==1.11.2 \
    requests==2.34.2 pyyaml==6.0.3 pandas==3.1.1 numpy==2.5.4 prometheus-client==0.26.0 \
    && rm -rf /wheels && useradd --create-home --uid 10001 elo
WORKDIR /app
COPY --from=builder /build/tools/stockfish/stockfish tools/stockfish/stockfish
COPY src/elo /app/elo
COPY params.yaml .
ENV PYTHONPATH=/app OPENBLAS_NUM_THREADS=1
USER elo
EXPOSE 8000 18000
CMD ["uvicorn", "elo.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
