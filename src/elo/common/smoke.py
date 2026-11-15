"""Verify MLflow parameter, metric and artifact logging through an MLproject."""

import mlflow

from elo.common.logging import get_logger


def main() -> None:
    """Use the experiment and run supplied by the MLflow project launcher."""
    logger = get_logger(__name__)
    with mlflow.start_run() as run:
        mlflow.log_param("milestone", "M1")
        mlflow.log_metric("smoke_success", 1.0)
        mlflow.log_text("Guess the Elo M1 smoke test\n", "smoke.txt")
        logger.info("Smoke run completed: %s", run.info.run_id)


if __name__ == "__main__":
    main()
