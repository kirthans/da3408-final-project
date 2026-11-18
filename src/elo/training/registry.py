"""Register the trained pyfunc and move registry aliases for rollback."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from elo.common.logging import get_logger
from elo.models.aggregate import GAME_META_COLUMNS
from elo.models.pyfunc import EloPyfunc

LOG = get_logger(__name__)
NAME = "elo-estimator"


def register():
    """Register a pyfunc with a signature as challenger or initial champion (DESIGN D10-D11)."""
    run_id = json.loads(Path("reports/train_xgb.json").read_text())["run_id"]
    metrics = json.loads(Path("reports/metrics.json").read_text())
    artifacts = {
        key: str(Path("models/xgb", file).resolve())
        for key, file in {
            "model": "model.ubj",
            "conformal": "conformal.json",
            "features": "feature_columns.json",
            "reference": "reference_stats.json",
        }.items()
    }
    moves = pd.read_parquet("data/features/moves/2019-09.parquet")
    gid = moves.iloc[0].game_id
    games = pd.read_parquet("data/interim/sample/2019-09.parquet")
    example = moves[moves.game_id.eq(gid)].merge(games[GAME_META_COLUMNS], on="game_id")
    # JSON round-trip gives the same primitive dtypes used by REST serving.
    example = pd.DataFrame(json.loads(example.to_json(orient="records")))
    # Nullable numeric context must accept OTB games with no base or increment.
    example[["tc_base", "tc_inc"]] = example[["tc_base", "tc_inc"]].astype(float)
    model = EloPyfunc()
    model.load_context(SimpleNamespace(artifacts=artifacts))
    signature = infer_signature(example, model.predict(None, example), params={"top_k": 5})
    with mlflow.start_run(run_id=run_id):
        logged = mlflow.pyfunc.log_model(
            name="estimator",
            python_model=model,
            artifacts=artifacts,
            code_paths=["src/elo"],
            signature=signature,
            input_example=example,
            pip_requirements=[
                "mlflow==3.18.0",
                "xgboost-cpu==3.4.1",
                "pandas==3.1.1",
                "numpy==2.5.4",
                "pyarrow==26.0.1",
            ],
        )
    client = MlflowClient()
    version = mlflow.register_model(logged.model_uri, NAME)
    for key, value in {
        "run_id": run_id,
        "test_mae": metrics["test_mae"],
        "coverage": metrics["coverage"],
        "trigger": os.getenv("ELO_TRIGGER", "manual"),
    }.items():
        client.set_model_version_tag(NAME, version.version, key, str(value))
    aliases = client.get_registered_model(NAME).aliases
    alias = "challenger" if "champion" in aliases else "champion"
    client.set_registered_model_alias(NAME, alias, version.version)
    Path("reports/register.json").write_text(
        json.dumps(dict(name=NAME, version=version.version, aliases=[alias]), indent=2) + "\n"
    )
    LOG.info("Registered %s version %s as %s", NAME, version.version, alias)


def promote(version):
    """Move champion to an existing version and retain its predecessor (DESIGN D11).

    Args:
        version: Registry version to serve; validated before any alias changes.

    Repeating the current promotion preserves the previous rollback target."""
    client = MlflowClient()
    version = str(version)
    # Validate before changing either alias; repeating a promotion keeps rollback useful.
    client.get_model_version(NAME, version)
    current = client.get_registered_model(NAME).aliases.get("champion")
    if client.get_registered_model(NAME).aliases.get("challenger") == version:
        client.delete_registered_model_alias(NAME, "challenger")
    if current == version:
        print(f"Version {version} is already champion.")
        return
    # Preserve the old serving target before switching champion for reload/rollback (DESIGN D11).
    if current is not None:
        client.set_registered_model_alias(NAME, "previous_champion", current)
    client.set_registered_model_alias(NAME, "champion", version)
    print(f"Champion promoted from {current} to version {version}.")


def reject(version):
    """Tag a rejected version and remove its challenger alias if present (DESIGN D11).

    Args:
        version: Rejected registry version; serving aliases are left intact."""
    client = MlflowClient()
    if client.get_registered_model(NAME).aliases.get("challenger") == str(version):
        client.delete_registered_model_alias(NAME, "challenger")
    client.set_model_version_tag(NAME, str(version), "gate", "rejected")


def reset(version):
    """Restore the baseline and clear aliases learned from replay months (DESIGN D11).

    Args:
        version: Baseline registry version to make champion."""
    promote(version)
    client = MlflowClient()
    for alias in ("previous_champion", "challenger"):
        if alias in client.get_registered_model(NAME).aliases:
            client.delete_registered_model_alias(NAME, alias)


def rollback():
    """Point champion at previous_champion when available, preserving other aliases (DESIGN D11)."""
    client = MlflowClient()
    previous = client.get_registered_model(NAME).aliases.get("previous_champion")
    if previous is None:
        print("No previous champion exists; champion is unchanged.")
        return
    client.set_registered_model_alias(NAME, "champion", previous)
    print(f"Champion rolled back to version {previous}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["register", "rollback", "promote"])
    parser.add_argument("--version", type=int)
    args = parser.parse_args()
    if args.action == "promote":
        if args.version is None or args.version < 1:
            parser.error("promote requires --version N (a positive integer)")
        promote(args.version)
    else:
        {"register": register, "rollback": rollback}[args.action]()
