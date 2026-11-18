"""Load registry artifacts via the tracking server, caching immutable versions."""

import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import mlflow
import xgboost as xgb
from mlflow import MlflowClient

from elo.models.conformal import radii
from elo.models.scoring import predict


def load_bundle(alias):
    """Resolve a live alias to an immutable, cached model bundle (DESIGN D11)."""
    version = MlflowClient().get_model_version_by_alias("elo-estimator", alias)
    return _load(str(version.version), version.tags.get("test_mae"))


@lru_cache(maxsize=4)
def _load(version, test_mae):
    """Download and cache artifacts by version and test MAE, rather than mutable alias."""
    root = Path(
        mlflow.artifacts.download_artifacts(artifact_uri=f"models:/elo-estimator/{version}")
    )
    config = __import__("yaml").safe_load((root / "MLmodel").read_text())
    artifacts = config["flavors"]["python_function"]["artifacts"]

    def read(name):
        """Decode one JSON artifact using its path recorded in the MLmodel manifest."""
        return json.loads((root / artifacts[name]["path"]).read_text())

    booster = xgb.Booster(params={"nthread": 2})
    booster.load_model(root / artifacts["model"]["path"])
    return SimpleNamespace(
        booster=booster,
        conformal=read("conformal"),
        columns=read("features"),
        reference=read("reference"),
        version=version,
        test_mae=float(test_mae),
    )


def score(bundle, frame):
    """Return predictions and per-row conformal radii using the bundle feature order."""
    return predict(bundle.booster, frame, bundle.columns), radii(frame, bundle.conformal)
