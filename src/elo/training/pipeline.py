"""MLproject baseline, parallel XGBoost sweep and held-out evaluation entry points."""

import argparse
import itertools
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import ray
import xgboost as xgb
import yaml
from mlflow import MlflowClient
from mlflow.entities import Metric
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from elo.common.config import load_params
from elo.common.errors import EloError
from elo.common.logging import get_logger
from elo.models.aggregate import CATEGORIES, FEATURE_COLUMNS, TIME_FEATURES
from elo.models.conformal import calibrate, radii
from elo.models.scoring import predict
from elo.training.dataset import write_json

logger = get_logger(__name__)
MODEL_DIR = Path("models/xgb")


class TrainingError(EloError):
    """A configured sweep cannot respect the local resource budget."""


def datasets(*names):
    return [pd.read_parquet(f"data/datasets/{name}.parquet") for name in names]


def baseline_predictions(train, target):
    """Fit all three baselines solely on training labels."""
    means = train.groupby("tc_class", observed=True).elo.mean()
    linear = make_pipeline(
        ColumnTransformer(
            [
                ("tc", OneHotEncoder(handle_unknown="ignore"), ["tc_class"]),
                ("numeric", "passthrough", ["score", "n_plies"]),
            ]
        ),
        LinearRegression(),
    ).fit(train, train.elo)
    return {
        "B0": np.full(len(target), train.elo.mean()),
        "B1": target.tc_class.astype(object).map(means).fillna(train.elo.mean()).to_numpy(),
        "B2": linear.predict(target),
    }


def train_baselines():
    train, val, test = datasets("train", "val", "test")
    predictions = {
        name: baseline_predictions(train, frame) for name, frame in [("val", val), ("test", test)]
    }
    with mlflow.start_run() as project:
        mlflow.set_tag("mlflow.runName", "baselines")
        report = {}
        for name in ["B0", "B1", "B2"]:
            with mlflow.start_run(run_name=name, nested=True) as run:
                metrics = {
                    f"{split}_mae": mean_absolute_error(frame.elo, predictions[split][name])
                    for split, frame in [("val", val), ("test", test)]
                }
                mlflow.log_metrics(metrics)
                report[name] = dict(run_id=run.info.run_id, **metrics)
        write_json("reports/baselines.json", report)
        mlflow.log_artifact("reports/baselines.json")
        logger.info("Baseline project %s: %s", project.info.run_id, report)


@ray.remote
def fit_trial(run_id, uri, train, val, settings, grid_params, seed):
    """A Ray task owns one child run and only its allocated XGBoost threads."""
    import time

    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    with mlflow.start_run(run_id=run_id):
        model = xgb.XGBRegressor(
            objective="reg:absoluteerror",
            tree_method="hist",
            enable_categorical=True,
            eval_metric="mae",
            n_estimators=settings["n_estimators"],
            early_stopping_rounds=settings["early_stopping_rounds"],
            n_jobs=settings["n_jobs_per_trial"],
            random_state=seed,
            **grid_params,
        )
        model.fit(
            train[FEATURE_COLUMNS],
            train.elo,
            eval_set=[(val[FEATURE_COLUMNS], val.elo)],
            verbose=False,
        )
        history = model.evals_result()["validation_0"]["mae"]
        # Batched explicit logging retains every iteration's step without thousands of HTTP calls.
        stamp = int(time.time() * 1000)
        for start in range(0, len(history), 500):
            client.log_batch(
                run_id,
                metrics=[
                    Metric("val_mae_iteration", float(value), stamp, step)
                    for step, value in enumerate(history[start : start + 500], start)
                ],
            )
        mae = mean_absolute_error(val.elo, model.predict(val[FEATURE_COLUMNS]))
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({"val_mae": mae, "best_iteration": model.best_iteration})
        return run_id, bytes(model.get_booster().save_raw(raw_format="ubj"))


def train_xgb():
    p = load_params()
    cfg = p["train"]
    settings = cfg["xgb"]
    threads = settings["n_jobs_per_trial"]
    if not 1 <= threads <= 12:
        raise TrainingError("n_jobs_per_trial must be within 1..12")
    train, val, cal = datasets("train", "val", "cal")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run() as parent:
        mlflow.set_tag("mlflow.runName", "xgb_sweep")
        client = MlflowClient()
        tasks = []
        ray.init(
            num_cpus=min(p["ray"]["num_workers"], int(os.getenv("ELO_RAY_CPUS", "12"))),
            include_dashboard=False,
            runtime_env={
                "env_vars": {"OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": "1"}
            },
        )
        try:
            train_ref, val_ref = ray.put(train), ray.put(val)
            keys = list(settings["grid"])
            for values in itertools.product(*(settings["grid"][k] for k in keys)):
                grid = dict(zip(keys, values))
                child = client.create_run(
                    parent.info.experiment_id,
                    tags={
                        "mlflow.parentRunId": parent.info.run_id,
                        "mlflow.runName": "xgb_" + "_".join(map(str, values)),
                    },
                )
                tasks.append(
                    fit_trial.options(num_cpus=threads).remote(
                        child.info.run_id,
                        mlflow.get_tracking_uri(),
                        train_ref,
                        val_ref,
                        settings,
                        grid,
                        p["seed"],
                    )
                )
            fitted = dict(ray.get(tasks))
        finally:
            ray.shutdown()
        best = mlflow.search_runs(
            [parent.info.experiment_id],
            filter_string=(
                f"tags.mlflow.parentRunId = '{parent.info.run_id}' "
                "and attributes.status = 'FINISHED'"
            ),
            order_by=["metrics.val_mae ASC"],
        ).iloc[0]
        booster = xgb.Booster()
        booster.load_model(bytearray(fitted[best.run_id]))
        # Preserve only trees through the validation-selected best iteration.
        booster = booster[: int(best["metrics.best_iteration"]) + 1]
        booster.save_model(MODEL_DIR / "model.ubj")
        residuals = np.abs(
            cal.elo - booster.predict(xgb.DMatrix(cal[FEATURE_COLUMNS], enable_categorical=True))
        )
        write_json(
            MODEL_DIR / "conformal.json", calibrate(residuals, cal.tc_class, cfg["conformal_alpha"])
        )
        write_json(MODEL_DIR / "feature_columns.json", FEATURE_COLUMNS)
        (MODEL_DIR / "reference_stats.json").write_bytes(
            Path("data/datasets/reference_stats.json").read_bytes()
        )
        importance = booster.get_score(importance_type="gain")
        write_json(MODEL_DIR / "gain_importance.json", importance)
        plt = plotting()
        fig, ax = plt.subplots(figsize=(9, 8))
        top = sorted(importance, key=importance.get)[-25:]
        ax.barh(top, [importance[k] for k in top])
        ax.set_xlabel("XGBoost gain")
        fig.tight_layout()
        fig.savefig(MODEL_DIR / "gain_importance.png")
        plt.close(fig)
        mlflow.log_artifacts(str(MODEL_DIR), artifact_path="model")
        best_params = {k: best[f"params.{k}"] for k in keys}
        mlflow.log_params(best_params)
        mlflow.log_metric("val_mae", best["metrics.val_mae"])
        write_json(
            "reports/train_xgb.json",
            dict(
                run_id=parent.info.run_id,
                best_run_id=best.run_id,
                best_params=best_params,
                val_mae=float(best["metrics.val_mae"]),
            ),
        )
        logger.info("Best validation result: %s", Path("reports/train_xgb.json").read_text())


def plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def flatten(mapping, prefix=""):
    return {
        key: value
        for k, v in mapping.items()
        for key, value in (
            flatten(v, prefix + k + ".").items() if isinstance(v, dict) else [(prefix + k, v)]
        )
    }


def evaluate():
    p = load_params()
    report = json.loads(Path("reports/train_xgb.json").read_text())
    model = xgb.Booster()
    model.load_model(MODEL_DIR / "model.ubj")
    cal = json.loads((MODEL_DIR / "conformal.json").read_text())
    (test,) = datasets("test")
    pred = predict(model, test)
    error = np.abs(test.elo.to_numpy() - pred)
    q = radii(test, cal)
    metrics = dict(
        test_mae=float(error.mean()),
        test_rmse=float(np.sqrt(mean_squared_error(test.elo, pred))),
        coverage=float((error <= q).mean()),
        mean_interval_width=float((2 * q).mean()),
    )
    groups = test.tc_class.astype(str).to_numpy()
    bands = ["<1200", "1200-1500", "1500-1800", "1800-2100", ">=2100"]
    bins = [-np.inf, 1200, 1500, 1800, 2100, np.inf]
    true_band = pd.cut(test.elo, bins, labels=bands, right=False)
    for kind, labels in [("tc", groups), ("band", true_band)]:
        for label in (CATEGORIES["tc_class"] if kind == "tc" else bands):
            mask = np.asarray(labels == label)
            if mask.any():
                metrics[f"mae_{kind}_{label}"] = float(error[mask].mean())
                if kind == "tc":
                    metrics[f"coverage_{label}"] = float((error[mask] <= q[mask]).mean())
    noclock = test.copy()
    noclock["has_clock"] = False
    noclock[[f"{s}_{c}" for s in ["own", "opp"] for c in TIME_FEATURES]] = np.nan
    metrics["test_noclock_mae"] = float(mean_absolute_error(test.elo, predict(model, noclock)))
    baselines = json.loads(Path("reports/baselines.json").read_text())
    for name, base in baselines.items():
        metrics[f"improvement_{name}"] = base["test_mae"] - metrics["test_mae"]
    metrics["ac1_ratio"] = metrics["test_mae"] / baselines["B1"]["test_mae"]
    metrics["ac1_pass"] = metrics["ac1_ratio"] <= 0.75
    metrics["ac2_pass"] = 0.75 <= metrics["coverage"] <= 0.85
    metrics["analysis_games_per_second"] = float(
        np.mean(
            [
                json.loads(path.read_text())["games_per_second"]
                for path in Path("reports/throughput").glob("*.json")
            ]
        )
    )
    replay = {}
    for month in p["data"]["replay_months"]:
        frame = pd.read_parquet(f"data/features/games/{month}.parquet")
        err = np.abs(frame.elo - predict(model, frame))
        replay[month] = dict(
            mae=float(err.mean()), coverage=float((err <= radii(frame, cal)).mean()), n=len(frame)
        )
    write_json("reports/replay_eval.json", replay)
    write_json("reports/metrics.json", metrics)
    plot_dir = Path("reports/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt = plotting()
    for name, y, ylabel in [
        ("residual", pred - test.elo, "Predicted minus true rating"),
        ("predicted", pred, "Predicted rating"),
    ]:
        fig, ax = plt.subplots()
        ax.scatter(test.elo, y, s=2, alpha=0.15)
        ax.set(xlabel="True rating", ylabel=ylabel)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{name}.png")
        plt.close(fig)
    fig, ax = plt.subplots()
    ax.bar(
        CATEGORIES["tc_class"],
        [metrics.get(f"coverage_{c}", np.nan) for c in CATEGORIES["tc_class"]],
    )
    ax.axhline(0.8, color="black")
    ax.set(ylabel="80% interval coverage", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(plot_dir / "coverage.png")
    plt.close(fig)
    matrix = confusion_matrix(
        true_band, pd.cut(pred, bins, labels=bands, right=False), labels=bands
    )
    fig, ax = plt.subplots()
    image = ax.imshow(matrix)
    ax.set(
        xticks=range(5),
        yticks=range(5),
        xticklabels=bands,
        yticklabels=bands,
        xlabel="Predicted band",
        ylabel="True band",
    )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(plot_dir / "confusion.png")
    plt.close(fig)
    # The evaluate MLproject run supplies provenance; metrics resume the sweep parent.
    os.environ.pop("MLFLOW_RUN_ID", None)
    with mlflow.start_run(run_id=report["run_id"]):
        mlflow.log_metrics(
            {k.replace(">=", "ge").replace("<", "lt"): float(v) for k, v in metrics.items()}
        )
        mlflow.log_params(flatten(p))
        lock = yaml.safe_load(Path("dvc.lock").read_text())
        dataset_md5 = next(
            out["md5"]
            for out in lock["stages"]["build_dataset"]["outs"]
            if out["path"] == "data/datasets"
        )
        mlflow.set_tags(
            dict(
                trigger=os.getenv("ELO_TRIGGER", "manual"),
                alert_name=os.getenv("ELO_ALERT_NAME", ""),
                train_months=json.dumps(p["train"]["fit_months"]),
                parent_champion_version=os.getenv("ELO_PARENT_CHAMPION_VERSION", ""),
                dvc_data_md5=dataset_md5,
            )
        )
        mlflow.log_artifacts(str(plot_dir), artifact_path="plots")
        for path in [
            "reports/dataset.json",
            "reports/metrics.json",
            "reports/replay_eval.json",
            "reports/baselines.json",
            "params.yaml",
        ]:
            mlflow.log_artifact(path)
    logger.info("Evaluation: %s", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("entry", choices=["train_baselines", "train_xgb", "evaluate"])
    globals()[parser.parse_args().entry]()
