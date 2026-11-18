"""Dataset construction and train reference distributions.

reference_stats.json maps feature names to {type, edges, proportions} for numeric
features. Edges are unique internal deciles, with implicit -inf/+inf endpoints;
proportions correspond to (-inf,e0], (e0,e1], ..., (elast,+inf), then NaN.
Categorical entries have {type, categories, proportions}; the final category is
__missing__. Fixed categories include zero-frequency levels. All proportions use
all fit-month rows before clock dropout as denominator. These bins can be reused
directly for PSI; synthetic training dropout must not appear as real-world drift.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from elo.common.config import load_params
from elo.common.logging import get_logger
from elo.models.aggregate import (
    FEATURE_COLUMNS,
    GAME_META_COLUMNS,
    drop_clocks,
    perspective_rows,
    side_features,
)

logger = get_logger(__name__)


def write_json(path, value):
    """Create parent directories and write readable JSON, rejecting nonfinite numbers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def aggregate_month(month, dropout=0, seed=42):
    """Build labelled perspective rows from one sampled and analyzed month (DESIGN D6).

    Args:
        month: Monthly partition name in YYYY-MM format.
        dropout: Fraction of games whose clock features are masked for training.
        seed: Seed for reproducible clock dropout.

    Returns:
        A labelled perspective dataframe and the rounded count of masked games.
        Labels are joined only after feature aggregation."""
    games = pd.read_parquet(
        f"data/interim/sample/{month}.parquet",
        columns=[
            *GAME_META_COLUMNS,
            "label_white_elo",
            "label_black_elo",
        ],
    )
    moves = pd.read_parquet(f"data/features/moves/{month}.parquet")
    games = games[games.game_id.isin(moves.game_id.unique())]
    moves, games = drop_clocks(moves, games, dropout, seed)
    moves = moves.merge(
        games[["game_id", "tc_base", "has_clock"]], on="game_id", validate="many_to_one"
    )
    rows = perspective_rows(side_features(moves), games)
    labels = games.set_index("game_id")
    rows["elo"] = np.where(
        rows.is_white,
        rows.game_id.map(labels.label_white_elo),
        rows.game_id.map(labels.label_black_elo),
    )
    rows["month"] = month
    return rows, round(len(games) * dropout)


def split_holdout(rows, seed):
    """Split games into deterministic disjoint holdout thirds (DESIGN §24 A4).

    Args:
        rows: Labelled perspective rows from the holdout month.
        seed: Salt for the stable SHA-256 game partition.

    Returns:
        Validation, calibration and test frames. Both perspectives of a game
        share one bucket, preventing leakage across the three splits."""
    bucket = rows.game_id.map(
        lambda gid: int.from_bytes(hashlib.sha256(f"{seed}:{gid}".encode()).digest()[:8], "big") % 3
    )
    return {name: rows[bucket == i].copy() for i, name in enumerate(["val", "cal", "test"])}


def reference_stats(frame):
    """Build fixed numeric deciles and categorical proportions for PSI (DESIGN D17).

    Args:
        frame: Fit-month feature rows before synthetic clock dropout.

    Returns:
        Distribution entries with missing bins and deduplicated numeric edges."""
    result = {}
    for name in FEATURE_COLUMNS:
        col = frame[name]
        if pd.api.types.is_bool_dtype(col.dtype) or isinstance(col.dtype, pd.CategoricalDtype):
            categories = (
                [False, True] if pd.api.types.is_bool_dtype(col.dtype) else list(col.cat.categories)
            )
            counts = [int(col.eq(c).sum()) for c in categories] + [int(col.isna().sum())]
            result[name] = dict(
                type="categorical",
                categories=categories + ["__missing__"],
                proportions=(np.array(counts) / len(col)).tolist(),
            )
        else:
            values = col.astype(float).dropna().to_numpy()
            edges = (
                np.unique(np.quantile(values, np.arange(0.1, 1, 0.1))).tolist()
                if len(values)
                else []
            )
            counts = np.bincount(
                np.searchsorted(edges, values, side="left"), minlength=len(edges) + 1
            )
            counts = np.append(counts, col.isna().sum())
            result[name] = dict(
                type="numeric", edges=edges, proportions=(counts / len(col)).tolist()
            )
    return result


def build_dataset():
    """Write fit rows, disjoint holdout thirds and pre-dropout references (DESIGN §24 A3-A4)."""
    p = load_params()
    train = p["train"]
    chunks, dropped = [], 0
    for month in train["fit_months"]:
        rows, n = aggregate_month(month, train["clock_dropout"], p["seed"])
        chunks.append(rows)
        dropped += n
    splits = {
        "train": pd.concat(chunks, ignore_index=True),
        **split_holdout(
            pd.read_parquet(f"data/features/games/{train['holdout_month']}.parquet"), p["seed"]
        ),
    }
    Path("data/datasets").mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        rows.to_parquet(f"data/datasets/{name}.parquet", index=False)
    reference = pd.concat(
        [pd.read_parquet(f"data/features/games/{m}.parquet") for m in train["fit_months"]],
        ignore_index=True,
    )
    write_json("data/datasets/reference_stats.json", reference_stats(reference))
    write_json(
        "reports/dataset.json",
        dict(
            clock_dropped_games=dropped,
            splits={
                name: dict(
                    rows=len(rows),
                    games=rows.game_id.nunique(),
                    label_mean=float(rows.elo.mean()),
                    label_std=float(rows.elo.std()),
                )
                for name, rows in splits.items()
            },
        ),
    )
    logger.info("Built dataset; dropped clocks for %s games", dropped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month")
    args = parser.parse_args()
    if args.month:
        rows, _ = aggregate_month(args.month)
        Path("data/features/games").mkdir(parents=True, exist_ok=True)
        rows.to_parquet(f"data/features/games/{args.month}.parquet", index=False)
        logger.info("Aggregated %s: %s rows", args.month, len(rows))
    else:
        build_dataset()
