"""Pure game aggregation shared by training, serving and move occlusion."""

import numpy as np
import pandas as pd

GAME_META_COLUMNS = [
    "game_id",
    "tc_base",
    "tc_inc",
    "tc_class",
    "result",
    "termination",
    "eco",
    "n_plies",
    "has_clock",
]
CATEGORIES = {
    "tc_class": ["blitz", "rapid", "classical"],
    "termination_group": ["normal", "time", "other"],
    "eco_group": [f"{letter}{digit}" for letter in "ABCDE" for digit in range(10)] + ["unknown"],
}
CLOCK_COLUMNS = ["clock_s", "time_spent_s", "clock_frac", "time_trouble"]
TIME_FEATURES = ["time_mean", "time_std", "time_trouble_rate", "final_clock_frac"]
SIDE_COLUMNS = [
    "cpl_mean",
    "cpl_median",
    "cpl_p90",
    "wp_loss_mean",
    "accuracy_mean",
    "best_rate",
    "inaccuracy_rate",
    "mistake_rate",
    "blunder_rate",
    "early_cpl_mean",
    "early_fraction",
    "legal_moves_mean",
    "material_change_std",
    *TIME_FEATURES,
    "n_moves",
]
FEATURE_COLUMNS = [
    "is_white",
    *[f"{side}_{name}" for side in ["own", "opp"] for name in SIDE_COLUMNS],
    "score",
    "termination_group",
    "tc_class",
    "tc_base",
    "tc_inc",
    "n_plies",
    "has_clock",
    "eco_group",
]


def side_features(moves: pd.DataFrame) -> pd.DataFrame:
    """Aggregate arbitrary ply subsets; tc_base may be joined onto moves by the caller."""
    m = moves.sort_values(["game_id", "side", "ply"]).copy()
    # Lichess per-move accuracy from win-probability loss in percentage points.
    m["accuracy"] = (103.1668 * np.exp(-0.04354 * m.wp_loss) - 3.1669).clip(0, 100)
    m["is_best"] = pd.to_numeric(m.is_best, errors="coerce").astype(float)
    m["opening_flag"] = m.phase.eq("opening").astype(float)
    m["opening_cpl"] = m.cpl.where(m.phase.eq("opening"))
    for kind in ["inaccuracy", "mistake", "blunder"]:
        m[kind] = m.move_class.eq(kind).astype(float)
    # Raw move rows carry enough information to infer base when metadata is absent.
    base = m.tc_base if "tc_base" in m else m.clock_s / m.clock_frac.replace(0, np.nan)
    m["time_fraction"] = m.time_spent_s / base.replace(0, np.nan)
    m["time_trouble"] = pd.to_numeric(m.time_trouble, errors="coerce").astype(float)
    if "has_clock" in m:
        m.loc[~m.has_clock, ["time_fraction", "time_trouble", "clock_frac"]] = np.nan
    g = m.groupby(["game_id", "side"], observed=True)
    out = g.agg(
        cpl_mean=("cpl", "mean"),
        cpl_median=("cpl", "median"),
        wp_loss_mean=("wp_loss", "mean"),
        accuracy_mean=("accuracy", "mean"),
        best_rate=("is_best", "mean"),
        inaccuracy_rate=("inaccuracy", "mean"),
        mistake_rate=("mistake", "mean"),
        blunder_rate=("blunder", "mean"),
        early_cpl_mean=("opening_cpl", "mean"),
        early_fraction=("opening_flag", "mean"),
        legal_moves_mean=("legal_moves", "mean"),
        material_change_std=("material_change", "std"),
        time_mean=("time_fraction", "mean"),
        time_std=("time_fraction", "std"),
        time_trouble_rate=("time_trouble", "mean"),
        n_moves=("ply", "size"),
    )
    out["cpl_p90"] = g.cpl.quantile(0.9)
    # nth preserves a null on the final observed ply, unlike groupby.last().
    out["final_clock_frac"] = (
        m.drop_duplicates(["game_id", "side"], keep="last")
        .set_index(["game_id", "side"])
        .clock_frac
    )
    return out[SIDE_COLUMNS]


def perspective_rows(side_feats: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Return two symmetric rows per game, with fixed categorical encodings and no labels."""
    parts = []
    meta = games[GAME_META_COLUMNS].drop_duplicates("game_id").set_index("game_id")
    for side, opponent in [("w", "b"), ("b", "w")]:
        own = side_feats.reindex(
            pd.MultiIndex.from_product([meta.index, [side]], names=["game_id", "side"])
        ).droplevel("side")
        opp = side_feats.reindex(
            pd.MultiIndex.from_product([meta.index, [opponent]], names=["game_id", "side"])
        ).droplevel("side")
        row = meta.join(own.add_prefix("own_")).join(opp.add_prefix("opp_"))
        row["is_white"] = side == "w"
        white_score = row.result.map({"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0})
        row["score"] = white_score if side == "w" else 1 - white_score
        term = row.termination.fillna("").str.lower()
        row["termination_group"] = np.select(
            [term.eq("normal"), term.str.contains("time")], ["normal", "time"], default="other"
        )
        eco = row.eco.fillna("").str[:2]
        row["eco_group"] = eco.where(eco.isin(CATEGORIES["eco_group"]), "unknown")
        for col, categories in CATEGORIES.items():
            row[col] = pd.Categorical(row[col], categories=categories)
        row.loc[~row.has_clock, [f"{s}_{c}" for s in ["own", "opp"] for c in TIME_FEATURES]] = (
            np.nan
        )
        parts.append(row[FEATURE_COLUMNS].reset_index())
    return pd.concat(parts, ignore_index=True)


def drop_clocks(moves: pd.DataFrame, games: pd.DataFrame, frac: float, seed: int):
    """Drop clocks for an exact rounded fraction of sorted game IDs without mutating inputs."""
    ids = np.sort(games.game_id.unique())
    chosen = np.random.default_rng(seed).choice(ids, size=round(len(ids) * frac), replace=False)
    m, g = moves.copy(), games.copy()
    for column in CLOCK_COLUMNS:
        m[column] = m[column].astype(float)
    m.loc[m.game_id.isin(chosen), CLOCK_COLUMNS] = np.nan
    g.loc[g.game_id.isin(chosen), "has_clock"] = False
    return m, g
