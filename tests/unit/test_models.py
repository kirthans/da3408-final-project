"""Hand-calculated aggregation, game isolation and calibration contract checks."""

import json

import numpy as np
import pandas as pd
import pytest

from elo.models.aggregate import (
    CLOCK_COLUMNS,
    FEATURE_COLUMNS,
    TIME_FEATURES,
    drop_clocks,
    perspective_rows,
    side_features,
)
from elo.models.conformal import calibrate, quantile
from elo.training import dataset
from elo.training.dataset import reference_stats, split_holdout
from elo.training.pipeline import baseline_predictions


@pytest.fixture
def frames():
    games = pd.DataFrame(
        [
            dict(
                game_id=g,
                tc_base=100,
                tc_inc=0,
                tc_class="blitz",
                result="1-0",
                termination="Normal",
                eco="B20",
                n_plies=4,
                has_clock=True,
            )
            for g in ["a", "b"]
        ]
    )
    moves = pd.DataFrame(
        [
            dict(
                game_id=g,
                side=s,
                ply=p,
                cpl=c,
                wp_loss=w,
                is_best=best,
                move_class=kind,
                phase=phase,
                legal_moves=20,
                material_change=0,
                clock_s=clock,
                time_spent_s=spent,
                clock_frac=clock / 100,
                time_trouble=clock < 30,
                tc_base=100,
            )
            for g in ["a", "b"]
            for s in ["w", "b"]
            for p, c, w, best, kind, phase, clock, spent in [
                (1, 0, 0, True, "ok", "opening", 90, 10),
                (3, 100, 20, False, "blunder", "middlegame", 20, 70),
            ]
        ]
    )
    return moves, games


def test_hand_calculation(frames):
    moves, games = frames
    features = side_features(moves)
    row = features.loc[("a", "w")]
    assert row.cpl_mean == 50
    assert row.cpl_median == 50
    assert row.cpl_p90 == 90
    assert row.accuracy_mean == pytest.approx(
        (99.9999 + 103.1668 * np.exp(-0.04354 * 20) - 3.1669) / 2
    )
    assert row.blunder_rate == 0.5
    assert row.early_cpl_mean == 0
    assert row.early_fraction == 0.5
    assert row.time_mean == pytest.approx(0.4)
    assert row.time_std == pytest.approx(np.std([0.1, 0.7], ddof=1))
    assert row.time_trouble_rate == 0.5
    assert row.final_clock_frac == 0.2
    rows = perspective_rows(features, games)
    white, black = rows[rows.is_white].set_index("game_id"), rows[~rows.is_white].set_index(
        "game_id"
    )
    for name in features.columns:
        np.testing.assert_allclose(white[f"own_{name}"], black[f"opp_{name}"], equal_nan=True)
    assert (white.score + black.score).eq(1).all()


def test_dropout_and_noclock(frames):
    moves, games = frames
    m, g = drop_clocks(moves, games, 0.5, 42)
    m2, g2 = drop_clocks(moves, games, 0.5, 42)
    pd.testing.assert_frame_equal(m, m2)
    pd.testing.assert_frame_equal(g, g2)
    chosen = g.loc[~g.has_clock, "game_id"]
    assert len(chosen) == 1
    assert m.loc[m.game_id.isin(chosen), CLOCK_COLUMNS].isna().all().all()
    pd.testing.assert_frame_equal(m.drop(columns=CLOCK_COLUMNS), moves.drop(columns=CLOCK_COLUMNS))
    no, meta = drop_clocks(moves, games, 1, 42)
    rows = perspective_rows(side_features(no), meta)
    assert not rows.has_clock.any()
    assert rows[[f"{s}_{c}" for s in ["own", "opp"] for c in TIME_FEATURES]].isna().all().all()
    null_best = moves.assign(is_best=None)
    assert side_features(null_best).best_rate.isna().all()


def test_occlusion_and_leakage(frames):
    moves, games = frames
    rows = perspective_rows(side_features(moves.iloc[1:]), games)
    assert len(rows) == 4
    assert list(rows[FEATURE_COLUMNS]) == FEATURE_COLUMNS
    assert rows.is_white.sum() == 2
    single = perspective_rows(
        side_features(moves[moves.game_id.eq("a")].iloc[1:]), games[games.game_id.eq("a")]
    )
    assert len(single) == 2
    assert not any(
        token in column
        for token in ["elo", "label", "hash", "event", "opening", "month", "role"]
        for column in FEATURE_COLUMNS
    )
    assert "opening" not in FEATURE_COLUMNS
    splits = split_holdout(
        pd.DataFrame({"game_id": np.repeat([f"g{i}" for i in range(200)], 2)}), 42
    )
    sets = [set(frame.game_id) for frame in splits.values()]
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i + 1 :])
    assert all(frame.groupby("game_id").size().eq(2).all() for frame in splits.values())
    stats = reference_stats(rows)
    assert all(sum(v["proportions"]) == pytest.approx(1) for v in stats.values())


def test_reference_excludes_training_clock_dropout(frames, tmp_path, monkeypatch):
    moves, games = frames
    rows = perspective_rows(side_features(moves), games)
    clean = pd.concat(
        [rows.assign(game_id=rows.game_id + str(i), elo=1500) for i in range(5)],
        ignore_index=True,
    )
    monkeypatch.chdir(tmp_path)
    game_dir = tmp_path / "data/features/games"
    game_dir.mkdir(parents=True)
    chunks = {}
    for month, chunk in [("2019-09", clean.iloc[:10]), ("2019-10", clean.iloc[10:])]:
        chunk.to_parquet(game_dir / f"{month}.parquet", index=False)
        dropped = chunk.copy()
        chosen = dropped.index[:2]  # One of five games loses clocks in each fit month.
        dropped.loc[chosen, "has_clock"] = False
        dropped.loc[chosen, [f"{s}_{c}" for s in ["own", "opp"] for c in TIME_FEATURES]] = np.nan
        chunks[month] = dropped
    clean.assign(has_clock=False).to_parquet(game_dir / "2019-12.parquet", index=False)
    monkeypatch.setattr(
        dataset,
        "load_params",
        lambda: dict(
            seed=42,
            train=dict(fit_months=list(chunks), holdout_month="2019-12", clock_dropout=0.2),
        ),
    )
    monkeypatch.setattr(dataset, "aggregate_month", lambda month, *args: (chunks[month], 1))
    dataset.build_dataset()
    train = pd.read_parquet("data/datasets/train.parquet")
    assert (~train.has_clock).mean() == pytest.approx(0.2)
    assert train.own_time_mean.isna().mean() == pytest.approx(0.2)
    actual = json.loads((tmp_path / "data/datasets/reference_stats.json").read_text())
    assert actual == reference_stats(clean)
    assert actual != reference_stats(train)


def test_conformal():
    rng = np.random.default_rng(42)
    residual = np.abs(rng.normal(size=5000))
    q = quantile(residual, 0.2)
    assert 0.78 <= (np.abs(rng.normal(size=20000)) <= q).mean() <= 0.82
    classes = np.array(["blitz"] * 2500 + ["rapid"] * 2490 + ["classical"] * 10)
    residual[2500:4990] *= 2
    c = calibrate(residual, classes)
    assert c["q"]["rapid"] > c["q"]["blitz"]
    assert c["q"]["classical"] == c["global_q"]
    assert quantile([1, 2, 3, 4], 0.2) == 4


def test_baseline_class_means():
    train = pd.DataFrame(
        dict(
            tc_class=["blitz", "blitz", "rapid", "rapid"],
            elo=[1000, 1200, 1600, 1800],
            score=[0, 1, 0, 1],
            n_plies=[20, 30, 40, 50],
        )
    )
    pred = baseline_predictions(train, train)
    np.testing.assert_allclose(pred["B1"], [1100, 1100, 1700, 1700])


def test_reference_bool_categories(monkeypatch):
    monkeypatch.setattr(dataset, "FEATURE_COLUMNS", ["has_clock", "is_white"])
    frame = pd.DataFrame(
        {
            name: pd.Series([False, True, None], dtype="boolean")
            for name in ["has_clock", "is_white"]
        }
    )
    for entry in reference_stats(frame).values():
        assert entry["type"] == "categorical"
        assert entry["categories"] == [False, True, "__missing__"]
        assert entry["proportions"] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
