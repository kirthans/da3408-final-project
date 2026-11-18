"""Real artifacts exercise serving and occlusion without a Stockfish process."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import xgboost as xgb

from elo.features.moves import build_move_rows
from elo.models import pyfunc
from elo.models.aggregate import GAME_META_COLUMNS, perspective_rows, side_features
from elo.models.pyfunc import EloPyfunc
from elo.pgn import PgnParser


def fixture_rows():
    games = PgnParser().parse(Path("tests/fixtures/pgn/features.pgn").read_text())
    rows = []
    for i, game in enumerate(games):
        meta = dict(
            game_id=f"game{i}",
            tc_base=300,
            tc_inc=2,
            tc_class="blitz",
            result=game["headers"]["Result"],
            termination="normal",
            eco="B20",
            n_plies=len(game["moves"]),
            has_clock=i == 0,
            fen=game["fen"],
        )
        analysis = dict(
            evals=[0] + [m["pgn_eval_cp"] or 0 for m in game["moves"]],
            best_moves=[None] * (len(game["moves"]) + 1),
        )
        rows.extend(
            {**r, **{k: v for k, v in meta.items() if k != "fen"}}
            for r in build_move_rows(meta, game["moves"], analysis)
        )
    return pd.DataFrame(rows)


@pytest.mark.skipif(not Path("models/xgb/model.ubj").exists(), reason="M4 artifacts absent")
def test_real_pyfunc():
    model = EloPyfunc()
    model.load_context(
        SimpleNamespace(
            artifacts={
                k: f"models/xgb/{v}"
                for k, v in {
                    "model": "model.ubj",
                    "conformal": "conformal.json",
                    "features": "feature_columns.json",
                }.items()
            }
        )
    )
    rows = fixture_rows()
    output = model.predict(None, rows, {"top_k": 1})
    pd.testing.assert_frame_equal(output, model.predict(None, rows, {"top_k": 1}))
    assert len(output) == 6
    for row in output.itertuples():
        assert row.lo <= row.elo <= row.hi
        assert abs(row.hi - row.lo - 2 * model.conformal["q"][row.tc_class]) <= 1
        influence = json.loads(row.influence)
        assert len(influence) <= 1
        for move in influence:
            assert (
                move["ply"]
                in rows[rows.game_id.eq(row.game_id) & rows.phase.ne("opening")].ply.values
            )
            assert move["side"] == row.side


@pytest.mark.parametrize("nonopening_count", [0, 1, 4])
def test_influence_excludes_opening_without_changing_elo(monkeypatch, nonopening_count):
    rows = fixture_rows()
    rows["phase"] = "opening"
    rows.loc[rows.index[:nonopening_count], "phase"] = "middlegame"
    original = rows.copy(deep=True)
    batches = []

    def capture_batch(batch):
        batches.append(batch.copy())
        return side_features(batch)

    class FeatureBooster:
        def predict(self, matrix):
            values = matrix.get_data().toarray()
            return 1800 - values[:, 0] + 10 * values[:, 1] + 20 * values[:, 2]

    model = EloPyfunc()
    model.columns = ["own_cpl_mean", "own_n_moves", "own_early_fraction"]
    model.booster = FeatureBooster()
    model.conformal = {"q": {"blitz": 100}, "global_q": 100}
    # Predict only the complete games as a reference, with no occlusion candidates.
    full = perspective_rows(side_features(rows), rows[GAME_META_COLUMNS])
    predictions = model.booster.predict(xgb.DMatrix(full[model.columns]))
    expected = {
        (row.game_id, "w" if row.is_white else "b"): round(float(pred))
        for row, pred in zip(full.itertuples(), predictions)
    }
    monkeypatch.setattr(pyfunc, "side_features", capture_batch)
    output = model.predict(None, rows, {"top_k": 5})
    count = 0
    for row in output.itertuples():
        assert row.elo == expected[(row.game_id, row.side)]
        items = json.loads(row.influence)
        count += len(items)
        candidates = rows[
            rows.game_id.eq(row.game_id) & rows.side.eq(row.side) & rows.phase.ne("opening")
        ]
        assert {item["ply"] for item in items} == set(candidates.ply)
    assert count == nonopening_count
    # Opening plies produce no leave-one-out variants, including the all-opening case.
    assert len(batches) == 1
    assert set(batches[0].game_id) == set(rows.game_id) | {
        f"{move.game_id}#{move.ply}" for move in rows[rows.phase.ne("opening")].itertuples()
    }
    pd.testing.assert_frame_equal(rows, original)
