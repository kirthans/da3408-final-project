"""Real artifacts exercise serving and occlusion without a Stockfish process."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from elo.features.moves import build_move_rows
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
            assert move["ply"] in rows[rows.game_id.eq(row.game_id)].ply.values
            assert move["side"] == row.side
