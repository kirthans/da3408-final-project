from pathlib import Path

import pytest

from elo.features.moves import build_move_rows
from elo.pgn import PgnParser


def test_hand_computed_features():
    game = PgnParser().parse(Path("tests/fixtures/pgn/features.pgn").read_text())[0]
    rows = build_move_rows(
        dict(game_id="a", tc_base=300, tc_inc=2, label_white_elo=1500),
        game["moves"],
        dict(evals=[0, 20, 0, -200, 0], best_moves=[None] * 5),
    )
    assert rows[0]["time_spent_s"] == 4
    assert rows[2]["time_spent_s"] == 5
    assert rows[2]["cpl"] == 200
    assert rows[2]["wp_loss"] == pytest.approx(17.6211637485)
    assert rows[2]["move_class"] == "blunder"
    assert rows[0]["legal_moves"] == 20
    assert rows[0]["material_mover"] == 0
    assert rows[0]["phase"] == "opening"
    assert rows[0]["clock_frac"] == pytest.approx(298 / 300)
    assert rows[0]["is_best"] is None
    assert not any("label" in key for row in rows for key in row)


def test_capture_and_no_clocks():
    game = PgnParser().parse(Path("tests/fixtures/pgn/features.pgn").read_text())[1]
    rows = build_move_rows(
        dict(game_id="b"), game["moves"], dict(evals=[0] * 5, best_moves=["e2e4"] + [None] * 4)
    )
    assert rows[0]["is_best"] is True
    assert rows[2]["material_mover"] == rows[2]["material_change"] == 1
    assert rows[3]["material_mover"] == 0
    assert rows[3]["material_change"] == 1
    for row in rows:
        for key in ("clock_s", "time_spent_s", "clock_frac", "time_trouble"):
            assert row[key] is None


@pytest.mark.parametrize(
    "after,category",
    [(-60, "inaccuracy"), (-120, "mistake"), (-200, "blunder"), (-300, "blunder"), (20, "ok")],
)
def test_loss_class(after, category):
    moves = PgnParser().parse_movetext("1. e4 *")
    row = build_move_rows(dict(game_id="x"), moves, dict(evals=[0, after], best_moves=[None] * 2))[
        0
    ]
    assert row["move_class"] == category
