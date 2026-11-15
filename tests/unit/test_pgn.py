from pathlib import Path

import pytest

from elo.common.errors import InvalidPgnError
from elo.pgn import PgnParser, sanitize_headers

FIXTURE = Path("tests/fixtures/pgn/features.pgn")


def test_multigame_clock_eval_and_sanitize():
    games = PgnParser().parse(FIXTURE.read_text())
    assert len(games) == 3
    first = games[0]["moves"][0]
    assert first == dict(ply=1, side="w", san="e4", uci="e2e4", clock_s=298.0, pgn_eval_cp=20)
    assert games[2]["moves"][-2]["pgn_eval_cp"] == -1000
    assert games[2]["moves"][-1]["pgn_eval_cp"] == -1000
    clean, labels = sanitize_headers(games[0]["headers"])
    assert labels == {"WhiteElo": "1500", "BlackElo": "1600"}
    assert (
        not {"White", "Black", "WhiteElo", "BlackElo", "BlackTitle", "WhiteRatingDiff"}
        & clean.keys()
    )
    assert all(m["clock_s"] is None for m in games[1]["moves"])
    assert PgnParser().parse_movetext("1. e4 {[%eval #3]} *")[0]["pgn_eval_cp"] == 1000


@pytest.mark.parametrize("text", ["", "garbage", "1. e4 e5 2. Ke3 *"])
def test_invalid(text):
    with pytest.raises(InvalidPgnError):
        PgnParser().parse(text)
