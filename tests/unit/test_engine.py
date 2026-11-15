from pathlib import Path
from unittest.mock import Mock, patch

import chess
import chess.engine
import pytest

from elo.common.errors import EngineError, EngineTimeoutError
from elo.engine import EngineAnalyzer
from elo.pgn import PgnParser


def test_fake_engine_and_reuse():
    fake = Mock()
    fake.analyse.return_value = {
        "score": chess.engine.PovScore(chess.engine.Cp(30), chess.BLACK),
        "pv": [chess.Move.from_uci("e2e4")],
    }
    with patch("chess.engine.SimpleEngine.popen_uci", return_value=fake):
        with EngineAnalyzer(nodes=123) as analyzer:
            moves = PgnParser().parse_movetext("1. e4 *")
            result = analyzer.analyse_game(chess.Board(), moves)
            assert result["evals"] == [-30, -30]
            assert result["best_moves"] == ["e2e4", "e2e4"]
            assert fake.analyse.call_count == 2
            assert fake.analyse.call_args.args[1].nodes == 123
            fake.configure.assert_called_once_with({"Threads": 1, "Hash": 16})
            moves[0]["pgn_eval_cp"] = 50
            result = analyzer.analyse_game(chess.Board(), moves)
            assert result == dict(
                evals=[0, 50], best_moves=[None, None], analysis_source="pgn_eval"
            )
            assert fake.analyse.call_count == 2
        fake.close.assert_called_once()


@pytest.mark.parametrize(
    "failure,error",
    [(TimeoutError(), EngineTimeoutError), (chess.engine.EngineTerminatedError(), EngineError)],
)
def test_restart(failure, error):
    broken, replacement = Mock(), Mock()
    broken.analyse.side_effect = failure
    with patch("chess.engine.SimpleEngine.popen_uci", side_effect=[broken, replacement]):
        with EngineAnalyzer() as analyzer:
            with pytest.raises(error):
                analyzer.analyse_game(chess.Board(), [])
            assert analyzer.engine is replacement
            broken.close.assert_called_once()


@pytest.mark.engine
@pytest.mark.skipif(
    not Path("tools/stockfish/stockfish").exists(), reason="Stockfish not installed"
)
def test_real_engine():
    with EngineAnalyzer(nodes=1000) as analyzer:
        result = analyzer.analyse_game(chess.Board(), PgnParser().parse_movetext("1. e4 e5 *"))
        assert len(result["evals"]) == 3
        assert result["analysis_source"] == "stockfish"
        assert result["best_moves"][0] in [m.uci() for m in chess.Board().legal_moves]
