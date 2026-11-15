from unittest.mock import Mock, patch

import pyarrow as pa

from elo.data.ray_analyze import MOVE_SCHEMA, AnalyzerActor


def test_actor_reuses_engine_and_skips_invalid_games():
    engine = Mock()
    engine.analyse_game.return_value = dict(
        evals=[0, 20], best_moves=[None, None], analysis_source="pgn_eval"
    )
    with patch("elo.data.ray_analyze.EngineAnalyzer", return_value=engine) as factory:
        actor = AnalyzerActor({"nodes": 100})
        batch = pa.Table.from_pylist(
            [
                dict(game_id="good", movetext="1. e4 {[%eval 0.2]} *"),
                dict(game_id="bad", movetext="garbage"),
            ]
        )
        result = actor(batch)
        assert result["_status"].to_pylist() == ["pgn_eval", "failed"]
        assert result.select(MOVE_SCHEMA.names).schema == MOVE_SCHEMA
        actor(batch.slice(0, 1))
        factory.assert_called_once_with(nodes=100)
        assert engine.analyse_game.call_count == 2
