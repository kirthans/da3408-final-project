"""Reusable, bounded Stockfish analysis."""

import time
from concurrent.futures import TimeoutError

import chess
import chess.engine

from elo.common.errors import EngineError, EngineTimeoutError
from elo.pgn import score_cp


class EngineAnalyzer:
    """Own one UCI process, restarting it after errors or game timeouts."""

    def __init__(
        self,
        path="tools/stockfish/stockfish",
        nodes=20000,
        threads=1,
        hash_mb=16,
        game_timeout_s=120,
    ):
        self.path, self.nodes = path, nodes
        self.threads, self.hash_mb = threads, hash_mb
        self.game_timeout_s = game_timeout_s
        self.engine = None
        self._start()

    def _start(self):
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(
                self.path, timeout=self.game_timeout_s
            )
            self.engine.configure({"Threads": self.threads, "Hash": self.hash_mb})
        except (OSError, chess.engine.EngineError, TimeoutError) as exc:
            self.close()
            raise EngineError(f"Cannot start Stockfish: {exc}") from exc

    def close(self):
        """Close without waiting for a possibly unresponsive UCI quit."""
        if self.engine is not None:
            self.engine.close()
            self.engine = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def analyse_game(self, board_start: chess.Board, moves: list[dict]) -> dict:
        """Return N+1 evaluations and best moves, all scores from White's POV.

        PGN comments describe positions after moves; the unannotated initial
        position uses a neutral 0 cp baseline when all comments are available.
        """
        board = board_start.copy()
        if moves and all(move.get("pgn_eval_cp") is not None for move in moves):
            return dict(
                evals=[0] + [move["pgn_eval_cp"] for move in moves],
                best_moves=[None] * (len(moves) + 1),
                analysis_source="pgn_eval",
            )
        deadline = time.monotonic() + self.game_timeout_s
        evals, best = [], []
        try:
            for index in range(len(moves) + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Per-game budget exhausted")
                self.engine.timeout = remaining
                info = self.engine.analyse(board, chess.engine.Limit(nodes=self.nodes))
                if time.monotonic() > deadline:
                    raise TimeoutError("Per-game budget exhausted")
                evals.append(score_cp(info["score"].white()))
                best.append(info["pv"][0].uci() if info.get("pv") else None)
                if index < len(moves):
                    board.push_uci(moves[index]["uci"])
        except (TimeoutError, chess.engine.EngineError, OSError) as exc:
            self.close()
            self._start()
            error = EngineTimeoutError if isinstance(exc, TimeoutError) else EngineError
            raise error(str(exc)) from exc
        return dict(evals=evals, best_moves=best, analysis_source="stockfish")
