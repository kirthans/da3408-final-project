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
        """Start a reusable engine with bounded resources (DESIGN D2).

        Args:
            path: UCI engine executable.
            nodes: Search nodes per position; callers supply the shared budget.
            threads: Engine worker threads.
            hash_mb: Engine hash-table size in MiB.
            game_timeout_s: Wall-clock budget for a whole game.

        Raises:
            EngineError: The UCI process cannot start or be configured."""
        self.path, self.nodes = path, nodes
        self.threads, self.hash_mb = threads, hash_mb
        self.game_timeout_s = game_timeout_s
        self.engine = None
        self._start()

    def _start(self):
        """Start and configure UCI, closing partial state before raising EngineError."""
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
        """Return this analyzer for deterministic context-manager cleanup."""
        return self

    def __exit__(self, *_):
        """Close the engine when leaving a context, including on exceptions."""
        self.close()

    def analyse_game(self, board_start: chess.Board, moves: list[dict]) -> dict:
        """Evaluate all N+1 positions from White's perspective with one game deadline.

        Complete PGN evaluation comments bypass Stockfish; the unannotated initial
        position uses a neutral 0 cp baseline in that path. An engine failure or
        timeout restarts the process before the domain error is raised.

        Args:
            board_start: Legal initial position; copied before applying moves.
            moves: Ordered plies with UCI notation and optional PGN evaluations.

        Returns:
            Evaluations, best moves and the pgn_eval or stockfish analysis source.

        Raises:
            EngineTimeoutError: The per-game wall-clock budget is exhausted.
            EngineError: The engine fails or cannot restart."""
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
