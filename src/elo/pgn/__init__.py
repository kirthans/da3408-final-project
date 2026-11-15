"""PGN parsing shared by batch analysis and serving."""

import io
from collections.abc import Mapping

import chess.pgn

from elo.common.errors import InvalidPgnError


def score_cp(score) -> int:
    """Map mate scores to a fixed signed 1000, ignoring distance to mate."""
    cp = score.score(mate_score=100000)
    return (1000 if cp > 0 else -1000) if score.is_mate() else cp


def sanitize_headers(headers: Mapping[str, str]) -> tuple[dict, dict | None]:
    """Separate rating labels and discard player identity and rating hints."""
    labels = {key: headers[key] for key in ("WhiteElo", "BlackElo") if key in headers}
    clean = {
        key: value
        for key, value in headers.items()
        if key not in ("White", "Black", "WhiteElo", "BlackElo")
        and not key.endswith(("RatingDiff", "Title"))
        and not key.startswith("Title")
    }
    return clean, labels or None


class PgnParser:
    """Parse legal mainlines with clocks and white-perspective evaluations."""

    def parse(self, text: str) -> list[dict]:
        """Return game records containing headers, starting FEN and move dictionaries."""
        stream, records = io.StringIO(text), []
        try:
            while (game := chess.pgn.read_game(stream)) is not None:
                if game.errors or not game.board().is_valid():
                    raise InvalidPgnError("Illegal PGN mainline or starting position")
                moves = []
                board = game.board()
                for ply, node in enumerate(game.mainline(), 1):
                    score = node.eval()
                    moves.append(
                        dict(
                            ply=ply,
                            side="w" if board.turn else "b",
                            san=board.san(node.move),
                            uci=node.move.uci(),
                            clock_s=node.clock(),
                            pgn_eval_cp=score_cp(score.white()) if score else None,
                        )
                    )
                    board.push(node.move)
                if not moves:
                    raise InvalidPgnError("PGN contains no moves")
                records.append(
                    dict(headers=dict(game.headers), fen=game.board().fen(), moves=moves)
                )
        except (ValueError, IndexError, TypeError) as exc:
            raise InvalidPgnError(str(exc)) from exc
        if not records:
            raise InvalidPgnError("PGN contains no games")
        return records

    def parse_movetext(self, text: str) -> list[dict]:
        """Parse one standard-start movetext, rejecting multi-game input."""
        games = self.parse(text)
        if len(games) != 1:
            raise InvalidPgnError("Expected one movetext")
        return games[0]["moves"]
