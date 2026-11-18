"""Pure per-ply feature transform shared by offline and online callers."""

import math

import chess

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def _material(board, color, pawns=True):
    """Sum one side's piece values, optionally excluding pawns for phase detection."""
    return sum(
        value * len(board.pieces(piece, color))
        for piece, value in VALUES.items()
        if pawns or piece != chess.PAWN
    )


def _wp(cp):
    # Lichess winning chance in percentage points, not a 0..1 probability.
    """Convert a centipawn score to Lichess winning-chance percentage points."""
    return 100 / (1 + math.exp(-0.00368208 * cp))


def build_move_rows(game_meta: dict, moves: list[dict], analysis: dict) -> list[dict]:
    """Compute mover-relative losses and clocks without copying labels or identity.

    Args:
        game_meta: Safe game ID, starting FEN and time-control fields.
        moves: Ordered legal plies with optional post-move clocks.
        analysis: N+1 White-relative evaluations and optional best moves.

    Returns:
        Per-ply feature dictionaries shared by batch and serving (DESIGN D19)."""
    board = chess.Board(game_meta.get("fen", chess.STARTING_FEN))
    base, inc = game_meta.get("tc_base"), game_meta.get("tc_inc", 0) or 0
    previous = {chess.WHITE: base, chess.BLACK: base}
    rows = []
    for index, move in enumerate(moves):
        color = board.turn
        sign = 1 if color else -1
        # Bound mate/extreme scores before the logistic transform; Black reverses the loss sign.
        # CPL below is nonnegative and capped so one lost position cannot dominate a game.
        before, after = [max(-1000, min(1000, x)) for x in analysis["evals"][index : index + 2]]
        loss = max(0, sign * (_wp(before) - _wp(after)))
        material = _material(board, color) - _material(board, not color)
        nonpawn = [_material(board, side, False) for side in (True, False)]
        phase = (
            "opening"
            if move["ply"] <= 20 and all(_material(board, side) == 39 for side in (True, False))
            else "endgame" if max(nonpawn) <= 13 else "middlegame"
        )
        legal, check = board.legal_moves.count(), board.is_check()
        board.push_uci(move["uci"])
        balance = _material(board, color) - _material(board, not color)
        clock = move.get("clock_s")
        spent = (
            max(0, previous[color] - clock + inc)
            if clock is not None and previous[color] is not None
            else None
        )
        previous[color] = clock
        best = analysis["best_moves"][index]
        rows.append(
            dict(
                game_id=game_meta["game_id"],
                ply=move["ply"],
                side=move["side"],
                san=move["san"],
                eval_before_cp=before,
                eval_after_cp=after,
                cpl=min(1000, max(0, sign * (before - after))),
                wp_loss=loss,
                is_best=move["uci"] == best if best is not None else None,
                # Lichess [-1, 1] losses 0.1/0.2/0.3 map to 5/10/15 wp points.
                move_class=(
                    "blunder"
                    if loss >= 15
                    else "mistake" if loss >= 10 else "inaccuracy" if loss >= 5 else "ok"
                ),
                legal_moves=legal,
                in_check=check,
                material_mover=balance,
                material_change=balance - material,
                clock_s=clock,
                time_spent_s=spent,
                clock_frac=clock / base if clock is not None and base else None,
                time_trouble=(
                    (clock < 30 or clock < 0.1 * base if base else clock < 30)
                    if clock is not None
                    else None
                ),
                phase=phase,
            )
        )
    return rows
