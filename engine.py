"""
engine.py — A minimax chess engine with alpha-beta pruning.

Features:
  * Material evaluation using standard piece values.
  * Positional evaluation using piece-square tables (PSTs).
  * Alpha-beta pruning with move ordering (captures searched first).
  * Quiescence search to mitigate the horizon effect on tactical positions.
  * Iterative-deepening-friendly `search` entry point with a depth limit.

The engine is intentionally self-contained and depends only on `python-chess`.
"""

from __future__ import annotations

import chess

# --------------------------------------------------------------------------- #
# Material values (centipawns). The king is given an effectively infinite
# value so it is never traded; checkmate is handled separately in evaluation.
# --------------------------------------------------------------------------- #
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# Large finite score used to represent checkmate. Kept well below integer
# limits so that "mate in N" can be encoded as MATE_SCORE - N.
MATE_SCORE = 1_000_000

# --------------------------------------------------------------------------- #
# Piece-square tables.
#
# Each table is written from White's point of view, with index 0 = a1 and
# index 63 = h8 (matching python-chess square numbering). Values are small
# centipawn bonuses/penalties encouraging good piece placement. For Black we
# mirror the square vertically via `chess.square_mirror`.
# --------------------------------------------------------------------------- #

PAWN_PST = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10, -20, -20,  10,  10,   5,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,   5,  10,  25,  25,  10,   5,   5,
     10,  10,  20,  30,  30,  20,  10,  10,
     50,  50,  50,  50,  50,  50,  50,  50,
      0,   0,   0,   0,   0,   0,   0,   0,
]

KNIGHT_PST = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_PST = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

ROOK_PST = [
      0,   0,   0,   5,   5,   0,   0,   0,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      5,  10,  10,  10,  10,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
]

QUEEN_PST = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -10,   5,   5,   5,   5,   5,   0, -10,
      0,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]

# King table for the middlegame: encourages castling and staying tucked away.
KING_PST = [
     20,  30,  10,   0,   0,  10,  30,  20,
     20,  20,   0,   0,   0,   0,  20,  20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
]

PST = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
    chess.KING: KING_PST,
}

# MVV-LVA: rough victim/attacker ordering value for captures used in move
# ordering. Higher is searched first.
_MVV_LVA_VICTIM = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(board: chess.Board) -> int:
    """Static evaluation of `board` from the side-to-move's perspective.

    A positive score means the side to move is better. Terminal positions
    (checkmate / draw) are scored here so the search can stop cleanly.
    """
    if board.is_checkmate():
        # Side to move has been mated -> worst possible outcome.
        return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material() or \
            board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0

    score = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]
        # Piece-square bonus: mirror the table for Black so both colours use
        # the same White-oriented tables.
        pst_square = square if piece.color == chess.WHITE else chess.square_mirror(square)
        positional = PST[piece.piece_type][pst_square]

        if piece.color == chess.WHITE:
            score += value + positional
        else:
            score -= value + positional

    # `score` is from White's perspective; flip it for Black to move.
    return score if board.turn == chess.WHITE else -score


# --------------------------------------------------------------------------- #
# Move ordering
# --------------------------------------------------------------------------- #
def _order_moves(board: chess.Board) -> list[chess.Move]:
    """Return legal moves ordered so promising moves are searched first.

    Ordering heuristics, best-first:
      1. Captures, ranked by MVV-LVA (most valuable victim, least valuable
         attacker).
      2. Promotions.
      3. Checks.
      4. Everything else.
    Good ordering dramatically increases the number of alpha-beta cutoffs.
    """
    def move_score(move: chess.Move) -> int:
        s = 0
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            # En passant: the victim square is empty, but it's still a pawn.
            victim_type = victim.piece_type if victim else chess.PAWN
            attacker_type = attacker.piece_type if attacker else chess.PAWN
            s += 10 * _MVV_LVA_VICTIM[victim_type] - _MVV_LVA_VICTIM[attacker_type]
            s += 1000  # Always try captures before quiet moves.
        if move.promotion:
            s += 900 + PIECE_VALUES.get(move.promotion, 0) // 100
        if board.gives_check(move):
            s += 50
        return s

    return sorted(board.legal_moves, key=move_score, reverse=True)


# --------------------------------------------------------------------------- #
# Quiescence search
# --------------------------------------------------------------------------- #
def _quiescence(board: chess.Board, alpha: int, beta: int) -> int:
    """Search only "noisy" moves (captures) until the position is quiet.

    This avoids the horizon effect: without it, the fixed-depth search might
    stop in the middle of a capture sequence and badly misjudge material.
    """
    stand_pat = evaluate(board)

    # Beta cutoff: the position is already good enough to refute.
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    # Only consider captures (and promotions) to keep the tree small.
    for move in _order_moves(board):
        if not board.is_capture(move) and not move.promotion:
            continue
        board.push(move)
        score = -_quiescence(board, -beta, -alpha)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


# --------------------------------------------------------------------------- #
# Negamax with alpha-beta pruning
# --------------------------------------------------------------------------- #
def _negamax(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    """Negamax formulation of minimax with alpha-beta pruning.

    Returns the evaluation of `board` from the side-to-move's perspective.
    """
    # Terminal node: stop and evaluate (handles mate/draw too).
    if board.is_game_over():
        return evaluate(board)

    if depth == 0:
        # Drop into quiescence search instead of returning a static eval, so
        # we don't stop mid-capture.
        return _quiescence(board, alpha, beta)

    best = -MATE_SCORE - 1
    for move in _order_moves(board):
        board.push(move)
        score = -_negamax(board, depth - 1, -beta, -alpha)
        board.pop()

        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break  # Beta cutoff — opponent won't allow this line.

    return best


# --------------------------------------------------------------------------- #
# Public search entry point
# --------------------------------------------------------------------------- #
def search(board: chess.Board, depth: int) -> chess.Move | None:
    """Return the best move for the side to move, searching to `depth` plies.

    Uses a root negamax loop so we can keep track of the actual best *move*
    (the recursive helper only returns scores).
    """
    best_move: chess.Move | None = None
    best_score = -MATE_SCORE - 1
    alpha, beta = -MATE_SCORE - 1, MATE_SCORE + 1

    for move in _order_moves(board):
        board.push(move)
        score = -_negamax(board, depth - 1, -beta, -alpha)
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move
        if best_score > alpha:
            alpha = best_score

    # Fallback: if for some reason no move was chosen (shouldn't happen unless
    # the game is already over), return the first legal move if any.
    if best_move is None:
        legal = list(board.legal_moves)
        return legal[0] if legal else None

    return best_move
