"""
engine.py — A minimax chess engine with alpha-beta pruning.

Search features:
  * Iterative deepening with a wall-clock time budget (anytime: always has a
    legal move ready, searches deeper when time allows).
  * Negamax + alpha-beta pruning.
  * Principal Variation Search (PVS) — null-window scout searches after the
    first move.
  * Transposition table (Zobrist-hashed) with depth-preferred replacement and
    mate-distance-corrected scores.
  * Move ordering: TT move -> captures (MVV-LVA) -> promotions -> killer moves
    -> history heuristic. Good ordering is what makes alpha-beta fast.
  * Quiescence search over captures to mitigate the horizon effect.

Evaluation features:
  * Material (standard piece values).
  * Piece-square tables for every piece type.
  * Tapered king safety: interpolate a middlegame king table (stay castled)
    toward an endgame king table (centralise) based on remaining material.
  * Bishop-pair bonus and doubled-pawn penalty.
  * Mate-distance scoring so the engine prefers faster mates / slower losses.

Only depends on `python-chess`.
"""

from __future__ import annotations

import os
import random
import time
from typing import NamedTuple

import chess
import chess.polyglot

# --------------------------------------------------------------------------- #
# Material values (centipawns).
# --------------------------------------------------------------------------- #
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# Score used to represent checkmate. "Mate in N plies" is encoded as
# MATE_SCORE - N so that the search prefers the quickest mate. MATE_BOUND is
# the threshold above which a score is considered "a mate" (used for TT
# correction and to stop deepening once a forced mate is found).
MATE_SCORE = 1_000_000
MATE_BOUND = MATE_SCORE - 1000

# --------------------------------------------------------------------------- #
# Piece-square tables (White's point of view, index 0 = a1 .. 63 = h8).
# For Black we mirror vertically via `chess.square_mirror`.
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

# Middlegame king table: encourages castling and staying tucked away.
KING_MG_PST = [
     20,  30,  10,   0,   0,  10,  30,  20,
     20,  20,   0,   0,   0,   0,  20,  20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
]

# Endgame king table: with few pieces left the king should be active and
# centralised, so we reward central squares instead of corners.
KING_EG_PST = [
    -50, -30, -30, -30, -30, -30, -30, -50,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -50, -40, -30, -20, -20, -30, -40, -50,
]

PST = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
}

# Phase weights for the tapered evaluation. The opening has a phase of 24;
# as pieces come off the phase drops toward 0 (pure endgame).
_PHASE_WEIGHTS = {
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
}
_PHASE_TOTAL = 24

BISHOP_PAIR_BONUS = 30
DOUBLED_PAWN_PENALTY = 12
TEMPO_BONUS = 8

# MVV-LVA victim/attacker ranks for capture ordering.
_MVV_LVA = {
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

    Positive = good for the side to move. This is a *static* score only;
    terminal positions (mate / stalemate) are handled by the search, which
    knows the ply distance needed for correct mate scoring.
    """
    white = 0
    black = 0

    white_bishops = 0
    black_bishops = 0
    # Pawn counts per file, used for the doubled-pawn penalty.
    white_pawn_files = [0] * 8
    black_pawn_files = [0] * 8
    phase = 0

    for square, piece in board.piece_map().items():
        ptype = piece.piece_type
        value = PIECE_VALUES[ptype]
        phase += _PHASE_WEIGHTS.get(ptype, 0)

        if piece.color == chess.WHITE:
            psq = square
            if ptype == chess.KING:
                positional = 0  # King handled separately (tapered) below.
            else:
                positional = PST[ptype][psq]
            white += value + positional
            if ptype == chess.BISHOP:
                white_bishops += 1
            elif ptype == chess.PAWN:
                white_pawn_files[chess.square_file(square)] += 1
        else:
            psq = chess.square_mirror(square)
            if ptype == chess.KING:
                positional = 0
            else:
                positional = PST[ptype][psq]
            black += value + positional
            if ptype == chess.BISHOP:
                black_bishops += 1
            elif ptype == chess.PAWN:
                black_pawn_files[chess.square_file(square)] += 1

    # Tapered king placement: blend middlegame and endgame tables by phase.
    phase = min(phase, _PHASE_TOTAL)
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is not None:
        white += _king_score(wk, phase)
    if bk is not None:
        black += _king_score(chess.square_mirror(bk), phase)

    # Bishop pair.
    if white_bishops >= 2:
        white += BISHOP_PAIR_BONUS
    if black_bishops >= 2:
        black += BISHOP_PAIR_BONUS

    # Doubled pawns (penalise each extra pawn on a file).
    for f in range(8):
        if white_pawn_files[f] > 1:
            white -= DOUBLED_PAWN_PENALTY * (white_pawn_files[f] - 1)
        if black_pawn_files[f] > 1:
            black -= DOUBLED_PAWN_PENALTY * (black_pawn_files[f] - 1)

    score = white - black  # White's perspective.
    score = score if board.turn == chess.WHITE else -score
    return score + TEMPO_BONUS  # Small bonus for having the move.


def _king_score(psq: int, phase: int) -> int:
    """Interpolate the king PST between middlegame and endgame by `phase`."""
    mg = KING_MG_PST[psq]
    eg = KING_EG_PST[psq]
    return (mg * phase + eg * (_PHASE_TOTAL - phase)) // _PHASE_TOTAL


# --------------------------------------------------------------------------- #
# Transposition table
# --------------------------------------------------------------------------- #
# Node-type flags for TT entries.
TT_EXACT = 0   # Score is exact (a PV node).
TT_LOWER = 1   # Score is a lower bound (beta cutoff / fail-high).
TT_UPPER = 2   # Score is an upper bound (fail-low).


class TTEntry(NamedTuple):
    depth: int
    flag: int
    score: int
    move: chess.Move | None


class SearchTimeout(Exception):
    """Raised internally to unwind the search when the time budget is spent."""


class SearchInfo(NamedTuple):
    move: chess.Move | None
    depth: int      # Last fully completed iterative-deepening depth.
    score: int      # Score (centipawns) from the side-to-move's view.
    nodes: int
    elapsed: float  # Seconds spent searching.


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class Engine:
    """A reusable search engine. Keep one instance per game so the
    transposition table, killer moves, and history heuristic persist across
    moves and make each search faster than the last."""

    # How often (in nodes) to check the clock. Checking every node would be
    # wasteful; 2047 is a cheap power-of-two mask.
    _TIME_CHECK_MASK = 2047

    def __init__(self, book_path: str | None = None) -> None:
        self.tt: dict[int, TTEntry] = {}
        # Two killer moves per ply (quiet moves that caused a cutoff).
        self.killers: list[list[chess.Move | None]] = [
            [None, None] for _ in range(64)
        ]
        # History heuristic: history[from_square][to_square] -> score.
        self.history: list[list[int]] = [[0] * 64 for _ in range(64)]
        self._deadline = 0.0
        self._nodes = 0
        # Optional Polyglot opening book (.bin). If the file is missing or
        # unreadable we silently fall back to pure search.
        self.book_path = book_path
        if book_path and os.path.isfile(book_path):
            print(f"Using opening book: {book_path}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def search(
        self,
        board: chess.Board,
        *,
        time_limit: float | None = 5.0,
        max_depth: int = 64,
    ) -> SearchInfo:
        """Find the best move using iterative deepening within `time_limit`
        seconds (and never deeper than `max_depth`).

        Returns a `SearchInfo`. The search operates on a copy of `board`, so
        the caller's board is never mutated.
        """
        start = time.monotonic()

        # --- Opening book: play a known move instantly if one exists. ---
        book_move = self._probe_book(board)
        if book_move is not None:
            return SearchInfo(book_move, 0, 0, 0, time.monotonic() - start)

        board = board.copy(stack=False)
        self._nodes = 0
        self._deadline = start + time_limit if time_limit else float("inf")

        # Decay history between moves so old data doesn't dominate.
        for row in self.history:
            for i in range(64):
                row[i] //= 2

        best_move: chess.Move | None = None
        best_score = 0
        completed_depth = 0

        legal = list(board.legal_moves)
        if not legal:
            return SearchInfo(None, 0, 0, 0, time.monotonic() - start)
        best_move = legal[0]  # Always have a fallback ready.

        for depth in range(1, max_depth + 1):
            try:
                score, move = self._search_root(board, depth)
            except SearchTimeout:
                break  # Time's up — keep the best move from the last depth.

            if move is not None:
                best_move, best_score = move, score
            completed_depth = depth

            # Stop early on a proven forced mate — deeper search won't help.
            if abs(best_score) >= MATE_BOUND:
                break

        return SearchInfo(
            best_move,
            completed_depth,
            best_score,
            self._nodes,
            time.monotonic() - start,
        )

    # ------------------------------------------------------------------ #
    # Root search (separate so we can recover the best *move*, not just a
    # score, and apply aspiration-free full-window search at the root).
    # ------------------------------------------------------------------ #
    def _search_root(
        self, board: chess.Board, depth: int
    ) -> tuple[int, chess.Move | None]:
        alpha, beta = -MATE_SCORE - 1, MATE_SCORE + 1
        best_score = -MATE_SCORE - 1
        best_move: chess.Move | None = None

        tt_move = self._tt_move(board)
        for move in self._order_moves(board, depth_ply=0, tt_move=tt_move):
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, ply=1)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if best_score > alpha:
                alpha = best_score

        return best_score, best_move

    # ------------------------------------------------------------------ #
    # Negamax with alpha-beta, PVS, and the transposition table.
    # ------------------------------------------------------------------ #
    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        allow_null: bool = True,
    ) -> int:
        # --- Clock check (cheap, periodic). ---
        self._nodes += 1
        if (self._nodes & self._TIME_CHECK_MASK) == 0 and \
                time.monotonic() >= self._deadline:
            raise SearchTimeout

        # --- Draw detection (repetition / 50-move). Guard with the halfmove
        # clock so we only pay for repetition checks when a draw is possible.
        if ply > 0 and board.halfmove_clock >= 4 and board.is_repetition(2):
            return 0
        if board.halfmove_clock >= 100 or board.is_insufficient_material():
            return 0

        # --- Check extension: searching one ply deeper when in check finds
        # tactics that would otherwise fall off the horizon, and guarantees we
        # never drop into quiescence (a static eval) while in check.
        in_check = board.is_check()
        if in_check:
            depth += 1

        alpha_orig = alpha
        key = chess.polyglot.zobrist_hash(board)

        # --- Transposition table probe. ---
        tt_move: chess.Move | None = None
        entry = self.tt.get(key)
        if entry is not None and entry.depth >= depth:
            score = _from_tt_score(entry.score, ply)
            if entry.flag == TT_EXACT:
                return score
            if entry.flag == TT_LOWER and score > alpha:
                alpha = score
            elif entry.flag == TT_UPPER and score < beta:
                beta = score
            if alpha >= beta:
                return score
        if entry is not None:
            tt_move = entry.move

        # --- Leaf: drop into quiescence search. ---
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply)

        # --- Null-move pruning. If we let the opponent move twice in a row
        # (we "pass") and our position is still good enough to cause a beta
        # cutoff, the real position is almost certainly a cutoff too — so we
        # skip it. Disabled when in check, in likely-zugzwang positions (no
        # non-pawn material, where passing can be *good*), and near mate.
        if (
            allow_null
            and not in_check
            and depth >= 3
            and beta < MATE_BOUND
            and self._has_non_pawn_material(board)
        ):
            reduction = 2 + (depth >= 6)
            board.push(chess.Move.null())
            score = -self._negamax(
                board, depth - 1 - reduction, -beta, -beta + 1, ply + 1,
                allow_null=False,
            )
            board.pop()
            if score >= beta:
                return beta

        # --- Generate & order moves; detect terminal nodes. ---
        moves = self._order_moves(board, depth_ply=ply, tt_move=tt_move)
        if not moves:
            if board.is_check():
                return -MATE_SCORE + ply  # Mated: prefer later (smaller) mates.
            return 0  # Stalemate.

        best_score = -MATE_SCORE - 1
        best_move: chess.Move | None = None
        first = True

        for move in moves:
            board.push(move)
            if first:
                # Full-window search for the first (best-ordered) move.
                score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                # PVS scout: try a null window first; re-search if it surprises.
                score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                if alpha < score < beta:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if best_score > alpha:
                alpha = best_score
            if alpha >= beta:
                # Beta cutoff. Reward quiet moves that cause cutoffs so they
                # are tried earlier next time (killer + history heuristics).
                if not board.is_capture(move) and move.promotion is None:
                    self._record_killer(ply, move)
                    self.history[move.from_square][move.to_square] += depth * depth
                break
            first = False

        # --- Store in the transposition table. ---
        if best_score <= alpha_orig:
            flag = TT_UPPER
        elif best_score >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT
        existing = self.tt.get(key)
        if existing is None or existing.depth <= depth:
            self.tt[key] = TTEntry(
                depth, flag, _to_tt_score(best_score, ply), best_move
            )

        return best_score

    # ------------------------------------------------------------------ #
    # Quiescence search — only captures and promotions until things settle.
    # ------------------------------------------------------------------ #
    def _quiescence(
        self, board: chess.Board, alpha: int, beta: int, ply: int
    ) -> int:
        self._nodes += 1
        if (self._nodes & self._TIME_CHECK_MASK) == 0 and \
                time.monotonic() >= self._deadline:
            raise SearchTimeout

        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        # Order captures by MVV-LVA only (cheap and effective here).
        captures = [
            m for m in board.legal_moves
            if board.is_capture(m) or m.promotion is not None
        ]
        captures.sort(key=lambda m: self._capture_score(board, m), reverse=True)

        for move in captures:
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, ply + 1)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    # ------------------------------------------------------------------ #
    # Move ordering helpers
    # ------------------------------------------------------------------ #
    def _order_moves(
        self,
        board: chess.Board,
        depth_ply: int,
        tt_move: chess.Move | None,
    ) -> list[chess.Move]:
        """Return legal moves best-first for maximum alpha-beta cutoffs."""
        killers = self.killers[depth_ply] if depth_ply < len(self.killers) else (None, None)

        def score(move: chess.Move) -> int:
            if tt_move is not None and move == tt_move:
                return 1_000_000  # Always try the TT/PV move first.
            if board.is_capture(move) or move.promotion is not None:
                return 100_000 + self._capture_score(board, move)
            if move == killers[0]:
                return 90_000
            if move == killers[1]:
                return 80_000
            return self.history[move.from_square][move.to_square]

        return sorted(board.legal_moves, key=score, reverse=True)

    def _capture_score(self, board: chess.Board, move: chess.Move) -> int:
        """MVV-LVA score for a capture/promotion, plus a promotion bonus."""
        s = 0
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        victim_type = victim.piece_type if victim else chess.PAWN  # en passant
        attacker_type = attacker.piece_type if attacker else chess.PAWN
        s += 10 * _MVV_LVA[victim_type] - _MVV_LVA[attacker_type]
        if move.promotion:
            s += 100 + PIECE_VALUES.get(move.promotion, 0) // 10
        return s

    def _record_killer(self, ply: int, move: chess.Move) -> None:
        if ply >= len(self.killers):
            return
        slot = self.killers[ply]
        if slot[0] != move:
            slot[1] = slot[0]
            slot[0] = move

    def _tt_move(self, board: chess.Board) -> chess.Move | None:
        entry = self.tt.get(chess.polyglot.zobrist_hash(board))
        return entry.move if entry else None

    def _probe_book(self, board: chess.Board) -> chess.Move | None:
        """Return a weighted-random move from the opening book, or None.

        The book is opened fresh each probe (cheap; avoids holding a file
        handle open for the whole game). Any error -> no book move.
        """
        if not self.book_path or not os.path.isfile(self.book_path):
            return None
        try:
            with chess.polyglot.open_reader(self.book_path) as reader:
                entry = reader.weighted_choice(board, random=random.Random())
                return entry.move
        except (IndexError, KeyError, OSError, ValueError):
            # IndexError/KeyError: position not in book. Others: bad file.
            return None

    @staticmethod
    def _has_non_pawn_material(board: chess.Board) -> bool:
        """True if the side to move has a piece other than pawns/king.

        Used to disable null-move pruning in pawn/king endgames, where passing
        the move can actually *help* (zugzwang) and would mislead the search.
        """
        side = board.occupied_co[board.turn]
        pieces = board.knights | board.bishops | board.rooks | board.queens
        return bool(side & pieces)


# --------------------------------------------------------------------------- #
# Mate-distance correction for TT scores.
#
# Mate scores are stored relative to the position they were found in, but the
# TT is shared across the whole tree. We convert to/from a ply-independent
# representation when writing/reading the table.
# --------------------------------------------------------------------------- #
def _to_tt_score(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score + ply
    if score <= -MATE_BOUND:
        return score - ply
    return score


def _from_tt_score(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score - ply
    if score <= -MATE_BOUND:
        return score + ply
    return score


# --------------------------------------------------------------------------- #
# Backwards-compatible module-level helpers (used by tests / simple callers).
# --------------------------------------------------------------------------- #
def search(board: chess.Board, depth: int) -> chess.Move | None:
    """Fixed-depth convenience wrapper. Prefer `Engine().search(...)` with a
    time limit for real play."""
    engine = Engine()
    info = engine.search(board, time_limit=None, max_depth=depth)
    return info.move
