"""
bot.py — Game handler.

Wraps the search engine in a small layer that:
  * Maintains a `chess.Board` reconstructed from the moves Lichess sends.
  * Owns one persistent `Engine` per game (so the transposition table,
    killer moves, and history heuristic carry over between moves).
  * Budgets thinking time from the clock and lets the engine deepen until the
    budget is spent (iterative deepening).
  * Returns the best move in UCI format for the Lichess API.

A `GameHandler` instance is created per game by `main.py`.
"""

from __future__ import annotations

import os

import chess

import engine

# Hard cap on search depth regardless of time, to bound worst-case move time.
MAX_DEPTH = 20

# Optional Polyglot opening book. Drop a `book.bin` next to this file (or set
# the BOOK_PATH env var) and the engine will use it for opening moves. If it's
# absent the bot plays openings from search, so this is purely opt-in.
BOOK_PATH = os.environ.get(
    "BOOK_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "book.bin"),
)

# Never spend more than this fraction of the remaining clock on one move, and
# never less than a small floor so we still search something when very low.
MAX_TIME_FRACTION = 0.05
MIN_BUDGET = 0.05
MAX_BUDGET = 8.0


def choose_time_budget(
    remaining_seconds: float | None, increment_seconds: float = 0.0
) -> float:
    """Decide how many seconds to think about the current move.

    Strategy: spend a small fraction of the remaining time plus most of the
    increment (which is replenished each move). This keeps us from flagging in
    long games while still using the increment in fast ones.
    """
    if remaining_seconds is None:
        return 2.0  # No clock info (e.g. correspondence) — a sane default.

    budget = remaining_seconds * MAX_TIME_FRACTION + increment_seconds * 0.8

    # When desperately low on time, move almost instantly.
    if remaining_seconds < 10:
        budget = min(budget, 0.2)
    elif remaining_seconds < 30:
        budget = min(budget, 1.0)

    return max(MIN_BUDGET, min(budget, MAX_BUDGET))


class GameHandler:
    """Holds the state for a single Lichess game and produces moves."""

    def __init__(self, game_id: str, bot_color: chess.Color):
        self.game_id = game_id
        self.bot_color = bot_color  # chess.WHITE or chess.BLACK
        self.board = chess.Board()
        # One engine for the whole game: its TT/history warm up over time.
        self.engine = engine.Engine(book_path=BOOK_PATH)

    def set_position(self, moves_str: str) -> None:
        """Rebuild the board from the space-separated UCI move list Lichess
        provides in the `gameState` event (e.g. "e2e4 e7e5 g1f3").

        Rebuilding from scratch each time is simple and robust against any
        missed events.
        """
        self.board = chess.Board()
        if moves_str:
            for uci in moves_str.split():
                try:
                    self.board.push_uci(uci)
                except ValueError:
                    # Ignore malformed/duplicate tokens defensively.
                    break

    def is_our_turn(self) -> bool:
        """True if it is the bot's turn to move in the current position."""
        return self.board.turn == self.bot_color and not self.board.is_game_over()

    def pick_move(
        self,
        remaining_seconds: float | None = None,
        increment_seconds: float = 0.0,
    ) -> str | None:
        """Return the best move in UCI string form, or None if no move.

        Also returns via the `last_info` attribute the search statistics for
        logging (depth reached, score, nodes, time).
        """
        budget = choose_time_budget(remaining_seconds, increment_seconds)
        info = self.engine.search(
            self.board, time_limit=budget, max_depth=MAX_DEPTH
        )
        self.last_info = info
        return info.move.uci() if info.move else None
