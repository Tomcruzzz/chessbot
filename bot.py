"""
bot.py — Game handler.

Wraps the search engine in a small layer that:
  * Maintains a `chess.Board` reconstructed from the moves Lichess sends.
  * Picks a search depth based on the clock (simple time management).
  * Returns the best move in UCI format for the Lichess API.

A `GameHandler` instance is created per game by `main.py`.
"""

from __future__ import annotations

import chess

import engine


# Time-management thresholds (in seconds of remaining clock). When we have
# plenty of time we search deeper; when low on time we search shallower so we
# never flag. Tune these to taste / hardware.
def choose_depth(remaining_seconds: float) -> int:
    """Pick a search depth from the remaining clock time.

    Conservative defaults that keep move times reasonable on typical hardware
    while still playing a meaningful game in blitz/rapid.
    """
    if remaining_seconds is None:
        return 4
    if remaining_seconds > 120:      # > 2 min: search deep
        return 5
    if remaining_seconds > 30:       # 30s–2min: medium
        return 4
    if remaining_seconds > 10:       # 10–30s: shallow
        return 3
    return 2                         # < 10s: emergency, move fast


class GameHandler:
    """Holds the state for a single Lichess game and produces moves."""

    def __init__(self, game_id: str, bot_color: chess.Color):
        self.game_id = game_id
        self.bot_color = bot_color  # chess.WHITE or chess.BLACK
        self.board = chess.Board()

    def set_position(self, moves_str: str) -> None:
        """Rebuild the board from the space-separated UCI move list Lichess
        provides in the `gameState` event (e.g. "e2e4 e7e5 g1f3").

        We rebuild from scratch each time, which is simple and robust against
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

    def pick_move(self, remaining_seconds: float | None = None) -> str | None:
        """Return the best move in UCI string form, or None if no move."""
        depth = choose_depth(remaining_seconds)
        move = engine.search(self.board, depth)
        return move.uci() if move else None
