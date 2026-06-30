"""
main.py — Entry point for the Lichess chess bot.

Responsibilities:
  * Read the Lichess API token from a `.env` file.
  * Open the main event stream (SSE) and react to:
      - `challenge`     -> accept (or decline) incoming challenges.
      - `gameStart`     -> spin up a handler thread for the new game.
  * For each game, stream the per-game events (SSE) and respond to:
      - `gameFull`      -> initial position + clock + colours.
      - `gameState`     -> a move was made; compute and play our reply.

Each game runs in its own thread so the bot can play several at once and keep
listening for new challenges on the main thread.
"""

from __future__ import annotations

import os
import threading

import berserk
import chess
from dotenv import load_dotenv

from bot import GameHandler


def create_client() -> berserk.Client:
    """Load the token from `.env` and return an authenticated berserk client."""
    load_dotenv()
    token = os.getenv("LICHESS_TOKEN")
    if not token:
        raise SystemExit(
            "LICHESS_TOKEN is not set. Copy .env.example to .env and add your "
            "Lichess bot API token (https://lichess.org/account/oauth/token)."
        )
    session = berserk.TokenSession(token)
    return berserk.Client(session=session)


class LichessBot:
    """Coordinates the account, the main event stream, and per-game threads."""

    def __init__(self, client: berserk.Client):
        self.client = client
        self.account = client.account.get()
        self.username = self.account["username"]
        print(f"Logged in as: {self.username}")

    # ------------------------------------------------------------------ #
    # Challenge handling
    # ------------------------------------------------------------------ #
    def handle_challenge(self, challenge: dict) -> None:
        """Accept standard challenges; decline variants we can't play."""
        challenge_id = challenge["id"]
        variant = challenge.get("variant", {}).get("key", "standard")
        challenger = challenge.get("challenger", {}).get("name", "?")

        # This simple engine only understands standard chess.
        if variant != "standard":
            print(f"Declining {variant} challenge from {challenger}")
            try:
                self.client.challenges.decline(
                    challenge_id, reason="variant"
                )
            except Exception as exc:  # noqa: BLE001 - log and continue
                print(f"  (decline failed: {exc})")
            return

        print(f"Accepting challenge {challenge_id} from {challenger}")
        try:
            self.client.challenges.accept(challenge_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  (accept failed: {exc})")

    # ------------------------------------------------------------------ #
    # Per-game loop (runs in its own thread)
    # ------------------------------------------------------------------ #
    def play_game(self, game_id: str) -> None:
        """Stream a single game's events and play our moves until it ends."""
        print(f"[{game_id}] starting game thread")
        handler: GameHandler | None = None

        try:
            for event in self.client.bots.stream_game_state(game_id):
                event_type = event["type"]

                if event_type == "gameFull":
                    # Determine our colour from which side we were assigned.
                    white_id = event["white"].get("id")
                    bot_color = (
                        chess.WHITE
                        if white_id == self.account["id"]
                        else chess.BLACK
                    )
                    handler = GameHandler(game_id, bot_color)
                    state = event["state"]
                    self._on_state(handler, state)

                elif event_type == "gameState":
                    if handler is None:
                        continue
                    self._on_state(handler, event)

                elif event_type == "chatLine":
                    pass  # Ignore chat.

        except Exception as exc:  # noqa: BLE001
            print(f"[{game_id}] game stream error: {exc}")

        print(f"[{game_id}] game thread finished")

    def _on_state(self, handler: GameHandler, state: dict) -> None:
        """React to a new game state: if it's our turn, compute and play."""
        status = state.get("status", "started")
        if status != "started":
            # Game over (mate, resign, draw, etc.).
            print(f"[{handler.game_id}] game over: {status}")
            return

        handler.set_position(state.get("moves", ""))

        if not handler.is_our_turn():
            return

        # Pull our remaining clock and increment (milliseconds) for time mgmt.
        if handler.bot_color == chess.WHITE:
            remaining_ms = state.get("wtime")
            increment_ms = state.get("winc")
        else:
            remaining_ms = state.get("btime")
            increment_ms = state.get("binc")
        remaining_seconds = _ms_to_seconds(remaining_ms)
        increment_seconds = _ms_to_seconds(increment_ms) or 0.0

        move = handler.pick_move(remaining_seconds, increment_seconds)
        if move is None:
            return

        info = getattr(handler, "last_info", None)
        if info is not None:
            print(
                f"[{handler.game_id}] playing {move}  "
                f"(depth {info.depth}, score {info.score}, "
                f"{info.nodes} nodes, {info.elapsed:.2f}s)"
            )
        else:
            print(f"[{handler.game_id}] playing {move}")

        try:
            self.client.bots.make_move(handler.game_id, move)
        except Exception as exc:  # noqa: BLE001
            print(f"[{handler.game_id}] make_move failed: {exc}")

    # ------------------------------------------------------------------ #
    # Main event stream
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Listen on the main event stream forever, dispatching events."""
        print("Listening for challenges and games...  (Ctrl+C to stop)")
        for event in self.client.bots.stream_incoming_events():
            event_type = event["type"]

            if event_type == "challenge":
                self.handle_challenge(event["challenge"])

            elif event_type == "gameStart":
                game_id = event["game"]["id"]
                thread = threading.Thread(
                    target=self.play_game, args=(game_id,), daemon=True
                )
                thread.start()

            elif event_type == "gameFinish":
                pass  # A game ended; its thread will exit on its own.


def _ms_to_seconds(value) -> float | None:
    """Convert a clock value from Lichess (ms int or datetime) to seconds.

    Lichess may send the clock as an integer number of milliseconds. berserk
    sometimes converts time fields to datetimes; we guard against that.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value / 1000.0
    # datetime.datetime -> seconds since midnight is meaningless here, so just
    # fall back to "plenty of time" rather than guessing.
    try:
        return value.hour * 3600 + value.minute * 60 + value.second + \
            value.microsecond / 1_000_000
    except AttributeError:
        return None


def main() -> None:
    client = create_client()
    bot = LichessBot(client)
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\nShutting down. Bye!")


if __name__ == "__main__":
    main()
