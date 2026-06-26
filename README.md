# Lichess Chess Bot

A simple but complete chess bot for [Lichess](https://lichess.org) written in
Python. It connects to the Lichess Bot API over Server-Sent Events (SSE),
automatically accepts challenges, and plays using a **minimax engine with
alpha-beta pruning and quiescence search**.

## Features

- **Lichess Bot API** integration via the official
  [`berserk`](https://github.com/lichess-org/berserk) client (SSE event
  streams).
- **Automatic challenge acceptance** (standard chess; variants are declined).
- **Multi-game support** — each game runs in its own thread.
- **Minimax search** with:
  - Alpha-beta pruning
  - Move ordering (captures first, via MVV-LVA, plus promotions/checks)
  - Quiescence search to avoid the horizon effect
  - Material + piece-square-table evaluation
  - Simple time management (depth chosen from the remaining clock)

## Project layout

```
chess-bot/
  main.py          # entry point — connects to Lichess, runs the event loop
  engine.py        # minimax + alpha-beta + evaluation + quiescence
  bot.py           # per-game handler — board state -> best move
  .env.example     # template: LICHESS_TOKEN=your_token_here
  requirements.txt # python-chess, berserk, python-dotenv
  README.md
```

## Setup

### 1. Create a Lichess BOT account

> **Important:** A Lichess account can only be upgraded to a BOT account if it
> has **never played a game**. Create a fresh account for the bot.

1. Create a new Lichess account for your bot.
2. Generate an API token at
   <https://lichess.org/account/oauth/token> with the
   **"Play games with the bot API"** (`bot:play`) scope.
3. Upgrade the account to a BOT account (one-time, irreversible):

   ```bash
   curl -d '' https://lichess.org/api/bot/account/upgrade \
     -H "Authorization: Bearer YOUR_TOKEN"
   ```

### 2. Install dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

(Drop `--break-system-packages` if you're using a virtual environment, which
is recommended:)

```bash
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure your token

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Then edit `.env` and set your token:

```
LICHESS_TOKEN=lip_xxxxxxxxxxxxxxxx
```

### 4. Run the bot

```bash
python main.py
```

You should see `Logged in as: <your-bot-name>` followed by
`Listening for challenges and games...`.

## Playing against the bot

1. Make sure the bot is running.
2. From another Lichess account, go to the bot's profile and send it a
   **challenge** (use a "Casual" or "Rated" real-time/correspondence game —
   standard variant).
3. The bot accepts automatically and starts playing.

You can also challenge it directly via URL:
`https://lichess.org/?user=YOUR_BOT_NAME#friend`

## Tuning

- **Strength vs. speed:** edit `choose_depth()` in [`bot.py`](bot.py) to raise
  or lower the search depth for each time bucket. Higher depth = stronger but
  slower.
- **Evaluation:** adjust `PIECE_VALUES` or the piece-square tables in
  [`engine.py`](engine.py) to change playing style.

## Notes & limitations

- Only **standard chess** is supported; variant challenges are declined.
- The engine has no opening book or endgame tablebases, so it plays purely on
  search + evaluation.
- Time management is deliberately conservative to avoid flagging; tune it for
  your hardware.

## License

Provided as-is for educational use.
