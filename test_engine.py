"""
test_engine.py — Correctness tests for the chess engine.

Run with either:
    python test_engine.py
    python -m unittest

These tests guard the search against regressions: tactical correctness (mates,
winning material), time-budget compliance, terminal handling, and the optional
opening book's graceful fallback. They use short time limits so the whole
suite finishes in a few seconds.
"""

from __future__ import annotations

import time
import unittest

import chess

import engine


def best_move(fen: str, time_limit: float = 1.5) -> engine.SearchInfo:
    """Search a FEN with a fresh engine and return the SearchInfo."""
    return engine.Engine().search(chess.Board(fen), time_limit=time_limit)


class TacticsTests(unittest.TestCase):
    def test_mate_in_1(self):
        # Back-rank mate: Ra8#.
        info = best_move("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1", 1.0)
        self.assertEqual(info.move, chess.Move.from_uci("a1a8"))
        self.assertGreaterEqual(info.score, engine.MATE_BOUND)

    def test_mate_in_2(self):
        # A forced mate should be found and reported as a mate score.
        info = best_move("r5rk/5p1p/5R2/4B3/8/8/7P/7K w - - 0 1", 3.0)
        self.assertGreaterEqual(info.score, engine.MATE_BOUND)

    def test_wins_hanging_queen(self):
        # Black's queen on d5 is undefended; Nxd5 wins it.
        info = best_move(
            "rnb1kbnr/pppp1ppp/8/3qp3/8/2N5/PPPPPPPP/R1BQKBNR w KQkq - 0 1", 1.5
        )
        self.assertEqual(info.move, chess.Move.from_uci("c3d5"))
        self.assertGreater(info.score, 500)  # Up roughly a queen.

    def test_recaptures(self):
        # After ...dxe4, White should recapture the pawn rather than lose it.
        info = best_move(
            "rnbqkbnr/ppp2ppp/8/8/4p3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 0 1", 1.5
        )
        self.assertEqual(info.move, chess.Move.from_uci("c3e4"))


class SearchBehaviourTests(unittest.TestCase):
    def test_returns_legal_move(self):
        info = best_move(chess.STARTING_FEN, 1.0)
        self.assertIn(info.move, chess.Board().legal_moves)
        self.assertGreaterEqual(info.depth, 1)

    def test_time_budget_respected(self):
        start = time.monotonic()
        info = engine.Engine().search(chess.Board(), time_limit=1.0)
        elapsed = time.monotonic() - start
        # Allow a small margin for finishing the in-flight node/iteration.
        self.assertLess(elapsed, 1.6)
        self.assertIsNotNone(info.move)

    def test_deeper_with_more_time(self):
        shallow = engine.Engine().search(chess.Board(), time_limit=0.3)
        deep = engine.Engine().search(chess.Board(), time_limit=2.0)
        self.assertGreaterEqual(deep.depth, shallow.depth)

    def test_no_move_when_game_over(self):
        # Stalemate: side to move has no legal moves.
        info = best_move("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 0.5)
        self.assertIsNone(info.move)

    def test_zugzwang_endgame_legal(self):
        # Null-move pruning must not corrupt pawnless K+R vs K.
        info = best_move("8/8/8/4k3/8/4K3/8/7R w - - 0 1", 1.0)
        self.assertIn(info.move, chess.Board("8/8/8/4k3/8/4K3/8/7R w - - 0 1").legal_moves)

    def test_does_not_mutate_input_board(self):
        board = chess.Board()
        before = board.fen()
        engine.Engine().search(board, time_limit=0.3)
        self.assertEqual(board.fen(), before)


class EvaluationTests(unittest.TestCase):
    def test_startpos_is_balanced(self):
        # Symmetric position: only the tempo bonus should remain.
        self.assertEqual(engine.evaluate(chess.Board()), engine.TEMPO_BONUS)

    def test_extra_queen_is_better(self):
        # White up a queen should evaluate clearly in White's favour.
        board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        self.assertGreater(engine.evaluate(board), 500)


class BookTests(unittest.TestCase):
    def test_missing_book_falls_back_to_search(self):
        eng = engine.Engine(book_path="this_file_does_not_exist.bin")
        info = eng.search(chess.Board(), time_limit=0.5)
        self.assertIn(info.move, chess.Board().legal_moves)


class CompatibilityTests(unittest.TestCase):
    def test_fixed_depth_wrapper(self):
        move = engine.search(chess.Board(), depth=3)
        self.assertIn(move, chess.Board().legal_moves)


if __name__ == "__main__":
    unittest.main(verbosity=2)
