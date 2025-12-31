"""
Some example classes for people who want to create a homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""
import chess
import chess.polyglot
from chess.engine import PlayResult, Limit
import random
import math
import logging
from dataclasses import dataclass
from typing import Optional

from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import MOVE

logger = logging.getLogger(__name__)


@dataclass
class TTEntry:
    depth: int
    value: int
    flag: str  # "EXACT", "LOWER", "UPPER"
    best_move_uci: Optional[str]


class AlphaBetaEngine(MinimalEngine):
    """
    Exam-grade AI chess engine:
    - Negamax minimax with alpha-beta pruning
    - Time-based depth selection
    - Evaluation: material + piece-square tables + basic king safety (+ tiny mobility)
    - Transposition table (cache)
    """

    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0,
    }

    # Simple PSTs (values in centipawns). White perspective; Black uses mirrored squares.
    # These are intentionally small and simple (good for exam explanation).
    PST = {
        chess.PAWN: [
             0,  0,  0,  0,  0,  0,  0,  0,
             5, 10, 10,-20,-20, 10, 10,  5,
             5, -5,-10,  0,  0,-10, -5,  5,
             0,  0,  0, 20, 20,  0,  0,  0,
             5,  5, 10, 25, 25, 10,  5,  5,
            10, 10, 20, 30, 30, 20, 10, 10,
            50, 50, 50, 50, 50, 50, 50, 50,
             0,  0,  0,  0,  0,  0,  0,  0
        ],
        chess.KNIGHT: [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ],
        chess.BISHOP: [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ],
        chess.ROOK: [
             0,  0,  5, 10, 10,  5,  0,  0,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
             5, 10, 10, 10, 10, 10, 10,  5,
             0,  0,  0,  0,  0,  0,  0,  0
        ],
        chess.QUEEN: [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
             -5,  0,  5,  5,  5,  5,  0, -5,
              0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ],
        chess.KING: [
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
             20, 20,  0,  0,  0,  0, 20, 20,
             20, 30, 10,  0,  0, 10, 30, 20
        ],
    }

    def __init__(self, *args, max_depth: int = 5, min_depth: int = 2, tt_max: int = 200000, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.tt_max = tt_max
        self.tt: dict[int, TTEntry] = {}
        self._rng = random.Random(42)

    def search(self,
               board: chess.Board,
               time_limit: Limit,
               ponder: bool,  # noqa: ARG002
               draw_offered: bool,
               root_moves: MOVE) -> PlayResult:
        # Root move restriction (lichess-bot may pass this sometimes)
        legal_moves = list(root_moves) if isinstance(root_moves, list) else list(board.legal_moves)
        if not legal_moves:
            legal_moves = list(board.legal_moves)

        if len(legal_moves) == 1:
            return PlayResult(legal_moves[0], None, draw_offered=draw_offered)

        depth = self._choose_depth(board, time_limit)
        logger.debug(f"AlphaBetaEngine: chosen depth={depth}")

        best_move = None
        best_score = -10**9

        alpha, beta = -10**9, 10**9

        # Move ordering: TT best move first + captures/checks
        ordered = self._order_moves(board, legal_moves)

        for move in ordered:
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)

        if best_move is None:
            best_move = self._rng.choice(legal_moves)

        return PlayResult(best_move, None, draw_offered=draw_offered)

    # ---------- Time management ----------
    def _choose_depth(self, board: chess.Board, time_limit: Limit) -> int:
        my_time, my_inc = 0.0, 0.0

        # time_limit.time is used for fixed-time searches; otherwise use clocks
        if isinstance(getattr(time_limit, "time", None), (int, float)):
            my_time = float(time_limit.time)
        else:
            if board.turn == chess.WHITE:
                my_time = float(time_limit.white_clock or 0)
                my_inc = float(time_limit.white_inc or 0)
            else:
                my_time = float(time_limit.black_clock or 0)
                my_inc = float(time_limit.black_inc or 0)

        # “effective time” heuristic
        effective = my_time + 3.0 * my_inc

        # Bullet / blitz / rapid-ish buckets
        if effective < 15:
            return max(2, self.min_depth)
        if effective < 45:
            return min(3, self.max_depth)
        if effective < 120:
            return min(4, self.max_depth)
        return self.max_depth

    # ---------- Negamax + AlphaBeta + TT ----------
    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int) -> int:
        alpha_orig = alpha

        # Terminal / leaf
        if depth <= 0 or board.is_game_over():
            return self._evaluate_from_side_to_move(board)

        key = chess.polyglot.zobrist_hash(board)
        entry = self.tt.get(key)
        if entry and entry.depth >= depth:
            if entry.flag == "EXACT":
                return entry.value
            if entry.flag == "LOWER":
                alpha = max(alpha, entry.value)
            elif entry.flag == "UPPER":
                beta = min(beta, entry.value)
            if alpha >= beta:
                return entry.value

        best_value = -10**9
        best_move_uci = None

        moves = self._order_moves(board, list(board.legal_moves), tt_entry=entry)

        for move in moves:
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()

            if score > best_value:
                best_value = score
                best_move_uci = move.uci()

            alpha = max(alpha, score)
            if alpha >= beta:
                break  # cutoff

        # Store TT
        flag = "EXACT"
        if best_value <= alpha_orig:
            flag = "UPPER"
        elif best_value >= beta:
            flag = "LOWER"

        self._tt_store(key, TTEntry(depth=depth, value=best_value, flag=flag, best_move_uci=best_move_uci))

        return best_value

    def _tt_store(self, key: int, entry: TTEntry) -> None:
        if len(self.tt) > self.tt_max:
            # simple cleanup (fast + safe)
            self.tt.clear()
        self.tt[key] = entry

    # ---------- Evaluation ----------
    def _evaluate_from_side_to_move(self, board: chess.Board) -> int:
        if board.is_checkmate():
            return -1_000_000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        # White perspective score
        score = 0

        # Material + PST
        for piece_type, base in self.PIECE_VALUES.items():
            for sq in board.pieces(piece_type, chess.WHITE):
                score += base + self._pst(piece_type, sq, chess.WHITE)
            for sq in board.pieces(piece_type, chess.BLACK):
                score -= base + self._pst(piece_type, sq, chess.BLACK)

        # King safety (very basic)
        score += self._king_safety(board, chess.WHITE)
        score -= self._king_safety(board, chess.BLACK)

        # Tiny mobility bonus (cheap version)
        score += self._mobility(board)

        # Convert to side-to-move
        return score if board.turn == chess.WHITE else -score

    def _pst(self, piece_type: int, square: int, color: bool) -> int:
        table = self.PST.get(piece_type)
        if not table:
            return 0
        idx = square if color == chess.WHITE else chess.square_mirror(square)
        return int(table[idx])

    def _king_safety(self, board: chess.Board, color: bool) -> int:
        king_sq = next(iter(board.pieces(chess.KING, color)), None)
        if king_sq is None:
            return 0

        # Reward castled-ish positions; penalize staying in center when queens exist
        queens_exist = bool(board.pieces(chess.QUEEN, chess.WHITE) or board.pieces(chess.QUEEN, chess.BLACK))
        bonus = 0

        if color == chess.WHITE:
            if king_sq in (chess.G1, chess.C1):
                bonus += 30
            if queens_exist and king_sq in (chess.E1, chess.D1):
                bonus -= 25
            bonus += self._pawn_shield(board, king_sq, color)
        else:
            if king_sq in (chess.G8, chess.C8):
                bonus += 30
            if queens_exist and king_sq in (chess.E8, chess.D8):
                bonus -= 25
            bonus += self._pawn_shield(board, king_sq, color)

        return bonus

    def _pawn_shield(self, board: chess.Board, king_sq: int, color: bool) -> int:
        # Looks for pawns in front of king (simple)
        bonus = 0
        file = chess.square_file(king_sq)
        rank = chess.square_rank(king_sq)
        direction = 1 if color == chess.WHITE else -1
        shield_rank = rank + direction

        for df in (-1, 0, 1):
            f = file + df
            if 0 <= f <= 7 and 0 <= shield_rank <= 7:
                sq = chess.square(f, shield_rank)
                if sq in board.pieces(chess.PAWN, color):
                    bonus += 8
                else:
                    bonus -= 4
        return bonus

    def _mobility(self, board: chess.Board) -> int:
        # very small weight so it doesn't dominate
        my_moves = board.legal_moves.count()
        board.push(chess.Move.null())
        opp_moves = board.legal_moves.count()
        board.pop()
        return (my_moves - opp_moves) * 1

    # ---------- Move ordering ----------
    def _order_moves(self, board: chess.Board, moves: list[chess.Move], tt_entry: Optional[TTEntry] = None) -> list[chess.Move]:
        tt_best = None
        if tt_entry and tt_entry.best_move_uci:
            try:
                tt_best = chess.Move.from_uci(tt_entry.best_move_uci)
            except Exception:
                tt_best = None

        def score(m: chess.Move) -> int:
            s = 0
            if tt_best and m == tt_best:
                s += 10_000
            if board.is_capture(m):
                s += 2000
            if board.gives_check(m):
                s += 200
            if m.promotion:
                s += 800
            return s

        return sorted(moves, key=score, reverse=True)


# Bot names and ideas from tom7's excellent eloWorld video

class RandomMove(MinimalEngine):
    """Get a random move."""

     # noqa: ARG002
    def search(self, board: chess.Board, *args) -> PlayResult:

        """Choose a random move."""
        return PlayResult(random.choice(list(board.legal_moves)), None)


class Alphabetical(MinimalEngine):
    """Get the first move when sorted by san representation."""

    def search(self, board: chess.Board, *args) -> PlayResult:  # noqa: ARG002
        """Choose the first move alphabetically."""
        moves = list(board.legal_moves)
        moves.sort(key=board.san)
        return PlayResult(moves[0], None)


class FirstMove(MinimalEngine):
    """Get the first move when sorted by uci representation."""

    def search(self, board: chess.Board, *args) -> PlayResult:  # noqa: ARG002
        """Choose the first move alphabetically in uci representation."""
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return PlayResult(moves[0], None)


class ComboEngine(MinimalEngine):
    """
    Get a move using multiple different methods.

    This engine demonstrates how one can use `time_limit`, `draw_offered`, and `root_moves`.
    """

    def search(self,
               board: chess.Board,
               time_limit: Limit,
               ponder: bool,  # noqa: ARG002
               draw_offered: bool,
               root_moves: MOVE) -> PlayResult:
        """
        Choose a move using multiple different methods.

        :param board: The current position.
        :param time_limit: Conditions for how long the engine can search (e.g. we have 10 seconds and search up to depth 10).
        :param ponder: Whether the engine can ponder after playing a move.
        :param draw_offered: Whether the bot was offered a draw.
        :param root_moves: If it is a list, the engine should only play a move that is in `root_moves`.
        :return: The move to play.
        """
        if isinstance(time_limit.time, int):
            my_time = time_limit.time
            my_inc = 0
        elif board.turn == chess.WHITE:
            my_time = time_limit.white_clock if isinstance(time_limit.white_clock, int) else 0
            my_inc = time_limit.white_inc if isinstance(time_limit.white_inc, int) else 0
        else:
            my_time = time_limit.black_clock if isinstance(time_limit.black_clock, int) else 0
            my_inc = time_limit.black_inc if isinstance(time_limit.black_inc, int) else 0

        possible_moves = root_moves if isinstance(root_moves, list) else list(board.legal_moves)

        if my_time / 60 + my_inc > 10:
            # Choose a random move.
            move = random.choice(possible_moves)
        else:
            # Choose the first move alphabetically in uci representation.
            possible_moves.sort(key=str)
            move = possible_moves[0]
        return PlayResult(move, None, draw_offered=draw_offered)

# --- compatibility for lichess-bot test harness ---
# test_bot/homemade.py imports ExampleEngine, so keep this alias.
ExampleEngine = AlphaBetaEngine

# --- compatibility for lichess-bot test harness ---
class ExampleEngine(AlphaBetaEngine):
    pass
