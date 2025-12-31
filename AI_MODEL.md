
# AI Model Explanation: AlphaBetaEngine

## 1) Overview
The bot plays chess using a search-based AI algorithm:
negamax (a minimax variant) with alpha-beta pruning.
It evaluates positions using a handcrafted evaluation function.

## 2) Minimax / Negamax
Chess is modeled as a turn-based zero-sum game.
Minimax searches a game tree to choose the move that maximizes my outcome
assuming the opponent plays optimally.

Negamax is a simplified form of minimax using the identity:
score(position) for current player = -score(position) for opponent after a move.
This allows one recursive function.

## 3) Alpha–Beta pruning
Alpha–beta pruning speeds up minimax by cutting branches that cannot affect
the final decision:
- alpha = best score found so far for the maximizing player
- beta  = best score found so far for the minimizing player
If alpha >= beta, further exploration of that branch is useless and is skipped.

Effect: with good move ordering, alpha–beta can search much deeper in the same time.

## 4) Evaluation function
Leaf nodes (or depth limit) are scored using:
- Material score (piece values)
- Piece-square tables (positional bonuses: central knights, advanced pawns, etc.)
- Basic king safety (bonus for castling, penalty for central king when queens exist)
- Small mobility bonus

Limitations:
- Heuristic evaluation can miss tactics (sacrifices, forced mates) if depth is too low
- Simple king safety is not fully accurate
- No quiescence search, so it can suffer from the “horizon effect”

## 5) Time management
The engine chooses depth based on remaining clock time and increment:
- very low time -> lower depth
- more time -> higher depth (up to a maximum)

This trades strength vs. speed safely across bullet/blitz/rapid.

## 6) Transposition table
Many different move orders reach the same position.
A transposition table caches the evaluation of positions using a Zobrist hash key.
This reduces repeated work, speeds up search, and effectively increases strength.

Limitations:
- Table is bounded and may be cleared when it grows too large
- Stored bounds (EXACT/LOWER/UPPER) can be imperfect if depth differs

## 7) Possible future improvements
- Iterative deepening + real time budget per move
- Quiescence search to reduce horizon effect
- Better move ordering (history heuristic, killer moves)
- Endgame tablebases (Syzygy) integration
- Opening book
