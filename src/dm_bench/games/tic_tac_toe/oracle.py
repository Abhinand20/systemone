"""Exact memoized minimax oracle, regret evaluation, and tactical classifier."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
import random
from typing import NamedTuple

from dm_bench.games.tic_tac_toe.engine import (
    INDEX_TO_KEY,
    Mark,
    check_winner,
    is_board_full,
    legal_indices,
    other_mark,
)

TACTICAL_CATEGORIES: tuple[str, ...] = (
    "immediate_win",
    "losing_position",
    "forced_block",
    "fork_create",
    "fork_defend",
    "forced_draw",
)


@lru_cache(maxsize=None)
def _minimax_cached(board: tuple[str, ...], player_to_move: Mark) -> int:
    winner = check_winner(board)
    if winner is not None:
        return 1 if winner == player_to_move else -1
    if is_board_full(board):
        return 0

    opp = other_mark(player_to_move)
    best = -2
    for idx, cell in enumerate(board):
        if cell == ".":
            next_board = board[:idx] + (player_to_move,) + board[idx + 1 :]
            val = -_minimax_cached(next_board, opp)
            if val > best:
                best = val
                if best == 1:
                    break
    return best


def minimax_value(board: Sequence[str], player_to_move: Mark) -> int:
    """Return exact minimax value (+1 win, 0 draw, -1 loss) for player_to_move."""
    if len(board) != 9:
        raise ValueError(f"Board must have length 9, got {len(board)}")
    if player_to_move not in ("X", "O"):
        raise ValueError(f"Invalid player_to_move: {player_to_move!r}")
    return _minimax_cached(tuple(board), player_to_move)


@dataclass(frozen=True, slots=True)
class PositionAnalysis:
    """Complete exact oracle analysis for a non-terminal Tic-Tac-Toe position."""

    state_value: int
    action_values: dict[str, int]
    optimal_actions: tuple[str, ...]
    tactical_category: str


class MoveEvaluation(NamedTuple):
    """Evaluation of a candidate move against the exact minimax oracle."""

    is_legal: bool
    is_optimal: bool
    legal_regret: int | None
    effective_regret: int


def analyze_position(board: Sequence[str], player_to_move: Mark) -> PositionAnalysis:
    """Compute minimax action values, optimal moves, and mutually exclusive tactical category."""
    board_tuple = tuple(board)
    if len(board_tuple) != 9:
        raise ValueError(f"Board must have length 9, got {len(board_tuple)}")
    if check_winner(board_tuple) is not None or is_board_full(board_tuple):
        raise ValueError("Cannot analyze a terminal Tic-Tac-Toe position")

    opp = other_mark(player_to_move)
    legal = legal_indices(board_tuple)
    action_values: dict[str, int] = {}
    has_immediate_win = False
    opponent_has_immediate_win = False

    for idx in legal:
        key = INDEX_TO_KEY[idx]
        next_board = board_tuple[:idx] + (player_to_move,) + board_tuple[idx + 1 :]
        if check_winner(next_board) == player_to_move:
            has_immediate_win = True
            action_values[key] = 1
        else:
            action_values[key] = -minimax_value(next_board, opp)

        opp_board = board_tuple[:idx] + (opp,) + board_tuple[idx + 1 :]
        if check_winner(opp_board) == opp:
            opponent_has_immediate_win = True

    state_value = max(action_values.values())
    optimal_actions = tuple(k for k, v in action_values.items() if v == state_value)

    if has_immediate_win:
        tactical_category = "immediate_win"
    elif state_value == -1:
        tactical_category = "losing_position"
    elif opponent_has_immediate_win:
        tactical_category = "forced_block"
    elif state_value == 1:
        tactical_category = "fork_create"
    elif min(action_values.values()) == -1:
        tactical_category = "fork_defend"
    else:
        tactical_category = "forced_draw"

    return PositionAnalysis(
        state_value=state_value,
        action_values=action_values,
        optimal_actions=optimal_actions,
        tactical_category=tactical_category,
    )


def evaluate_move(
    board: Sequence[str],
    player_to_move: Mark,
    action_key: str | None,
) -> MoveEvaluation:
    """Return (is_legal, is_optimal, legal_regret, effective_regret) for action_key."""
    analysis = analyze_position(board, player_to_move)
    if action_key is not None and action_key in analysis.action_values:
        chosen_value = analysis.action_values[action_key]
        regret = analysis.state_value - chosen_value
        return MoveEvaluation(
            is_legal=True,
            is_optimal=action_key in analysis.optimal_actions,
            legal_regret=regret,
            effective_regret=regret,
        )

    return MoveEvaluation(
        is_legal=False,
        is_optimal=False,
        legal_regret=None,
        effective_regret=analysis.state_value - (-1),
    )


def choose_minimax_action(
    board: Sequence[str],
    player_to_move: Mark,
    rng: random.Random | None = None,
) -> str:
    """Choose a minimax-optimal action key (uniformly via rng if provided, else deterministic first)."""
    analysis = analyze_position(board, player_to_move)
    if rng is not None:
        return rng.choice(analysis.optimal_actions)
    return analysis.optimal_actions[0]
