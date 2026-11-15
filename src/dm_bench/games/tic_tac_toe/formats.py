"""Text state serializers for Tic-Tac-Toe (T5 prompt representation variants)."""

from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Literal

from dm_bench.games.tic_tac_toe.engine import (
    CELL_KEYS,
    INDEX_TO_KEY,
    KEY_TO_INDEX,
    Mark,
    TTTState,
)

TEXT_FORMATS: tuple[str, ...] = (
    "json_canonical",
    "ascii_grid",
    "coord_list",
    "json_with_history",
)


def _validate_inputs(board: Sequence[str], player_to_move: Mark) -> tuple[str, ...]:
    if len(board) != 9:
        raise ValueError(f"Board must have length 9, got {len(board)}")
    out = tuple(board)
    for cell in out:
        if cell not in ("X", "O", "."):
            raise ValueError(f"Invalid board cell: {cell!r}")
    if player_to_move not in ("X", "O"):
        raise ValueError(f"Invalid player_to_move: {player_to_move!r}")
    return out


def _normalize_history_keys(move_history: Sequence[int | str] | None) -> list[str]:
    if not move_history:
        return []
    keys: list[str] = []
    for item in move_history:
        if isinstance(item, int) and not isinstance(item, bool):
            if item not in INDEX_TO_KEY:
                raise ValueError(f"Invalid move_history index: {item!r}")
            keys.append(INDEX_TO_KEY[item])
        elif isinstance(item, str):
            if item not in KEY_TO_INDEX:
                raise ValueError(f"Invalid move_history key: {item!r}")
            keys.append(item)
        else:
            raise ValueError(f"Unsupported move_history entry: {item!r}")
    return keys


def _cell_label(cell: str) -> str:
    return "empty" if cell == "." else cell


def _build_structured_board(validated_board: tuple[str, ...]) -> dict[str, dict[str, str]]:
    return {
        "R1": {
            "R1C1": _cell_label(validated_board[0]),
            "R1C2": _cell_label(validated_board[1]),
            "R1C3": _cell_label(validated_board[2]),
        },
        "R2": {
            "R2C1": _cell_label(validated_board[3]),
            "R2C2": _cell_label(validated_board[4]),
            "R2C3": _cell_label(validated_board[5]),
        },
        "R3": {
            "R3C1": _cell_label(validated_board[6]),
            "R3C2": _cell_label(validated_board[7]),
            "R3C3": _cell_label(validated_board[8]),
        },
    }


def format_ttt_state(
    board: Sequence[str],
    player_to_move: Mark,
    text_format: str = "json_canonical",
    move_history: Sequence[int | str] | None = None,
) -> str:
    """Serialize a Tic-Tac-Toe position into one of the four T5 text formats."""
    validated_board = _validate_inputs(board, player_to_move)

    if text_format == "json_canonical":
        payload = {
            "game": "tic_tac_toe",
            "player_to_move": player_to_move,
            "board": _build_structured_board(validated_board),
        }
        return json.dumps(payload)

    if text_format == "ascii_grid":
        lines = [
            "Game: tic_tac_toe",
            f"Player to move: {player_to_move}",
            "     C1  C2  C3",
            f"R1 | {validated_board[0]} | {validated_board[1]} | {validated_board[2]} |",
            f"R2 | {validated_board[3]} | {validated_board[4]} | {validated_board[5]} |",
            f"R3 | {validated_board[6]} | {validated_board[7]} | {validated_board[8]} |",
        ]
        return "\n".join(lines)

    if text_format == "coord_list":
        lines = [
            "Game: tic_tac_toe",
            f"Player to move: {player_to_move}",
            "Board cells:",
        ]
        for key, cell in zip(CELL_KEYS, validated_board):
            label = _cell_label(cell)
            lines.append(f"- {key}: {label}")
        return "\n".join(lines)

    if text_format == "json_with_history":
        payload = {
            "game": "tic_tac_toe",
            "player_to_move": player_to_move,
            "board": _build_structured_board(validated_board),
            "move_history": _normalize_history_keys(move_history),
        }
        return json.dumps(payload)

    raise ValueError(f"Unsupported text_format: {text_format!r}. Expected one of {TEXT_FORMATS}")


def format_state(state: TTTState, text_format: str = "json_canonical") -> str:
    """Convenience wrapper around format_ttt_state for a TTTState instance."""
    return format_ttt_state(
        board=state.board,
        player_to_move=state.player_to_move,
        text_format=text_format,
        move_history=state.move_history,
    )
