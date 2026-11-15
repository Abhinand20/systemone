"""Deterministic 3x3 Tic-Tac-Toe rules, D4 symmetries, and step transitions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Mark = Literal["X", "O"]
CellValue = Literal["X", "O", "."]
ActionMode = Literal["all_9", "legal_only"]
TerminationReason = Literal["win", "draw", "illegal_forfeit"]

CELL_KEYS: tuple[str, ...] = (
    "R1C1",
    "R1C2",
    "R1C3",
    "R2C1",
    "R2C2",
    "R2C3",
    "R3C1",
    "R3C2",
    "R3C3",
)

KEY_TO_INDEX: dict[str, int] = {key: idx for idx, key in enumerate(CELL_KEYS)}
INDEX_TO_KEY: dict[int, str] = {idx: key for idx, key in enumerate(CELL_KEYS)}

ACTION_CRITERIA: dict[str, str] = {
    "R1C1": "Top-left cell (row 1, column 1)",
    "R1C2": "Top-center cell (row 1, column 2)",
    "R1C3": "Top-right cell (row 1, column 3)",
    "R2C1": "Middle-left cell (row 2, column 1)",
    "R2C2": "Center cell (row 2, column 2)",
    "R2C3": "Middle-right cell (row 2, column 3)",
    "R3C1": "Bottom-left cell (row 3, column 1)",
    "R3C2": "Bottom-center cell (row 3, column 2)",
    "R3C3": "Bottom-right cell (row 3, column 3)",
}

WIN_LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)

D4_NAMES: tuple[str, ...] = (
    "identity",
    "rot90_cw",
    "rot180",
    "rot270_cw",
    "flip_horizontal",
    "flip_vertical",
    "transpose_main",
    "transpose_anti",
)


def _build_d4_forward_maps() -> tuple[tuple[int, ...], ...]:
    """Build forward maps old_idx -> new_idx for all 8 D4 dihedral symmetries."""
    transforms = (
        lambda r, c: (r, c),          # 0: identity
        lambda r, c: (c, 2 - r),      # 1: 90 deg CW
        lambda r, c: (2 - r, 2 - c),  # 2: 180 deg
        lambda r, c: (2 - c, r),      # 3: 270 deg CW
        lambda r, c: (r, 2 - c),      # 4: horizontal flip (left <-> right)
        lambda r, c: (2 - r, c),      # 5: vertical flip (top <-> bottom)
        lambda r, c: (c, r),          # 6: main-diagonal transpose
        lambda r, c: (2 - c, 2 - r),  # 7: anti-diagonal transpose
    )
    maps: list[tuple[int, ...]] = []
    for fn in transforms:
        mapping = [0] * 9
        for r in range(3):
            for c in range(3):
                nr, nc = fn(r, c)
                mapping[3 * r + c] = 3 * nr + nc
        maps.append(tuple(mapping))
    return tuple(maps)


D4_FORWARD_MAPS: tuple[tuple[int, ...], ...] = _build_d4_forward_maps()


def other_mark(mark: Mark) -> Mark:
    """Return the opposing player mark."""
    if mark == "X":
        return "O"
    if mark == "O":
        return "X"
    raise ValueError(f"Invalid mark: {mark!r}")


def _validate_board(board: Sequence[str]) -> tuple[str, ...]:
    if len(board) != 9:
        raise ValueError(f"Board must have length 9, got {len(board)}")
    out = tuple(board)
    for cell in out:
        if cell not in ("X", "O", "."):
            raise ValueError(f"Invalid board cell value: {cell!r}")
    return out


@dataclass(frozen=True, slots=True)
class TTTState:
    """Immutable Tic-Tac-Toe game state."""

    board: tuple[str, ...]
    player_to_move: Mark
    move_history: tuple[int, ...] = ()
    starting_player: Mark = "X"

    def __post_init__(self) -> None:
        object.__setattr__(self, "board", _validate_board(self.board))
        if self.player_to_move not in ("X", "O"):
            raise ValueError(f"Invalid player_to_move: {self.player_to_move!r}")
        if self.starting_player not in ("X", "O"):
            raise ValueError(f"Invalid starting_player: {self.starting_player!r}")
        history = tuple(self.move_history)
        for idx in history:
            if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0 or idx > 8:
                raise ValueError(f"Invalid move_history index: {idx!r}")
        object.__setattr__(self, "move_history", history)


@dataclass(frozen=True, slots=True)
class StepOutcome:
    """Result of applying an action in a Tic-Tac-Toe state."""

    next_state: TTTState
    reward: float
    done: bool
    action_is_legal: bool
    winner: Mark | None
    termination_reason: TerminationReason | None


def new_ttt_state(starting_player: Mark = "X") -> TTTState:
    """Create a fresh empty 3x3 Tic-Tac-Toe state."""
    return TTTState(
        board=(".",) * 9,
        player_to_move=starting_player,
        move_history=(),
        starting_player=starting_player,
    )


def check_winner(board: Sequence[str]) -> Mark | None:
    """Return 'X' or 'O' if a 3-in-a-row winner exists, otherwise None."""
    for a, b, c in WIN_LINES:
        cell = board[a]
        if cell != "." and cell == board[b] == board[c]:
            return cell  # type: ignore[return-value]
    return None


def is_board_full(board: Sequence[str]) -> bool:
    """Return True if no empty cells remain on the board."""
    return "." not in board


def is_terminal(board: Sequence[str]) -> bool:
    """Return True if the board has a winner or is full."""
    return check_winner(board) is not None or is_board_full(board)


def legal_indices(board: Sequence[str]) -> tuple[int, ...]:
    """Return empty board indices in row-major order."""
    return tuple(idx for idx, cell in enumerate(board) if cell == ".")


def legal_action_keys(board: Sequence[str]) -> tuple[str, ...]:
    """Return legal cell keys in R1C1..R3C3 order."""
    return tuple(INDEX_TO_KEY[idx] for idx in legal_indices(board))


def action_criteria_for_state(
    board: Sequence[str],
    action_mode: ActionMode = "all_9",
) -> dict[str, str]:
    """Return ordered R1C1..R3C3 action criteria for all_9 or legal_only mode."""
    if action_mode == "all_9":
        return dict(ACTION_CRITERIA)
    if action_mode == "legal_only":
        allowed = set(legal_action_keys(board))
        return {k: v for k, v in ACTION_CRITERIA.items() if k in allowed}
    raise ValueError(f"Unsupported action_mode: {action_mode!r}")


def apply_d4_board(board: Sequence[str], sym_id: int) -> tuple[str, ...]:
    """Apply D4 forward symmetry sym_id (0..7) to a 9-cell board."""
    if not 0 <= sym_id < 8:
        raise ValueError(f"sym_id must be in 0..7, got {sym_id!r}")
    validated = _validate_board(board)
    fwd = D4_FORWARD_MAPS[sym_id]
    transformed = ["."] * 9
    for old_idx, new_idx in enumerate(fwd):
        transformed[new_idx] = validated[old_idx]
    return tuple(transformed)


def apply_d4_history(move_history: Sequence[int], sym_id: int) -> tuple[int, ...]:
    """Apply D4 forward symmetry sym_id (0..7) to a sequence of board indices."""
    if not 0 <= sym_id < 8:
        raise ValueError(f"sym_id must be in 0..7, got {sym_id!r}")
    fwd = D4_FORWARD_MAPS[sym_id]
    return tuple(fwd[idx] for idx in move_history)


def apply_d4_state(state: TTTState, sym_id: int) -> TTTState:
    """Apply D4 symmetry sym_id identically to both board and move_history."""
    return TTTState(
        board=apply_d4_board(state.board, sym_id),
        player_to_move=state.player_to_move,
        move_history=apply_d4_history(state.move_history, sym_id),
        starting_player=state.starting_player,
    )


def swap_marks_board(board: Sequence[str]) -> tuple[str, ...]:
    """Swap 'X' and 'O' across the board while preserving '.'."""
    validated = _validate_board(board)
    return tuple("O" if c == "X" else ("X" if c == "O" else ".") for c in validated)


def swap_marks_state(state: TTTState) -> TTTState:
    """Swap 'X' and 'O' across board, player_to_move, and starting_player."""
    return TTTState(
        board=swap_marks_board(state.board),
        player_to_move=other_mark(state.player_to_move),
        move_history=state.move_history,
        starting_player=other_mark(state.starting_player),
    )


def canonical_d4_key(board: Sequence[str], player_to_move: Mark) -> str:
    """Return canonical key invariant under all 8 D4 symmetries and X<->O swap."""
    if player_to_move not in ("X", "O"):
        raise ValueError(f"Invalid player_to_move: {player_to_move!r}")
    # Normalize so the current player to move is always 'X'.
    norm_board = swap_marks_board(board) if player_to_move == "O" else _validate_board(board)
    return min("".join(apply_d4_board(norm_board, sym_id)) for sym_id in range(8))


def step_ttt(state: TTTState, action_key: str | None) -> StepOutcome:
    """Advance TTTState by one action; illegal or None actions trigger immediate forfeit."""
    if action_key is None or action_key not in KEY_TO_INDEX:
        return StepOutcome(
            next_state=state,
            reward=-1.0,
            done=True,
            action_is_legal=False,
            winner=other_mark(state.player_to_move),
            termination_reason="illegal_forfeit",
        )

    idx = KEY_TO_INDEX[action_key]
    if is_terminal(state.board) or state.board[idx] != ".":
        return StepOutcome(
            next_state=state,
            reward=-1.0,
            done=True,
            action_is_legal=False,
            winner=other_mark(state.player_to_move),
            termination_reason="illegal_forfeit",
        )

    next_board_list = list(state.board)
    next_board_list[idx] = state.player_to_move
    next_board = tuple(next_board_list)
    next_history = state.move_history + (idx,)
    next_state = TTTState(
        board=next_board,
        player_to_move=other_mark(state.player_to_move),
        move_history=next_history,
        starting_player=state.starting_player,
    )

    winner = check_winner(next_board)
    if winner is not None:
        return StepOutcome(
            next_state=next_state,
            reward=1.0,
            done=True,
            action_is_legal=True,
            winner=winner,
            termination_reason="win",
        )

    if is_board_full(next_board):
        return StepOutcome(
            next_state=next_state,
            reward=0.0,
            done=True,
            action_is_legal=True,
            winner=None,
            termination_reason="draw",
        )

    return StepOutcome(
        next_state=next_state,
        reward=0.0,
        done=False,
        action_is_legal=True,
        winner=None,
        termination_reason=None,
    )
