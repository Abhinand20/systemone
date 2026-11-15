"""Unit tests for Tic-Tac-Toe engine, D4 symmetries, minimax oracle, and tactical classifier."""

from __future__ import annotations

import random

import pytest

from dm_bench.games.tic_tac_toe.engine import (
    ACTION_CRITERIA,
    CELL_KEYS,
    D4_FORWARD_MAPS,
    INDEX_TO_KEY,
    KEY_TO_INDEX,
    TTTState,
    action_criteria_for_state,
    apply_d4_board,
    apply_d4_state,
    canonical_d4_key,
    new_ttt_state,
    other_mark,
    step_ttt,
    swap_marks_state,
)
from dm_bench.games.tic_tac_toe.oracle import (
    TACTICAL_CATEGORIES,
    analyze_position,
    choose_minimax_action,
    evaluate_move,
    minimax_value,
)


def test_cell_keys_and_action_criteria() -> None:
    assert len(CELL_KEYS) == 9
    assert CELL_KEYS[0] == "R1C1"
    assert CELL_KEYS[4] == "R2C2"
    assert CELL_KEYS[8] == "R3C3"
    assert len(KEY_TO_INDEX) == 9
    assert len(INDEX_TO_KEY) == 9
    for idx, key in enumerate(CELL_KEYS):
        assert KEY_TO_INDEX[key] == idx
        assert INDEX_TO_KEY[idx] == key
    assert tuple(ACTION_CRITERIA.keys()) == CELL_KEYS

    board = ("X", ".", "O", ".", "X", ".", ".", ".", "O")
    assert tuple(action_criteria_for_state(board, "all_9").keys()) == CELL_KEYS
    legal_criteria = action_criteria_for_state(board, "legal_only")
    assert tuple(legal_criteria.keys()) == ("R1C2", "R2C1", "R2C3", "R3C1", "R3C2")


def test_state_validation() -> None:
    with pytest.raises(ValueError):
        TTTState(board=(".",) * 8, player_to_move="X")
    with pytest.raises(ValueError):
        TTTState(board=(".",) * 8 + ("?",), player_to_move="X")
    with pytest.raises(ValueError):
        TTTState(board=(".",) * 9, player_to_move="Z")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TTTState(board=(".",) * 9, player_to_move="X", move_history=(9,))


def test_d4_symmetries_group_and_replay_equivariance() -> None:
    assert len(D4_FORWARD_MAPS) == 8
    assert len(set(D4_FORWARD_MAPS)) == 8
    for fwd in D4_FORWARD_MAPS:
        assert sorted(fwd) == list(range(9))

    # Replay equivariance: step_ttt(apply_d4(s), mapped_key) == apply_d4(step_ttt(s, key))
    state = new_ttt_state("X")
    state = step_ttt(state, "R1C1").next_state
    state = step_ttt(state, "R2C2").next_state
    state = step_ttt(state, "R1C2").next_state

    for sym_id in range(8):
        transformed = apply_d4_state(state, sym_id)
        mapped_idx = D4_FORWARD_MAPS[sym_id][KEY_TO_INDEX["R3C1"]]
        mapped_key = INDEX_TO_KEY[mapped_idx]

        out_orig = step_ttt(state, "R3C1")
        out_trans = step_ttt(transformed, mapped_key)
        assert out_trans.next_state == apply_d4_state(out_orig.next_state, sym_id)


def test_canonical_d4_key_invariance() -> None:
    state = TTTState(
        board=("X", "O", ".", ".", "X", ".", ".", ".", "O"),
        player_to_move="X",
        move_history=(0, 1, 4, 8),
        starting_player="X",
    )
    base_key = canonical_d4_key(state.board, state.player_to_move)
    swapped = swap_marks_state(state)
    assert canonical_d4_key(swapped.board, swapped.player_to_move) == base_key

    for sym_id in range(8):
        s_d4 = apply_d4_state(state, sym_id)
        assert canonical_d4_key(s_d4.board, s_d4.player_to_move) == base_key
        s_d4_swapped = swap_marks_state(s_d4)
        assert canonical_d4_key(s_d4_swapped.board, s_d4_swapped.player_to_move) == base_key


def test_step_ttt_legal_win_draw_and_illegal_forfeit() -> None:
    # Immediate win
    win_state = TTTState(
        board=("X", "X", ".", "O", "O", ".", ".", ".", "."),
        player_to_move="X",
        move_history=(0, 3, 1, 4),
        starting_player="X",
    )
    win_out = step_ttt(win_state, "R1C3")
    assert win_out.done is True
    assert win_out.action_is_legal is True
    assert win_out.reward == 1.0
    assert win_out.winner == "X"
    assert win_out.termination_reason == "win"
    assert win_out.next_state.board[2] == "X"
    assert win_out.next_state.move_history == (0, 3, 1, 4, 2)

    # Illegal move on occupied cell -> forfeit
    forfeit_occupied = step_ttt(win_state, "R1C1")
    assert forfeit_occupied.done is True
    assert forfeit_occupied.action_is_legal is False
    assert forfeit_occupied.reward == -1.0
    assert forfeit_occupied.winner == "O"
    assert forfeit_occupied.termination_reason == "illegal_forfeit"
    assert forfeit_occupied.next_state == win_state

    # None or invalid key -> forfeit
    for bad_action in (None, "INVALID", "R4C1"):
        forfeit_bad = step_ttt(win_state, bad_action)
        assert forfeit_bad.done is True
        assert forfeit_bad.action_is_legal is False
        assert forfeit_bad.reward == -1.0
        assert forfeit_bad.winner == other_mark(win_state.player_to_move)
        assert forfeit_bad.termination_reason == "illegal_forfeit"

    # Draw
    draw_state = TTTState(
        board=("X", "O", "X", "X", "O", "O", "O", "X", "."),
        player_to_move="X",
        move_history=(0, 1, 2, 4, 3, 5, 7, 6),
        starting_player="X",
    )
    draw_out = step_ttt(draw_state, "R3C3")
    assert draw_out.done is True
    assert draw_out.action_is_legal is True
    assert draw_out.reward == 0.0
    assert draw_out.winner is None
    assert draw_out.termination_reason == "draw"


def test_minimax_empty_board_and_symmetry_invariance() -> None:
    empty = (".",) * 9
    assert minimax_value(empty, "X") == 0
    assert minimax_value(empty, "O") == 0

    analysis = analyze_position(empty, "X")
    assert analysis.state_value == 0
    assert analysis.optimal_actions == CELL_KEYS
    assert all(val == 0 for val in analysis.action_values.values())
    assert analysis.tactical_category == "forced_draw"

    # Check D4 & mark-swap invariance across representative tactical boards
    sample_boards = [
        (("X", "X", ".", "O", "O", ".", ".", ".", "."), "X"),  # immediate_win
        (("X", ".", "O", ".", "O", ".", "X", ".", "X"), "O"),  # losing_position (X threatens R2C1 & R3C2)
        (("X", "X", ".", "O", ".", ".", ".", ".", "."), "O"),  # forced_block
        (("X", ".", ".", ".", "O", ".", ".", ".", "X"), "X"),  # fork_create
        (("X", ".", ".", ".", ".", ".", ".", ".", "."), "O"),  # fork_defend
        (("X", "O", "X", "O", "X", "X", "O", "X", "O")[:-1] + (".",), "O"),  # forced_draw
    ]
    for board, player in sample_boards:
        base_analysis = analyze_position(board, player)
        assert base_analysis.tactical_category in TACTICAL_CATEGORIES
        swapped = swap_marks_state(TTTState(board=board, player_to_move=player))
        swapped_analysis = analyze_position(swapped.board, swapped.player_to_move)
        assert swapped_analysis.state_value == base_analysis.state_value
        assert swapped_analysis.tactical_category == base_analysis.tactical_category

        for sym_id in range(8):
            sym_board = apply_d4_board(board, sym_id)
            sym_analysis = analyze_position(sym_board, player)
            assert sym_analysis.state_value == base_analysis.state_value
            assert sym_analysis.tactical_category == base_analysis.tactical_category
            expected_opt = {
                INDEX_TO_KEY[D4_FORWARD_MAPS[sym_id][KEY_TO_INDEX[k]]]
                for k in base_analysis.optimal_actions
            }
            assert set(sym_analysis.optimal_actions) == expected_opt


def test_tactical_categories_examples() -> None:
    # 1. immediate_win
    a1 = analyze_position(("X", "X", ".", "O", "O", ".", ".", ".", "."), "X")
    assert a1.tactical_category == "immediate_win"
    assert a1.optimal_actions == ("R1C3",)

    # 2. losing_position
    a2 = analyze_position(("X", ".", ".", "O", ".", ".", "X", ".", "X"), "O")
    #Wait: let's check a clean losing position: X at R1C1, R3C1, R3C3; O at R1C2, R2C2? Wait:
    # If O at R1C2, R2C2, X at R1C1, R3C1, R3C3 -> O wins with R3C2? Let's use X at (0,6,8), O at (1,5).
    a2 = analyze_position(("X", "O", ".", ".", ".", "O", "X", ".", "X"), "O")
    assert a2.state_value == -1
    assert a2.tactical_category == "losing_position"

    # 3. forced_block
    a3 = analyze_position(("X", "X", ".", ".", "O", ".", ".", ".", "."), "O")
    assert a3.tactical_category == "forced_block"
    assert a3.optimal_actions == ("R1C3",)

    # 4. fork_create
    a4 = analyze_position(("X", ".", ".", ".", "O", ".", ".", ".", "X"), "X")
    # X at 0,8; O at 4; X to move:Wait—0,4,8: O at 4, X at 0,8 -> is R1C3 or R3C1 a win? X at 0,8, O at 4 is a draw!
    # Classic fork_create: X at R1C1 (0), O at R1C2 (1), X at R2C2 (4), O at R3C3 (8) -> X plays R3C1 (6) or R2C1 (3)
    a4 = analyze_position(("X", "O", ".", ".", "X", ".", ".", ".", "O"), "X")
    assert a4.tactical_category == "fork_create"
    assert a4.state_value == 1

    # 5. fork_defend
    a5 = analyze_position(("X", ".", ".", ".", ".", ".", ".", ".", "."), "O")
    assert a5.tactical_category == "fork_defend"
    assert a5.optimal_actions == ("R2C2",)

    # 6. forced_draw
    a6 = analyze_position((".",) * 9, "X")
    assert a6.tactical_category == "forced_draw"


def test_evaluate_move_regrets() -> None:
    # Board where X has immediate win at R1C3, and other moves may draw or lose.
    board = ("X", "X", ".", "O", "O", ".", ".", ".", ".")
    ev_opt = evaluate_move(board, "X", "R1C3")
    assert ev_opt == (True, True, 0, 0)

    ev_sub = evaluate_move(board, "X", "R2C3")
    assert ev_sub.is_legal is True
    assert ev_sub.is_optimal is False
    assert ev_sub.legal_regret == 1  # state_value (+1) - draw (0)
    assert ev_sub.effective_regret == 1

    ev_lose = evaluate_move(board, "X", "R3C1")
    assert ev_lose.is_legal is True
    assert ev_lose.is_optimal is False
    assert ev_lose.legal_regret == 2  # state_value (+1) - loss (-1)
    assert ev_lose.effective_regret == 2

    ev_illegal = evaluate_move(board, "X", "R1C1")
    assert ev_illegal.is_legal is False
    assert ev_illegal.is_optimal is False
    assert ev_illegal.legal_regret is None
    assert ev_illegal.effective_regret == 2  # state_value (+1) - (-1)

    ev_none = evaluate_move(board, "X", None)
    assert ev_none == (False, False, None, 2)


def test_minimax_never_loses_in_500_games() -> None:
    rng = random.Random(20261114)
    for game_idx in range(500):
        starting = "X" if game_idx % 2 == 0 else "O"
        state = new_ttt_state(starting)
        minimax_mark = "X" if (game_idx // 2) % 2 == 0 else "O"
        vs_minimax = game_idx >= 250

        while True:
            if state.player_to_move == minimax_mark or vs_minimax:
                action = choose_minimax_action(state.board, state.player_to_move, rng=rng)
            else:
                legal = [
                    INDEX_TO_KEY[i]
                    for i, c in enumerate(state.board)
                    if c == "."
                ]
                action = rng.choice(legal)

            outcome = step_ttt(state, action)
            assert outcome.action_is_legal is True
            state = outcome.next_state
            if outcome.done:
                if vs_minimax:
                    assert outcome.termination_reason == "draw"
                    assert outcome.winner is None
                else:
                    assert outcome.winner != other_mark(minimax_mark)
                break
