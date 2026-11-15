"""Unit tests for Tic-Tac-Toe 512x512 Pillow renderer and T5 text formatters."""

from __future__ import annotations

import hashlib
import io
import json

from PIL import Image
import pytest

from dm_bench.games.tic_tac_toe.engine import CELL_KEYS, TTTState
from dm_bench.games.tic_tac_toe.formats import TEXT_FORMATS, format_state, format_ttt_state
from dm_bench.games.tic_tac_toe.renderer import (
    IMAGE_SIZE,
    RENDER_VARIANTS,
    render_state_png,
    render_ttt_image,
    render_ttt_png,
)


def test_all_render_variants_valid_512_rgb_and_deterministic() -> None:
    board = ("X", ".", "O", ".", "X", ".", ".", ".", "O")
    hashes_by_variant: dict[str, str] = {}

    for variant in RENDER_VARIANTS:
        png_a = render_ttt_png(board, "X", variant=variant, position_id="pos_0042")
        png_b = render_ttt_png(board, "X", variant=variant, position_id="pos_0042")
        assert png_a == png_b
        assert len(png_a) < 16 * 1024 * 1024

        with Image.open(io.BytesIO(png_a)) as img:
            assert img.format == "PNG"
            assert img.mode == "RGB"
            assert img.size == IMAGE_SIZE

        hashes_by_variant[variant] = hashlib.sha256(png_a).hexdigest()

        # Changing player_to_move changes the banner and rendered PNG bytes.
        png_o = render_ttt_png(board, "O", variant=variant, position_id="pos_0042")
        assert png_o != png_a

    # All 5 variants are visually distinct.
    assert len(set(hashes_by_variant.values())) == len(RENDER_VARIANTS)


def test_variant_background_colors_and_state_wrapper() -> None:
    board = ("X", "O", ".", ".", "X", ".", ".", ".", ".")
    state = TTTState(board=board, player_to_move="O", move_history=(0, 1, 4), starting_player="X")

    img_clean = render_ttt_image(board, "O", variant="clean_default", position_id="pos_0001")
    assert img_clean.getpixel((0, 0)) == (250, 250, 252)

    img_dark = render_ttt_image(board, "O", variant="dark_low_contrast", position_id="pos_0001")
    assert img_dark.getpixel((0, 0)) == (26, 29, 36)

    img_thin = render_ttt_image(board, "O", variant="thin_alt_font", position_id="pos_0001")
    assert img_thin.getpixel((0, 0)) == (253, 251, 247)

    assert render_state_png(state, variant="clean_default", position_id="pos_0001") == render_ttt_png(
        board, "O", variant="clean_default", position_id="pos_0001"
    )


def test_position_id_determinism_for_jitter_and_tilt() -> None:
    board = ("X", ".", "O", ".", "X", ".", ".", ".", "O")
    # Different position_ids should deterministically vary thin_alt_font jitter
    png_1 = render_ttt_png(board, "X", variant="thin_alt_font", position_id="pos_0001")
    png_2 = render_ttt_png(board, "X", variant="thin_alt_font", position_id="pos_0002")
    assert png_1 != png_2
    assert png_1 == render_ttt_png(board, "X", variant="thin_alt_font", position_id="pos_0001")


def test_invalid_render_variant_raises() -> None:
    with pytest.raises(ValueError):
        render_ttt_png((".",) * 9, "X", variant="unknown_variant")


def test_all_four_text_formats() -> None:
    board = ("X", ".", "O", ".", "X", ".", ".", ".", "O")
    history = (0, 2, 4, 8)
    state = TTTState(board=board, player_to_move="X", move_history=history, starting_player="X")

    expected_board = {
        "R1": {"R1C1": "X", "R1C2": "empty", "R1C3": "O"},
        "R2": {"R2C1": "empty", "R2C2": "X", "R2C3": "empty"},
        "R3": {"R3C1": "empty", "R3C2": "empty", "R3C3": "O"},
    }

    # 1. json_canonical
    jc = format_ttt_state(board, "X", "json_canonical")
    assert json.loads(jc) == {
        "game": "tic_tac_toe",
        "player_to_move": "X",
        "board": expected_board,
    }
    assert format_state(state, "json_canonical") == jc

    # 2. ascii_grid
    ag = format_ttt_state(board, "X", "ascii_grid")
    assert ag == "\n".join(
        [
            "Game: tic_tac_toe",
            "Player to move: X",
            "     C1  C2  C3",
            "R1 | X | . | O |",
            "R2 | . | X | . |",
            "R3 | . | . | O |",
        ]
    )

    # 3. coord_list
    cl = format_ttt_state(board, "X", "coord_list")
    lines = cl.splitlines()
    assert lines[0] == "Game: tic_tac_toe"
    assert lines[1] == "Player to move: X"
    assert lines[2] == "Board cells:"
    assert len(lines) == 12
    assert lines[3] == "- R1C1: X"
    assert lines[4] == "- R1C2: empty"
    assert lines[5] == "- R1C3: O"
    assert lines[11] == "- R3C3: O"
    for key, line in zip(CELL_KEYS, lines[3:]):
        assert line.startswith(f"- {key}: ")

    # 4. json_with_history
    jh = format_ttt_state(board, "X", "json_with_history", move_history=history)
    assert json.loads(jh) == {
        "game": "tic_tac_toe",
        "player_to_move": "X",
        "board": expected_board,
        "move_history": ["R1C1", "R1C3", "R2C2", "R3C3"],
    }
    assert format_state(state, "json_with_history") == jh
    assert len(TEXT_FORMATS) == 4


def test_invalid_text_format_raises() -> None:
    with pytest.raises(ValueError):
        format_ttt_state((".",) * 9, "X", text_format="markdown_table")
