"""Deterministic 512x512 RGB PNG renderer for Tic-Tac-Toe and T3 visual variants."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
import hashlib
import io
import os
from typing import Literal

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from dm_bench.games.tic_tac_toe.engine import Mark, TTTState

IMAGE_SIZE: tuple[int, int] = (512, 512)
RENDER_VARIANTS: tuple[str, ...] = (
    "clean_default",
    "labeled_margins",
    "dark_low_contrast",
    "thin_alt_font",
    "tilt_blur",
)

# Board geometry in 512x512 coordinates (56..456 x 92..492 -> 400x400 px).
_BOARD_X0 = 56
_BOARD_Y0 = 92
_BOARD_SPAN = 400
_X_BOUNDARIES = tuple(_BOARD_X0 + round(i * _BOARD_SPAN / 3) for i in range(4))
_Y_BOUNDARIES = tuple(_BOARD_Y0 + round(i * _BOARD_SPAN / 3) for i in range(4))

_SANS_FONT_PATHS = (
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)

_SERIF_ITALIC_FONT_PATHS = (
    "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",
    "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
    "/System/Library/Fonts/Times.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
)


@lru_cache(maxsize=32)
def _load_font(size: int, serif_italic: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = _SERIF_ITALIC_FONT_PATHS if serif_italic else _SANS_FONT_PATHS
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _validate_inputs(board: Sequence[str], player_to_move: Mark, variant: str) -> tuple[str, ...]:
    if len(board) != 9:
        raise ValueError(f"Board must have length 9, got {len(board)}")
    out = tuple(board)
    for cell in out:
        if cell not in ("X", "O", "."):
            raise ValueError(f"Invalid board cell: {cell!r}")
    if player_to_move not in ("X", "O"):
        raise ValueError(f"Invalid player_to_move: {player_to_move!r}")
    if variant not in RENDER_VARIANTS:
        raise ValueError(f"Unsupported render variant: {variant!r}. Expected one of {RENDER_VARIANTS}")
    return out


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    center_xy: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str,
) -> None:
    draw.text(center_xy, text, fill=fill, font=font, anchor="mm")


def _draw_vector_board(
    board: tuple[str, ...],
    player_to_move: Mark,
    *,
    bg_color: str,
    grid_color: str,
    grid_width: int,
    x_color: str,
    o_color: str,
    header_color: str,
    labeled_margins: bool = False,
) -> Image.Image:
    # Render at 2x supersampling for crisp anti-aliased strokes, then downsample to 512x512.
    scale = 2
    img = Image.new("RGB", (IMAGE_SIZE[0] * scale, IMAGE_SIZE[1] * scale), color=bg_color)
    draw = ImageDraw.Draw(img)

    header_font = _load_font(28 * scale, serif_italic=False)
    header_y = (30 if labeled_margins else 38) * scale
    _draw_centered_text(
        draw,
        (256 * scale, header_y),
        f"Player to move: {player_to_move}",
        header_font,
        header_color,
    )

    if labeled_margins:
        label_font = _load_font(20 * scale, serif_italic=False)
        margin_color = "#4B5563"
        for col_idx in range(3):
            cx = (_X_BOUNDARIES[col_idx] + _X_BOUNDARIES[col_idx + 1]) / 2 * scale
            _draw_centered_text(draw, (cx, 72 * scale), f"C{col_idx + 1}", label_font, margin_color)
        for row_idx in range(3):
            cy = (_Y_BOUNDARIES[row_idx] + _Y_BOUNDARIES[row_idx + 1]) / 2 * scale
            _draw_centered_text(draw, (28 * scale, cy), f"R{row_idx + 1}", label_font, margin_color)

    # Outer border and inner grid lines.
    bx0 = _X_BOUNDARIES[0] * scale
    bx1 = _X_BOUNDARIES[3] * scale
    by0 = _Y_BOUNDARIES[0] * scale
    by1 = _Y_BOUNDARIES[3] * scale
    gw = grid_width * scale

    draw.rectangle([bx0, by0, bx1, by1], outline=grid_color, width=gw)
    for i in (1, 2):
        xi = _X_BOUNDARIES[i] * scale
        yi = _Y_BOUNDARIES[i] * scale
        draw.line([(xi, by0), (xi, by1)], fill=grid_color, width=gw)
        draw.line([(bx0, yi), (bx1, yi)], fill=grid_color, width=gw)

    # Vector marks inside each cell.
    stroke_w = 12 * scale
    pad = 28 * scale
    for idx, cell in enumerate(board):
        if cell == ".":
            continue
        r, c = divmod(idx, 3)
        cx0 = _X_BOUNDARIES[c] * scale + pad
        cy0 = _Y_BOUNDARIES[r] * scale + pad
        cx1 = _X_BOUNDARIES[c + 1] * scale - pad
        cy1 = _Y_BOUNDARIES[r + 1] * scale - pad
        if cell == "X":
            draw.line([(cx0, cy0), (cx1, cy1)], fill=x_color, width=stroke_w)
            draw.line([(cx1, cy0), (cx0, cy1)], fill=x_color, width=stroke_w)
            # Round endcaps for clean geometric strokes.
            cap_r = stroke_w / 2
            for px, py in ((cx0, cy0), (cx1, cy1), (cx1, cy0), (cx0, cy1)):
                draw.ellipse([px - cap_r, py - cap_r, px + cap_r, py + cap_r], fill=x_color)
        else:
            draw.ellipse([cx0, cy0, cx1, cy1], outline=o_color, width=stroke_w)

    return img.resize(IMAGE_SIZE, resample=Image.Resampling.LANCZOS)


def _cell_jitter(position_id: str, cell_idx: int) -> tuple[int, int]:
    digest = hashlib.sha256(f"{position_id}|{cell_idx}".encode("utf-8")).digest()
    dx = (digest[0] % 13) - 6
    dy = (digest[1] % 13) - 6
    return dx, dy


def _tilt_angle(board: tuple[str, ...], player_to_move: Mark, position_id: str) -> float:
    payload = f"{position_id}|{''.join(board)}|{player_to_move}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return 5.0 if (digest[0] % 2 == 0) else -5.0


def _draw_thin_alt_font(
    board: tuple[str, ...],
    player_to_move: Mark,
    position_id: str,
) -> Image.Image:
    img = Image.new("RGB", IMAGE_SIZE, color="#FDFBF7")
    draw = ImageDraw.Draw(img)

    header_font = _load_font(26, serif_italic=True)
    _draw_centered_text(
        draw,
        (256, 38),
        f"Player to move: {player_to_move}",
        header_font,
        "#374151",
    )

    grid_color = "#6B7280"
    bx0, bx1 = _X_BOUNDARIES[0], _X_BOUNDARIES[3]
    by0, by1 = _Y_BOUNDARIES[0], _Y_BOUNDARIES[3]
    draw.rectangle([bx0, by0, bx1, by1], outline=grid_color, width=1)
    for i in (1, 2):
        draw.line([(_X_BOUNDARIES[i], by0), (_X_BOUNDARIES[i], by1)], fill=grid_color, width=1)
        draw.line([(bx0, _Y_BOUNDARIES[i]), (bx1, _Y_BOUNDARIES[i])], fill=grid_color, width=1)

    mark_font = _load_font(86, serif_italic=True)
    for idx, cell in enumerate(board):
        if cell == ".":
            continue
        r, c = divmod(idx, 3)
        cx = (_X_BOUNDARIES[c] + _X_BOUNDARIES[c + 1]) / 2
        cy = (_Y_BOUNDARIES[r] + _Y_BOUNDARIES[r + 1]) / 2
        dx, dy = _cell_jitter(position_id, idx)
        color = "#1E3A8A" if cell == "X" else "#991B1B"
        _draw_centered_text(draw, (cx + dx, cy + dy), cell, mark_font, color)

    return img


def render_ttt_image(
    board: Sequence[str],
    player_to_move: Mark,
    variant: str = "clean_default",
    position_id: str = "",
) -> Image.Image:
    """Render a deterministic 512x512 RGB Pillow Image for the requested T3 variant."""
    validated_board = _validate_inputs(board, player_to_move, variant)

    if variant == "clean_default":
        return _draw_vector_board(
            validated_board,
            player_to_move,
            bg_color="#FAFAFC",
            grid_color="#1F2937",
            grid_width=5,
            x_color="#1D4ED8",
            o_color="#B91C1C",
            header_color="#111827",
            labeled_margins=False,
        )

    if variant == "labeled_margins":
        return _draw_vector_board(
            validated_board,
            player_to_move,
            bg_color="#FAFAFC",
            grid_color="#1F2937",
            grid_width=5,
            x_color="#1D4ED8",
            o_color="#B91C1C",
            header_color="#111827",
            labeled_margins=True,
        )

    if variant == "dark_low_contrast":
        return _draw_vector_board(
            validated_board,
            player_to_move,
            bg_color="#1A1D24",
            grid_color="#2E3340",
            grid_width=4,
            x_color="#4B5263",
            o_color="#565E70",
            header_color="#6B7280",
            labeled_margins=False,
        )

    if variant == "thin_alt_font":
        return _draw_thin_alt_font(validated_board, player_to_move, position_id)

    # variant == "tilt_blur"
    base = _draw_vector_board(
        validated_board,
        player_to_move,
        bg_color="#FAFAFC",
        grid_color="#1F2937",
        grid_width=5,
        x_color="#1D4ED8",
        o_color="#B91C1C",
        header_color="#111827",
        labeled_margins=False,
    )
    angle = _tilt_angle(validated_board, player_to_move, position_id)
    rotated = base.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(250, 250, 252),
    )
    return rotated.filter(ImageFilter.GaussianBlur(radius=1.4))


def render_ttt_png(
    board: Sequence[str],
    player_to_move: Mark,
    variant: str = "clean_default",
    position_id: str = "",
) -> bytes:
    """Render a deterministic 512x512 RGB PNG byte string."""
    img = render_ttt_image(
        board=board,
        player_to_move=player_to_move,
        variant=variant,
        position_id=position_id,
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def render_state_png(
    state: TTTState,
    variant: str = "clean_default",
    position_id: str = "",
) -> bytes:
    """Convenience wrapper around render_ttt_png for a TTTState instance."""
    return render_ttt_png(
        board=state.board,
        player_to_move=state.player_to_move,
        variant=variant,
        position_id=position_id,
    )
