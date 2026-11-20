"""The empty-session pane: a centered block-font logo and a command tip.

Shown by ``AdCodeApp._render`` whenever the transcript has nothing to paint
(fresh start, after ``/clear``).  The tip is picked once per empty episode and
handed in by the caller, so the pane never flickers between frames.

The logo is a dot-matrix wordmark.  A terminal cell is roughly twice as tall
as it is wide, so a full block would draw a dot twice as tall as wide; the
glyphs are therefore encoded with half blocks (``▀`` / ``▄``), which makes
every dot visually square and keeps the wordmark short and wide.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from adcode.i18n import t
from adcode.theme import Palette
from adcode.tui.canvas import DIM, Canvas, text_width

if TYPE_CHECKING:
    from adcode.commands import Command

# Lowercase glyphs on a 5 x 7 grid.  Lowercase keeps the wordmark short, and
# the ascender of "d" gives it the rhythm a wordmark wants.
_LETTERS = {
    'a': ('     ',
          ' ████',
          '    █',
          '█████',
          '█   █',
          '█   █',
          '█████'),
    'c': ('█████',
          '█    ',
          '█    ',
          '█    ',
          '█    ',
          '█    ',
          '█████'),
    'd': ('    █',
          '    █',
          '█████',
          '█   █',
          '█   █',
          '█   █',
          '█████'),
    'e': ('     ',
          '█████',
          '█   █',
          '█████',
          '█    ',
          '█    ',
          '█████'),
    'o': ('     ',
          ' ███ ',
          '█   █',
          '█   █',
          '█   █',
          '█   █',
          ' ███ '),
}
_LOGO_TEXT = 'adcode'
_LETTER_WIDTH = 5
_LETTER_HEIGHT = 7
# One blank column between two letters.
_LETTER_GAP = 1


def _bitmap() -> list[str]:
    """The wordmark as a grid of blocks and spaces, one row per dot row."""
    rows: list[str] = []
    for y in range(_LETTER_HEIGHT):
        row = ''
        for index, ch in enumerate(_LOGO_TEXT):
            if index:
                row += ' ' * _LETTER_GAP
            row += _LETTERS[ch][y]
        rows.append(row)
    return rows


def _half_blocks(bitmap: list[str]) -> list[str]:
    """Fold two dot rows into one cell row of half blocks."""
    rows: list[str] = []
    for y in range(0, len(bitmap), 2):
        top = bitmap[y]
        bottom = bitmap[y + 1] if y + 1 < len(bitmap) else ''
        row = ''
        for x in range(len(top)):
            up = top[x] == '█'
            down = bottom[x] == '█' if x < len(bottom) else False
            if up and down:
                row += '█'
            elif up:
                row += '▀'
            elif down:
                row += '▄'
            else:
                row += ' '
        rows.append(row)
    return rows


# The wordmark, ready to print: four cell rows of half blocks.
LOGO_ROWS: tuple[str, ...] = tuple(_half_blocks(_bitmap()))
LOGO_HEIGHT = len(LOGO_ROWS)
LOGO_WIDTH = len(LOGO_ROWS[0]) if LOGO_ROWS else 0

# The gradient: a diagonal sweep, dim at the top left and bright at the
# bottom right, so the wordmark reads as lit rather than flat.
_DIM_ALPHA = 72
_BRIGHT_ALPHA = 255
_ALPHA_STEPS = 16


def _alpha_at(row: int, column: int) -> int:
    """The opacity of one logo cell."""
    if LOGO_HEIGHT <= 1:
        return _BRIGHT_ALPHA
    # The horizontal falloff is weaker than the vertical one, so the wordmark
    # still reads as a single band of light.
    diagonal = row + 0.4 * (LOGO_WIDTH - 1 - column) / max(1, LOGO_WIDTH - 1)
    span = (LOGO_HEIGHT - 1) + 0.4
    t = min(1.0, max(0.0, diagonal / span))
    level = int(round(t * (_ALPHA_STEPS - 1)))
    alpha = _DIM_ALPHA + (_BRIGHT_ALPHA - _DIM_ALPHA) * level / (_ALPHA_STEPS - 1)
    return int(round(alpha))


def paint_welcome(canvas: Canvas, top: int, bottom: int, width: int,
                  palette: Palette, tip: 'Command | None') -> None:
    """Paint the centered logo and one tip line into the transcript region."""
    rows = bottom - top
    if rows < LOGO_HEIGHT:
        return
    # The logo sits below the middle of the pane: the eye reads it as the
    # resting place of an empty conversation, not as a banner.
    y = top + int((rows - LOGO_HEIGHT) * 0.62)
    x0 = max(0, (width - LOGO_WIDTH) // 2)
    for row_index, text in enumerate(LOGO_ROWS):
        x = x0
        column = 0
        while column < len(text):
            if text[column] == ' ':
                x += 1
                column += 1
                continue
            alpha = _alpha_at(row_index, column)
            # Neighbouring cells with the same opacity share one draw call.
            run = 1
            while (column + run < len(text) and text[column + run] != ' '
                   and _alpha_at(row_index, column + run) == alpha):
                run += 1
            canvas.draw_text(x, y, text[column:column + run], palette.accent,
                             None, 0, alpha)
            x += run
            column += run
        y += 1
    if bottom - y < 2 or tip is None:
        return
    y += 1
    usage = tip.usage or tip.name
    segments = (
        (t('try '), palette.muted, DIM),
        (usage, palette.accent, 0),
        (f' — {t(tip.description)}', palette.subtext, DIM),
    )
    total = sum(text_width(text) for text, _, _ in segments)
    x = max(0, (width - total) // 2)
    for text, colour, attrs in segments:
        x += canvas.draw_text(x, y, text, colour, None, attrs)
