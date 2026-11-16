"""The welcome pane: lowercase wordmark, square dots, gradient and the tip."""

import random

from adcode.commands import COMMANDS, random_tip
from adcode.tui.canvas import Canvas
from adcode.tui.welcome import (
    LOGO_HEIGHT, LOGO_ROWS, LOGO_WIDTH, _bitmap, paint_welcome,
)


def row_text(canvas, y, width):
    cells = (canvas.cell(x, y) for x in range(width))
    return ''.join(cell.ch if cell is not None else ' ' for cell in cells)


def slice_text(canvas, y, x0, width):
    cells = (canvas.cell(x, y) for x in range(x0, x0 + width))
    return ''.join(cell.ch if cell is not None else ' ' for cell in cells)


class Tip:
    def __init__(self, name='/keys', usage='', description='Configure key bindings'):
        self.name = name
        self.usage = usage
        self.description = description


def brightness(colour):
    return sum(colour)


def test_the_wordmark_is_lowercase():
    bitmap = _bitmap()
    assert len(bitmap) == 7
    # "adcode": six lowercase glyphs, five columns each, one column apart.
    assert len(bitmap[0]) == 6 * 5 + 5
    assert set(''.join(bitmap)) <= {'█', ' '}


def test_logo_rows_are_half_blocks():
    # Half blocks keep every dot visually square: a cell is twice as tall as
    # it is wide, so a half cell is as tall as it is wide.
    assert LOGO_HEIGHT == 4
    assert LOGO_WIDTH == 35
    for row in LOGO_ROWS:
        assert len(row) == LOGO_WIDTH
        assert set(row) <= {' ', '█', '▀', '▄'}


def test_paint_welcome_centers_the_logo(palette):
    canvas = Canvas(90, 30, palette)
    canvas.clear()
    paint_welcome(canvas, 0, 30, 90, palette, Tip())
    x0 = (90 - LOGO_WIDTH) // 2
    painted = [y for y in range(30)
               if any(ch in row_text(canvas, y, 90) for ch in '█▀▄')]
    assert painted == list(range(painted[0], painted[0] + LOGO_HEIGHT))
    for y, row in zip(painted, LOGO_ROWS):
        assert slice_text(canvas, y, x0, LOGO_WIDTH) == row


def test_the_logo_sits_below_the_middle(palette):
    canvas = Canvas(90, 30, palette)
    canvas.clear()
    paint_welcome(canvas, 0, 30, 90, palette, Tip())
    painted = [y for y in range(30)
               if any(ch in row_text(canvas, y, 90) for ch in '█▀▄')]
    centre = (painted[0] + painted[-1]) / 2
    assert centre > 15          # the pane centre is 14.5
    assert centre < 30 - 4      # but it stays well above the bottom


def test_the_logo_carries_a_light_gradient(palette):
    canvas = Canvas(90, 30, palette)
    canvas.clear()
    paint_welcome(canvas, 0, 30, 90, palette, Tip())
    x0 = (90 - LOGO_WIDTH) // 2
    top_left = canvas.cell(x0 + 1, 16)
    bottom_right = canvas.cell(x0 + LOGO_WIDTH - 2, 19)
    assert top_left is not None and bottom_right is not None
    # The dim end is far from the accent colour, the bright end close to it.
    assert brightness(bottom_right.fg) > brightness(top_left.fg)
    assert brightness(top_left.fg) < brightness(palette.accent)


def test_paint_welcome_draws_one_tip_line(palette):
    canvas = Canvas(90, 30, palette)
    canvas.clear()
    paint_welcome(canvas, 0, 30, 90, palette, Tip())
    lines = [row_text(canvas, y, 90).rstrip() for y in range(30)]
    tips = [line for line in lines if 'try ' in line]
    assert len(tips) == 1
    assert '/keys' in tips[0] and 'Configure key bindings' in tips[0]
    # A usage string replaces the bare command name.
    canvas = Canvas(90, 30, palette)
    canvas.clear()
    paint_welcome(canvas, 0, 30, 90, palette, Tip(usage='/save [name]'))
    lines = [row_text(canvas, y, 90) for y in range(30)]
    assert any('/save [name]' in line for line in lines)


def test_paint_welcome_skips_a_viewport_that_cannot_hold_the_logo(palette):
    canvas = Canvas(90, 3, palette)
    canvas.clear()
    paint_welcome(canvas, 0, 3, 90, palette, Tip())
    assert all(row_text(canvas, y, 90).strip() == '' for y in range(3))


def test_paint_welcome_drops_the_tip_when_there_is_no_room(palette):
    canvas = Canvas(90, LOGO_HEIGHT, palette)
    canvas.clear()
    paint_welcome(canvas, 0, LOGO_HEIGHT, 90, palette, Tip())
    text = '\n'.join(row_text(canvas, y, 90) for y in range(LOGO_HEIGHT))
    assert '█' in text
    assert 'try ' not in text


def test_random_tip_never_suggests_quit():
    random.seed(1)
    picked = {random_tip().name for _ in range(400)}
    assert picked
    assert all(name in {item.name for item in COMMANDS} for name in picked)
    assert '/quit' not in picked
