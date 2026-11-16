"""A dialog whose body outgrows its panel must scroll.

The /help output is the long case: on a normal 30 row terminal more than half
of it used to be clipped with no way to reach it, because a message dialog had
no scroll state and the wheel was swallowed by the modal.
"""

from __future__ import annotations

import pytest

from adcode.theme import build_palette
from adcode.tui.canvas import Canvas
from adcode.tui.dialogs import ConfirmDialog, MessageDialog, ScrollingBody
from adcode.tui.input import KeyEvent

MESSAGE = '\n\n'.join(f'paragraph {index:02d} of a long message'
                      for index in range(40))


def wheel(action: str) -> KeyEvent:
    return KeyEvent(kind='mouse', action=action, x=10, y=5)


def key(name: str) -> KeyEvent:
    return KeyEvent(kind='key', key=name)


def render(dialog, width: int = 100, height: int = 30) -> str:
    palette = build_palette('dark')
    canvas = Canvas(width, height, palette)
    dialog.render(canvas, width, height, palette, 0.0)
    rows = []
    for y in range(height):
        rows.append(''.join(
            (canvas.cell(x, y).ch if canvas.cell(x, y) else ' ')
            for x in range(width)).rstrip())
    return '\n'.join(rows)


# ── the message body ───────────────────────────────────────
def test_a_long_message_scrolls_with_the_wheel():
    dialog = MessageDialog('Help', MESSAGE)
    frame = render(dialog)
    assert 'paragraph 00' in frame
    assert 'paragraph 39' not in frame

    for _ in range(40):
        dialog.handle(wheel('wheel_down'))

    frame = render(dialog)
    assert 'paragraph 39' in frame
    assert 'paragraph 00' not in frame
    assert not dialog.closed


def test_the_wheel_goes_back_up_and_stops_at_the_top():
    dialog = MessageDialog('Help', MESSAGE)
    render(dialog)
    dialog.handle(wheel('wheel_down'))
    dialog.handle(wheel('wheel_down'))
    assert dialog.body.top == 6
    dialog.handle(wheel('wheel_up'))
    assert dialog.body.top == 3
    for _ in range(10):
        dialog.handle(wheel('wheel_up'))
    assert dialog.body.top == 0
    assert 'paragraph 00' in render(dialog)


def test_the_scroll_stops_at_the_last_line():
    dialog = MessageDialog('Help', MESSAGE)
    render(dialog)
    for _ in range(200):
        dialog.handle(wheel('wheel_down'))
    body = dialog.body
    assert body.top == body.total - body.rows
    assert body.top > 0
    frame = render(dialog)
    assert 'paragraph 39' in frame


def test_page_keys_and_home_end_scroll():
    dialog = MessageDialog('Help', MESSAGE)
    render(dialog)
    dialog.handle(key('pgdn'))
    page = dialog.body.rows - 1
    assert dialog.body.top == page
    dialog.handle(key('pgup'))
    assert dialog.body.top == 0
    dialog.handle(key('end'))
    assert dialog.body.top == dialog.body.total - dialog.body.rows
    dialog.handle(key('home'))
    assert dialog.body.top == 0
    dialog.handle(key('down'))
    assert dialog.body.top == 1
    dialog.handle(key('up'))
    assert dialog.body.top == 0


def test_a_long_body_draws_a_scrollbar():
    dialog = MessageDialog('Help', MESSAGE)
    assert '┃' in render(dialog)


def test_a_short_message_never_scrolls():
    dialog = MessageDialog('Info', 'nothing to see here')
    frame = render(dialog)
    assert '┃' not in frame
    assert 'enter / esc close' in frame
    assert 'wheel scroll' not in frame
    dialog.handle(wheel('wheel_down'))
    dialog.handle(key('pgdn'))
    assert dialog.body.top == 0


def test_closing_still_works_after_scrolling():
    for name in ('enter', 'esc', 'q'):
        dialog = MessageDialog('Help', MESSAGE)
        render(dialog)
        dialog.handle(wheel('wheel_down'))
        assert dialog.body.top > 0
        dialog.handle(key(name))
        assert dialog.closed


def test_confirm_dialog_scrolls_a_long_question():
    asked: list[bool] = []
    dialog = ConfirmDialog('Sure?', MESSAGE, asked.append)
    frame = render(dialog)
    assert 'paragraph 00' in frame and 'paragraph 39' not in frame
    dialog.handle(wheel('wheel_down'))
    dialog.handle(key('end'))
    assert 'paragraph 39' in render(dialog)
    dialog.handle(key('y'))
    assert asked == [True]


@pytest.mark.parametrize('width,height', [(60, 20), (100, 30), (140, 45)])
def test_the_scroll_position_survives_a_resize(width, height):
    dialog = MessageDialog('Help', MESSAGE)
    render(dialog, 100, 30)
    for _ in range(5):
        dialog.handle(wheel('wheel_down'))
    before = dialog.body.top
    frame = render(dialog, width, height)
    assert dialog.body.top <= max(0, dialog.body.total - dialog.body.rows)
    assert frame.strip()
    assert dialog.body.top >= min(before, dialog.body.total - dialog.body.rows)


def test_the_body_is_wrapped_once_per_width():
    body = ScrollingBody()
    first = body.lines(MESSAGE, 60)
    assert body.lines(MESSAGE, 60) is first
    assert body.lines(MESSAGE, 80) is not first

def test_a_wheel_before_the_first_frame_still_scrolls():
    # The body is measured while the dialog is drawn.  A wheel event that
    # arrives before that first frame must not be swallowed: the offset is
    # kept and the geometry of the frame clamps it.
    dialog = MessageDialog('Help', MESSAGE)
    dialog.handle(wheel('wheel_down'))
    assert dialog.body.top > 0
    frame = render(dialog)
    assert 'paragraph 00' not in frame
    assert dialog.body.top <= dialog.body.total - dialog.body.rows
