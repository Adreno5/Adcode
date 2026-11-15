"""Simulated keyboard and mouse behaviour against the real app loop.

Every action goes through a raw terminal sequence parsed by AdCode's own input
layer.  The assertions check both the prompt state and the composited frame, so
a key that updates state but never reaches the screen still fails.
"""

from __future__ import annotations

from bench.script import ScriptedResponder, Turn

LONG_ANSWER = '\n'.join(
    f'{index + 1:02d}. streamed transcript row for scroll testing -- '
    f'the quick brown fox jumps over the lazy dog.'
    for index in range(80))


def test_keyboard_editing_selection_cut_paste_and_undo(harness):
    harness.type_text('helo')
    harness.wait_prompt('helo', 4)

    harness.press('left')
    harness.wait_prompt(cursor=3)
    harness.type_text('l')
    harness.wait_prompt('hello', 4)

    harness.press('home')
    harness.wait_prompt(cursor=0)
    harness.press('delete')
    harness.wait_prompt('ello', 0)
    harness.type_text('H')
    harness.wait_prompt('Hello', 1)

    harness.press('ctrl+end')
    harness.wait_prompt(cursor=5)
    harness.type_text(' world')
    harness.wait_prompt('Hello world', 11)
    assert '> Hello world' in harness.screen_text()

    harness.press('shift+left')
    harness.wait_until(
        lambda: harness.app is not None
        and harness.app.prompt.selection() == (10, 11),
        timeout=5.0, description='shift+left to extend the selection')
    harness.press('shift+left')
    harness.wait_until(
        lambda: harness.app is not None
        and harness.app.prompt.selection() == (9, 11),
        timeout=5.0, description='a second shift+left to extend further')

    harness.press('shift+home')
    harness.wait_until(
        lambda: harness.app is not None
        and harness.app.prompt.selection() == (0, 11),
        timeout=5.0, description='shift+home to select the line')
    assert '> Hello world' in harness.screen_text()

    harness.press('ctrl+x')
    harness.wait_prompt('', 0)
    assert harness.clipboard == 'Hello world'

    harness.press('ctrl+v')
    harness.wait_prompt('Hello world', 11)
    harness.press('ctrl+z')
    harness.wait_prompt('', 0)
    harness.press('ctrl+y')
    harness.wait_prompt('Hello world', 11)
    assert len(harness.requests) == 0


def test_slash_palette_completion_and_command_side_effect(harness):
    harness.type_text('/th')
    harness.wait_until(
        lambda: 'Set the thinking display mode' in harness.screen_text(),
        timeout=10.0, description='the slash completion palette')
    frame = harness.screen.latest
    assert frame is not None
    assert '/thinking' in frame.text()

    harness.press('tab')
    harness.wait_prompt('/thinking ')
    harness.type_text('full')
    harness.wait_prompt('/thinking full')
    harness.press('enter')

    harness.wait_for_text('Thinking display: full')
    harness.wait_until(
        lambda: harness.app is not None
        and harness.app.config.thinking.mode == 'full',
        timeout=10.0, description='the thinking mode to change')
    assert harness.workspace.config_json()['thinking']['mode'] == 'full'
    assert len(harness.requests) == 0


def test_mouse_wheel_scrolls_the_transcript_and_returns_to_tail(harness):
    harness.gateway.responder = ScriptedResponder([
        Turn(thinking='A long answer fills the viewport.', text=LONG_ANSWER),
    ])
    harness.submit('render a long answer', timeout=25.0)
    harness.wait_for_text('80. streamed transcript row')
    assert harness.app is not None and harness.app.scroll == 0
    assert harness.screen.latest is not None
    assert not harness.screen.latest.vt_mismatch

    harness.wheel('wheel_up')
    harness.wheel('wheel_up')
    harness.wait_until(
        lambda: harness.app is not None and harness.app.scroll > 0,
        timeout=10.0, description='the wheel to scroll up')
    harness.wait_rendered()
    assert '↑' in harness.screen_text()

    harness.wheel('wheel_down')
    harness.wheel('wheel_down')
    harness.wait_until(
        lambda: harness.app is not None
        and harness.app.follow and harness.app.scroll == 0,
        timeout=10.0, description='the wheel to return to the tail')
    harness.wait_for_text('80. streamed transcript row')


def test_ctrl_c_twice_exits_the_real_loop(harness):
    harness.type_text('draft')
    harness.wait_prompt('draft')
    harness.press('ctrl+c')
    harness.wait_prompt('')

    harness.press('ctrl+c')
    harness.wait_until(
        lambda: harness.app is not None and harness.app.exit_armed,
        timeout=5.0, description='the first ctrl+c to arm exit')
    harness.wait_for_text('ctrl+c again to exit')

    harness.press('ctrl+c')
    harness.wait_until(lambda: harness.finished.is_set(), timeout=10.0,
                       description='the app loop to exit')
    assert harness.screen.left
    assert harness.workspace.config_path.exists()
    assert harness.error is None


def test_resize_reflows_the_canvas_and_frames(harness):
    harness.resize(72, 20)
    harness.wait_rendered()
    frame = harness.screen.latest
    assert frame is not None
    assert (frame.width, frame.height) == (72, 20)
    assert max(len(line) for line in frame.lines) <= 72
    assert 'AdCode' in frame.text()
    assert 'ready' in frame.text()

    harness.resize(120, 40)
    harness.wait_rendered()
    frame = harness.screen.latest
    assert frame is not None
    assert (frame.width, frame.height) == (120, 40)
    assert max(len(line) for line in frame.lines) <= 120
    assert 'Harness / harness-model' in frame.text()
    assert 'ready' in frame.text()


def test_multiline_input_paste_and_the_portable_newline_gesture(harness):
    harness.type_text('line1')
    harness.press('ctrl+j')
    harness.type_text('line2')
    harness.wait_prompt('line1\nline2')
    harness.paste('\npasted')
    harness.wait_prompt('line1\nline2\npasted')
    assert len(harness.requests) == 0

    harness.type_text('\\')
    harness.wait_prompt('line1\nline2\npasted\\')
    harness.press('enter')
    harness.wait_prompt('line1\nline2\npasted\n')
    assert len(harness.requests) == 0

    harness.press('shift+enter')
    harness.wait_prompt('line1\nline2\npasted\n\n')
    assert len(harness.requests) == 0

    harness.press('enter')
    harness.wait_until(lambda: len(harness.requests) == 1, timeout=15.0,
                       description='the multiline message to submit')
    harness.wait_idle(20.0)
    assert harness.requests[0].last_user_text() == \
        'line1\nline2\npasted'


def test_modal_dialogs_own_the_wheel_and_the_keyboard(harness):
    """A floating dialog must not leak events to the page underneath it."""
    harness.gateway.responder = ScriptedResponder([
        Turn(thinking='A long answer fills the viewport.', text=LONG_ANSWER),
    ])
    harness.submit('render a long answer', timeout=25.0)
    harness.wait_for_text('80. streamed transcript row')
    assert harness.app is not None
    assert harness.app.scroll == 0 and harness.app.follow

    harness.run_command('/context')
    harness.wait_until(lambda: harness.app is not None
                       and harness.app.overlay is not None, timeout=10.0,
                       description='the context dialog')
    harness.wait_rendered()

    # The wheel over a dialog must not scroll the transcript behind it.
    harness.wheel('wheel_up')
    harness.wheel('wheel_up')
    harness.wheel('wheel_down')
    harness.wait_rendered()
    assert harness.app.scroll == 0
    assert harness.app.follow

    # Nor may keys reach the prompt; they belong to the dialog.
    harness.type_text('leaked')
    harness.press('pgup')
    harness.wait_rendered()
    assert harness.app.prompt.text == ''
    assert harness.app.scroll == 0
    assert harness.app.overlay is not None

    # Closing the dialog must not submit anything that was typed at it.
    harness.press('enter')
    harness.wait_until(lambda: harness.app is not None
                       and harness.app.overlay is None, timeout=10.0,
                       description='the context dialog to close')
    harness.wait_rendered()
    assert harness.app.prompt.text == ''
    assert harness.app.scroll == 0 and harness.app.follow
    assert len(harness.requests) == 1
