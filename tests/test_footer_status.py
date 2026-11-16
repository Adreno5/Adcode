"""The status line under the input.

Three things live on that line:

- the green state of a finished task, which stays until the user types;
- the progress bar of a command that reports a percentage;
- the light that runs along the line while the agent works.
"""

import time

import pytest

from adcode.tui.app import AdCodeApp, _percent_in


@pytest.fixture()
def app(tmp_path):
    instance = AdCodeApp(cwd=str(tmp_path), resume=False,
                         config_path=str(tmp_path / 'config.json'))
    try:
        yield instance
    finally:
        instance.input_thread.stop()
        instance.store.save()


def green_columns(app, columns=60, y=4):
    default = tuple(app.palette.background)
    return [x for x in range(columns)
            if tuple(app.canvas.cell(x, y).bg) != default]


# ── the finished state ─────────────────────────────────────
def test_a_finished_task_turns_the_line_green(app):
    app._apply_event('turn_end', {'turn': 1, 'status': 'done'})
    assert app.footer_state == 'done'
    app.footer_state_at = time.time() - 1.0      # the wipe is over
    app.canvas.resize(60, 10)
    app.canvas.clear()
    app._draw_footer(60, 9, 0)
    assert len(green_columns(app, 60, 9)) == 60


def test_an_interrupted_task_does_not_turn_it_green(app):
    app._apply_event('turn_end', {'turn': 1, 'status': 'interrupted'})
    assert app.footer_state == 'normal'


def test_the_green_runs_in_from_the_left(app):
    app.config.footer.done_wipe = 0.5
    app._apply_event('turn_end', {'turn': 1, 'status': 'done'})
    app.footer_state_at = time.time() - 0.25     # half of the wipe
    cover = app._footer_done_cover(60, time.time())
    assert 0 < cover < 60
    # It grows from the left edge.
    app.footer_state_at = time.time() - 0.1
    early = app._footer_done_cover(60, time.time())
    assert early < cover


def test_the_green_stays_until_the_user_types(app):
    app._apply_event('turn_end', {'turn': 1, 'status': 'done'})
    app._footer_input()
    assert app.footer_state == 'fading'
    # The fade removes the green from the left to the right.
    app.footer_state_at = time.time() - 0.25
    cover = app._footer_done_cover(60, time.time())
    assert 0 < cover < 60
    app.footer_state_at = time.time() - 0.6
    assert app._footer_done_cover(60, time.time()) == 0
    # And the line is normal again.
    app._footer_tick(time.time())
    assert app.footer_state == 'normal'


def test_the_font_is_white_on_the_green(app):
    app._apply_event('turn_end', {'turn': 1, 'status': 'done'})
    assert tuple(app._footer_color('muted')) == (255, 255, 255)
    app.footer_state = 'normal'
    assert tuple(app._footer_color('muted')) != (255, 255, 255)


def test_the_green_colour_is_configurable(app):
    app.config.footer.done_color = '#ff8800'
    background, _ = app._footer_colors()
    assert tuple(background) == (255, 136, 0)


# ── the progress bar ────────────────────────────────────────
def test_a_percentage_is_read_from_the_output():
    assert _percent_in('tests/test_x.py::test_y [ 45%]') == 45
    assert _percent_in('Building [12/20] 60%') == 60
    assert _percent_in('no number here') is None
    assert _percent_in('999%') is None


def test_a_rising_percentage_shows_the_bar(app):
    app._apply_event('tool_start', {'call_id': 'c1', 'name': 'bash',
                                     'display': 'bash', 'args': {}})
    for value in (10, 30, 50):
        app._apply_event('tool_output', {'call_id': 'c1',
                                          'text': f'step [ {value}%]'})
    assert app._footer_progress(time.time()) == 50
    # The bar is as long as the percentage.
    app.canvas.resize(60, 10)
    app.canvas.clear()
    app._draw_footer(60, 9, 0)
    assert len(green_columns(app, 60, 9)) == 30


def test_a_jittering_percentage_is_not_a_progress(app):
    app._apply_event('tool_start', {'call_id': 'c1', 'name': 'bash',
                                     'display': 'bash', 'args': {}})
    for value in (45, 80, 12):
        app._apply_event('tool_output', {'call_id': 'c1',
                                          'text': f'Downloading pkg ({value}%)'})
    # It is not a total progress: the light comes back instead.
    assert app._footer_progress(time.time()) is None


def test_one_percentage_is_not_enough(app):
    app._apply_event('tool_start', {'call_id': 'c1', 'name': 'bash',
                                     'display': 'bash', 'args': {}})
    app._apply_event('tool_output', {'call_id': 'c1', 'text': 'step [ 45%]'})
    assert app._footer_progress(time.time()) is None


def test_the_bar_names_the_command(app):
    app.status_args = {'command': 'pytest -q tests/'}
    assert app._progress_label() == 'pytest'
    app.status_args = {'command': 'python -m pytest -q'}
    assert app._progress_label() == 'pytest'
    app.status_args = {'command': '/usr/bin/pip install -e .'}
    assert app._progress_label() == 'pip'


def test_the_progress_is_forgotten_with_the_command(app):
    app._apply_event('tool_start', {'call_id': 'c1', 'name': 'bash',
                                     'display': 'bash', 'args': {}})
    app._apply_event('tool_output', {'call_id': 'c1', 'text': 'step [ 45%]'})
    app._apply_event('tool_end', {'call_id': 'c1', 'name': 'bash'})
    assert app._footer_progress(time.time()) is None
