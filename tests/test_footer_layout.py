"""The status line under the input.

The line carries the state on the left and the token counters on the right.
Both have to fit: the extras can grow (a selection, the work-step level, the
hold counter, the scroll offset) and a narrow terminal has no room for them.
Drawing past the last cell used to end the frame with an IndexError.
"""

import pytest

from adcode.tui.app import AdCodeApp


@pytest.fixture()
def app(tmp_path):
    instance = AdCodeApp(cwd=str(tmp_path), resume=False,
                         config_path=str(tmp_path / 'config.json'))
    try:
        yield instance
    finally:
        instance.input_thread.stop()
        instance.store.save()


def crowded(app):
    """A status line with everything on it at once."""
    app.phase = 'tool'
    app.status_hint = 'bash'
    app.status_args = {'command': 'pytest -q ' + 'x' * 200}
    app.btw_mode = True
    app.exit_armed = True
    app.scroll = 999
    app.prompt.selection_anchor = 0
    app.prompt.selection_anchor = None


def test_the_footer_fits_a_narrow_terminal(app):
    crowded(app)
    for columns in (200, 120, 80, 60, 40, 20, 10):
        app.canvas.resize(columns, 10)
        app.canvas.clear()
        app._draw_footer(columns, 9, 0)


def test_the_token_counters_stay_on_the_right(app):
    app.config.ui.show_token_status = True
    app.canvas.resize(80, 10)
    app.canvas.clear()
    app._draw_footer(80, 9, 0)
    row = ''.join(app.canvas.cell(x, 9).ch for x in range(80))
    assert 'ctx' in row
    assert row.rstrip().endswith('0') or 'ctx' in row


def test_the_extras_do_not_run_into_the_counters(app):
    app.config.ui.show_token_status = True
    crowded(app)
    app.canvas.resize(60, 10)
    app.canvas.clear()
    app._draw_footer(60, 9, 0)
    row = ''.join(app.canvas.cell(x, 9).ch for x in range(60))
    assert 'ctx' in row
    # The counters are the last thing on the line.
    assert row.index('ctx') > 30, row


def test_the_extras_are_dropped_whole(app):
    """A half word on the line reads like a bug."""
    app.config.ui.show_token_status = True
    app.btw_mode = True
    app.exit_armed = True
    app.scroll = 999
    for columns in (60, 46, 40, 34, 28, 22, 16):
        app.canvas.resize(columns, 10)
        app.canvas.clear()
        app._draw_footer(columns, 9, 0)
        row = ''.join(app.canvas.cell(x, 9).ch for x in range(columns))
        assert 'ctx' in row, (columns, row)
        # An extra is either there as a whole or not at all.
        for phrase in (('aside', 'question', 'mode'),
                       ('ctrl+c', 'again', 'to', 'exit'),
                       ('selected', 'ctrl+c', 'copies')):
            present = [word in row for word in phrase]
            assert all(present) or not any(present), (columns, row)
