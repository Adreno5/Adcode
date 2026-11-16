"""The light that runs along the status line while the agent works.

It sweeps the bottom row from the left to the right and crosses the screen
once per second.  The wait for the provider has no light: nothing is
happening yet.
"""

import time

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


def lit_cells(app, columns=60, y=4):
    """The columns of the status row that carry the light."""
    default = tuple(app.palette.background)
    return [x for x in range(columns)
            if tuple(app.canvas.cell(x, y).bg) != default]


def draw(app, phase, columns=60, y=4):
    app.phase = phase
    app.canvas.clear()
    app.canvas.palette = app.palette
    app._draw_status_sweep(columns, y, phase)


def test_the_light_runs_while_the_agent_works(app, monkeypatch):
    import adcode.tui.app as app_module

    # A fixed moment in the sweep, with the light on the line.
    monkeypatch.setattr(app_module.time, 'time', lambda: 100.3)
    for phase in ('thinking', 'answering', 'tool', 'compacting', 'aside'):
        draw(app, phase)
        assert lit_cells(app), phase


def test_the_light_waits_for_the_provider(app):
    draw(app, 'connecting')
    assert not lit_cells(app)
    draw(app, 'idle')
    assert not lit_cells(app)


def test_the_light_moves_to_the_right(app, monkeypatch):
    import adcode.tui.app as app_module

    clock = {'now': 100.2}
    monkeypatch.setattr(app_module.time, 'time', lambda: clock['now'])
    draw(app, 'thinking')
    first = lit_cells(app)
    clock['now'] += 0.3
    app.canvas.clear()
    app._draw_status_sweep(60, 4, 'thinking')
    second = lit_cells(app)
    assert first and second
    assert first != second
    # The light travels to the right: its head moved past the first one.
    assert max(second) > max(first)


def test_the_light_crosses_the_line_once_a_second(app, monkeypatch):
    import adcode.tui.app as app_module

    clock = {'now': 100.0}
    monkeypatch.setattr(app_module.time, 'time', lambda: clock['now'])
    clock['now'] = 100.3
    draw(app, 'thinking')
    start = lit_cells(app)
    clock['now'] += 0.4
    app.canvas.clear()
    app._draw_status_sweep(60, 4, 'thinking')
    later = lit_cells(app)
    assert start and later
    assert max(later) > max(start)
    # A second later it has crossed the line and starts again at the left.
    clock['now'] = 101.25
    app.canvas.clear()
    app._draw_status_sweep(60, 4, 'thinking')
    wrapped = lit_cells(app)
    assert min(wrapped) < min(later)


def test_the_light_stays_on_the_status_row(app, monkeypatch):
    import adcode.tui.app as app_module

    # The light enters from beyond the left edge, so the clock has to be
    # pinned to a moment where it is on the line.
    monkeypatch.setattr(app_module.time, 'time', lambda: 100.3)
    draw(app, 'thinking')
    assert lit_cells(app, y=4)
    assert not lit_cells(app, y=3)
