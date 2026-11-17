"""The status line under the input.

Three things live on that line:

- the green state of a finished task, which stays until the user types;
- the progress bar of a command that reports a percentage;
- the light that runs along the line while the agent works.
"""

import time

import pytest

from infra.policy.predictor import DurationPredictor

from adcode.tui.app import AdCodeApp, _percent_in


@pytest.fixture()
def app(tmp_path, monkeypatch):
    # the wait predictor persists between sessions; a test must not read the
    # learnings of the real one
    monkeypatch.setattr('adcode.paths.predictor_file',
                        lambda: tmp_path / 'predictor.json')
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


def test_the_fade_leaves_no_green_behind(app):
    """The last fading frame still carries a sliver of the cover.

    Leaving the state does not schedule a frame by itself, so without an
    explicit request that sliver stays on the status line -- over the
    separator between the status word and the extras -- until the user types.
    """
    app.config.footer.done_wipe = 0.4
    app._apply_event('turn_end', {'turn': 1, 'status': 'done'})
    app._footer_input()
    assert app.footer_state == 'fading'

    # A frame taken in the middle of the fade keeps a piece of the green.
    app.footer_state_at = time.time() - 0.39
    app.canvas.resize(60, 10)
    app.canvas.clear()
    app._draw_footer(60, 9, 0)
    assert green_columns(app, 60, 9)

    # The fade is over: the frame after it has to be drawn again.
    app.dirty = False
    app.footer_state_at = time.time() - 1.0
    app._tick_animation()
    assert app.footer_state == 'normal'
    assert app.dirty is True

    app.canvas.clear()
    app._draw_footer(60, 9, 0)
    assert green_columns(app, 60, 9) == []


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


# ── the light walks with the tokens ────────────────────────
def test_the_light_walks_with_the_tokens(app):
    """One cycle of the light per N tokens."""
    app.config.footer.sweep_tokens_per_cycle = 100
    app._stream_tokens = 0
    assert app._sweep_position('answering', 0.0) == pytest.approx(0.0)
    app._stream_tokens = 25
    assert app._sweep_position('answering', 0.0) == pytest.approx(0.25)
    app._stream_tokens = 100
    assert app._sweep_position('answering', 0.0) == pytest.approx(0.0)
    app._stream_tokens = 150
    assert app._sweep_position('answering', 0.0) == pytest.approx(0.5)


def test_the_token_mode_only_moves_while_the_model_generates(app):
    app.config.footer.sweep_tokens_per_cycle = 100
    app._stream_tokens = 30
    # A running command has no tokens: the light keeps its own rhythm.
    position = app._sweep_position('tool', 0.25)
    assert position != pytest.approx(0.3)


def test_the_streamed_tokens_are_counted(app):
    app._apply_event('turn_start', {'turn': 1, 'text': 'hi'})
    app._apply_event('text_delta', {'text': 'hello world'})
    assert app._stream_tokens > 0
    before = app._stream_tokens
    app._apply_event('text_delta', {'text': 'more'})
    assert app._stream_tokens > before


def test_the_token_rate_follows_the_usage(app):
    app._apply_event('turn_start', {'turn': 1, 'text': 'hi'})
    app._token_samples = [(time.time() - 2.0, 0), (time.time(), 120)]
    assert app._token_rate() == pytest.approx(60.0, abs=2.0)


def test_the_light_follows_the_token_rate(app):
    app.config.footer.sweep_tok_speed = True
    app.config.footer.sweep_period = 1.0
    app.config.footer.sweep_tok_rate = 60.0
    app._token_samples = [(time.time() - 2.0, 0), (time.time(), 120)]
    assert app._sweep_period('answering') == pytest.approx(1.0, rel=0.05)
    app._token_samples = [(time.time() - 2.0, 0), (time.time(), 600)]
    assert app._sweep_period('answering') == pytest.approx(0.2, rel=0.05)


def test_without_a_rate_the_light_moves_slowly(app):
    app.config.footer.sweep_tok_speed = True
    app.config.footer.sweep_period = 1.0
    app.config.footer.sweep_slow_factor = 6.0
    assert app._token_rate() == 0.0
    assert app._sweep_period('answering') == pytest.approx(6.0, rel=0.01)


def test_the_line_breathes_while_connecting(app):
    levels = []
    for step in range(12):
        app.canvas.clear()
        app._draw_connect_breath(60, 9, step * 0.1)
        cells = [app.canvas.cell(x, 9) for x in range(60)]
        levels.append(max(sum(abs(a - b) for a, b in zip(c.bg,
                                                         app.palette.background))
                         for c in cells))
    # It fades in and out instead of staying flat.
    assert max(levels) > min(levels) * 2


def test_the_rate_is_measured_while_the_model_writes(app):
    """The speed of the model, measured from the tokens that stream."""
    app._apply_event('turn_start', {'turn': 1, 'text': 'hi'})
    app._apply_event('text_delta', {'text': 'word ' * 200})
    assert app._stream_rate() > 0
    # The rate of the fake clock: the sample is old enough to be measured.
    app._stream_started = time.time() - 2.0
    assert app._token_rate() == pytest.approx(100.0, rel=0.5)


def test_without_tokens_there_is_no_rate(app):
    app._apply_event('turn_start', {'turn': 1, 'text': 'hi'})
    assert app._token_rate() == 0.0


# ── the wait estimate and the cache hit ────────────────────
def _finished_request(app, elapsed=2.0):
    app._apply_event('status', {'status': 'connecting'})
    app.connect_started = time.time() - elapsed
    app._apply_event('usage', {'input_tokens': 12_000,
                               'cached_input_tokens': 9_000,
                               'output_tokens': 800,
                               'cache_hit_rate': 0.75})


def test_a_finished_request_teaches_the_wait_estimate(app):
    assert app._connect_eta() == ''            # nothing learned yet
    _finished_request(app)
    assert app._last_cache_hit == 0.75
    assert app._predictor.n == 1
    # One sample is not an estimate either; the second one is.
    _finished_request(app)
    assert app._predictor.n == 2
    assert app._connect_eta().startswith(' ~')


def test_the_cache_hit_shows_in_the_extras(app):
    app._apply_event('usage', {'input_tokens': 1_000,
                               'cached_input_tokens': 250,
                               'output_tokens': 100,
                               'cache_hit_rate': 0.25})
    assert app._last_cache_hit == 0.25
    app.canvas.resize(60, 10)
    app.canvas.clear()
    app._draw_footer(60, 9, 0)
    row = ''.join(app.canvas.cell(x, 9).ch for x in range(60))
    assert 'cache 25%' in row


def test_the_estimate_takes_the_learned_sizes(app):
    app._predictor = DurationPredictor(w1=0.0, w2=1.0, w3=0.0, b=1.0, n=5)
    app._last_usage = {'input_tokens': 10_000, 'cached_input_tokens': 0,
                       'output_tokens': 2_000}
    eta = app._connect_eta()
    assert eta == ' ~3.0s'                     # w2 * 2k + b, at the floor
