"""Frame budget and the configurable frame rate."""

from __future__ import annotations

import statistics
import time

from adcode.config import AppConfig, UiSettings
from adcode.session import TranscriptItem


def _frames(app, count=40):
    timings = []
    for _ in range(count):
        started = time.perf_counter()
        app._render()
        timings.append((time.perf_counter() - started) * 1000)
    return timings


def test_a_full_frame_stays_inside_the_60fps_budget(harness):
    """A long transcript must still render well inside 16.7 ms."""
    app = harness.app
    for index in range(60):
        app.standalone_items.append(
            TranscriptItem(kind='user', turn=1, text=f'question {index}'))
        app.standalone_items.append(
            TranscriptItem(kind='assistant', turn=1, text='answer ' * 40))
    app.prompt.set_text('a fairly long prompt line ' * 30)
    time.sleep(0.2)

    timings = _frames(app)
    median = statistics.median(timings)
    assert median < 16.7, f'a frame took {median:.2f} ms'
    # The prompt is wrapped once and reused; wrapping it again every frame
    # was the most expensive part of the frame.
    assert median < 5.0, f'a frame took {median:.2f} ms'


def test_the_prompt_layout_is_reused_between_frames(harness):
    app = harness.app
    app.prompt.set_text('hello world')
    app._render()
    first = app._prompt_cache
    app._render()
    assert app._prompt_cache is first
    app.prompt.set_text('hello there')
    app._render()
    assert app._prompt_cache is not first


def test_the_frame_interval_follows_the_setting(harness):
    app = harness.app
    app.config.ui.fps = 60
    assert app._render_interval() == 1 / 60
    app.config.ui.fps = 30
    assert app._render_interval() == 1 / 30
    app.config.ui.fps = 120
    assert app._render_interval() == 1 / 120


def test_the_frame_rate_is_clamped_in_the_config():
    assert UiSettings.from_dict({}).fps == 60
    assert UiSettings.from_dict({'fps': 24}).fps == 24
    assert UiSettings.from_dict({'fps': 0}).fps == 10   # the minimum
    assert UiSettings.from_dict({'fps': 10_000}).fps == 240


def test_the_frame_rate_is_part_of_the_saved_config(tmp_path):
    from adcode.config import ConfigStore
    path = tmp_path / 'config.json'
    store = ConfigStore(path)
    store.config = AppConfig()
    store.config.ui.fps = 90
    store.save()
    assert ConfigStore(path).load().ui.fps == 90
