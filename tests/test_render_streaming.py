"""Frame-streaming behaviour of the real app: partial flush, zero idle cost.

These drive the production ``AdCodeApp`` on the simulated terminal, so they
exercise the whole path (widgets -> canvas -> diff -> flush) rather than the
renderer in isolation.
"""

from __future__ import annotations

import pytest

import adcode.tui.app as app_module
from bench.script import ScriptedResponder, Turn
from tests.test_render_equivalence import LegacyCanvas, grid


def _bytes_written(harness) -> int:
    frame = harness.screen.latest
    return len(frame.raw) if frame is not None else 0


def _frame_count(harness) -> int:
    return len(harness.screen.frames)


def _force_render(harness, timeout: float = 5.0) -> None:
    harness.app.dirty = True
    harness.wait_until(lambda: not harness.app.dirty, timeout=timeout,
                       description='a render pass')


def test_an_idle_frame_writes_nothing(harness):
    harness.wait_rendered()
    harness.wait_idle()
    _force_render(harness)
    frames = _frame_count(harness)
    for _ in range(3):
        _force_render(harness)
    # A flush is only issued when the frame carries bytes; an unchanged UI
    # produces no frame at all.
    assert _frame_count(harness) == frames


def test_typing_repaints_far_less_than_a_full_frame(harness):
    harness.wait_rendered()
    harness.wait_idle()
    full_frame = _bytes_written(harness)
    assert full_frame > 1_000, full_frame

    harness.type_text('hello')
    harness.wait_prompt('hello')
    _force_render(harness)
    typed = _bytes_written(harness)
    assert typed < full_frame / 2, f'{typed} bytes vs a {full_frame}-byte frame'
    assert 'hello' in harness.screen_text()


def test_scrolling_keeps_the_screen_correct(harness):
    harness.wait_rendered()
    for index in range(6):
        harness.type_text(f'line {index}\n')
    harness.wait_rendered()
    frame = harness.screen.latest
    assert frame is not None
    assert not frame.vt_mismatch, 'incremental payload must match the canvas'
    assert frame.vt_text == frame.text()


def test_same_scenario_renders_identically_to_the_old_renderer(harness_factory,
                                                               monkeypatch):
    """A/B against the pre-refactor canvas on a real typed interaction."""
    script = [
        Turn(thinking='Think briefly.', text='Answer with **markup** and 中文.'),
    ]

    def run() -> tuple[str, list[str]]:
        harness = harness_factory()
        harness.gateway.responder = ScriptedResponder(list(script))
        harness.wait_rendered()
        harness.submit('hello there')
        harness.wait_idle(20.0)
        harness.wait_for_text('Answer with markup and 中文.')
        return harness.screen_text(), [text for _, text in harness.transcript_texts()]

    monkeypatch.setattr(app_module, 'Canvas', LegacyCanvas)
    legacy_text, legacy_items = run()
    monkeypatch.undo()
    new_text, new_items = run()

    assert new_text == legacy_text
    assert new_items == legacy_items
