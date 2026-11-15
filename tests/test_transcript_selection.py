"""Selecting text in the transcript and copying it."""

from __future__ import annotations

import time

from adcode.session import TranscriptItem


def _transcript(harness, text='hello selection world'):
    app = harness.app
    app.loop.items.append(TranscriptItem(kind='user', turn=1, text=text))
    app.dirty = True
    harness.wait_until(
        lambda: text in harness.screen_text(),
        timeout=5.0, description='the line to reach the screen')
    return app


def test_dragging_selects_the_text_under_the_mouse(harness):
    app = _transcript(harness)
    # The transcript starts below the two header rows.
    harness.drag(3, 3, 14, 3)
    harness.wait_until(
        lambda: app._selection_range() is not None,
        timeout=5.0, description='the selection to start')
    assert 'hello sele' in app._selection_text()


def test_ctrl_c_copies_the_selected_text(harness):
    app = _transcript(harness)
    harness.drag(3, 3, 14, 3)
    harness.wait_until(
        lambda: app._selection_range() is not None,
        timeout=5.0, description='the selection to start')
    harness.press('ctrl+c')
    harness.wait_until(
        lambda: harness.clipboard != '',
        timeout=5.0, description='the selection to reach the clipboard')
    assert 'hello sele' in harness.clipboard


def test_the_footer_shows_the_selection_size(harness):
    app = _transcript(harness)
    harness.drag(3, 3, 14, 3)
    harness.wait_until(
        lambda: 'selected' in harness.screen_text(),
        timeout=5.0, description='the footer to show the selection')


def test_a_click_outside_the_transcript_clears_the_selection(harness):
    app = _transcript(harness)
    harness.drag(3, 3, 14, 3)
    harness.wait_until(
        lambda: app._selection_range() is not None,
        timeout=5.0, description='the selection to start')
    # The footer row is below the transcript.
    harness.click(5, 30)
    harness.wait_until(
        lambda: app._selection_range() is None,
        timeout=5.0, description='the selection to clear')


def test_the_selection_survives_a_scroll(harness):
    app = _transcript(harness)
    harness.drag(3, 3, 14, 3)
    harness.wait_until(
        lambda: app._selection_range() is not None,
        timeout=5.0, description='the selection to start')
    before = app._selection_text()
    app.scroll = 3
    app.dirty = True
    time.sleep(0.3)
    assert app._selection_text() == before
