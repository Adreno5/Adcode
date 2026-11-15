"""The command palette (ctrl+p) and the one key quit (ctrl+q)."""

from __future__ import annotations

import time

from harness.app import AppHarness


def test_ctrl_p_opens_the_command_palette(harness):
    harness.press('ctrl+p')
    harness.wait_until(
        lambda: harness.overlay is not None
        and type(harness.overlay).__name__ == 'CommandPalette',
        timeout=5.0, description='the command palette to open')
    harness.wait_until(
        lambda: 'Command palette' in harness.screen_text(),
        timeout=5.0, description='the palette to be drawn')


def test_typing_narrows_the_command_list(harness):
    harness.press('ctrl+p')
    harness.wait_until(
        lambda: harness.overlay is not None,
        timeout=5.0, description='the command palette to open')
    harness.type_text('goal')
    harness.wait_until(
        lambda: harness.overlay is not None
        and harness.overlay.filter == 'goal',
        timeout=5.0, description='the filter to reach the palette')
    visible = harness.overlay._visible()
    assert visible
    assert all('goal' in item.label.lower() for item in visible)


def test_enter_runs_the_selected_command(harness):
    harness.press('ctrl+p')
    harness.wait_until(
        lambda: harness.overlay is not None,
        timeout=5.0, description='the command palette to open')
    harness.type_text('effort')
    harness.press('enter')
    # /effort without an argument opens its own chooser.
    harness.wait_until(
        lambda: harness.overlay is not None
        and type(harness.overlay).__name__ == 'SelectDialog',
        timeout=5.0, description='the effort chooser to open')


def test_esc_closes_the_palette(harness):
    harness.press('ctrl+p')
    harness.wait_until(
        lambda: harness.overlay is not None,
        timeout=5.0, description='the command palette to open')
    harness.press('esc')
    harness.wait_until(
        lambda: harness.overlay is None,
        timeout=5.0, description='the palette to close')


def test_ctrl_q_quits_the_app(harness):
    harness.press('ctrl+q')
    harness.wait_until(
        lambda: harness.app is not None and not harness.app.running,
        timeout=5.0, description='the app to stop')


def test_ctrl_q_quits_even_with_an_overlay_open(harness):
    harness.press('ctrl+p')
    harness.wait_until(
        lambda: harness.overlay is not None,
        timeout=5.0, description='the command palette to open')
    harness.press('ctrl+q')
    harness.wait_until(
        lambda: harness.app is not None and not harness.app.running,
        timeout=5.0, description='the app to stop')
