"""Temporary-config behaviour driven through the real command and settings UI."""

from __future__ import annotations

from pathlib import Path


def test_effort_command_persists_and_shapes_the_next_request(harness):
    harness.run_command('/effort high')
    harness.wait_for_text('Reasoning effort: high')
    harness.wait_until(
        lambda: harness.workspace.config_json().get('reasoning', {}).get(
            'effort') == 'high',
        timeout=10.0, description='the effort to be persisted in temp config')
    assert harness.app is not None
    assert harness.app.config.reasoning.effort == 'high'

    harness.submit('shape the next request')
    assert harness.requests
    assert harness.requests[-1].reasoning_effort == 'high'
    assert str(harness.workspace.config_path).startswith(
        str(harness.workspace.root))
    assert harness.workspace.config_path != Path.home() / '.adcode' / 'config.json'


def test_settings_panel_switches_the_theme_in_the_temp_config(harness):
    harness.run_command('/config')
    harness.wait_until(lambda: harness.overlay is not None, timeout=10.0,
                       description='the settings overlay')
    harness.wait_for_text('Appearance')

    harness.press('down')
    harness.wait_until(
        lambda: harness.overlay is not None
        and getattr(harness.overlay, 'row_key', '') == 'ui.theme',
        timeout=5.0, description='the Theme row to be selected')
    harness.press('enter')
    harness.wait_until(
        lambda: 'ui.theme' in getattr(harness.overlay, 'expanded', []),
        timeout=5.0, description='the Theme row to expand')
    harness.press('down')
    harness.press('down')
    harness.press('enter')

    harness.wait_until(
        lambda: harness.app is not None
        and harness.app.config.ui.theme == 'light',
        timeout=10.0, description='the light theme to be selected')
    harness.wait_until(
        lambda: harness.app is not None and harness.app.palette.name == 'light'
        and not harness.app.dirty,
        timeout=10.0, description='the light frame to render')
    assert harness.workspace.config_json()['ui']['theme'] == 'light'

    harness.press('ctrl+s')
    harness.wait_until(
        lambda: harness.overlay is None and not harness.app.dirty,
        timeout=10.0, description='the settings overlay to close')
    assert harness.app.palette.name == 'light'
    assert harness.workspace.config_json()['ui']['theme'] == 'light'
    frame = harness.screen.latest
    assert frame is not None
    assert frame.cells(5)[0].bg == harness.app.palette.background
    assert not frame.vt_mismatch


def test_escape_restores_the_values_captured_when_an_item_expanded(harness):
    harness.run_command('/config')
    harness.wait_until(lambda: harness.overlay is not None, timeout=10.0,
                       description='the settings overlay')
    harness.press('down')
    harness.wait_until(
        lambda: getattr(harness.overlay, 'row_key', '') == 'ui.theme',
        timeout=5.0, description='the Theme row to be selected')
    harness.press('enter')
    harness.wait_until(
        lambda: 'ui.theme' in getattr(harness.overlay, 'expanded', []),
        timeout=5.0, description='the Theme row to expand')
    harness.press('down')
    harness.press('down')
    harness.wait_until(
        lambda: getattr(harness.overlay, 'row_key', '') == 'ui.theme::light',
        timeout=5.0, description='the Light option to be highlighted')
    assert harness.app is not None
    assert harness.app.config.ui.theme == 'dark'

    # This Esc is not closing the overlay: it cancels the expanded Theme item
    # and restores the snapshot taken when it was opened.
    harness.press('esc')
    harness.wait_until(
        lambda: harness.overlay is not None
        and getattr(harness.overlay, 'expanded', []) == [],
        timeout=5.0, description='the Theme item to collapse')
    assert harness.app is not None
    assert harness.app.config.ui.theme == 'dark'
    assert harness.workspace.config_json()['ui']['theme'] == 'dark'

    harness.press('esc')
    harness.wait_until(lambda: harness.overlay is None, timeout=5.0,
                       description='the settings overlay to close')
