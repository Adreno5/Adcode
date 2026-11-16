"""Temporary-config behaviour driven through the real command and settings UI."""

from __future__ import annotations

from pathlib import Path


def _model_on_disk(harness, model_id):
    for provider in harness.workspace.config_json().get('providers', []):
        for model in provider.get('models', []):
            if model.get('id') == model_id:
                return model
    return None


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


def test_settings_panel_keeps_the_cursor_on_the_edited_item(harness):
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
    harness.press('enter')

    # Choosing an option with no children collapses the item; the cursor has
    # to stay on that item instead of falling back to the top of the tree.
    harness.wait_until(
        lambda: getattr(harness.overlay, 'row_key', '') == 'ui.theme',
        timeout=5.0, description='the cursor to stay on the Theme row')
    harness.press('down')
    harness.wait_until(
        lambda: getattr(harness.overlay, 'row_key', '') == 'ui.accent',
        timeout=5.0, description='the next row after Theme to be selected')

    harness.press('esc')
    harness.wait_until(lambda: harness.overlay is None, timeout=5.0,
                       description='the settings overlay to close')


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

def test_a_setting_is_persisted_without_pressing_ctrl_s(harness):
    # The panel writes the file as soon as a value changes; there is
    # nothing to save by hand.
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
    harness.press('enter')
    harness.wait_until(
        lambda: harness.workspace.config_json()['ui']['theme'] == 'light',
        timeout=10.0,
        description='the theme to be persisted without ctrl+s')
    # The panel is still open: the change was saved, not committed.
    assert harness.overlay is not None
    harness.press('esc')
    harness.wait_until(lambda: harness.overlay is None, timeout=5.0,
                       description='the settings overlay to close')
    assert harness.workspace.config_json()['ui']['theme'] == 'light'

def test_a_model_edit_is_persisted_without_pressing_ctrl_s(harness):
    # The model editor writes the file as soon as a value changes, too.
    harness.run_command('/model edit')
    harness.wait_until(lambda: harness.overlay is not None, timeout=10.0,
                       description='the model editor')
    editor = harness.overlay
    assert type(editor).__name__ == 'ModelEditorOverlay'
    model_id = editor.model.id
    editor.index = 2                       # the "Vision input" toggle
    harness.press('enter')
    harness.wait_until(
        lambda: _model_on_disk(harness, model_id) is not None
        and _model_on_disk(harness, model_id).get('vision')
        == editor.model.vision,
        timeout=10.0,
        description='the model to be persisted without ctrl+s')
    # The panel is still open: the change was saved, not committed.
    assert harness.overlay is not None
