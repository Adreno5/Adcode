"""^H opens the saved sessions and the bindings panel rebinds chords live."""

from __future__ import annotations

from adcode.session import SessionPayload


def test_ctrl_h_lists_saved_sessions(harness):
    harness.app.session_store.save(SessionPayload(name='alpha', messages=[
        {'role': 'user', 'content': 'hello'},
    ]))
    harness.press('ctrl+h')
    harness.wait_until(
        lambda: harness.overlay is not None
        and type(harness.overlay).__name__ == 'SelectDialog',
        timeout=5.0, description='the sessions browser to open')
    harness.wait_for_text('alpha')


def test_ctrl_h_without_sessions_says_so(harness):
    harness.press('ctrl+h')
    harness.wait_for_text('No saved sessions.')
    assert harness.overlay is None


def test_keys_panel_rebinds_quit_to_ctrl_e(harness):
    harness.run_command('/keys')
    harness.wait_until(
        lambda: harness.overlay is not None
        and type(harness.overlay).__name__ == 'KeybindingsPanel',
        timeout=5.0, description='the bindings panel to open')
    # The first row is quit; capture a new chord for it and close the panel.
    harness.press('enter')
    harness.wait_until(
        lambda: harness.overlay is not None and harness.overlay.capture_mode,
        timeout=5.0, description='the panel to capture a key')
    harness.press('ctrl+e')
    harness.press('esc')
    harness.wait_until(
        lambda: harness.overlay is None,
        timeout=5.0, description='the panel to close')
    harness.press('ctrl+e')
    harness.wait_until(
        lambda: harness.app is not None and not harness.app.running,
        timeout=5.0, description='the app to quit with the new chord')
