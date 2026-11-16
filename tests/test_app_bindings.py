"""App-level dispatch: bindings resolve through the keymap and ^H opens the
sessions browser; the panel's rebinds reach the dispatcher immediately."""

from __future__ import annotations

import pytest

from adcode.session import SessionPayload
from adcode.tui.app import AdCodeApp
from adcode.tui.input import KeyEvent


def key(name):
    return KeyEvent(kind='key', key=name)


@pytest.fixture()
def app(tmp_path):
    from adcode import paths as paths_module
    # Keep the session store, goals and image cache out of the real ~/.adcode.
    paths_module.set_app_dir(tmp_path / 'adcode-home')
    instance = AdCodeApp(cwd=str(tmp_path), resume=False,
                         config_path=str(tmp_path / 'config.json'))
    try:
        yield instance
    finally:
        instance.input_thread.stop()
        paths_module.set_app_dir(None)


def test_ctrl_h_opens_the_sessions_browser(app):
    app.session_store.save(SessionPayload(name='alpha', messages=[
        {'role': 'user', 'content': 'hello'},
    ]))
    app._handle_event(key('ctrl+h'))
    assert type(app.overlay).__name__ == 'SelectDialog'
    assert 'alpha' in [item.label for item in app.overlay.items]


def test_ctrl_h_without_sessions_notifies(app):
    app._handle_event(key('ctrl+h'))
    assert app.overlay is None
    assert any('No saved sessions' in item.text
               for item in app.standalone_items if item.kind == 'notice')


def test_rebind_through_the_panel_takes_effect_immediately(app):
    app.open_keybindings()
    panel = app.overlay
    panel.index = panel._selectable()[0]        # quit is the first row
    panel.handle(key('enter'))
    panel.handle(key('ctrl+e'))
    # The panel notified the app: the config is saved and the keymap rebuilt.
    assert app.keymap.key_of('quit') == 'ctrl+e'
    reloaded = app.store.load()
    assert reloaded.keys.quit == 'ctrl+e'
    app.dismiss_overlay()
    app._handle_event(key('ctrl+e'))
    assert app.running is False                 # the new quit chord works


def test_the_old_quit_chord_becomes_inert(app):
    app.config.keys.quit = 'ctrl+e'
    app.config_changed()
    app._handle_event(key('ctrl+q'))
    assert app.running is True


def test_quit_is_suppressed_while_the_panel_captures(app):
    app.open_keybindings()
    panel = app.overlay
    panel.handle(key('enter'))                  # capture a key for quit
    app._handle_event(key('ctrl+q'))
    assert app.running is True                  # routed into the panel
    assert panel.capture_mode is False          # it applied as a rebind
    assert app.config.keys.quit == 'ctrl+q'
    panel.handle(key('esc'))                    # close the panel
    app._handle_event(key('ctrl+q'))
    assert app.running is False


def test_shift_aliases_keep_selection_behind_a_rebind(app):
    app.config.keys.move_left = 'ctrl+s'
    app.config_changed()
    app.prompt.set_text('word')
    app.prompt.cursor = len(app.prompt.text)
    app._handle_event(KeyEvent(kind='key', key='shift+left', shift=True))
    assert app.prompt.anchor is not None
