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
    # Keep the session store, goals and image cache out of the real ~/.adcode
    # -- and out of the project: the state never lands inside the project
    # the app is working on, so it goes next to it.
    paths_module.set_app_dir(tmp_path.parent / f'{tmp_path.name}-adcode-home')
    instance = AdCodeApp(cwd=str(tmp_path), resume=False,
                         config_path=str(tmp_path / 'config.json'))
    try:
        yield instance
    finally:
        instance.input_thread.stop()
        paths_module.set_app_dir(None)


class StubLoop:
    """Stands in for the agent loop: records what the app would send."""

    def __init__(self):
        self.sent = []

    def submit(self, text, mode):
        self.sent.append((text, mode))

def test_a_message_that_starts_with_a_path_is_sent_to_the_model(app):
    app.loop = StubLoop()
    app.prompt.set_text('/home/me/notes.md explain this')
    app._handle_enter(key('enter'))
    assert app.loop.sent == [('/home/me/notes.md explain this', 'chat')]
    assert app.prompt.text == ''
    assert not [item for item in app.standalone_items
                if item.kind == 'notice' and 'Unknown command' in item.text]

def test_a_registered_name_still_runs_the_command(app):
    app.loop = StubLoop()
    app.submit('/help')
    assert app.loop.sent == []
    assert type(app.overlay).__name__ == 'MessageDialog'

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


def test_ctrl_comma_opens_the_settings_browser(app):
    app._handle_event(key('ctrl+,'))
    assert type(app.overlay).__name__ == 'SettingsOverlay'


def test_ctrl_comma_is_rebindable(app):
    app.config.keys.config = 'ctrl+g'
    app.config_changed()
    app._handle_event(key('ctrl+,'))
    assert app.overlay is None
    app._handle_event(key('ctrl+g'))
    assert type(app.overlay).__name__ == 'SettingsOverlay'


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


def test_alt_c_copies_latest_assistant_reply(app, monkeypatch):
    from adcode.tui import app as app_module
    from adcode.session import TranscriptItem

    copied = {}
    monkeypatch.setattr(app_module, 'set_clipboard',
                        lambda text: copied.setdefault('text', text) or True)
    app.standalone_items[:] = [
        TranscriptItem(kind='assistant', turn=1, text='# first\nold'),
        TranscriptItem(kind='tool', turn=2, text='a tool call'),
        TranscriptItem(kind='assistant', turn=2, text='# latest\n**markdown**'),
    ]
    app._handle_event(key('alt+c'))
    assert copied['text'] == '# latest\n**markdown**'
    assert 'copied last reply' in app.status_hint


def test_alt_c_uses_live_answer_when_no_committed_reply(app, monkeypatch):
    from adcode.tui import app as app_module

    copied = {}
    monkeypatch.setattr(app_module, 'set_clipboard',
                        lambda text: copied.setdefault('text', text) or True)
    app.standalone_items[:] = []
    app.live_answer_text = 'streaming **answer**'
    app._handle_event(key('alt+c'))
    assert copied['text'] == 'streaming **answer**'


def modifier(name, action):
    return KeyEvent(kind='modifier', key=name, action=action)


def test_holding_alt_reveals_details_and_release_restores(app, monkeypatch):
    config_before = app.config
    # A normal, non-revealing render uses the live config object.
    app._handle_event(modifier('alt', 'press'))
    assert app._alt_pressed_at is not None
    assert app._alt_details_shown is False

    # Move the press into the past so the hold threshold is crossed.
    app._alt_pressed_at -= app._alt_reveal_seconds + 0.1
    app._tick_alt_reveal()
    assert app._alt_details_shown is True

    reveal_cfg, version = app._reveal_config()
    assert reveal_cfg is not config_before
    assert reveal_cfg.thinking.mode == 'full'
    assert reveal_cfg.thinking.after_completion == 'keep'
    assert reveal_cfg.tools.mode == 'both'
    assert reveal_cfg.tools.chain == 'separate'
    assert reveal_cfg.tools.result_mode == 'full'
    # The live config is untouched, so nothing is persisted.
    assert config_before.thinking.mode == app.config.thinking.mode

    app._handle_event(modifier('alt', 'release'))
    assert app._alt_pressed_at is None
    assert app._alt_details_shown is False


def test_reveal_config_renders_the_transcript(app):
    from adcode.session import TranscriptItem

    app.standalone_items[:] = [
        TranscriptItem(kind='assistant', turn=1, text='hello there'),
    ]
    app._alt_details_shown = True
    # Rendering with the swapped-in config must work: it is a throwaway
    # dataclass replacement, not a mutated live config.
    lines = app._transcript_lines()
    assert any('hello there' in ''.join(span.text for span in line)
               for line in lines)
    app._alt_details_shown = False
    assert app._transcript_lines()


def test_quick_alt_tap_does_not_reveal(app):
    app._handle_event(modifier('alt', 'press'))
    app._tick_alt_reveal()
    assert app._alt_details_shown is False
