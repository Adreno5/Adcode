"""Clipboard behaviour: fallbacks, paste bindings and every input surface."""

import pytest

from adcode.tui import clipboard
from adcode.tui.dialogs import DialogItem, MultiSelectDialog, SelectDialog, TextPrompt
from adcode.tui.input import KeyEvent
from adcode.tui.settings import SettingsOverlay


@pytest.fixture()
def paste_text(monkeypatch):
    """Replace the clipboard with a value the test controls.

    Every surface reads the clipboard through a module level name, so the
    name is patched where it is used.
    """
    holder = {'text': ''}

    def fake() -> str:
        return holder['text']

    import adcode.tui.app as app_module
    import adcode.tui.dialogs as dialogs_module
    import adcode.tui.provider_ui as provider_module
    import adcode.tui.settings as settings_module
    for module in (app_module, dialogs_module, provider_module, settings_module):
        monkeypatch.setattr(module, 'get_clipboard', fake, raising=False)
    return holder


def key(name):
    return KeyEvent(kind='key', key=name)


# the clipboard module
def test_copy_is_kept_for_the_next_paste(monkeypatch):
    """A copy works even when the machine has no clipboard tool."""
    monkeypatch.setattr(clipboard, '_write_pyperclip', lambda text: False)
    monkeypatch.setattr(clipboard, '_write_commands', lambda: [])
    monkeypatch.setattr(clipboard, '_write_tk', lambda text: False)
    monkeypatch.setattr(clipboard, '_read_pyperclip', lambda: None)
    monkeypatch.setattr(clipboard, '_read_commands', lambda: [])
    monkeypatch.setattr(clipboard, '_read_tk', lambda: None)

    assert clipboard.set_clipboard('copied text') is False
    assert clipboard.get_clipboard() == 'copied text'


def test_trailing_newline_is_stripped(monkeypatch):
    """A helper that appends a newline must not submit the message."""
    monkeypatch.setattr(clipboard, '_read_pyperclip', lambda: None)
    monkeypatch.setattr(clipboard, '_read_commands',
                        lambda: [['true'], ['true']])
    monkeypatch.setattr(clipboard, '_run', lambda *args, **kwargs: 'text\n')

    assert clipboard.get_clipboard() == 'text'


def test_system_clipboard_wins_over_the_cache(monkeypatch):
    monkeypatch.setattr(clipboard, '_read_pyperclip', lambda: 'from system\n')
    monkeypatch.setattr(clipboard, '_cached', 'from cache')

    assert clipboard.get_clipboard() == 'from system'


def test_paste_keys_are_recognised():
    assert clipboard.is_paste_key('ctrl+v')
    assert clipboard.is_paste_key('shift+insert')
    assert clipboard.is_paste_key('ctrl+shift+v')
    assert not clipboard.is_paste_key('ctrl+c')


# the prompt
def test_ctrl_v_pastes_into_the_prompt(harness):
    harness.clipboard = 'pasted text'
    harness.press('ctrl+v')
    harness.wait_prompt('pasted text', 11)


def test_shift_insert_pastes_into_the_prompt(harness):
    harness.clipboard = 'terminal paste'
    harness.press('shift+insert')
    harness.wait_prompt('terminal paste', 14)


def test_paste_replaces_the_selection(harness):
    harness.type_text('old text')
    harness.wait_prompt('old text', 8)
    harness.press('home')
    harness.press('shift+right', 'shift+right', 'shift+right')
    harness.wait_until(
        lambda: harness.app is not None
        and harness.app.prompt.selection() == (0, 3),
        timeout=5.0, description='the selection to cover three characters')
    harness.clipboard = 'new'
    harness.press('ctrl+v')
    # the caret sits right behind the pasted text
    harness.wait_prompt('new text', 3)


def test_an_empty_clipboard_attaches_the_image_instead(harness):
    harness.clipboard = ''
    harness.press('ctrl+v')
    harness.wait_until(
        lambda: harness.app is not None and harness.app.prompt.text == '',
        timeout=5.0, description='the prompt to stay empty')


def test_bracketed_paste_still_reaches_the_prompt(harness):
    harness.paste('from the terminal')
    harness.wait_prompt('from the terminal', 17)


# the overlays
def test_ctrl_v_pastes_into_the_text_prompt(paste_text):
    submitted = []
    dialog = TextPrompt('Name', 'Label', '', submitted.append)
    paste_text['text'] = 'pasted value'
    dialog.handle(key('ctrl+v'))
    assert dialog.text == 'pasted value'
    dialog.handle(key('enter'))
    assert submitted == ['pasted value']


def test_ctrl_v_pastes_into_the_select_filter(paste_text):
    dialog = SelectDialog('Pick', [DialogItem(1, 'one')], lambda value: None)
    paste_text['text'] = 'needle'
    dialog.handle(key('ctrl+v'))
    assert dialog.filter == 'needle'


def test_ctrl_v_pastes_into_the_multi_select_filter(paste_text):
    dialog = MultiSelectDialog('Pick', [DialogItem(1, 'one')], lambda value: None)
    paste_text['text'] = 'needle'
    dialog.handle(key('ctrl+v'))
    assert dialog.filter == 'needle'


def test_ctrl_v_pastes_into_the_manual_entry(paste_text):
    dialog = MultiSelectDialog('Pick', [], lambda value: None,
                               allow_manual=True)
    dialog.handle(key('enter'))       # open the manual entry
    paste_text['text'] = 'custom-model'
    dialog.handle(key('ctrl+v'))
    assert dialog.manual_text == 'custom-model'


def test_ctrl_v_pastes_into_the_settings_editor(config, paste_text):
    overlay = SettingsOverlay(config, lambda: None)
    node = overlay._find_node('thinking.label')
    overlay._begin_edit(node, 'Thinking')
    paste_text['text'] = 'Reasoning'
    overlay.handle(key('ctrl+v'))
    assert overlay.edit_value == 'Reasoning'
    overlay.handle(key('enter'))
    assert config.thinking.label == 'Reasoning'


def test_ctrl_v_pastes_into_the_model_editor(paste_text):
    from adcode.config import ModelConfig, ProviderConfig
    from adcode.tui.provider_ui import ModelEditorOverlay

    provider = ProviderConfig(name='P', base_url='http://x',
                              models=[ModelConfig(id='m')])
    editor = ModelEditorOverlay(provider, provider.models[0], lambda: None)
    editor.editing = True
    editor.edit_text = ''
    paste_text['text'] = '4096'
    editor.handle(key('ctrl+v'))
    assert editor.edit_text == '4096'
