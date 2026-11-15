"""Regression tests: a modal overlay owns every keyboard and mouse event.

The pass-through bug had two halves.  The input handler checked for a wheel
event before it checked for an overlay, so scrolling over a dialog silently
moved the transcript underneath it.  And keys were handled one per frame, so
keys typed at a dialog could still be queued when the dialog closed and then
land in the prompt behind it -- or submit a message.
"""

from __future__ import annotations

import pytest

from adcode.config import ModelConfig, ProviderConfig
from adcode.session import TranscriptItem
from adcode.tui.app import AdCodeApp
from adcode.tui.dialogs import (
    DialogItem, MessageDialog, MultiSelectDialog, SelectDialog,
)
from adcode.tui.input import KeyEvent
from adcode.tui.provider_ui import ModelEditorOverlay
from adcode.tui.settings import SettingsOverlay


def wheel(action: str = 'wheel_up') -> KeyEvent:
    return KeyEvent(kind='mouse', x=10, y=5, action=action, button=64)


def key(name: str) -> KeyEvent:
    return KeyEvent(kind='key', key=name)


def text(value: str) -> KeyEvent:
    return KeyEvent(kind='text', text=value)


@pytest.fixture()
def app(tmp_path):
    instance = AdCodeApp(cwd=str(tmp_path), resume=False,
                         config_path=str(tmp_path / 'config.json'))
    # Long enough that any stray wheel or page key can scroll it.
    instance.standalone_items = [
        TranscriptItem(kind='notice', text=f'row {index} of a long transcript')
        for index in range(60)
    ]
    try:
        yield instance
    finally:
        instance.input_thread.stop()
        instance.store.save()


# ── the page still works when nothing is modal ─────────────
def test_wheel_and_page_keys_scroll_without_a_modal(app):
    app._handle_event(wheel('wheel_up'))
    assert app.scroll > 0 and not app.follow

    app._handle_event(wheel('wheel_down'))
    assert app.scroll == 0 and app.follow

    app._handle_event(key('pgup'))
    assert app.scroll > 0


# ── the wheel belongs to the modal ─────────────────────────
def test_wheel_never_scrolls_the_page_behind_a_modal(app):
    dialog = MessageDialog('Info', 'a modal owns the wheel')
    app.open_overlay(dialog)
    for _ in range(3):
        app._handle_event(wheel('wheel_up'))
    app._handle_event(wheel('wheel_down'))
    assert app.scroll == 0
    assert app.follow
    assert app.overlay is dialog


def test_page_keys_never_scroll_the_page_behind_a_modal(app):
    dialog = MessageDialog('Info', 'a modal owns the page keys')
    app.open_overlay(dialog)
    app._handle_event(key('pgup'))
    app._handle_event(key('pgdn'))
    assert app.scroll == 0
    assert app.follow


def test_wheel_moves_a_select_dialog_list():
    items = [DialogItem(index, f'item {index}') for index in range(5)]
    picked = []
    dialog = SelectDialog('Pick', items, picked.append)
    dialog.handle(wheel('wheel_down'))
    assert dialog.index == 1
    dialog.handle(wheel('wheel_up'))
    dialog.handle(wheel('wheel_up'))
    assert dialog.index == 4        # wraps like the arrow keys
    assert not dialog.closed
    dialog.handle(key('enter'))
    assert picked == [4]


def test_wheel_moves_a_multiselect_dialog():
    selected = []
    items = [DialogItem('a', 'A'), DialogItem('b', 'B'), DialogItem('c', 'C')]
    dialog = MultiSelectDialog('Pick', items, selected.append)
    dialog.handle(wheel('wheel_down'))
    dialog.handle(wheel('wheel_down'))
    dialog.handle(wheel('wheel_up'))
    assert dialog.index == 1
    dialog.handle(key('enter'))
    assert selected == [[]]         # moving alone never toggles anything


def test_wheel_walks_the_settings_tree(app):
    overlay = SettingsOverlay(app.config, lambda: None)
    app.open_overlay(overlay)
    rows = overlay._flatten()
    selectable = overlay._selectable(rows)
    app._handle_event(wheel('wheel_down'))
    top = overlay.row
    assert top == selectable[0]
    app._handle_event(wheel('wheel_down'))
    assert overlay.row == selectable[1]
    app._handle_event(wheel('wheel_up'))
    assert overlay.row == top


def test_wheel_walks_the_model_editor():
    provider = ProviderConfig(name='P', base_url='http://x',
                              models=[ModelConfig(id='m')])
    editor = ModelEditorOverlay(provider, provider.models[0], lambda: None)
    editor.handle(wheel('wheel_down'))
    assert editor.index == 1
    editor.handle(wheel('wheel_up'))
    assert editor.index == 0


# ── the keyboard belongs to the modal ──────────────────────
def test_typing_at_a_modal_never_reaches_the_prompt(app):
    dialog = MessageDialog('Info', 'modal')
    app.open_overlay(dialog)
    for char in 'hello':
        app._handle_event(text(char))
    app._handle_event(key('enter'))
    assert app.prompt.text == ''
    assert app.overlay is None


def test_background_dismissal_drops_input_aimed_at_the_spinner(app):
    """A modal that closes itself must not hand queued keys to the prompt."""
    dialog = MessageDialog('Loading models', 'fetching…')
    app.open_overlay(dialog)
    # Pressed while the spinner was on screen, still queued when it closed.
    app.input_queue.put(text('x'))
    app.input_queue.put(key('enter'))
    app.dismiss_overlay(dialog, drop_input=True)
    assert app.overlay is None
    assert app.input_queue.empty()
    assert app.prompt.text == ''


def test_closing_a_modal_with_a_key_keeps_input_typed_after_it(app):
    """Esc closes the dialog; what follows belongs to the prompt, in order."""
    dialog = MessageDialog('Info', 'modal')
    app.open_overlay(dialog)
    app.input_queue.put(key('esc'))
    app.input_queue.put(text('h'))
    app._dispatch_input(None)
    assert app.overlay is None
    assert app.prompt.text == 'h'


def test_dismiss_only_closes_the_overlay_it_was_given(app):
    first = MessageDialog('one', 'a')
    second = MessageDialog('two', 'b')
    app.open_overlay(first)
    app.open_overlay(second)
    app.dismiss_overlay(first)      # a stale reference must not close `second`
    assert app.overlay is second
    app.dismiss_overlay(second)
    assert app.overlay is None


def test_replacing_a_modal_from_its_callback_keeps_the_new_one(app):
    """A dialog that opens the next dialog is not cleared by its own close."""
    holder = {}

    def chain(value: object) -> None:
        holder['next'] = MessageDialog('next', 'b')
        app.open_overlay(holder['next'])

    dialog = SelectDialog('Pick', [DialogItem(1, 'one')], chain)
    app.open_overlay(dialog)
    app._handle_event(key('enter'))
    assert app.overlay is holder['next']
    assert app.overlay is not None and not app.overlay.closed
