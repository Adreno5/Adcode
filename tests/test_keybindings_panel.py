"""The bindings panel: rebinds land in the settings and notify the app."""

from adcode.config import KeysSettings
from adcode.tui.canvas import Canvas
from adcode.tui.input import KeyEvent
from adcode.tui.keybindings_panel import KeybindingsPanel
from adcode.tui.keymap import BINDINGS


def key(name):
    return KeyEvent(kind='key', key=name)


def text(value):
    return KeyEvent(kind='text', text=value)


def make_panel():
    settings = KeysSettings()
    changes = []
    panel = KeybindingsPanel(settings, lambda: changes.append(1))
    return panel, settings, changes


def select(panel, action):
    for index, row in enumerate(panel.rows):
        if row.kind == 'binding' and row.binding.action == action:
            panel.index = index
            return
    raise AssertionError(f'no row for {action}')


def test_the_panel_lists_every_registered_action():
    panel, _, _ = make_panel()
    rows = panel.rows
    actions = [row.binding.action for row in rows if row.kind == 'binding']
    headers = [row.group for row in rows if row.kind == 'header']
    assert actions == [item.action for item in BINDINGS]
    assert headers == ['General', 'Clipboard', 'Editing', 'Navigation']
    assert len(rows) == len(BINDINGS) + len(headers)


def test_rebind_applies_and_notifies():
    panel, settings, changes = make_panel()
    select(panel, 'palette')
    panel.handle(key('enter'))
    assert panel.capture_mode
    panel.handle(key('ctrl+e'))
    assert settings.palette == 'ctrl+e'
    assert changes == [1]
    assert not panel.capture_mode


def test_esc_cancels_the_capture():
    panel, settings, _ = make_panel()
    select(panel, 'palette')
    panel.handle(key('enter'))
    panel.handle(key('esc'))
    assert not panel.capture_mode
    assert settings.palette == 'ctrl+p'


def test_capture_rejects_typing_and_reserved_keys():
    panel, settings, changes = make_panel()
    select(panel, 'palette')
    panel.handle(key('enter'))
    panel.handle(text('a'))
    assert panel.capture_mode
    assert settings.palette == 'ctrl+p'
    assert 'reserved for typing' in panel.message
    panel.handle(key('enter'))
    assert panel.capture_mode
    assert 'fixed gesture' in panel.message
    assert changes == []
    panel.handle(key('ctrl+e'))
    assert settings.palette == 'ctrl+e'
    assert changes == [1]


def test_a_taken_over_key_unbinds_the_previous_holder():
    panel, settings, changes = make_panel()
    select(panel, 'palette')
    panel.handle(key('enter'))
    panel.handle(key('ctrl+h'))            # sessions holds ctrl+h
    assert panel.mode == 'conflict'
    panel.handle(key('esc'))               # backing out changes nothing
    assert settings.palette == 'ctrl+p'
    assert settings.sessions == 'ctrl+h'
    panel.handle(key('enter'))
    panel.handle(key('ctrl+h'))
    panel.handle(key('enter'))             # take it over
    assert settings.palette == 'ctrl+h'
    assert settings.sessions == 'none'
    assert panel.message == 'Sessions is now unbound'
    assert changes == [1]


def test_reset_restores_the_default_key():
    panel, settings, _ = make_panel()
    select(panel, 'palette')
    panel.handle(key('enter'))
    panel.handle(key('ctrl+e'))
    panel.handle(text('r'))
    assert settings.palette == 'ctrl+p'


def test_reset_unbinds_the_action_holding_the_default():
    panel, settings, _ = make_panel()
    settings.palette = 'ctrl+z'            # undo's default, by hand
    settings.undo = 'none'
    select(panel, 'undo')
    panel.handle(text('r'))
    assert settings.undo == 'ctrl+z'
    assert settings.palette == 'none'


def test_list_keys_scroll_and_close():
    panel, _, _ = make_panel()
    first = panel.index
    panel.handle(key('down'))
    assert panel.index != first
    panel.handle(key('pgdn'))
    panel.handle(key('end'))
    assert panel.index == panel._selectable()[-1]
    panel.handle(key('home'))
    assert panel.index == panel._selectable()[0]
    panel.handle(key('esc'))
    assert panel.closed


def test_render_draws_the_rows(palette):
    panel, _, _ = make_panel()
    canvas = Canvas(100, 30, palette)
    panel.render(canvas, 100, 30, palette, 0.0)
    select(panel, 'palette')
    panel.handle(key('enter'))
    panel.render(canvas, 100, 30, palette, 0.0)
    panel.handle(key('ctrl+h'))
    panel.render(canvas, 100, 30, palette, 0.0)
