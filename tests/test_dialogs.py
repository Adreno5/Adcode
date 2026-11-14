from adcode.config import ModelConfig, ProviderConfig
from adcode.theme import build_palette
from adcode.tui.canvas import Canvas
from adcode.tui.dialogs import (
    ConfirmDialog, DialogItem, MessageDialog, MultiSelectDialog, SelectDialog,
    TextPrompt,
)
from adcode.tui.input import KeyEvent
from adcode.tui.provider_ui import ModelEditorOverlay


def canvas():
    return Canvas(100, 30, build_palette('dark'))


def render(overlay):
    surface = canvas()
    overlay.render(surface, 100, 30, surface.palette, 0.0)
    frame = surface.render()
    assert frame.startswith('\x1b[H')
    return frame


def test_message_dialog_render_and_close():
    dialog = MessageDialog('Info', '**Bold** and `code`\n\nsecond line')
    render(dialog)
    dialog.handle(KeyEvent(kind='key', key='enter'))
    assert dialog.closed


def test_confirm_dialog():
    results = []
    dialog = ConfirmDialog('Sure?', 'Remove it.', results.append)
    render(dialog)
    dialog.handle(KeyEvent(kind='key', key='y'))
    assert results == [True]
    dialog = ConfirmDialog('Sure?', 'Remove it.', results.append)
    dialog.handle(KeyEvent(kind='key', key='n'))
    assert results == [True, False]


def test_select_dialog_filter_navigation_and_select():
    selected = []
    items = [DialogItem(index, f'item {index}', 'desc') for index in range(20)]
    dialog = SelectDialog('Pick', items, selected.append)
    render(dialog)
    dialog.handle(KeyEvent(kind='key', key='down'))
    dialog.handle(KeyEvent(kind='key', key='down'))
    dialog.handle(KeyEvent(kind='key', key='enter'))
    assert selected == [2]
    selected.clear()
    dialog = SelectDialog('Pick', items, selected.append)
    for char in '1':
        dialog.handle(KeyEvent(kind='text', text=char))
    dialog.handle(KeyEvent(kind='key', key='enter'))
    assert selected == [1]


def test_multiselect_toggle_and_confirm():
    submitted = []
    items = [DialogItem('a', 'A'), DialogItem('b', 'B')]
    dialog = MultiSelectDialog('Pick', items, submitted.append)
    render(dialog)
    dialog.handle(KeyEvent(kind='text', text=' '))
    dialog.handle(KeyEvent(kind='key', key='down'))
    dialog.handle(KeyEvent(kind='text', text=' '))
    dialog.handle(KeyEvent(kind='key', key='enter'))
    assert submitted == [['a', 'b']]


def test_multiselect_manual_entry():
    submitted = []
    dialog = MultiSelectDialog('Pick', [], submitted.append, allow_manual=True)
    render(dialog)
    dialog.handle(KeyEvent(kind='key', key='enter'))  # open manual input
    for char in 'custom-model':
        dialog.handle(KeyEvent(kind='text', text=char))
    dialog.handle(KeyEvent(kind='key', key='enter'))
    dialog.handle(KeyEvent(kind='key', key='enter'))
    assert submitted == [['custom-model']]


def test_text_prompt_editing_and_submit():
    submitted = []
    dialog = TextPrompt('Name', 'Label', 'abc', submitted.append)
    render(dialog)
    dialog.handle(KeyEvent(kind='key', key='left'))
    dialog.handle(KeyEvent(kind='text', text='X'))
    dialog.handle(KeyEvent(kind='key', key='enter'))
    assert submitted == ['abXc']


def test_model_editor_toggle_and_render():
    provider = ProviderConfig(name='P', base_url='http://x',
                              models=[ModelConfig(id='m')])
    model = provider.models[0]
    saved = []
    editor = ModelEditorOverlay(provider, model, lambda: saved.append(1))
    render(editor)
    editor.handle(KeyEvent(kind='key', key='down'))
    editor.handle(KeyEvent(kind='key', key='down'))
    editor.handle(KeyEvent(kind='key', key='enter'))  # vision toggle
    assert model.vision is True
    editor.handle(KeyEvent(kind='key', key='ctrl+s'))
    assert saved == [1]
