from adcode.tui.input import KeyEvent
from adcode.tui.settings import SettingsOverlay, build_settings_tree


def key(name):
    return KeyEvent(kind='key', key=name)


def text(value):
    return KeyEvent(kind='text', text=value)


def find_row(overlay, key_name):
    rows = overlay._flatten()
    for index, row in enumerate(rows):
        if row.key == key_name:
            return index, row
    raise AssertionError(f'row not found: {key_name}')


def focus(overlay, key_name):
    index, _ = find_row(overlay, key_name)
    overlay.row = index
    overlay.row_key = key_name


def walk_to(overlay, key_name):
    """Reach a row the way a user does: with the down key only."""
    for _ in range(500):
        rows = overlay._flatten()
        if 0 <= overlay.row < len(rows) and rows[overlay.row].key == key_name:
            return
        overlay.handle(key('down'))
    raise AssertionError(f'could not walk to {key_name}')


def current(overlay):
    rows = overlay._flatten()
    return rows[overlay.row].key


def test_tree_contains_all_groups(config):
    tree = build_settings_tree(config)
    groups = [node.label for node in tree if node.kind == 'group']
    assert groups == ['Appearance', 'Work steps', 'Footer status', 'Hold',
                      'Failover', 'Reasoning', 'Context',
                      'Sub-agents and teams', 'Skills', 'MCP']


def test_expand_and_choose(config):
    overlay = SettingsOverlay(config, lambda: None)
    focus(overlay, 'thinking.mode')
    overlay.handle(key('enter'))
    index, _ = find_row(overlay, 'thinking.mode::partial')
    overlay.row = index
    overlay.row_key = 'thinking.mode::partial'
    overlay.handle(key('enter'))
    assert config.thinking.mode == 'partial'
    # Child settings of the selected option are visible.
    keys = [row.key for row in overlay._flatten()]
    assert 'thinking.partial.from' in keys
    assert 'thinking.partial.to' in keys
    # Escape reverts the whole item.
    overlay.handle(key('esc'))
    assert config.thinking.mode == 'label'
    assert overlay.expanded == []


def test_enter_on_title_confirms(config):
    overlay = SettingsOverlay(config, lambda: None)
    focus(overlay, 'tools.mode')
    overlay.handle(key('enter'))
    index, _ = find_row(overlay, 'tools.mode::hidden')
    overlay.row = index
    overlay.row_key = 'tools.mode::hidden'
    overlay.handle(key('enter'))
    assert config.tools.mode == 'hidden'
    assert overlay.expanded == []  # childless option confirms and collapses


def test_nested_animation_children(config):
    overlay = SettingsOverlay(config, lambda: None)
    focus(overlay, 'thinking.mode')
    overlay.handle(key('enter'))
    focus(overlay, 'thinking.animation')
    overlay.handle(key('enter'))
    keys = [row.key for row in overlay._flatten()]
    assert 'thinking.animation::rotating' in keys
    index, _ = find_row(overlay, 'thinking.animation::rotating')
    overlay.row = index
    overlay.row_key = 'thinking.animation::rotating'
    overlay.handle(key('enter'))
    keys = [row.key for row in overlay._flatten()]
    assert 'thinking.rotation' in keys
    assert 'thinking.rotation_ms' in keys


def test_number_edit_select_all_and_clamp(config):
    overlay = SettingsOverlay(config, lambda: None)
    focus(overlay, 'context.threshold')
    overlay.handle(key('enter'))
    assert overlay.editing
    overlay.handle(text('7'))
    overlay.handle(text('0'))
    overlay.handle(key('enter'))
    assert config.context.compact_threshold == 70
    focus(overlay, 'context.threshold')
    overlay.handle(key('enter'))
    overlay.handle(text('5'))
    overlay.handle(key('enter'))
    assert config.context.compact_threshold == 40


def test_invalid_number_keeps_editor(config):
    overlay = SettingsOverlay(config, lambda: None)
    focus(overlay, 'context.threshold')
    overlay.handle(key('enter'))
    overlay.handle(text('abc'))
    overlay.handle(key('enter'))
    assert overlay.editing
    assert overlay.edit_error
    overlay.handle(key('esc'))
    assert not overlay.editing


def test_bool_expands_and_selects(config):
    overlay = SettingsOverlay(config, lambda: None)
    focus(overlay, 'context.auto')
    overlay.handle(key('enter'))
    rows = overlay._flatten()
    assert any(row.key == 'context.auto::False' for row in rows)
    index, _ = find_row(overlay, 'context.auto::False')
    overlay.row = index
    overlay.row_key = 'context.auto::False'
    overlay.handle(key('enter'))
    assert config.context.auto_compact is False


def test_text_list_editing(config):
    overlay = SettingsOverlay(config, lambda: None)
    focus(overlay, 'thinking.mode')
    overlay.handle(key('enter'))
    focus(overlay, 'thinking.label')
    overlay.handle(key('enter'))
    overlay.handle(text('Reasoning'))
    overlay.handle(key('enter'))
    assert config.thinking.label == 'Reasoning'
    focus(overlay, 'thinking.animation')
    overlay.handle(key('enter'))
    focus(overlay, 'thinking.rotation_ms')
    overlay.handle(key('enter'))
    overlay.handle(text('100,200,300'))
    overlay.handle(key('enter'))
    assert config.thinking.rotation_ms == [100, 200, 300]


def test_accent_and_custom_hex(config):
    overlay = SettingsOverlay(config, lambda: None)
    focus(overlay, 'ui.accent::#ff0000' if any(
        row.key == 'ui.accent::#ff0000' for row in overlay._flatten())
        else 'ui.accent')
    # Expand the accent choice and select the Sand preset.
    index, _ = find_row(overlay, 'ui.accent')
    overlay.row = index
    overlay.row_key = 'ui.accent'
    overlay.handle(key('enter'))
    preset = next(row.key for row in overlay._flatten()
                  if row.key.startswith('ui.accent::#'))
    index, _ = find_row(overlay, preset)
    overlay.row = index
    overlay.row_key = preset
    overlay.handle(key('enter'))
    assert config.ui.accent.startswith('#')
    focus(overlay, 'ui.accent_custom')
    overlay.handle(key('enter'))
    overlay.handle(text('#010203'))
    overlay.handle(key('enter'))
    assert config.ui.accent == '#010203'


def test_first_down_reaches_the_first_setting(config):
    overlay = SettingsOverlay(config, lambda: None)
    overlay.handle(key('down'))
    assert current(overlay) == 'ui.theme'


def test_cursor_stays_after_choosing_a_bool_value(config):
    overlay = SettingsOverlay(config, lambda: None)
    walk_to(overlay, 'context.auto')
    overlay.handle(key('enter'))
    walk_to(overlay, 'context.auto::False')
    overlay.handle(key('enter'))
    assert config.context.auto_compact is False
    assert overlay.expanded == []
    # The option row is gone; the cursor must sit on its parent, not on the
    # first row of the tree.
    assert current(overlay) == 'context.auto'
    overlay.handle(key('down'))
    assert current(overlay) == 'context.threshold'


def test_cursor_stays_after_editing_a_number(config):
    overlay = SettingsOverlay(config, lambda: None)
    walk_to(overlay, 'context.threshold')
    overlay.handle(key('enter'))
    overlay.handle(text('70'))
    overlay.handle(key('enter'))
    assert config.context.compact_threshold == 70
    assert current(overlay) == 'context.threshold'
    overlay.handle(key('down'))
    assert current(overlay) == 'context.keep'


def test_cursor_stays_after_switching_a_choice_branch(config):
    overlay = SettingsOverlay(config, lambda: None)
    walk_to(overlay, 'thinking.mode')
    overlay.handle(key('enter'))
    walk_to(overlay, 'thinking.mode::partial')
    overlay.handle(key('enter'))
    assert config.thinking.mode == 'partial'
    assert current(overlay) == 'thinking.mode::partial'
    overlay.handle(key('down'))
    assert current(overlay) == 'thinking.partial.from'


def test_escape_leaves_the_cursor_on_the_collapsed_item(config):
    overlay = SettingsOverlay(config, lambda: None)
    walk_to(overlay, 'context.auto')
    overlay.handle(key('enter'))
    walk_to(overlay, 'context.auto::True')
    overlay.handle(key('esc'))
    assert overlay.expanded == []
    assert current(overlay) == 'context.auto'
