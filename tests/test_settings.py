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
    assert groups == ['Recents', 'Appearance', 'Work steps', 'Footer status',
                      'Hold', 'Failover', 'Reasoning', 'Context',
                      'Sub-agents and teams', 'Skills', 'Security', 'MCP']


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
    # Escape closes the item and keeps what was chosen: every change is
    # applied and saved as it is made, so there is nothing to take back.
    overlay.handle(key('esc'))
    assert config.thinking.mode == 'partial'
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
    # The pinned Recents section comes first: the theme is its first entry.
    assert current(overlay) == 'recent:ui.theme'


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

def test_the_work_steps_group_starts_with_the_collapse_toggle(config):
    overlay = SettingsOverlay(config, lambda: None)
    keys = [row.key for row in overlay._flatten()]
    start = keys.index('group:Work steps')
    assert keys[start + 1] == 'steps.collapse'
    assert keys[start + 2] == 'steps.preset'
    assert keys[start + 3] == 'thinking.mode'
    assert keys[start + 4] == 'thinking.after_completion'
    focus(overlay, 'steps.preset')
    overlay.handle(key('enter'))
    options = [row.key for row in overlay._flatten() if row.kind == 'option']
    assert options == [f'steps.preset::{level}'
                       for level in ('concise', 'standard', 'detailed', 'custom')]


def test_the_collapse_toggle_is_no_presets_business(config):
    overlay = SettingsOverlay(config, lambda: None)
    walk_to(overlay, 'steps.preset')
    overlay.handle(key('enter'))
    walk_to(overlay, 'steps.preset::detailed')
    overlay.handle(key('enter'))
    assert config.steps.preset == 'detailed'
    # A preset keeps the thinking and tool settings read-only, but the
    # toggle above it stays the user's own switch.
    walk_to(overlay, 'thinking.mode')
    overlay.handle(key('enter'))
    assert 'thinking.mode::full' not in [row.key for row in overlay._flatten()]
    walk_to(overlay, 'steps.collapse')
    overlay.handle(key('enter'))
    index, _ = find_row(overlay, 'steps.collapse::True')
    overlay.row, overlay.row_key = index, 'steps.collapse::True'
    overlay.handle(key('enter'))
    assert config.steps.collapse_after_completion is True

def test_a_preset_writes_its_settings_and_locks_them(config):
    overlay = SettingsOverlay(config, lambda: None)
    walk_to(overlay, 'steps.preset')
    overlay.handle(key('enter'))
    walk_to(overlay, 'steps.preset::standard')
    overlay.handle(key('enter'))
    assert config.steps.preset == 'standard'
    assert config.thinking.mode == 'partial'
    assert config.tools.chain == 'separate'
    # The settings the preset owns are read-only: they show their value but
    # do not open, and the panel says why.
    walk_to(overlay, 'thinking.mode')
    overlay.handle(key('enter'))
    assert overlay.editing is False
    assert 'thinking.mode::full' not in [row.key for row in overlay._flatten()]
    assert config.thinking.mode == 'partial'
    assert 'Custom' in overlay.notice
    walk_to(overlay, 'tools.chain')
    overlay.handle(key('enter'))
    assert config.tools.chain == 'separate'

def test_custom_unlocks_the_work_step_settings(config):
    config.steps.preset = 'concise'
    config.thinking.mode = 'label'
    config.thinking.label_tokens = True
    overlay = SettingsOverlay(config, lambda: None)
    walk_to(overlay, 'steps.preset')
    overlay.handle(key('enter'))
    walk_to(overlay, 'steps.preset::custom')
    overlay.handle(key('enter'))
    assert config.steps.preset == 'custom'
    # Nothing else moved: the hand-edited values stay as they were.
    assert config.thinking.mode == 'label'
    assert config.thinking.label_tokens is True
    walk_to(overlay, 'thinking.mode')
    overlay.handle(key('enter'))
    walk_to(overlay, 'thinking.mode::full')
    overlay.handle(key('enter'))
    assert config.thinking.mode == 'full'

def test_choosing_a_preset_collapses_what_it_locked(config):
    overlay = SettingsOverlay(config, lambda: None)
    walk_to(overlay, 'thinking.mode')
    overlay.handle(key('enter'))
    walk_to(overlay, 'steps.preset')
    overlay.handle(key('enter'))
    walk_to(overlay, 'steps.preset::concise')
    overlay.handle(key('enter'))
    # An expansion of a setting the preset just locked must not linger: its
    # rows are gone, and the lock would refuse every edit inside it anyway.
    assert overlay.expanded == []
    assert config.thinking.mode == 'label'        # written by the preset


def test_the_thinking_display_is_two_stages(config):
    overlay = SettingsOverlay(config, lambda: None)
    walk_to(overlay, 'thinking.mode')
    overlay.handle(key('enter'))
    # The finished-thought choice is its own item now, not a child of the
    # live block's options.
    rows = [row for row in overlay._flatten()
            if row.key == 'thinking.after_completion']
    assert [(row.kind, row.depth) for row in rows] == [('setting', 0)], rows
    walk_to(overlay, 'thinking.mode::partial')
    overlay.handle(key('enter'))
    assert config.thinking.mode == 'partial'
    # The second stage offers the same four modes.
    walk_to(overlay, 'thinking.after_completion')
    overlay.handle(key('enter'))
    options = [row.key for row in overlay._flatten()
               if row.key.startswith('thinking.after_completion::')]
    assert options == ['thinking.after_completion::hidden',
                       'thinking.after_completion::label',
                       'thinking.after_completion::partial',
                       'thinking.after_completion::full'], options
    walk_to(overlay, 'thinking.after_completion::label')
    overlay.handle(key('enter'))
    assert config.thinking.after_completion == 'label'
    assert config.thinking.mode == 'partial'      # the live block is untouched

def test_the_token_count_belongs_to_the_label(config):
    overlay = SettingsOverlay(config, lambda: None)
    walk_to(overlay, 'thinking.mode')
    overlay.handle(key('enter'))
    options = [row.key for row in overlay._flatten() if row.kind == 'option']
    assert 'thinking.mode::summary' not in options
    # The label owns the count: it is one of its settings, not a mode.
    walk_to(overlay, 'thinking.mode::label')
    overlay.handle(key('enter'))
    assert config.thinking.mode == 'label'
    walk_to(overlay, 'thinking.label_tokens')
    overlay.handle(key('enter'))
    walk_to(overlay, 'thinking.label_tokens::True')
    overlay.handle(key('enter'))
    assert config.thinking.label_tokens is True

def test_every_label_position_offers_the_token_count(config):
    overlay = SettingsOverlay(config, lambda: None)
    walk_to(overlay, 'thinking.mode')
    overlay.handle(key('enter'))
    walk_to(overlay, 'thinking.mode::full')
    overlay.handle(key('enter'))
    # The label collapse after the thought finishes shows a label too, so it
    # carries the same switch (and the same setting).
    walk_to(overlay, 'thinking.after_completion')
    overlay.handle(key('enter'))
    walk_to(overlay, 'thinking.after_completion::label')
    overlay.handle(key('enter'))
    assert config.thinking.after_completion == 'label'
    walk_to(overlay, 'thinking.label_tokens')
    overlay.handle(key('enter'))
    walk_to(overlay, 'thinking.label_tokens::True')
    overlay.handle(key('enter'))
    assert config.thinking.label_tokens is True


def test_the_access_levels_are_in_the_security_group(config):
    tree = build_settings_tree(config)
    keys = [node.key for node in tree]
    assert 'security.access' in keys, 'the default level'
    assert 'security.project_access' in keys, 'the level of this project'
    node = next(node for node in tree if node.key == 'security.access')
    assert [choice.value for choice in node.choices] == ['readonly',
                                                         'workspace', 'full']


def test_the_default_level_is_chosen_in_the_browser(config):
    overlay = SettingsOverlay(config, lambda: None)
    focus(overlay, 'security.access')
    overlay.handle(key('enter'))
    index, _ = find_row(overlay, 'security.access::readonly')
    overlay.row = index
    overlay.row_key = 'security.access::readonly'
    overlay.handle(key('enter'))
    assert config.security.access == 'readonly'


def test_the_project_rule_is_its_own_setting(config, tmp_path, monkeypatch):
    """The level of the project this session is in."""
    from adcode import paths

    monkeypatch.setattr(paths, 'PROJECT_ROOT', str(tmp_path))
    overlay = SettingsOverlay(config, lambda: None)
    focus(overlay, 'security.project_access')
    overlay.handle(key('enter'))
    index, _ = find_row(overlay, 'security.project_access::workspace')
    overlay.row = index
    overlay.row_key = 'security.project_access::workspace'
    overlay.handle(key('enter'))
    assert config.security.access_by_project[str(tmp_path.resolve())] == 'workspace'

    # and back to the default
    focus(overlay, 'security.project_access')
    overlay.handle(key('enter'))
    index, _ = find_row(overlay, 'security.project_access::')
    overlay.row = index
    overlay.row_key = 'security.project_access::'
    overlay.handle(key('enter'))
    assert str(tmp_path.resolve()) not in config.security.access_by_project
