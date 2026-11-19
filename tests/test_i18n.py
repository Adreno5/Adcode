"""The interface language: the setting, the translation table, the tree."""

from __future__ import annotations

import pytest

from adcode.config import AppConfig, config_from_dict, config_to_dict
from adcode.i18n import (
    LOCALES, current_locale, normalize_locale, set_locale, t, tf,
)
from adcode.tui.settings import SettingsOverlay, build_settings_tree


@pytest.fixture(autouse=True)
def english():
    """Every test starts in English and leaves the language as it was."""
    set_locale('en')
    yield
    set_locale(None)


def key(name):
    from adcode.tui.input import KeyEvent
    return KeyEvent(kind='key', key=name)


# ── the setting ─────────────────────────────────────────────

def test_the_language_is_part_of_the_configuration():
    config = AppConfig()
    assert config.ui.locale == 'en'
    config.ui.locale = 'zh'
    data = config_to_dict(config)
    assert data['ui']['locale'] == 'zh'
    assert config_from_dict(data).ui.locale == 'zh'


def test_an_unknown_language_falls_back_to_english():
    config = config_from_dict({'ui': {'locale': 'klingon'}})
    assert config.ui.locale == 'en'
    assert config_from_dict({'ui': {'locale': 'zh-CN'}}).ui.locale == 'zh'


def test_normalize_locale_accepts_the_usual_spellings():
    assert normalize_locale('en') == 'en'
    assert normalize_locale('zh') == 'zh'
    assert normalize_locale('zh-CN') == 'zh'
    assert normalize_locale('zh_CN') == 'zh'
    assert normalize_locale('') == 'en'
    assert normalize_locale(None) == 'en'


def test_the_locale_override_wins_over_the_configuration():
    from adcode.config import set_current_config
    config = AppConfig()
    set_current_config(config)
    try:
        assert current_locale() == 'en'
        set_locale('zh')
        assert current_locale() == 'zh'
        # the configured language is used again once the override is gone
        set_locale(None)
        assert current_locale() == 'en'
    finally:
        set_current_config(None)


# ── the translation ─────────────────────────────────────────

def test_english_stays_english():
    set_locale('en')
    assert t('Theme') == 'Theme'
    assert t('a string nobody translated') == 'a string nobody translated'


def test_chinese_is_translated():
    set_locale('zh')
    assert t('Theme') == '主题'
    assert t('Settings') == '设置'
    # an untranslated string is left alone instead of breaking the interface
    assert t('a string nobody translated') == 'a string nobody translated'


def test_templates_are_filled_in_after_the_translation():
    set_locale('zh')
    assert tf('Session renamed to {name}.', name='demo') == '会话已重命名为 demo。'


def test_every_shipped_language_has_a_name():
    from adcode.i18n import locale_label
    assert locale_label('en') == 'English'
    assert locale_label('zh') == '中文'
    assert set(LOCALES) == {'en', 'zh'}


# ── the settings browser ────────────────────────────────────

def test_the_settings_tree_offers_the_language(config):
    tree = build_settings_tree(config)
    node = next(item for item in tree if item.key == 'ui.locale')
    assert node.path == 'ui.locale'
    assert {choice.value for choice in node.choices} == {'en', 'zh'}


def test_changing_the_language_translates_the_tree(config):
    overlay = SettingsOverlay(config, lambda: None)
    rows = overlay._flatten()
    theme = next(row for row in rows if row.key == 'ui.theme')
    assert theme.node.label == 'Theme'

    config.ui.locale = 'zh'
    rows = overlay._flatten()
    theme = next(row for row in rows if row.key == 'ui.theme')
    assert theme.node.label == 'Theme'      # the tree keeps its English keys
    assert overlay._tr(theme.node.label) == '主题'


def test_the_recents_section_is_pinned_to_the_top(config):
    tree = build_settings_tree(config)
    assert tree[0].kind == 'group'
    assert tree[0].label == 'Recents'
    # the pinned entries come right after the group header
    keys = [node.key for node in tree[1:8]]
    assert keys[:6] == ['recent:ui.theme', 'recent:ui.locale',
                        'recent:ui.accent', 'recent:steps.preset',
                        'recent:reasoning.effort', 'recent:tools.chain']


def test_a_pinned_entry_writes_the_setting_it_shows(config):
    overlay = SettingsOverlay(config, lambda: None)
    rows = overlay._flatten()
    index = next(index for index, row in enumerate(rows)
                 if row.key == 'recent:ui.theme')
    overlay.row = index
    overlay.row_key = 'recent:ui.theme'
    overlay.handle(key('enter'))
    index = next(index for index, row in enumerate(overlay._flatten())
                 if row.key == 'recent:ui.theme::light')
    overlay.row = index
    overlay.row_key = 'recent:ui.theme::light'
    overlay.handle(key('enter'))
    # the copy and the original are the same setting
    assert config.ui.theme == 'light'


def test_the_pinned_entries_are_the_ones_of_the_tree(config):
    tree = build_settings_tree(config)
    by_key = {node.key: node for node in tree}
    recent = by_key['recent:ui.locale']
    original = by_key['ui.locale']
    assert recent.path == original.path
    assert [choice.value for choice in recent.choices] == \
           [choice.value for choice in original.choices]
