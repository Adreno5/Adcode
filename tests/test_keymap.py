"""The binding registry, key-name validation and the dispatch resolver."""

from adcode.config import ConfigStore, KeysSettings
from adcode.tui.keymap import (
    BINDINGS, UNBOUND, KeyMap, canonical_key, validate_key_name,
)


def test_settings_and_registry_defaults_match():
    settings = KeysSettings()
    for item in BINDINGS:
        assert getattr(settings, item.action) == item.default


def test_default_resolution_and_reverse_lookup():
    keymap = KeyMap(KeysSettings())
    assert keymap.key_of('palette') == 'ctrl+p'
    assert keymap.action_of('ctrl+p') == 'palette'
    assert keymap.action_of('ctrl+h') == 'sessions'
    assert keymap.action_of('ctrl+q') == 'quit'
    assert keymap.key_of('config') == 'ctrl+,'
    assert keymap.action_of('ctrl+,') == 'config'


def test_rebind_moves_the_action():
    settings = KeysSettings()
    settings.palette = 'ctrl+e'
    keymap = KeyMap(settings)
    assert keymap.key_of('palette') == 'ctrl+e'
    assert keymap.action_of('ctrl+e') == 'palette'
    assert keymap.action_of('ctrl+p') is None


def test_aliases_resolve_to_their_action():
    keymap = KeyMap(KeysSettings())
    assert keymap.action_of('shift+left') == 'move_left'
    assert keymap.action_of('shift+right') == 'move_right'
    assert keymap.action_of('ctrl+shift+v') == 'paste'
    assert keymap.action_of('shift+insert') == 'paste'
    assert keymap.action_of('alt+enter') == 'newline'
    assert keymap.action_of('shift+enter') == 'newline'


def test_a_rebound_action_keeps_its_fixed_aliases():
    settings = KeysSettings()
    settings.paste = 'ctrl+e'
    keymap = KeyMap(settings)
    assert keymap.action_of('ctrl+e') == 'paste'
    assert keymap.action_of('ctrl+v') == 'paste'
    assert keymap.action_of('shift+insert') == 'paste'


def test_an_unbound_action_has_no_key():
    settings = KeysSettings()
    settings.sessions = UNBOUND
    keymap = KeyMap(settings)
    assert keymap.key_of('sessions') == ''
    assert keymap.action_of('ctrl+h') is None


def test_canonical_key_orders_modifiers():
    assert canonical_key('Ctrl+H') == 'ctrl+h'
    assert canonical_key('alt+ctrl+x') == 'ctrl+alt+x'
    assert canonical_key(' shift+up ') == 'shift+up'
    assert canonical_key('') == ''


def test_validate_accepts_chords_and_special_keys():
    for name in ('ctrl+h', 'alt+v', 'ctrl+shift+s', 'f5', 'pgdn', 'up',
                 'shift+insert', 'ctrl+space', 'ctrl+]', 'alt+7', 'backspace'):
        assert validate_key_name(name) is None, name

def test_validate_accepts_a_chord_on_punctuation():
    # The punctuation keys carry no control byte, so only a chord can
    # reach the app.
    for name in ('ctrl+,', 'ctrl+.', 'ctrl+/', 'ctrl+shift+,', 'alt+,'):
        assert validate_key_name(name) is None, name

def test_validate_rejects_bare_punctuation():
    for name in (',', '.', '/'):
        assert validate_key_name(name) is not None, name


def test_validate_rejects_bare_and_reserved_names():
    for name in ('a', '5', 'space', 'enter', 'esc', 'shift+a', '', 'ctrl+',
                 'hyper+t', 'bogus', 'none'):
        assert validate_key_name(name) is not None, name


def test_from_dict_falls_back_to_defaults():
    settings = KeysSettings.from_dict({
        'palette': 'ctrl+e', 'quit': 'nope!!', 'cut': 'none',
        'undo': 'Ctrl+Z', 'sessions': None, 'redo': 'shift+up',
    })
    assert settings.palette == 'ctrl+e'
    assert settings.quit == 'ctrl+q'      # invalid name
    assert settings.cut == 'none'         # explicit unbound
    assert settings.undo == 'ctrl+z'      # canonicalised
    assert settings.sessions == 'ctrl+h'  # missing
    assert settings.redo == 'shift+up'    # special key without ctrl


def test_config_roundtrip_keeps_keys(tmp_path):
    store = ConfigStore(tmp_path / 'config.json')
    config = store.load()
    config.keys.palette = 'ctrl+e'
    config.keys.sessions = 'none'
    store.save()
    reloaded = ConfigStore(tmp_path / 'config.json').load()
    assert reloaded.keys.palette == 'ctrl+e'
    assert reloaded.keys.sessions == 'none'
    assert reloaded.keys.quit == 'ctrl+q'
