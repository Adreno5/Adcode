"""Central registry of application-level key bindings.

Every shortcut the application handles outside of the overlay widgets is
declared here as a :class:`Binding`.  The configured keys live in
``config.keys`` (see :class:`adcode.config.KeysSettings`); :class:`KeyMap`
resolves them for dispatch and the bindings panel (``/keys``) edits the
same fields.

Keys that are part of widget navigation (the arrows, enter and esc inside
the dialogs) are not bindings -- they stay fixed.  ``enter`` and ``esc``
remain structural gestures app-wide: submit and cancel are never rebound.
"""

from __future__ import annotations

from dataclasses import dataclass

# Modifier order the input parser produces; canonical for stored key names.
_MODIFIERS = ('ctrl', 'alt', 'shift')

# Named keys a binding may use without any modifier.
_SPECIAL_KEYS = frozenset({
    'up', 'down', 'left', 'right', 'home', 'end', 'pgup', 'pgdn',
    'insert', 'delete', 'backspace', 'tab', *(f'f{n}' for n in range(1, 13)),
})

# Control-character keys the parser can emit beyond the letters.
_CTRL_SYMBOLS = frozenset({'\\', ']', '^', '_', 'space'})

# Punctuation a chord can carry.  These keys have no control byte in the
# legacy encoding, so they only reach the app through the kitty keyboard
# protocol (``ctrl+,`` is reported as ``CSI 44;5 u``).
_CTRL_PUNCTUATION = frozenset(
    '`~!@#$%^&*()-_=+[{]}\\|;\':",.<>/?')

# Fixed gestures and the sentinel the panel writes to clear a binding.
_RESERVED_KEYS = frozenset({'enter', 'esc', 'space'})
UNBOUND = 'none'


@dataclass(frozen=True)
class Binding:
    """One rebindable application shortcut."""

    action: str                    # stable id, matches a KeysSettings field
    label: str                     # short name shown in the panel and /help
    description: str
    default: str                   # canonical key name
    aliases: tuple[str, ...] = ()  # fixed extra keys that trigger the action
    group: str = 'General'


BINDINGS: tuple[Binding, ...] = (
    # ── general ─────────────────────────────────────────────
    Binding('quit', 'Quit', 'exit AdCode immediately, whatever is on screen',
            'ctrl+q'),
    Binding('palette', 'Command palette', 'open the / command palette',
            'ctrl+p'),
    Binding('config', 'Settings', 'open the settings browser',
            'ctrl+,'),
    Binding('sessions', 'Sessions', 'browse and load saved sessions',
            'ctrl+h'),
    Binding('exit_chord', 'Copy, clear or exit',
            'copy selection, clear the input, or press twice to exit',
            'ctrl+c'),
    Binding('exit_eof', 'Exit on EOF', 'exit when the input is empty',
            'ctrl+d'),
    Binding('redraw', 'Redraw', 'force a full repaint', 'ctrl+l'),
    # ── clipboard ───────────────────────────────────────────
    Binding('paste', 'Paste',
            'paste text, or attach a copied screenshot',
            'ctrl+v', aliases=('ctrl+v', 'ctrl+shift+v', 'shift+insert'),
            group='Clipboard'),
    Binding('attach_image', 'Attach clipboard image',
            'attach the clipboard image to the message', 'alt+v',
            group='Clipboard'),
    Binding('copy_last', 'Copy last reply',
            'copy the latest assistant reply as Markdown', 'alt+c',
            group='Clipboard'),
    Binding('cut', 'Cut', 'cut the selected input text', 'ctrl+x',
            group='Clipboard'),
    # ── editing ─────────────────────────────────────────────
    Binding('select_all', 'Select all', 'select the whole input', 'ctrl+a',
            group='Editing'),
    Binding('undo', 'Undo', 'undo the last input edit', 'ctrl+z',
            group='Editing'),
    Binding('redo', 'Redo', 'redo an undone input edit', 'ctrl+y',
            group='Editing'),
    Binding('newline', 'Newline', 'insert a newline in the input', 'ctrl+j',
            aliases=('alt+enter', 'shift+enter'), group='Editing'),
    Binding('complete', 'Complete', 'complete the selected command', 'tab',
            group='Editing'),
    Binding('delete_backward', 'Delete backward',
            'delete the character before the cursor', 'backspace',
            group='Editing'),
    Binding('delete_forward', 'Delete forward',
            'delete the character at the cursor', 'delete',
            group='Editing'),
    # ── navigation ──────────────────────────────────────────
    Binding('scroll_up', 'Scroll up', 'scroll the transcript up a page',
            'pgup', group='Navigation'),
    Binding('scroll_down', 'Scroll down', 'scroll the transcript down a page',
            'pgdn', group='Navigation'),
    Binding('history_up', 'History up',
            'previous history entry, or completion up', 'up',
            group='Navigation'),
    Binding('history_down', 'History down',
            'next history entry, or completion down', 'down',
            group='Navigation'),
    Binding('move_left', 'Cursor left', 'move or select left', 'left',
            aliases=('shift+left',), group='Navigation'),
    Binding('move_right', 'Cursor right', 'move or select right', 'right',
            aliases=('shift+right',), group='Navigation'),
    Binding('line_start', 'Line start', 'move or select to the line start',
            'home', aliases=('shift+home',), group='Navigation'),
    Binding('line_end', 'Line end', 'move or select to the line end', 'end',
            aliases=('shift+end',), group='Navigation'),
    Binding('jump_start', 'Input start', 'jump to the start of the input',
            'ctrl+home', group='Navigation'),
    Binding('jump_end', 'Input end', 'jump to the end of the input',
            'ctrl+end', group='Navigation'),
)

_BINDING_BY_ACTION = {item.action: item for item in BINDINGS}


def binding_of(action: str) -> Binding:
    """The registry entry for an action id."""
    return _BINDING_BY_ACTION[action]


def canonical_key(name: str) -> str:
    """Lower-case a key name and order its modifiers ctrl, alt, shift."""
    parts = [part for part in (name or '').strip().lower().split('+') if part]
    if not parts:
        return ''
    modifiers = [part for part in parts[:-1] if part in _MODIFIERS]
    if len(modifiers) == len(parts) - 1:
        ordered = [mod for mod in _MODIFIERS if mod in modifiers]
        return '+'.join([*ordered, parts[-1]])
    return '+'.join(parts)


def validate_key_name(name: str) -> str | None:
    """Return an error message for an unusable binding, or None if valid.

    Printable characters and the structural enter/esc gestures can never be
    bindings: bare characters are reserved for typing, enter/esc submit and
    cancel everywhere.
    """
    name = canonical_key(name)
    if not name:
        return 'empty key name'
    if name in _RESERVED_KEYS:
        return f'{name} is a fixed gesture and cannot be bound'
    if name in _SPECIAL_KEYS:
        return None
    parts = name.split('+')
    *modifiers, base = parts
    if not all(mod in _MODIFIERS for mod in modifiers):
        return f'{name!r} is not a key name'
    if len(set(modifiers)) != len(modifiers):
        return f'{name!r} has a duplicate modifier'
    if base in _SPECIAL_KEYS:
        return None
    symbol = base in _CTRL_SYMBOLS
    word = len(base) == 1 and (base.isalpha() or base.isdigit())
    punctuation = len(base) == 1 and base in _CTRL_PUNCTUATION
    if symbol:
        return None if 'ctrl' in modifiers else \
            f'{name!r} needs the ctrl modifier'
    if word and ('ctrl' in modifiers or 'alt' in modifiers):
        return None
    if punctuation and ('ctrl' in modifiers or 'alt' in modifiers):
        # Punctuation carries no legacy control byte, so only a chord can
        # reach the app -- and only a terminal that speaks the kitty
        # keyboard protocol can report one.
        return None
    if not word:
        return f'{name!r} is not a key name'
    return f'{name!r} is reserved for typing; use a ctrl/alt/function key'


class KeyMap:
    """Resolved bindings for the current settings, ready for dispatch."""

    def __init__(self, settings: object) -> None:
        self._by_action: dict[str, str] = {}
        self._by_key: dict[str, str] = {}
        for item in BINDINGS:
            raw = getattr(settings, item.action, None)
            if not isinstance(raw, str) or raw == UNBOUND:
                self._by_action[item.action] = ''
            else:
                self._by_action[item.action] = canonical_key(raw)
        # Primaries claim their keys first; aliases fill only the keys no
        # binding holds, so a rebound action never loses to its own alias.
        for item in BINDINGS:
            key = self._by_action[item.action]
            if key and key not in self._by_key:
                self._by_key[key] = item.action
        for item in BINDINGS:
            for alias in item.aliases:
                self._by_key.setdefault(alias, item.action)

    def key_of(self, action: str) -> str:
        """The configured key for an action, '' when unbound."""
        return self._by_action.get(action, '')

    def action_of(self, key: str) -> str | None:
        """The action a canonical key name triggers, None when none does."""
        return self._by_key.get(key)
