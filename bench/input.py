"""Scripted terminal input for the headless application harness.

Input is fed as the raw byte sequences a real terminal would send (arrows,
control bytes, SGR mouse reports and bracketed paste) and parsed by the
application's own :class:`adcode.tui.input.KeyParser`.  The harness never
touches the OS input queue, so it can run in CI with no console attached.
"""

from __future__ import annotations

import queue
import threading
from typing import Iterable

from adcode.tui.input import KeyEvent, KeyParser, _CTRL_NAMES

# Reverse the application's control-character table: 'ctrl+a' -> '\x01'.
_CONTROL_SEQUENCES = {name: char for char, name in _CTRL_NAMES.items()}

_KEY_SEQUENCES = {
    'enter': '\r',
    'tab': '\t',
    'backspace': '\x7f',
    'esc': '\x1b',
    'up': '\x1b[A',
    'down': '\x1b[B',
    'right': '\x1b[C',
    'left': '\x1b[D',
    'home': '\x1b[H',
    'end': '\x1b[F',
    'insert': '\x1b[2~',
    'shift+insert': '\x1b[2;2~',
    'delete': '\x1b[3~',
    'pgup': '\x1b[5~',
    'pgdn': '\x1b[6~',
    'f1': '\x1b[11~',
    'f2': '\x1b[12~',
    'f3': '\x1b[13~',
    'f4': '\x1b[14~',
    'f5': '\x1b[15~',
    'f6': '\x1b[17~',
    'f7': '\x1b[18~',
    'f8': '\x1b[19~',
    'f9': '\x1b[20~',
    'f10': '\x1b[21~',
    'f11': '\x1b[23~',
    'f12': '\x1b[24~',
    'shift+tab': '\x1b[Z',
    'alt+enter': '\x1b\r',
    'ctrl+left': '\x1b[1;5D',
    'ctrl+right': '\x1b[1;5C',
    'ctrl+up': '\x1b[1;5A',
    'ctrl+down': '\x1b[1;5B',
    'shift+left': '\x1b[1;2D',
    'shift+right': '\x1b[1;2C',
    'shift+up': '\x1b[1;2A',
    'shift+down': '\x1b[1;2B',
    'ctrl+home': '\x1b[1;5H',
    'ctrl+end': '\x1b[1;5F',
    'shift+home': '\x1b[1;2H',
    'shift+end': '\x1b[1;2F',
    # Chords on punctuation have no legacy control byte; a terminal that
    # speaks the kitty keyboard protocol reports them as CSI <code>;<mods> u.
    'ctrl+,': '\x1b[44;5u',
    'ctrl+.': '\x1b[46;5u',
    'ctrl+/': '\x1b[47;5u',
    'ctrl+shift+v': '\x1b[118;6u',
}

# Keys with no portable byte sequence; delivered as a typed KeyEvent.
_DIRECT_KEYS = {
    'shift+enter',
    'ctrl+space',
}


def key_sequence(name: str) -> str:
    """Return the canonical byte sequence for a key name."""
    if name in _KEY_SEQUENCES:
        return _KEY_SEQUENCES[name]
    if name in _CONTROL_SEQUENCES:
        return _CONTROL_SEQUENCES[name]
    lowered = name.lower()
    if lowered.startswith('alt+') and len(lowered) == 5:
        return '\x1b' + lowered[-1]
    if name.startswith('ctrl+') and len(name) == 6 and name[-1].isalpha():
        return chr(ord(name[-1].lower()) - 96)
    raise KeyError(f'no byte sequence for key {name!r}')


def mouse_sequence(action: str, x: int = 10, y: int = 10,
                   button: int = 0) -> str:
    """Build an SGR mouse report (1-based coordinates)."""
    x = max(1, int(x))
    y = max(1, int(y))
    if action == 'wheel_up':
        return f'\x1b[<64;{x};{y}M'
    if action == 'wheel_down':
        return f'\x1b[<65;{x};{y}M'
    if action == 'move':
        return f'\x1b[<{32 + button};{x};{y}M'
    if action == 'press':
        return f'\x1b[<{button};{x};{y}M'
    if action == 'release':
        return f'\x1b[<{button};{x};{y}m'
    raise ValueError(f'unknown mouse action {action!r}')


class ScriptedInput:
    """Stands in for :class:`adcode.tui.input.InputThread`."""

    def __init__(self, events: 'queue.Queue[KeyEvent]') -> None:
        self.events = events
        self.parser = KeyParser()
        self.started = False
        self.stopped = False
        self.sent: list[KeyEvent] = []
        self.chunks: list[str] = []
        self._lock = threading.RLock()

    # ── InputThread interface ───────────────────────────────
    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def join(self, timeout: float | None = None) -> None:  # noqa: ARG002
        return None

    def is_alive(self) -> bool:
        return False

    # ── feeding ─────────────────────────────────────────────
    def feed(self, data: str, *, flush: bool = False) -> list[KeyEvent]:
        with self._lock:
            self.chunks.append(data)
            events = self.parser.feed(data)
            if flush:
                events.extend(self.parser.flush_pending())
        self._deliver(events)
        return events

    def flush(self) -> list[KeyEvent]:
        with self._lock:
            events = self.parser.flush_pending()
        self._deliver(events)
        return events

    def _deliver(self, events: Iterable[KeyEvent]) -> None:
        for event in events:
            if self.stopped:
                return
            self.sent.append(event)
            self.events.put(event)

    # ── high-level actions ──────────────────────────────────
    def type_text(self, text: str) -> None:
        """Type literal text; newlines become Enter, as on a real terminal."""
        for char in text:
            if char == '\n':
                self.feed('\r')
            elif char == '\t':
                self.feed('\t')
            else:
                self.feed(char)

    def press(self, name: str) -> None:
        if name in _DIRECT_KEYS:
            flags = {
                'ctrl': name.startswith('ctrl+'),
                'alt': name.startswith('alt+'),
                'shift': name.startswith('shift+'),
            }
            self._deliver([KeyEvent(kind='key', key=name, **flags)])
            return
        sequence = key_sequence(name)
        self.feed(sequence, flush=(name == 'esc'))

    def press_all(self, *names: str) -> None:
        for name in names:
            self.press(name)

    def paste(self, text: str) -> None:
        self.feed(f'\x1b[200~{text}\x1b[201~')

    def wheel(self, action: str = 'wheel_up', x: int = 10, y: int = 10) -> None:
        if action not in ('wheel_up', 'wheel_down'):
            raise ValueError('wheel action must be wheel_up or wheel_down')
        self.feed(mouse_sequence(action, x, y))

    def click(self, x: int = 10, y: int = 10, button: int = 0,
              release: bool = True) -> None:
        self.feed(mouse_sequence('press', x, y, button))
        if release:
            self.feed(mouse_sequence('release', x, y, button))

    def drag(self, x1: int, y1: int, x2: int, y2: int,
             button: int = 0) -> None:
        self.feed(mouse_sequence('press', x1, y1, button))
        self.feed(mouse_sequence('move', x2, y2, button))
        self.feed(mouse_sequence('release', x2, y2, button))

    def key_event(self, key: str, **flags: bool) -> None:
        self._deliver([KeyEvent(kind='key', key=key, **flags)])
