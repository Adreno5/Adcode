"""Browser keystrokes as the terminal events the app already parses.

A browser sends a key name ("ctrl+c"), literal text, a paste, or a wheel
notch.  Each one becomes the byte sequence a terminal would send and goes
through the app's own KeyParser, so a button in the browser and the same
chord on a real keyboard take exactly the same path through the app.
"""

from __future__ import annotations

import queue
import threading

from adcode.tui.input import (
    KeyEvent, KeyParser, canonical_key, key_sequence, split_key,
)

_WHEEL_CODES = {'wheel_up': 64, 'wheel_down': 65}


def mouse_sequence(action: str, x: int, y: int, button: int = 0) -> str:
    """An SGR mouse report (1-based cell coordinates)."""
    x = max(1, int(x))
    y = max(1, int(y))
    if action in _WHEEL_CODES:
        return f'\x1b[<{_WHEEL_CODES[action]};{x};{y}M'
    if action == 'release':
        return f'\x1b[<{button};{x};{y}m'
    return f'\x1b[<{button};{x};{y}M'


class WebInput:
    """Stands in for InputThread: the events arrive from WebSocket clients."""

    def __init__(self, events: 'queue.Queue[KeyEvent]') -> None:
        self.events = events
        self.parser = KeyParser()
        self._lock = threading.Lock()
        self._stopped = False

    # -- InputThread interface ------------------------------------
    def start(self) -> None:
        pass

    def stop(self) -> None:
        self._stopped = True

    def join(self, timeout: float | None = None) -> None:
        return None

    def is_alive(self) -> bool:
        return False

    # -- feeding ---------------------------------------------------
    def feed(self, data: str) -> None:
        """Parse one complete sequence and deliver the events it carries."""
        if not data or self._stopped:
            return
        with self._lock:
            events = self.parser.feed(data)
            # Every browser message is a whole sequence: an Esc that arrived
            # on its own is a key press, not the prefix of a longer chord.
            events.extend(self.parser.flush_pending())
        for event in events:
            self.events.put(event)

    def press(self, name: str) -> None:
        """Deliver a canonical key name, e.g. "up" or "ctrl+shift+left"."""
        try:
            sequence = key_sequence(name)
        except KeyError:
            self._deliver_direct(name)
            return
        self.feed(sequence)

    def type_text(self, text: str) -> None:
        self.feed(text)

    def paste(self, text: str) -> None:
        if not text:
            return
        self.feed('\x1b[200~' + text + '\x1b[201~')

    def modifier(self, name: str, action: str) -> None:
        """Deliver a modifier press or release.

        The terminal reports these through the kitty keyboard protocol; a
        browser has no key hold state to read, so its key bar sends them as the
        button goes down and up.
        """
        if action not in ('press', 'release'):
            return
        if name not in ('ctrl', 'alt', 'shift'):
            return
        if self._stopped:
            return
        self.events.put(KeyEvent(kind='modifier', key=name, action=action))

    def mouse(self, action: str, x: int, y: int, button: int = 0) -> None:
        if action in _WHEEL_CODES or action == 'press' or action == 'release':
            self.feed(mouse_sequence(action, x, y, button))

    def modifier(self, name: str, action: str) -> None:
        """Deliver a modifier key press/release ("alt", "ctrl", "shift").

        Browsers report keydown/keyup for the modifier keys themselves; the
        terminal path gets the same information from kitty's event-type flag.
        Encoding it as the matching CSI u sequence keeps both paths identical.
        """
        codes = {'shift': 57441, 'ctrl': 57442, 'alt': 57443}
        code = codes.get(name)
        if code is None or action not in ('press', 'release'):
            return
        if action == 'press':
            # mods includes the modifier itself (1 + alt bit = 3).
            bits = {'shift': 1, 'ctrl': 4, 'alt': 2}
            self.feed(f'\x1b[{code};{1 + bits[name]}:1 u')
        else:
            self.feed(f'\x1b[{code};1:3 u')

    # -- helpers ---------------------------------------------------
    def _deliver_direct(self, name: str) -> None:
        """Deliver a key name no terminal can encode (e.g. shift+enter)."""
        try:
            modifiers, base = split_key(name)
        except KeyError:
            return
        canonical = canonical_key(modifiers, base)
        self.events.put(KeyEvent(
            kind='key', key=canonical,
            ctrl='ctrl' in modifiers, alt='alt' in modifiers,
            shift='shift' in modifiers,
        ))
