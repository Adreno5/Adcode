"""Cross-platform keyboard, paste and mouse input for the full-screen UI.

Input arrives on a dedicated reader thread and is converted into
:class:`KeyEvent` objects.  The parser understands ANSI/VT sequences (arrows,
Home/End, PageUp/PageDown, modified keys, SGR mouse reports and bracketed
paste) and the legacy Windows scan codes returned by ``msvcrt`` when virtual
terminal input is unavailable.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass

_IS_WIN = os.name == 'nt'

if _IS_WIN:
    import ctypes
    import msvcrt

    _kernel32 = ctypes.windll.kernel32
    _STD_INPUT_HANDLE = -10


@dataclass(frozen=True)
class KeyEvent:
    kind: str = 'key'          # key | text | paste | mouse | resize
    key: str = ''              # canonical key name for kind == 'key'
    text: str = ''             # literal text for kind == 'text' / 'paste'
    ctrl: bool = False
    alt: bool = False
    shift: bool = False
    # Mouse fields.
    button: int = -1
    x: int = 0
    y: int = 0
    action: str = ''           # press | release | move | wheel_up | wheel_down
    mods: int = 0


_CTRL_NAMES = {
    '\x00': 'ctrl+space',
    '\x01': 'ctrl+a', '\x02': 'ctrl+b', '\x03': 'ctrl+c', '\x04': 'ctrl+d',
    '\x05': 'ctrl+e', '\x06': 'ctrl+f', '\x07': 'ctrl+g', '\x08': 'backspace',
    '\x09': 'tab', '\x0a': 'ctrl+j', '\x0b': 'ctrl+k', '\x0c': 'ctrl+l',
    '\x0d': 'enter', '\x0e': 'ctrl+n', '\x0f': 'ctrl+o', '\x10': 'ctrl+p',
    '\x11': 'ctrl+q', '\x12': 'ctrl+r', '\x13': 'ctrl+s', '\x14': 'ctrl+t',
    '\x15': 'ctrl+u', '\x16': 'ctrl+v', '\x17': 'ctrl+w', '\x18': 'ctrl+x',
    '\x19': 'ctrl+y', '\x1a': 'ctrl+z', '\x1c': 'ctrl+\\', '\x1d': 'ctrl+]',
    '\x1e': 'ctrl+^', '\x1f': 'ctrl+_',
}

_CSI_LETTER_KEYS = {
    'A': 'up', 'B': 'down', 'C': 'right', 'D': 'left',
    'F': 'end', 'H': 'home', 'P': 'f1', 'Q': 'f2', 'R': 'f3', 'S': 'f4',
    'Z': 'shift+tab',
}

_CSI_TILDE_KEYS = {
    1: 'home', 2: 'insert', 3: 'delete', 4: 'end', 5: 'pgup', 6: 'pgdn',
    7: 'home', 8: 'end', 11: 'f1', 12: 'f2', 13: 'f3', 14: 'f4', 15: 'f5',
    17: 'f6', 18: 'f7', 19: 'f8', 20: 'f9', 21: 'f10', 23: 'f11', 24: 'f12',
}

_LEGACY_WIN_KEYS = {
    'H': '\x1b[A', 'P': '\x1b[B', 'K': '\x1b[D', 'M': '\x1b[C',
    'G': '\x1b[H', 'O': '\x1b[F', 'I': '\x1b[5~', 'Q': '\x1b[6~',
    'S': '\x1b[3~', 'R': '\x1b[2~',
    ';': '\x1b[11~', '<': '\x1b[12~', '=': '\x1b[13~', '>': '\x1b[14~',
    '?': '\x1b[15~', '@': '\x1b[17~', 'A': '\x1b[18~', 'B': '\x1b[19~',
    'C': '\x1b[20~', 'D': '\x1b[21~', 'E': '\x1b[23~', 'F': '\x1b[24~',
}


def _modified_key(name: str, mods: int) -> tuple[str, bool, bool, bool]:
    """Map an xterm modifier parameter (1 + bitmask) onto our flags."""
    value = max(1, mods) - 1
    shift = bool(value & 1)
    alt = bool(value & 2)
    ctrl = bool(value & 4)
    prefix = ''
    if ctrl:
        prefix += 'ctrl+'
    if alt:
        prefix += 'alt+'
    if shift and name not in ('shift+tab',):
        prefix += 'shift+'
    return prefix + name, ctrl, alt, shift


class KeyParser:
    """Incremental parser for VT input sequences."""

    def __init__(self) -> None:
        self._buffer = ''
        self._esc_at = 0.0

    def feed(self, chunk: str, now: float | None = None) -> list[KeyEvent]:
        now = time.monotonic() if now is None else now
        self._buffer += chunk
        if chunk.endswith('\x1b'):
            self._esc_at = now
        return self._parse(now, final=False)

    def flush_pending(self, now: float | None = None) -> list[KeyEvent]:
        """Resolve an unfinished escape prefix as a stand-alone key."""
        now = time.monotonic() if now is None else now
        return self._parse(now, final=True)

    def pending_since(self) -> float:
        return self._esc_at

    def _parse(self, now: float, final: bool) -> list[KeyEvent]:
        events: list[KeyEvent] = []
        while self._buffer:
            if self._buffer.startswith('\x1b'):
                consumed = self._parse_escape(events, now, final)
                if consumed == 0:
                    break
                self._buffer = self._buffer[consumed:]
                continue
            ch = self._buffer[0]
            self._buffer = self._buffer[1:]
            events.append(self._plain_char(ch))
        return events

    def _plain_char(self, ch: str) -> KeyEvent:
        if ch in _CTRL_NAMES:
            name = _CTRL_NAMES[ch]
            return KeyEvent(kind='key', key=name, ctrl=name.startswith('ctrl+'))
        if ch == '\x1b':
            return KeyEvent(kind='key', key='esc')
        if ch == '\x7f':
            return KeyEvent(kind='key', key='backspace')
        return KeyEvent(kind='text', text=ch)

    def _parse_escape(self, events: list[KeyEvent], now: float, final: bool) -> int:
        buffer = self._buffer
        if len(buffer) == 1:
            if final or now - self._esc_at >= 0.035:
                events.append(KeyEvent(kind='key', key='esc'))
                return 1
            return 0

        # Alt+<char>
        second = buffer[1]
        if second not in ('[', 'O', ']', 'P'):
            if len(buffer) < 2:
                return 0
            last = self._plain_char(second)
            if last.kind == 'key':
                base = last.key
                if base.startswith('ctrl+'):
                    name = 'alt+' + base
                else:
                    name = 'alt+' + base
                events.append(KeyEvent(kind='key', key=name, alt=True,
                                       ctrl=last.ctrl, shift=last.shift))
            else:
                events.append(KeyEvent(kind='key', key='alt+' + second.lower(),
                                       alt=True, shift=second.isupper()))
            return 2

        if second == 'O':
            if len(buffer) < 3:
                return 0
            code = buffer[2]
            name = _CSI_LETTER_KEYS.get(code)
            if name:
                events.append(KeyEvent(kind='key', key=name))
            return 3

        if second == ']':
            # OSC sequence (e.g. terminal replies): swallow through BEL/ST.
            end = buffer.find('\x07')
            st = buffer.find('\x1b\\')
            stop = min([value for value in (end, st) if value != -1], default=-1)
            if stop == -1:
                return len(buffer) if final else 0
            consumed = stop + (2 if buffer[stop:stop + 2] == '\x1b\\' else 1)
            return consumed

        if second == 'P':
            # Device control / application sequences: ignore them.
            end = buffer.find('\x1b\\')
            if end == -1:
                return len(buffer) if final else 0
            return end + 2

        # CSI
        if len(buffer) < 3:
            return 0
        intro = buffer[2]
        if intro == '<':
            return self._parse_mouse(buffer, events)
        if intro == 'M':
            # Legacy X10 mouse report (6 bytes total).
            if len(buffer) < 6:
                return len(buffer) if final else 0
            return self._parse_x10_mouse(buffer, events)
        # Find the final byte in 0x40..0x7e; it can be the first byte after '['.
        index = 2
        while index < len(buffer):
            code = ord(buffer[index])
            if 0x40 <= code <= 0x7E:
                break
            index += 1
        if index >= len(buffer):
            return len(buffer) if final else 0

        params = buffer[2:index]
        final_byte = buffer[index]
        consumed = index + 1

        if final_byte == '~':
            # Bracketed paste.
            if params == '200':
                start = buffer.find('\x1b[201~', 2)
                if start == -1:
                    return len(buffer) if final else 0
                text = buffer[6:start]
                events.append(KeyEvent(kind='paste', text=text))
                return start + 6
            try:
                number = int(params.split(';')[0]) if params else 0
            except ValueError:
                number = 0
            name = _CSI_TILDE_KEYS.get(number)
            if name:
                mods = 1
                parts = params.split(';')
                if len(parts) > 1:
                    try:
                        mods = int(parts[1])
                    except ValueError:
                        mods = 1
                final_name, ctrl, alt, shift = _modified_key(name, mods)
                events.append(KeyEvent(kind='key', key=final_name, ctrl=ctrl,
                                       alt=alt, shift=shift, mods=mods))
            return consumed

        name = _CSI_LETTER_KEYS.get(final_byte)
        if name is None:
            return consumed
        parts = params.split(';') if params else []
        mods = 1
        if len(parts) > 1:
            try:
                mods = int(parts[1])
            except ValueError:
                mods = 1
        final_name, ctrl, alt, shift = _modified_key(name, mods)
        events.append(KeyEvent(kind='key', key=final_name, ctrl=ctrl, alt=alt,
                               shift=shift, mods=mods))
        return consumed

    def _parse_mouse(self, buffer: str, events: list[KeyEvent]) -> int:
        end = buffer.find('M', 3)
        end_release = buffer.find('m', 3)
        if end == -1 or (end_release != -1 and end_release < end):
            end = end_release
        if end == -1:
            return len(buffer) if len(buffer) < 64 else 0
        body = buffer[3:end]
        try:
            button, x, y = (int(value) for value in body.split(';'))
        except ValueError:
            return end + 1
        base = button & 0b1100_0011
        action = 'press'
        if button & 64:
            action = 'wheel_up' if base == 64 else 'wheel_down'
        elif button & 32:
            action = 'move'
        elif buffer[end] == 'm':
            action = 'release'
        events.append(KeyEvent(
            kind='mouse', button=base, x=x, y=y, action=action,
            shift=bool(button & 4), alt=bool(button & 8), ctrl=bool(button & 16),
        ))
        return end + 1

    def _parse_x10_mouse(self, buffer: str, events: list[KeyEvent]) -> int:
        try:
            button = ord(buffer[3]) - 32
            x = ord(buffer[4]) - 32
            y = ord(buffer[5]) - 32
        except (IndexError, TypeError):
            return len(buffer)
        action = 'press'
        if button & 64:
            action = 'wheel_up' if (button & 3) == 0 else 'wheel_down'
        elif button & 3 == 3:
            action = 'release'
        events.append(KeyEvent(kind='mouse', button=button & 3, x=x, y=y,
                               action=action))
        return 6


def _win_read_char() -> str:
    ch = msvcrt.getwch()
    if ch in ('\x00', '\xe0'):
        code = msvcrt.getwch()
        return _LEGACY_WIN_KEYS.get(code, '')
    code = ord(ch)
    if 0xD800 <= code <= 0xDBFF:
        low = msvcrt.getwch()
        return ch + low
    return ch


def _win_input_ready() -> bool:
    try:
        return msvcrt.kbhit()
    except Exception:
        return False


class InputThread(threading.Thread):
    """Reads stdin on a daemon thread and pushes :class:`KeyEvent` objects."""

    def __init__(self, event_queue: 'queue.Queue[KeyEvent]') -> None:
        super().__init__(name='adcode-input', daemon=True)
        self._events = event_queue
        self._stop = threading.Event()
        self._parser = KeyParser()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        if _IS_WIN:
            self._run_windows()
        else:
            self._run_posix()

    def _deliver(self, events: list[KeyEvent]) -> None:
        for event in events:
            if self._stop.is_set():
                return
            self._events.put(event)

    def _run_windows(self) -> None:
        while not self._stop.is_set():
            try:
                ch = _win_read_char()
            except Exception:
                time.sleep(0.01)
                continue
            events = self._parser.feed(ch)
            if not ch:
                continue
            if ch == '\x1b' and not _win_input_ready():
                # Give escape sequences a short window to arrive.
                deadline = time.monotonic() + 0.03
                while time.monotonic() < deadline and not _win_input_ready():
                    time.sleep(0.004)
                if not _win_input_ready():
                    events.extend(self._parser.flush_pending())
            self._deliver(events)

    def _run_posix(self) -> None:
        import select
        import sys
        fd = sys.stdin.fileno()
        decoder = __import__('codecs').getincrementaldecoder('utf-8')(errors='replace')
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([fd], [], [], 0.05)
            except Exception:
                time.sleep(0.02)
                continue
            if not ready:
                if self._parser.pending_since() and \
                        time.monotonic() - self._parser.pending_since() >= 0.035:
                    self._deliver(self._parser.flush_pending())
                continue
            try:
                data = os.read(fd, 4096)
            except OSError:
                time.sleep(0.01)
                continue
            if not data:
                continue
            chunk = decoder.decode(data)
            events = self._parser.feed(chunk)
            if self._parser.pending_since() and \
                    time.monotonic() - self._parser.pending_since() >= 0.035:
                events.extend(self._parser.flush_pending())
            self._deliver(events)


# ── clipboard helpers ──────────────────────────────────────


def get_clipboard() -> str:
    try:
        import pyperclip
        return pyperclip.paste()
    except Exception:
        pass
    try:
        import subprocess
        import sys
        if _IS_WIN:
            result = subprocess.check_output(
                ['powershell', '-NoProfile', '-Command', 'Get-Clipboard'],
                text=True, encoding='utf-8-sig', stderr=subprocess.DEVNULL,
                timeout=5,
            )
        elif sys.platform == 'darwin':
            result = subprocess.check_output(
                ['pbpaste'], text=True, encoding='utf-8-sig',
                stderr=subprocess.DEVNULL, timeout=5,
            )
        else:
            result = subprocess.check_output(
                ['xclip', '-selection', 'clipboard', '-o'], text=True,
                encoding='utf-8-sig', stderr=subprocess.DEVNULL, timeout=5,
            )
        return result.rstrip('\n')
    except Exception:
        return ''


def set_clipboard(text: str) -> bool:
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        return False
