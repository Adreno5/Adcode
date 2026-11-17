"""Cross-platform keyboard, paste and mouse input for the full-screen UI.

Input arrives on a dedicated reader thread and is converted into
:class:`KeyEvent` objects.  The parser understands ANSI/VT sequences (arrows,
Home/End, PageUp/PageDown, modified keys, SGR mouse reports and bracketed
paste) and the legacy Windows scan codes returned by ``msvcrt`` when virtual
terminal input is unavailable.

Windows consoles need one extra hop: ``msvcrt.getwch`` reads through
``ReadConsole``, which silently discards mouse records, so conhost's wheel
never reaches the VT parser.  The reader therefore drains mouse records from
the console input buffer itself and turns them into wheel events, leaving
keyboard records for ``msvcrt``.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass

from adcode.tui import clipboard

_IS_WIN = os.name == 'nt'

if _IS_WIN:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _kernel32 = ctypes.windll.kernel32
    _STD_INPUT_HANDLE = -10

    _KEY_EVENT = 0x0001
    _MOUSE_EVENT = 0x0002
    _MOUSE_WHEELED = 0x0004
    _MOUSE_HWHEELED = 0x0008
    _SHIFT_PRESSED = 0x0010
    _ALT_PRESSED = 0x0003        # left | right alt
    _CTRL_PRESSED = 0x000C       # left | right ctrl

    class _COORD(ctypes.Structure):
        _fields_ = [('X', ctypes.c_short), ('Y', ctypes.c_short)]

    class _MOUSE_EVENT_RECORD(ctypes.Structure):
        _fields_ = [
            ('dwMousePosition', _COORD),
            ('dwButtonState', wintypes.DWORD),
            ('dwControlKeyState', wintypes.DWORD),
            ('dwEventFlags', wintypes.DWORD),
        ]

    class _KEY_EVENT_RECORD(ctypes.Structure):
        _fields_ = [
            ('bKeyDown', wintypes.BOOL),
            ('wRepeatCount', wintypes.WORD),
            ('wVirtualKeyCode', wintypes.WORD),
            ('wVirtualScanCode', wintypes.WORD),
            ('UnicodeChar', wintypes.WCHAR),
            ('dwControlKeyState', wintypes.DWORD),
        ]

    class _INPUT_EVENT(ctypes.Union):
        _fields_ = [
            ('KeyEvent', _KEY_EVENT_RECORD),
            ('MouseEvent', _MOUSE_EVENT_RECORD),
        ]

    class _INPUT_RECORD(ctypes.Structure):
        _fields_ = [('EventType', wintypes.WORD), ('Event', _INPUT_EVENT)]

    def _win_console_input() -> int | None:
        """Return the console input buffer handle, or None when stdin is a pipe."""
        handle = _kernel32.GetStdHandle(_STD_INPUT_HANDLE)
        if not handle or handle == ctypes.c_void_p(-1).value:
            return None
        return handle

    def _win_peek_record(handle: int) -> _INPUT_RECORD | None:
        record = _INPUT_RECORD()
        read = wintypes.DWORD()
        if not _kernel32.PeekConsoleInputW(handle, ctypes.byref(record), 1,
                                           ctypes.byref(read)):
            return None
        return record if read.value else None

    def _win_read_record(handle: int) -> _INPUT_RECORD | None:
        record = _INPUT_RECORD()
        read = wintypes.DWORD()
        if not _kernel32.ReadConsoleInputW(handle, ctypes.byref(record), 1,
                                           ctypes.byref(read)):
            return None
        return record if read.value else None

    def _win_wheel_event(mouse: _MOUSE_EVENT_RECORD) -> KeyEvent | None:
        """Convert a console mouse record into a wheel event, if it is one."""
        flags = mouse.dwEventFlags
        if not flags & _MOUSE_WHEELED or flags & _MOUSE_HWHEELED:
            return None
        # The high word of dwButtonState holds a signed wheel delta.
        delta = ctypes.c_short((mouse.dwButtonState >> 16) & 0xFFFF).value
        if not delta:
            return None
        keys = mouse.dwControlKeyState
        return KeyEvent(
            kind='mouse',
            x=mouse.dwMousePosition.X + 1,
            y=mouse.dwMousePosition.Y + 1,
            action='wheel_up' if delta > 0 else 'wheel_down',
            shift=bool(keys & _SHIFT_PRESSED),
            alt=bool(keys & _ALT_PRESSED),
            ctrl=bool(keys & _CTRL_PRESSED),
        )

    def _win_drain_records(events: list['KeyEvent']) -> None:
        """Consume console records that ``ReadConsole`` would silently drop.

        ``msvcrt.getwch`` filters every non-keyboard record, so a wheel moved
        on a Windows console would otherwise never reach the app.  Key-down
        records are left in place for ``msvcrt``; key releases, mouse, focus
        and window-size records are consumed here so the waitable input handle
        does not stay permanently signaled.
        """
        handle = _win_console_input()
        if handle is None:
            return
        while True:
            record = _win_peek_record(handle)
            if record is None:
                return
            if record.EventType == _KEY_EVENT:
                if record.Event.KeyEvent.bKeyDown:
                    return  # msvcrt owns keyboard input
                if _win_read_record(handle) is None:
                    return
                continue
            if _win_read_record(handle) is None:
                return
            if record.EventType != _MOUSE_EVENT:
                continue
            event = _win_wheel_event(record.Event.MouseEvent)
            if event is not None:
                events.append(event)



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
    # Backspace is \x7f (DEL) on modern terminals; \x08 is ctrl+h, so it can
    # be bound like any other chord.
    '\x00': 'ctrl+space',
    '\x01': 'ctrl+a', '\x02': 'ctrl+b', '\x03': 'ctrl+c', '\x04': 'ctrl+d',
    '\x05': 'ctrl+e', '\x06': 'ctrl+f', '\x07': 'ctrl+g', '\x08': 'ctrl+h',
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

# Keys the kitty keyboard protocol reports by code point instead of by
# character (``CSI <code> ; <mods> u``).  Every other code is the printable
# character itself, so ``ctrl+,`` arrives as ``CSI 44;5 u``: the legacy
# control-byte table has no entry for the comma, so no terminal can deliver
# that chord any other way.
_KITTY_KEYS = {
    27: 'esc',
    13: 'enter',
    9: 'tab',
    8: 'backspace',
    127: 'backspace',
    32: 'space',
}

# Kitty's private-use codes for the modifier keys themselves.  With flag 8
# ("report all keys as escape codes") plus flag 2 (event types) a terminal
# reports pressing and releasing Alt as its own CSI u sequence, which is the
# only way a terminal app can tell "Alt is held" from "Alt was tapped".
_KITTY_MODIFIERS = {
    57441: 'shift', 57442: 'ctrl', 57443: 'alt',
    57447: 'shift', 57448: 'ctrl', 57449: 'alt',
}

# event-type sub-parameter (the part after the colon in ``mods:event``).
_KITTY_EVENT_PRESS = 1
_KITTY_EVENT_REPEAT = 2
_KITTY_EVENT_RELEASE = 3

# With flag 8 the terminal reports the functional keys as private-use code
# points as well, so their legacy sequences never arrive and the names have to
# come from this table.
_KITTY_FUNCTIONAL = {
    57344: 'esc', 57345: 'enter', 57346: 'tab', 57347: 'backspace',
    57348: 'insert', 57349: 'delete', 57350: 'left', 57351: 'right',
    57352: 'up', 57353: 'down', 57354: 'pgup', 57355: 'pgdn',
    57356: 'home', 57357: 'end',
}
_KITTY_FUNCTIONAL.update({57364 + index: f'f{index + 1}'
                          for index in range(35)})

# Unicode private use areas.  Kitty reports its functional keys inside the BMP
# one, so a private-use code point that is in no table above is a key this app
# has no name for -- caps lock, a media key -- and must never become text.
_PRIVATE_USE_RANGES = ((0xE000, 0xF8FF), (0xF0000, 0xFFFFD),
                       (0x100000, 0x10FFFD))


def _is_private_use(code: int) -> bool:
    return any(low <= code <= high for low, high in _PRIVATE_USE_RANGES)

# Bits of the 1 + bitmask modifier parameter.
_KITTY_SHIFT_BIT = 1
_KITTY_ALT_BIT = 2
_KITTY_CTRL_BIT = 4
_KITTY_SUPER_BIT = 8

_LEGACY_WIN_KEYS = {
    'H': '\x1b[A', 'P': '\x1b[B', 'K': '\x1b[D', 'M': '\x1b[C',
    'G': '\x1b[H', 'O': '\x1b[F', 'I': '\x1b[5~', 'Q': '\x1b[6~',
    'S': '\x1b[3~', 'R': '\x1b[2~',
    ';': '\x1b[11~', '<': '\x1b[12~', '=': '\x1b[13~', '>': '\x1b[14~',
    '?': '\x1b[15~', '@': '\x1b[17~', 'A': '\x1b[18~', 'B': '\x1b[19~',
    'C': '\x1b[20~', 'D': '\x1b[21~', 'E': '\x1b[23~', 'F': '\x1b[24~',
}

# Bytes that can appear inside the parameters of a control sequence (SGR mouse
# reports, modified keys, terminal replies).  A run of these after a committed
# Esc is the tail of a report that was split across two reads.
_REPORT_PARAMETERS = frozenset('0123456789;:<=>?')


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


def _kitty_text(code: int, shifted: bool, code_field: list[str],
                fields: list[str]) -> str:
    """The characters a text key press generates.

    Flag 16 makes the terminal spell the text out, which is the only exact
    answer: the keyboard layout, dead keys and input methods all live in the
    terminal.  Without it the best available guess is the alternate code point
    reported for a shifted key (flag 4) and, failing that, the shift applied to
    a letter by hand.
    """
    if len(fields) > 2 and fields[2]:
        try:
            return ''.join(chr(int(part)) for part in fields[2].split(':')
                           if part)
        except (ValueError, OverflowError):
            pass
    if shifted and len(code_field) > 1 and code_field[1]:
        try:
            return chr(int(code_field[1]))
        except (ValueError, OverflowError):
            pass
    char = chr(code)
    upper = char.upper()
    if shifted and upper != char and len(upper) == 1:
        return upper
    return char


def _kitty_key_event(params: str) -> KeyEvent | None:
    """Turn a kitty keyboard protocol sequence into an event.

    The protocol reports a key as

        CSI <code>[:<shifted>[:<base>]] [;<mods>[:<event>]] [;<text>] u

    where <code> is the code point of the key, <mods> is one plus a bitmask,
    <event> is 1 press, 2 repeat or 3 release, and <text> is the text the key
    produced as colon separated code points.

    Two flags decide what a terminal sends:

    * flag 2 adds the event type, so a held key reports repeats and every key
      also reports its release;
    * flag 8 reports every key as a CSI u sequence -- text keys, the modifier
      keys and the functional keys -- instead of the legacy encoding.  The
      terminal then sends no plain text bytes at all, so the text of a press
      has to be rebuilt here (or read from <text> when the terminal honours
      flag 16 as well).
    """
    fields = params.split(';')
    code_field = fields[0].split(':')
    try:
        code = int(code_field[0])
    except ValueError:
        return None

    mods = 1
    event_type = _KITTY_EVENT_PRESS
    if len(fields) > 1:
        mod_field = fields[1].split(':')
        try:
            mods = int(mod_field[0])
        except ValueError:
            mods = 1
        if len(mod_field) > 1:
            try:
                event_type = int(mod_field[1])
            except ValueError:
                event_type = _KITTY_EVENT_PRESS

    # The modifier keys themselves: the only way to tell a held Alt from a
    # tapped one.
    modifier = _KITTY_MODIFIERS.get(code)
    if modifier is not None:
        if event_type == _KITTY_EVENT_RELEASE:
            return KeyEvent(kind='modifier', key=modifier, action='release',
                            mods=mods)
        if event_type == _KITTY_EVENT_REPEAT:
            # A held modifier repeats; the press already started the hold.
            return None
        return KeyEvent(kind='modifier', key=modifier, action='press',
                        mods=mods)

    # A release ends a key that already fired on its press, but a repeat is the
    # key still being held: Backspace and the arrows auto-repeat this way, so a
    # repeat has to behave exactly like another press.
    if event_type == _KITTY_EVENT_RELEASE:
        return None

    base = _KITTY_FUNCTIONAL.get(code) or _KITTY_KEYS.get(code)
    if base is not None and base != 'space':
        name, ctrl, alt, shift = _modified_key(base, mods)
        return KeyEvent(kind='key', key=name, ctrl=ctrl, alt=alt,
                        shift=shift, mods=mods)

    value = max(1, mods) - 1
    shift = bool(value & _KITTY_SHIFT_BIT)
    alt = bool(value & _KITTY_ALT_BIT)
    ctrl = bool(value & _KITTY_CTRL_BIT)

    if len(fields) > 2 and fields[2]:
        # The terminal says this press produced text: an input method commit,
        # an AltGr character or a plain character.  Insert it.
        text = _kitty_text(code, shift, code_field, fields)
        return KeyEvent(kind='text', text=text, mods=mods) if text else None

    if value & _KITTY_SUPER_BIT:
        # No super bindings exist, and inventing text for a shortcut would
        # type a character the user never asked for.
        return None

    if code < 32 or code == 127:
        # A control code point without chord bits: nothing to insert.
        return None

    if _is_private_use(code) or 0xD800 <= code <= 0xDFFF:
        # An unnamed functional key, or half of a surrogate pair: not text.
        return None

    if ctrl or alt:
        base_char = 'space' if code == 32 else chr(code)
        name, ctrl, alt, shift = _modified_key(base_char, mods)
        return KeyEvent(kind='key', key=name, ctrl=ctrl, alt=alt,
                        shift=shift, mods=mods)

    return KeyEvent(kind='text', text=_kitty_text(code, shift, code_field,
                                                  fields), mods=mods)


class KeyParser:
    """Incremental parser for VT input sequences."""

    # A lone Esc is only resolved once nothing followed it for this long.
    _ESC_TIMEOUT = 0.09
    # An unfinished control sequence (mouse report, key report, terminal
    # reply) is never typed into the prompt; after this long it is dropped.
    _CSI_TIMEOUT = 0.4
    # A bracketed paste can legitimately take seconds to stream in.
    _PASTE_TIMEOUT = 10.0
    # Right after an Esc was committed, a sequence tail may still arrive: it
    # belongs to the report that Esc interrupted, so swallow it.
    _TAIL_WINDOW = 0.5

    def __init__(self) -> None:
        self._buffer = ''
        self._esc_at = 0.0
        self._started = 0.0
        self._tail_until = 0.0
        self._tail_final = False

    def feed(self, chunk: str, now: float | None = None) -> list[KeyEvent]:
        now = time.monotonic() if now is None else now
        if not self._buffer:
            # A new (possibly escape-prefixed) sequence starts here.
            self._started = now
            self._esc_at = now
        self._buffer += chunk
        events = self._parse(now, final=False)
        # Any leftover bytes are an unfinished escape sequence; remember when it
        # started so the reader thread can resolve it as a stand-alone key.
        self._esc_at = self._started if self._buffer else 0.0
        return events

    def flush_pending(self, now: float | None = None) -> list[KeyEvent]:
        """Resolve an unfinished escape prefix as a stand-alone key."""
        now = time.monotonic() if now is None else now
        events = self._parse(now, final=True)
        self._esc_at = 0.0
        return events

    def flush_due(self, now: float | None = None) -> list[KeyEvent]:
        """Flush a pending prefix that has waited longer than its window."""
        if not self._buffer:
            return []
        now = time.monotonic() if now is None else now
        if now - self._esc_at < self._timeout_for(self._buffer):
            return []
        return self.flush_pending(now)

    def _timeout_for(self, buffer: str) -> float:
        if buffer.startswith('\x1b[200~'):
            return self._PASTE_TIMEOUT
        if len(buffer) > 1 and buffer[1] in '[O]P':
            return self._CSI_TIMEOUT
        return self._ESC_TIMEOUT

    def pending_since(self) -> float:
        return self._esc_at

    def _abandon(self, buffer: str, now: float, final: bool) -> int:
        """Drop an unfinished control sequence instead of typing its tail.

        Splitting a report across two reads is normal on a busy terminal; the
        tail must never reach the prompt as text, and it must not resolve as a
        stray Esc either.
        """
        if not final and now - self._esc_at < self._CSI_TIMEOUT:
            return 0
        # The remaining bytes of this report may still arrive; swallow them.
        self._tail_until = now + self._TAIL_WINDOW
        self._tail_final = False
        return len(buffer)

    def _parse(self, now: float, final: bool) -> list[KeyEvent]:
        events: list[KeyEvent] = []
        while self._buffer:
            if self._buffer.startswith('\x1b'):
                consumed = self._parse_escape(events, now, final)
                if consumed == 0:
                    break
                self._buffer = self._buffer[consumed:]
                continue
            if self._tail_until and now <= self._tail_until:
                if self._looks_like_sequence_tail(self._buffer):
                    # The Esc that opened this report was already delivered;
                    # drop the rest of it rather than typing it into the prompt.
                    self._buffer = '\x1b' + self._buffer
                    consumed = self._parse_escape([], now, final)
                    if consumed == 0:
                        break
                    self._buffer = self._buffer[consumed:]
                    continue
                consumed = self._swallow_report_parameters()
                if consumed:
                    self._buffer = self._buffer[consumed:]
                    continue
            elif self._tail_until:
                # The tail window has passed; later bytes are ordinary typing.
                self._tail_until = 0.0
                self._tail_final = False
            ch = self._buffer[0]
            self._buffer = self._buffer[1:]
            events.append(self._plain_char(ch))
        return events

    def _swallow_report_parameters(self) -> int:
        """Consume the parameter/final bytes of an abandoned report.

        Used when a report was cut short mid-parameters and its tail arrives
        after the parser stopped waiting for it.  Only a report-shaped tail is
        swallowed: a lone printable typed after Esc -- ``/`` right after closing
        a dialog, say -- is ordinary typing and must reach the prompt.
        """
        buffer = self._buffer
        if self._tail_final:
            # The parameters were swallowed on an earlier read; this is the
            # final byte that closes the report.
            self._tail_final = False
            if 0x40 <= ord(buffer[0]) <= 0x7E:
                self._tail_until = 0.0
                return 1
        index = 0
        while index < len(buffer) and buffer[index] in _REPORT_PARAMETERS:
            index += 1
        if 0 < index < len(buffer) and 0x40 <= ord(buffer[index]) <= 0x7E:
            self._tail_until = 0.0
            return index + 1
        if index >= 2 and index == len(buffer):
            # A numeric run with no final byte yet: keep waiting for the rest.
            self._tail_final = True
            return index
        # Anything shorter is a keypress, not the tail of a report.
        self._tail_until = 0.0
        return 0

    @staticmethod
    def _looks_like_sequence_tail(buffer: str) -> bool:
        """True when ``buffer`` looks like the rest of a split control sequence.

        Guards the swallow-after-Esc path so ordinary typing stays intact: only
        a CSI/SS3 shape qualifies, not an arbitrary ``[`` the user typed.
        """
        if not buffer:
            return False
        if buffer[0] == 'O':
            return len(buffer) > 1 and buffer[1].isalpha()
        if buffer[0] not in '[]P':
            return False
        if len(buffer) == 1:
            return True
        return buffer[1] in '0123456789;<M'

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
            if final or now - self._esc_at >= self._ESC_TIMEOUT:
                self._tail_until = now + self._TAIL_WINDOW
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
                return self._abandon(buffer, now, final)
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
                return self._abandon(buffer, now, final)
            consumed = stop + (2 if buffer[stop:stop + 2] == '\x1b\\' else 1)
            return consumed

        if second == 'P':
            # Device control / application sequences: ignore them.
            end = buffer.find('\x1b\\')
            if end == -1:
                return self._abandon(buffer, now, final)
            return end + 2

        # CSI
        if len(buffer) < 3:
            return self._abandon(buffer, now, final)
        intro = buffer[2]
        if intro == '<':
            return self._parse_mouse(buffer, events, now, final)
        if intro == 'M':
            # Legacy X10 mouse report (6 bytes total).
            if len(buffer) < 6:
                return self._abandon(buffer, now, final)
            return self._parse_x10_mouse(buffer, events)
        # Find the final byte in 0x40..0x7e; it can be the first byte after '['.
        index = 2
        while index < len(buffer):
            code = ord(buffer[index])
            if 0x40 <= code <= 0x7E:
                break
            index += 1
        if index >= len(buffer):
            return self._abandon(buffer, now, final)

        params = buffer[2:index]
        final_byte = buffer[index]
        consumed = index + 1

        if final_byte == 'u':
            # The kitty keyboard protocol: CSI <code>[:alt][:base] [;mods] u.
            event = _kitty_key_event(params)
            if event is not None:
                events.append(event)
            return consumed

        if final_byte == '~':
            # Bracketed paste.
            if params == '200':
                start = buffer.find('\x1b[201~', 2)
                if start == -1:
                    # Guard against a paste whose terminator never arrives, but
                    # give a large paste time to stream in first.
                    if not final and now - self._esc_at < self._PASTE_TIMEOUT:
                        return 0
                    text = buffer[6:]
                    if text:
                        events.append(KeyEvent(kind='paste', text=text))
                    return len(buffer)
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

    def _parse_mouse(self, buffer: str, events: list[KeyEvent],
                     now: float, final: bool) -> int:
        end = buffer.find('M', 3)
        end_release = buffer.find('m', 3)
        if end == -1 or (end_release != -1 and end_release < end):
            end = end_release
        if end == -1:
            return self._abandon(buffer, now, final)
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


def _win_wait_input(seconds: float) -> None:
    """Block until console input arrives (or the timeout elapses).

    The console input buffer is waitable and becomes signaled for mouse
    records too, so the reader wakes as soon as the wheel moves instead of
    polling ``kbhit`` (which only reports keyboard events).
    """
    handle = _win_console_input()
    if handle is None:
        time.sleep(seconds)
        return
    wait = _kernel32.WaitForSingleObject(handle, max(1, int(seconds * 1000)))
    if wait not in (0, 0x00000102):  # WAIT_OBJECT_0, WAIT_TIMEOUT
        # Not a waitable console handle (redirected stdin): fall back to sleep.
        time.sleep(seconds)


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
            events: list[KeyEvent] = []
            # ReadConsole drops mouse records, so take the wheel out of the
            # console buffer before asking msvcrt for the next keystroke.
            _win_drain_records(events)
            if _win_input_ready():
                try:
                    ch = _win_read_char()
                except Exception:
                    time.sleep(0.01)
                    continue
                if ch:
                    events.extend(self._parser.feed(ch))
                self._deliver(events)
                continue
            if not events:
                # Sleep until the console signals a record; kbhit() only knows
                # about keyboard input, so polling it would miss the wheel.
                _win_wait_input(0.03)
            # Resolve a prefix that has waited past its own window: a lone Esc
            # key press, or a report the terminal split across two writes.
            events.extend(self._parser.flush_due())
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
                self._deliver(self._parser.flush_due())
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
            events.extend(self._parser.flush_due())
            self._deliver(events)


# ── sequence building ─────────────────────────────────────
# The inverse of the parser above: the bytes a terminal sends for a canonical
# key name.  A real terminal encodes these chords itself, but an input surface
# that is not a terminal -- an on-screen key bar, a browser -- has to produce
# them, and going through the same bytes keeps a single parsing path.

_MODIFIER_BITS = {'ctrl': 4, 'alt': 2, 'shift': 1}

# The order _modified_key writes into a key name.
_MODIFIER_ORDER = ('ctrl', 'alt', 'shift')

_CSI_LETTER_KEYS_INV = {'up': 'A', 'down': 'B', 'right': 'C', 'left': 'D',
                        'home': 'H', 'end': 'F'}

_TILDE_KEY_CODES = {'insert': 2, 'delete': 3, 'pgup': 5, 'pgdn': 6,
                    'f1': 11, 'f2': 12, 'f3': 13, 'f4': 14, 'f5': 15,
                    'f6': 17, 'f7': 18, 'f8': 19, 'f9': 20, 'f10': 21,
                    'f11': 23, 'f12': 24}

# Keys that have a plain byte of their own and a kitty code point.
_PLAIN_KEY_BYTES = {'esc': '\x1b', 'enter': '\r', 'tab': '\t',
                    'backspace': '\x7f', 'space': ' '}

_KITTY_KEY_CODES = {'esc': 27, 'enter': 13, 'tab': 9, 'backspace': 127,
                    'space': 32}


def split_key(name: str) -> tuple[tuple[str, ...], str]:
    """Split "ctrl+shift+left" into (("ctrl", "shift"), "left")."""
    if not name:
        raise KeyError(name)
    parts = name.split('+')
    base = parts[-1]
    modifiers = parts[:-1]
    if not base or set(modifiers) - set(_MODIFIER_ORDER):
        raise KeyError(name)
    return tuple(mod for mod in _MODIFIER_ORDER if mod in modifiers), base


def canonical_key(modifiers, base: str) -> str:
    """Build a key name in the modifier order the parser produces."""
    order = [mod for mod in _MODIFIER_ORDER if mod in modifiers]
    return '+'.join([*order, base])


def key_sequence(name: str) -> str:
    """Return the bytes a terminal sends for a canonical key name.

    "a", "ctrl+c", "alt+up", "shift+tab" and "f5" all map to the sequence
    KeyParser turns back into that same name, so a caller that is not a
    terminal can drive the app exactly like a keyboard does.  Chords on
    punctuation and on control characters travel through the kitty keyboard
    protocol, the only encoding that can carry them.

    Raises KeyError for names no terminal encoding can express.
    """
    modifiers, base = split_key(name)
    bits = sum(_MODIFIER_BITS[mod] for mod in modifiers)
    if base in _CSI_LETTER_KEYS_INV:
        letter = _CSI_LETTER_KEYS_INV[base]
        if bits:
            return f'\x1b[1;{bits + 1}{letter}'
        return '\x1b[' + letter
    if base == 'tab' and bits == _MODIFIER_BITS['shift']:
        # shift+tab is old enough to have a sequence of its own.
        return '\x1b[Z'
    if base in _TILDE_KEY_CODES:
        code = _TILDE_KEY_CODES[base]
        if bits:
            return f'\x1b[{code};{bits + 1}~'
        return f'\x1b[{code}~'
    if base in _KITTY_KEY_CODES:
        if not bits:
            return _PLAIN_KEY_BYTES[base]
        return f'\x1b[{_KITTY_KEY_CODES[base]};{bits + 1}u'
    if len(base) == 1 and ' ' <= base < '\x7f':
        if not bits:
            return base
        return f'\x1b[{ord(base)};{bits + 1}u'
    raise KeyError(name)


# ── clipboard helpers ──────────────────────────────────────

# The clipboard lives in its own module; the names are re-exported here so
# every input surface can reach them through the input layer.
get_clipboard = clipboard.get_clipboard
set_clipboard = clipboard.set_clipboard
is_paste_key = clipboard.is_paste_key
PASTE_KEYS = clipboard.PASTE_KEYS
