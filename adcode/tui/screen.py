"""Terminal lifecycle and frame output.

On Windows the console is driven through ``WriteConsoleW`` so Unicode output is
exact regardless of the active code page; on POSIX systems frames are written
as UTF-8 bytes.  The screen enters the alternate buffer, hides the cursor and
enables bracketed paste plus SGR mouse reporting, so the app owns every cell
on every frame.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass

_IS_WIN = os.name == 'nt'

if _IS_WIN:
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.windll.kernel32
    _STD_INPUT_HANDLE = -10
    _STD_OUTPUT_HANDLE = -11

    _ENABLE_PROCESSED_INPUT = 0x0001
    _ENABLE_LINE_INPUT = 0x0002
    _ENABLE_ECHO_INPUT = 0x0004
    _ENABLE_WINDOW_INPUT = 0x0008
    _ENABLE_MOUSE_INPUT = 0x0010
    _ENABLE_VIRTUAL_TERMINAL_INPUT = 0x0200
    _ENABLE_EXTENDED_FLAGS = 0x0080

    _ENABLE_PROCESSED_OUTPUT = 0x0001
    _ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
    _DISABLE_NEWLINE_AUTO_RETURN = 0x0008

    class _COORD(ctypes.Structure):
        _fields_ = [('X', ctypes.c_short), ('Y', ctypes.c_short)]

    class _SMALL_RECT(ctypes.Structure):
        _fields_ = [
            ('Left', ctypes.c_short), ('Top', ctypes.c_short),
            ('Right', ctypes.c_short), ('Bottom', ctypes.c_short),
        ]

    class _CONSOLE_SCREEN_BUFFER_INFO(ctypes.Structure):
        _fields_ = [
            ('dwSize', _COORD),
            ('dwCursorPosition', _COORD),
            ('wAttributes', ctypes.c_ushort),
            ('srWindow', _SMALL_RECT),
            ('dwMaximumWindowSize', _COORD),
        ]


@dataclass
class TerminalSize:
    columns: int
    rows: int

    def __iter__(self):
        yield self.columns
        yield self.rows


class TerminalScreen:
    """Owns terminal modes and writes complete frames."""

    def __init__(self) -> None:
        self._entered = False
        self._old_out_mode: int | None = None
        self._old_in_mode: int | None = None
        self._out_handle = None
        self._in_handle = None
        self._last_size = TerminalSize(80, 24)

    # ── size ────────────────────────────────────────────────
    def size(self) -> TerminalSize:
        if _IS_WIN:
            try:
                handle = self._out_handle
                if handle:
                    info = _CONSOLE_SCREEN_BUFFER_INFO()
                    if _kernel32.GetConsoleScreenBufferInfo(
                            handle, ctypes.byref(info)):
                        columns = max(1, info.srWindow.Right - info.srWindow.Left + 1)
                        rows = max(1, info.srWindow.Bottom - info.srWindow.Top + 1)
                        self._last_size = TerminalSize(columns, rows)
                        return self._last_size
            except Exception:
                pass
        try:
            size = shutil.get_terminal_size(fallback=(80, 24))
            self._last_size = TerminalSize(max(1, size.columns), max(1, size.lines))
        except Exception:
            pass
        return self._last_size

    # ── modes ───────────────────────────────────────────────
    def enter(self) -> None:
        if self._entered:
            return
        self._entered = True
        if _IS_WIN:
            self._prepare_windows()
        else:
            self._prepare_posix()
        self.write(
            '\x1b[?1049h'       # alternate screen buffer
            '\x1b[?25l'         # hide cursor
            '\x1b[?2004h'       # bracketed paste
            '\x1b[?1000h'       # basic mouse reporting
            '\x1b[?1006h'       # SGR extended mouse coordinates
            '\x1b[2J\x1b[H'
        )

    def leave(self) -> None:
        if not self._entered:
            return
        self._entered = False
        self.write(
            '\x1b[?1006l\x1b[?1000l\x1b[?2004l'
            '\x1b[?25h'
            '\x1b[0m'
            '\x1b[?1049l'
        )
        if _IS_WIN:
            self._restore_windows()
        else:
            self._restore_posix()

    def _prepare_windows(self) -> None:
        self._out_handle = _kernel32.GetStdHandle(_STD_OUTPUT_HANDLE)
        self._in_handle = _kernel32.GetStdHandle(_STD_INPUT_HANDLE)
        out_mode = wintypes.DWORD()
        in_mode = wintypes.DWORD()
        if _kernel32.GetConsoleMode(self._out_handle, ctypes.byref(out_mode)):
            self._old_out_mode = int(out_mode.value)
        if _kernel32.GetConsoleMode(self._in_handle, ctypes.byref(in_mode)):
            self._old_in_mode = int(in_mode.value)
        new_out = _ENABLE_PROCESSED_OUTPUT | _ENABLE_VIRTUAL_TERMINAL_PROCESSING
        _kernel32.SetConsoleMode(self._out_handle, new_out)
        new_in = (
            _ENABLE_WINDOW_INPUT | _ENABLE_EXTENDED_FLAGS
            | _ENABLE_VIRTUAL_TERMINAL_INPUT | _ENABLE_MOUSE_INPUT
        )
        # Clear line/echo/processed input so Ctrl+C arrives as a keystroke.
        _kernel32.SetConsoleMode(self._in_handle, new_in)

    def _restore_windows(self) -> None:
        try:
            if self._old_out_mode is not None:
                _kernel32.SetConsoleMode(self._out_handle, self._old_out_mode)
            if self._old_in_mode is not None:
                _kernel32.SetConsoleMode(self._in_handle, self._old_in_mode)
        except Exception:
            pass

    def _prepare_posix(self) -> None:
        try:
            import termios
            import tty
            fd = sys.stdin.fileno()
            self._old_termios = termios.tcgetattr(fd)
            tty.setraw(fd)
        except Exception:
            self._old_termios = None

    def _restore_posix(self) -> None:
        try:
            import termios
            if getattr(self, '_old_termios', None) is not None:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN,
                                  self._old_termios)
        except Exception:
            pass

    # ── output ──────────────────────────────────────────────
    def write(self, text: str) -> None:
        if not text:
            return
        if _IS_WIN:
            try:
                buffer = ctypes.create_unicode_buffer(text)
                # WriteConsoleW counts UTF-16 code units, not code points.
                length = len(text.encode('utf-16-le')) // 2
                written = wintypes.DWORD()
                ok = _kernel32.WriteConsoleW(
                    self._out_handle,
                    buffer,
                    length,
                    ctypes.byref(written),
                    None,
                )
                if ok and written.value:
                    return
            except Exception:
                pass
        try:
            data = text.encode('utf-8', errors='replace')
            try:
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
            except Exception:
                sys.stdout.write(text)
                sys.stdout.flush()
        except Exception:
            pass

    def flush_frame(self, frame: str) -> None:
        self.write(frame)

    def set_title(self, title: str) -> None:
        safe = title.replace('\x07', '')
        self.write(f'\x1b]0;{safe}\x07')

    def bell(self) -> None:
        self.write('\x07')

    # ── cursor ─────────────────────────────────────────────
    def set_cursor(self, x: int, y: int, visible: bool = True) -> None:
        """Place the hardware cursor after drawing (1-based terminal coords)."""
        self.write(f'\x1b[{max(1, y + 1)};{max(1, x + 1)}H' + ('\x1b[?25h' if visible else '\x1b[?25l'))

    def hide_cursor(self) -> None:
        self.write('\x1b[?25l')
