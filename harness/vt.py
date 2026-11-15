"""A small ANSI/VT screen emulator for verifying serialized TUI frames.

The AdCode canvas emits a full frame as ``CSI H`` followed by one SGR-styled
row per line, separated by CRLF.  This module replays that byte stream into a
cell grid, so tests can assert on *what the terminal would show* and not just on
the canvas object that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

from adcode.tui.canvas import char_width

DEFAULT_FG = (255, 255, 255)
DEFAULT_BG = (0, 0, 0)


@dataclass
class VTCell:
    ch: str = ' '
    fg: tuple[int, int, int] = DEFAULT_FG
    bg: tuple[int, int, int] = DEFAULT_BG
    attrs: int = 0
    cont: bool = False


class VirtualScreen:
    """Replays ANSI output into a fixed-size cell grid."""

    def __init__(self, width: int = 80, height: int = 24) -> None:
        self.width = max(1, width)
        self.height = max(1, height)
        self.grid: list[list[VTCell]] = []
        self.x = 0
        self.y = 0
        self.pending_wrap = False
        self.fg = DEFAULT_FG
        self.bg = DEFAULT_BG
        self.attrs = 0
        self._blank()

    # ── grid ───────────────────────────────────────────────
    def _blank_row(self) -> list[VTCell]:
        return [VTCell(' ', self.fg, self.bg, 0, False)
                for _ in range(self.width)]

    def _blank(self) -> None:
        self.grid = [self._blank_row() for _ in range(self.height)]
        self.x = self.y = 0
        self.pending_wrap = False

    def resize(self, width: int, height: int) -> None:
        width, height = max(1, width), max(1, height)
        if width == self.width and height == self.height:
            return
        old, old_w, old_h = self.grid, self.width, self.height
        self.width, self.height = width, height
        self.grid = [self._blank_row() for _ in range(height)]
        for y in range(min(height, old_h)):
            for x in range(min(width, old_w)):
                self.grid[y][x] = old[y][x]
        self.x = min(self.x, width - 1)
        self.y = min(self.y, height - 1)
        self.pending_wrap = False

    # ── feeding ────────────────────────────────────────────
    def feed(self, data: str) -> None:
        index = 0
        length = len(data)
        while index < length:
            ch = data[index]
            if ch == '\x1b':
                index = self._escape(data, index)
            elif ch == '\r':
                self.x = 0
                self.pending_wrap = False
                index += 1
            elif ch == '\n':
                self._linefeed()
                self.pending_wrap = False
                index += 1
            elif ch == '\b':
                self.x = max(0, self.x - 1)
                self.pending_wrap = False
                index += 1
            elif ch == '\t':
                self.x = min(self.width, ((self.x // 8) + 1) * 8)
                self.pending_wrap = False
                index += 1
            elif ch == '\x07':
                index += 1
            else:
                self._put(ch)
                index += 1

    def _escape(self, data: str, start: int) -> int:
        if start + 1 >= len(data):
            return start + 1
        second = data[start + 1]
        if second == '[':
            index = start + 2
            while index < len(data):
                code = ord(data[index])
                if 0x40 <= code <= 0x7E:
                    final = data[index]
                    params = data[start + 2:index]
                    self._csi(final, params)
                    return index + 1
                index += 1
            return len(data)
        if second == ']':
            index = start + 2
            while index < len(data):
                if data[index] == '\x07':
                    return index + 1
                if data[index] == '\x1b' and index + 1 < len(data) \
                        and data[index + 1] == '\\':
                    return index + 2
                index += 1
            return len(data)
        if second in ('P', '^', '_'):
            index = start + 2
            marker = data.find('\x1b\\', index)
            return len(data) if marker == -1 else marker + 2
        return start + 2

    def _csi(self, final: str, params: str) -> None:
        if final in ('H', 'f'):
            parts = params.split(';') if params else []
            row = self._int(parts[0] if len(parts) > 0 else '1', 1)
            col = self._int(parts[1] if len(parts) > 1 else '1', 1)
            self.y = max(0, min(self.height - 1, row - 1))
            self.x = max(0, min(self.width - 1, col - 1))
            self.pending_wrap = False
            return
        if final == 'J':
            mode = self._int(params or '0', 0)
            if mode in (2, 3):
                self._blank()
            elif mode == 0:
                for x in range(self.x, self.width):
                    self.grid[self.y][x] = VTCell(' ', self.fg, self.bg, 0, False)
                for y in range(self.y + 1, self.height):
                    self.grid[y] = self._blank_row()
            return
        if final == 'K':
            mode = self._int(params or '0', 0)
            if mode == 0:
                for x in range(self.x, self.width):
                    self.grid[self.y][x] = VTCell(' ', self.fg, self.bg, 0, False)
            elif mode == 2:
                self.grid[self.y] = self._blank_row()
            return
        if final == 'm':
            self._sgr(params)
            return
        # Mode set/reset, cursor visibility, etc. do not change the grid.

    def _sgr(self, params: str) -> None:
        values = [self._int(part, 0) for part in params.split(';')] if params else [0]
        index = 0
        while index < len(values):
            value = values[index]
            if value == 0:
                self.fg = DEFAULT_FG
                self.bg = DEFAULT_BG
                self.attrs = 0
            elif value == 1:
                self.attrs |= 1
            elif value == 2:
                self.attrs |= 2
            elif value == 3:
                self.attrs |= 4
            elif value == 4:
                self.attrs |= 8
            elif value == 7:
                self.attrs |= 32
            elif value == 9:
                self.attrs |= 16
            elif value == 22:
                self.attrs &= ~(1 | 2)
            elif value == 39:
                self.fg = DEFAULT_FG
            elif value == 49:
                self.bg = DEFAULT_BG
            elif value in (38, 48) and index + 1 < len(values) \
                    and values[index + 1] == 2 and index + 4 < len(values):
                color = (values[index + 2], values[index + 3], values[index + 4])
                if value == 38:
                    self.fg = color
                else:
                    self.bg = color
                index += 4
            index += 1

    @staticmethod
    def _int(value: str, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    # ── cell painting ──────────────────────────────────────
    def _linefeed(self) -> None:
        if self.y + 1 >= self.height:
            self.grid.pop(0)
            self.grid.append(self._blank_row())
            self.y = self.height - 1
        else:
            self.y += 1

    def _put(self, ch: str) -> None:
        if not ch:
            return
        width = char_width(ch)
        if width == 0:
            # Combining mark: attach it to the previous cell.
            if self.x > 0:
                previous = self.grid[self.y][self.x - 1]
                if not previous.cont:
                    previous.ch += ch
            return
        if self.pending_wrap:
            self.x = 0
            self._linefeed()
            self.pending_wrap = False
        if self.x + width > self.width:
            self.x = 0
            self._linefeed()
        cell = self.grid[self.y][self.x]
        cell.ch = ch
        cell.fg = self.fg
        cell.bg = self.bg
        cell.attrs = self.attrs
        cell.cont = False
        if width == 2 and self.x + 1 < self.width:
            right = self.grid[self.y][self.x + 1]
            right.ch = ''
            right.fg = self.fg
            right.bg = self.bg
            right.attrs = self.attrs
            right.cont = True
        self.x += width
        if self.x >= self.width:
            self.x = self.width
            self.pending_wrap = True

    # ── snapshots ──────────────────────────────────────────
    def rows(self) -> tuple[tuple[VTCell, ...], ...]:
        return tuple(tuple(cell for cell in row) for row in self.grid)

    def line(self, y: int) -> str:
        if not (0 <= y < self.height):
            return ''
        return ''.join(cell.ch for cell in self.grid[y] if not cell.cont).rstrip()

    def text(self) -> str:
        return '\n'.join(self.line(y) for y in range(self.height))

    def nonempty_lines(self) -> list[str]:
        return [line for line in self.text().splitlines() if line.strip()]

    def cursor(self) -> tuple[int, int]:
        return (self.x, self.y)
