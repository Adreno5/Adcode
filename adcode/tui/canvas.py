"""A small software renderer for terminal cells.

The canvas is the drawing bus.  Every frame starts blank, every widget draws
into it through primitives such as :meth:`Canvas.draw_text` and
:meth:`Canvas.fill_rect`, and the screen layer eventually serializes the final
cell grid into one ANSI string.  Draw operations support an alpha channel, so
widgets can fade text or tint panels over whatever was drawn beneath them,
similar to a compositor.
"""

from __future__ import annotations

import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from adcode.theme import Palette, RGB, blend_alpha

# Text attribute bit flags.
BOLD = 1
DIM = 2
ITALIC = 4
UNDERLINE = 8
STRIKE = 16
REVERSE = 32


def char_width(ch: str) -> int:
    """Display width of a single character (0, 1 or 2 cells)."""
    if not ch:
        return 0
    code = ord(ch)
    if code == 0:
        return 0
    if code < 32 or code == 0x7F:
        return 0
    if unicodedata.combining(ch):
        return 0
    if unicodedata.east_asian_width(ch) in ('W', 'F'):
        return 2
    if 0x1F300 <= code <= 0x1FAFF or 0x2600 <= code <= 0x27BF:
        # Emoji and pictographs; most terminals render these double-width.
        if unicodedata.east_asian_width(ch) in ('W', 'F'):
            return 2
    return 1


def text_width(text: str) -> int:
    return sum(char_width(ch) for ch in text)


def truncate_text(text: str, width: int, ellipsis: str = '') -> str:
    """Cut ``text`` to ``width`` display cells, optionally adding an ellipsis."""
    if width <= 0:
        return ''
    if text_width(text) <= width:
        return text
    budget = max(0, width - text_width(ellipsis))
    out = ''
    used = 0
    for ch in text:
        w = char_width(ch)
        if used + w > budget:
            break
        out += ch
        used += w
    return out + ellipsis


@dataclass(slots=True)
class Cell:
    ch: str = ' '
    fg: RGB = (255, 255, 255)
    bg: RGB = (0, 0, 0)
    attrs: int = 0
    # A continuation cell carries the right half of a double-width glyph.  Its
    # character is not emitted; the previous cell's glyph occupies it.
    cont: bool = False

    def copy(self) -> 'Cell':
        return Cell(self.ch, self.fg, self.bg, self.attrs, self.cont)


class Canvas:
    """A fixed-size cell buffer with alpha-aware drawing primitives."""

    def __init__(self, width: int, height: int, palette: Palette) -> None:
        self.palette = palette
        self.width = max(1, width)
        self.height = max(1, height)
        self._cells: list[Cell] = []
        self._clips: list[tuple[int, int, int, int]] = []
        self._blank()

    # ── lifecycle ───────────────────────────────────────────
    def _blank(self) -> None:
        self._cells = [
            Cell(' ', self.palette.text, self.palette.background)
            for _ in range(self.width * self.height)
        ]

    def clear(self, bg: RGB | None = None) -> None:
        color = bg if bg is not None else self.palette.background
        fg = self.palette.text
        for cell in self._cells:
            cell.ch = ' '
            cell.fg = fg
            cell.bg = color
            cell.attrs = 0
            cell.cont = False

    def resize(self, width: int, height: int) -> None:
        width, height = max(1, width), max(1, height)
        if width == self.width and height == self.height:
            return
        old_w, old_h = self.width, self.height
        old = self._cells
        self.width, self.height = width, height
        self._cells = [
            Cell(' ', self.palette.text, self.palette.background)
            for _ in range(width * height)
        ]
        for y in range(min(height, old_h)):
            for x in range(min(width, old_w)):
                self._cells[y * width + x] = old[y * old_w + x]

    # ── clipping ────────────────────────────────────────────
    @contextmanager
    def clip(self, x: int, y: int, width: int, height: int) -> Iterator[None]:
        """Restrict all drawing to the given rectangle for the block."""
        self._clips.append((x, y, width, height))
        try:
            yield
        finally:
            self._clips.pop()

    def _visible(self, x: int, y: int) -> bool:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        for cx, cy, cw, ch in self._clips:
            if not (cx <= x < cx + cw and cy <= y < cy + ch):
                return False
        return True

    # ── cell access ─────────────────────────────────────────
    def cell(self, x: int, y: int) -> Cell | None:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self._cells[y * self.width + x]
        return None

    def _clear_wide_char(self, x: int, y: int, bg: RGB) -> None:
        """Repair a neighbouring wide glyph when overwriting half of it."""
        cell = self.cell(x, y)
        if cell is None:
            return
        if cell.cont:
            left = self.cell(x - 1, y)
            if left is not None and not left.cont:
                left.ch = ' '
                left.fg = self.palette.text
                left.bg = bg
        elif char_width(cell.ch) == 2:
            right = self.cell(x + 1, y)
            if right is not None:
                right.cont = False
                right.ch = ' '
                right.fg = self.palette.text
                right.bg = bg

    def _put(self, x: int, y: int, ch: str, fg: RGB, bg: RGB, attrs: int) -> None:
        if not self._visible(x, y):
            return
        cell = self.cell(x, y)
        if cell is None:
            return
        self._clear_wide_char(x, y, bg)
        cell.ch = ch
        cell.fg = fg
        cell.bg = bg
        cell.attrs = attrs
        cell.cont = False
        if char_width(ch) == 2:
            self._clear_wide_char(x + 1, y, bg)
            right = self.cell(x + 1, y)
            if right is not None and self._visible(x + 1, y):
                right.ch = ''
                right.cont = True
                right.fg = fg
                right.bg = bg
                right.attrs = attrs

    def put(self, x: int, y: int, ch: str, fg: RGB | None = None,
            bg: RGB | None = None, attrs: int = 0, alpha: int = 255) -> None:
        """Composite a single character over the cell at ``(x, y)``."""
        if not self._visible(x, y):
            return
        cell = self.cell(x, y)
        if cell is None:
            return
        dest_bg = cell.bg
        new_bg = dest_bg
        if bg is not None and alpha > 0:
            new_bg = blend_alpha(bg, dest_bg, alpha) if alpha < 255 else bg
        else:
            bg = dest_bg
        base_fg = fg if fg is not None else cell.fg
        new_fg = blend_alpha(base_fg, new_bg, alpha) if alpha < 255 else base_fg
        if alpha <= 0:
            return
        self._put(x, y, ch, new_fg, new_bg, attrs if attrs else cell.attrs)

    # ── primitives ──────────────────────────────────────────
    def fill_rect(self, x: int, y: int, width: int, height: int,
                  color: RGB, alpha: int = 255) -> None:
        if width <= 0 or height <= 0 or alpha <= 0:
            return
        for row in range(y, y + height):
            for col in range(x, x + width):
                if not self._visible(col, row):
                    continue
                cell = self.cell(col, row)
                if cell is None:
                    continue
                cell.bg = blend_alpha(color, cell.bg, alpha) if alpha < 255 else color
                if alpha < 255:
                    # Keep existing glyphs but fade them into the new tint.
                    cell.fg = blend_alpha(cell.fg, cell.bg, alpha)

    def fill_row(self, y: int, color: RGB, alpha: int = 255,
                 x: int = 0, width: int | None = None) -> None:
        self.fill_rect(x, y, width if width is not None else self.width - x, 1,
                       color, alpha)

    def draw_text(self, x: int, y: int, text: str, fg: RGB | None = None,
                  bg: RGB | None = None, attrs: int = 0, alpha: int = 255,
                  max_width: int | None = None) -> int:
        """Draw one line of text; returns the advance in cells."""
        if not text:
            return 0
        col = x
        limit = None if max_width is None else x + max_width
        index = 0
        while index < len(text):
            ch = text[index]
            index += 1
            if ch == '\n':
                break
            if ch == '\r':
                continue
            if ch == '\t':
                spaces = 4 - (col % 4)
                for _ in range(spaces):
                    if limit is not None and col >= limit:
                        break
                    self.put(col, y, ' ', fg, bg, attrs, alpha)
                    col += 1
                continue
            width = char_width(ch)
            if width == 0:
                # Combining mark: append to the previous cell when possible.
                prev = self.cell(col - 1, y) if col > x else None
                if prev is not None and not prev.cont:
                    prev.ch += ch
                continue
            if limit is not None and col + width > limit:
                break
            self.put(col, y, ch, fg, bg, attrs, alpha)
            col += width
        return col - x

    def hline(self, x: int, y: int, width: int, color: RGB,
              alpha: int = 255, char: str = '─') -> None:
        for offset in range(max(0, width)):
            self.put(x + offset, y, char, color, None, 0, alpha)

    def vline(self, x: int, y: int, height: int, color: RGB,
              alpha: int = 255, char: str = '│') -> None:
        for offset in range(max(0, height)):
            self.put(x, y + offset, char, color, None, 0, alpha)

    def box(self, x: int, y: int, width: int, height: int, border: RGB,
            bg: RGB | None = None, alpha: int = 255, title: str = '',
            title_color: RGB | None = None, title_attrs: int = BOLD) -> None:
        """Draw a rounded?  No - a plain square-cornered panel border."""
        if width < 2 or height < 2:
            return
        if bg is not None:
            self.fill_rect(x, y, width, height, bg, alpha)
        self.hline(x + 1, y, width - 2, border, alpha)
        self.hline(x + 1, y + height - 1, width - 2, border, alpha)
        self.vline(x, y + 1, height - 2, border, alpha)
        self.vline(x + width - 1, y + 1, height - 2, border, alpha)
        corners = {('tl', x, y, '┌'), ('tr', x + width - 1, y, '┐'),
                   ('bl', x, y + height - 1, '└'), ('br', x + width - 1, y + height - 1, '┘')}
        for _, cx, cy, glyph in corners:
            self.put(cx, cy, glyph, border, bg, 0, alpha)
        if title and width > 4:
            text = truncate_text(title, width - 4)
            self.draw_text(x + 2, y, text, title_color or border, bg,
                           title_attrs, alpha, max_width=width - 4)

    # ── serialization ───────────────────────────────────────
    def render_row(self, y: int) -> str:
        """Return the ANSI text for one row."""
        if not (0 <= y < self.height):
            return ''
        parts: list[str] = []
        current_fg: RGB | None = None
        current_bg: RGB | None = None
        current_attrs = -1
        for x in range(self.width):
            cell = self._cells[y * self.width + x]
            if cell.cont:
                continue
            attrs = cell.attrs
            if (cell.fg != current_fg or cell.bg != current_bg
                    or attrs != current_attrs):
                codes = ['0']
                if attrs & BOLD:
                    codes.append('1')
                if attrs & DIM:
                    codes.append('2')
                if attrs & ITALIC:
                    codes.append('3')
                if attrs & UNDERLINE:
                    codes.append('4')
                if attrs & STRIKE:
                    codes.append('9')
                if attrs & REVERSE:
                    codes.append('7')
                fg = cell.fg
                bg = cell.bg
                if attrs & REVERSE:
                    fg, bg = bg, fg
                codes.append(f'38;2;{fg[0]};{fg[1]};{fg[2]}')
                codes.append(f'48;2;{bg[0]};{bg[1]};{bg[2]}')
                parts.append('\x1b[' + ';'.join(codes) + 'm')
                current_fg, current_bg, current_attrs = cell.fg, cell.bg, attrs
            parts.append(cell.ch)
        if current_attrs != -1:
            parts.append('\x1b[0m')
        return ''.join(parts)

    def render(self) -> str:
        """Serialize the whole canvas as one frame payload."""
        rows = [self.render_row(y) for y in range(self.height)]
        return '\x1b[H' + '\r\n'.join(rows)
