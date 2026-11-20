"""A small software renderer for terminal cells.

The canvas is the drawing bus.  Every frame starts blank, every widget draws
into it through primitives such as :meth:`Canvas.draw_text` and
:meth:`Canvas.fill_rect`, and the screen layer eventually serializes the final
cell grid into one ANSI string.  Draw operations support an alpha channel, so
widgets can fade text or tint panels over whatever was drawn beneath them,
similar to a compositor.

Storage is split into two independent planes over parallel lists of plain ints:
glyphs (:meth:`draw_text`) and surfaces (:meth:`fill_rect`).  Filling a rect
writes only the background plane, drawing text writes only the glyph plane, so
neither primitive pays for the other's bookkeeping, and colour comparisons are
integer comparisons instead of tuple allocations.

Two mechanisms keep a frame cheap:

``lazy blanking``
    :meth:`Canvas.clear` is O(1).  It bumps a frame stamp, and a cell whose
    stamp is older than the current frame is *logically* blank — it is only
    rewritten when something draws on it.  Clearing a 120x40 grid therefore
    costs a counter increment instead of 4 800 field assignments.

``incremental flush``
    :meth:`Canvas.render_diff` emits only the rows whose serialized content
    actually changed, each prefixed with an absolute cursor position.  A frame
    that changes nothing writes zero bytes; streaming one line of answer text
    rewrites one row instead of all 40.  :meth:`Canvas.render` still produces a
    complete frame for callers that need a full repaint.
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

# Per-cell flag bits.
_CONT = 1          # cell carries the right half of a double-width glyph

_ATTR_CODES = ((BOLD, '1'), (DIM, '2'), (ITALIC, '3'), (UNDERLINE, '4'),
               (STRIKE, '9'), (REVERSE, '7'))


def _pack(rgb: RGB) -> int:
    """Pack an RGB tuple into one int so cell state compares as integers."""
    return (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]


def _unpack(value: int) -> RGB:
    return ((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF)


# Alpha blending runs per cell for separators, scrollbars and glow rows, and
# the same handful of (colour, surface, alpha) triples repeat thousands of
# times per frame.  Memoizing them removes the float round() from the hot path.
_BLEND_CACHE: dict[tuple[RGB, RGB, int], RGB] = {}


def _blend(foreground: RGB, background: RGB, alpha: int) -> RGB:
    if alpha >= 255:
        return foreground
    if alpha <= 0:
        return background
    key = (foreground, background, alpha)
    blended = _BLEND_CACHE.get(key)
    if blended is None:
        blended = blend_alpha(foreground, background, alpha)
        if len(_BLEND_CACHE) > 8192:
            _BLEND_CACHE.clear()
        _BLEND_CACHE[key] = blended
    return blended


def is_control(ch: str) -> bool:
    """True for C0/C1 control characters that must never reach the terminal."""
    if not ch:
        return False
    code = ord(ch[0])
    return code < 32 or code == 0x7F or 0x80 <= code <= 0x9F


# Width lookups are memoized per code point.  ``east_asian_width`` costs a
# CPython call per glyph and used to run two to three times per character drawn.
_WIDTH_CACHE: dict[int, int] = {}


def _measure_width(ch: str) -> int:
    if unicodedata.combining(ch):
        return 0
    # Most emoji and pictographs are already reported as wide by the UCD.
    return 2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1


def char_width(ch: str) -> int:
    """Display width of a single character (0, 1 or 2 cells)."""
    if not ch:
        return 0
    code = ord(ch[0])
    if code < 128:
        # ASCII: the overwhelming majority of every frame.
        return 0 if code < 32 or code == 0x7F else 1
    width = _WIDTH_CACHE.get(code)
    if width is None:
        width = _measure_width(ch[0])
        _WIDTH_CACHE[code] = width
    return width


def text_width(text: str) -> int:
    if text.isascii():
        # Every printable ASCII byte occupies exactly one cell.
        return len(text) if text.isprintable() else sum(
            1 for ch in text if ' ' <= ch < '\x7f')
    return sum(char_width(ch) for ch in text)


def truncate_text(text: str, width: int, ellipsis: str = '') -> str:
    """Cut ``text`` to ``width`` display cells, optionally adding an ellipsis."""
    if width <= 0:
        return ''
    if text.isascii() and text.isprintable():
        # Fast path: one cell per byte, so slicing is the whole job.
        if len(text) <= width:
            return text
        if not ellipsis:
            return text[:width]
        budget = max(0, width - len(ellipsis))
        return text[:budget] + ellipsis
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
        self._fg_blank = _pack(palette.text)
        self._bg_blank = _pack(palette.background)
        size = self.width * self.height
        # Glyph plane and surface plane over parallel int lists.
        self._ch: list[str] = [' '] * size
        self._fg: list[int] = [self._fg_blank] * size
        self._bg: list[int] = [self._bg_blank] * size
        self._at: list[int] = [0] * size
        self._gw = bytearray([1]) * size      # glyph width in cells
        self._fl = bytearray(size)            # _CONT flags
        self._st: list[int] = [0] * size      # frame stamp; older == blank
        self._frame = 1
        # Clip rectangle, kept pre-intersected: no stack scan per glyph.
        self._cx0, self._cy0 = 0, 0
        self._cx1, self._cy1 = self.width, self.height
        self._clips: list[tuple[int, int, int, int]] = []
        # Incremental flush state: last serialized text per row, plus which
        # rows changed since that serialization.
        self._rows: list[str | None] = [None] * self.height
        self._touched = bytearray([1]) * self.height
        self._sgr_cache: dict[tuple[int, int, int], str] = {}
        self._blank_cache: tuple[int, int, str] | None = None

    # ── lifecycle ───────────────────────────────────────────
    def clear(self, bg: RGB | None = None) -> None:
        """Start a new frame.

        Nothing is rewritten: bumping the stamp makes every cell logically
        blank, so untouched cells cost nothing.  Rows that still hold visible
        text must however be re-serialized (to blank them on screen), so they
        are marked touched up front.
        """
        palette = self.palette
        self._bg_blank = _pack(palette.background if bg is None else bg)
        self._fg_blank = _pack(palette.text)
        self._frame += 1
        self._sgr_cache.clear()
        blank = self._blank_text()
        touched = self._touched
        rows = self._rows
        for y in range(self.height):
            if rows[y] != blank:
                touched[y] = 1

    def _blank_text(self) -> str:
        """ANSI for a row of blank cells (one line, no leading cursor move)."""
        cached = self._blank_cache
        if (cached is not None and cached[0] == self._fg_blank
                and cached[1] == self._bg_blank):
            return cached[2]
        text = self._row_text(0, force_blank=True)
        self._blank_cache = (self._fg_blank, self._bg_blank, text)
        return text

    def resize(self, width: int, height: int) -> None:
        width, height = max(1, width), max(1, height)
        if width == self.width and height == self.height:
            return
        old_w, old_h = self.width, self.height
        old = (self._ch, self._fg, self._bg, self._at, self._gw, self._fl, self._st)
        self.width, self.height = width, height
        size = width * height
        self._ch = [' '] * size
        self._fg = [self._fg_blank] * size
        self._bg = [self._bg_blank] * size
        self._at = [0] * size
        self._gw = bytearray([1]) * size
        self._fl = bytearray(size)
        self._st = [0] * size
        self._frame += 1
        stamps = self._st
        for y in range(min(height, old_h)):
            source = y * old_w
            target = y * width
            for x in range(min(width, old_w)):
                i, j = source + x, target + x
                if old[6][i]:
                    self._ch[j] = old[0][i]
                    self._fg[j] = old[1][i]
                    self._bg[j] = old[2][i]
                    self._at[j] = old[3][i]
                    self._gw[j] = old[4][i]
                    self._fl[j] = old[5][i]
                    stamps[j] = self._frame
        self._rows = [None] * height
        self._touched = bytearray([1]) * height
        self._sgr_cache.clear()
        self._blank_cache = None
        self._cx0, self._cy0, self._cx1, self._cy1 = 0, 0, width, height
        self._clips.clear()

    # ── clipping ────────────────────────────────────────────
    @contextmanager
    def clip(self, x: int, y: int, width: int, height: int) -> Iterator[None]:
        """Restrict all drawing to the given rectangle for the block."""
        previous = (self._cx0, self._cy0, self._cx1, self._cy1)
        self._clips.append(previous)
        self._cx0 = max(previous[0], x)
        self._cy0 = max(previous[1], y)
        self._cx1 = min(previous[2], x + width)
        self._cy1 = min(previous[3], y + height)
        try:
            yield
        finally:
            self._clips.pop()
            self._cx0, self._cy0, self._cx1, self._cy1 = previous

    def _visible(self, x: int, y: int) -> bool:
        return self._cx0 <= x < self._cx1 and self._cy0 <= y < self._cy1

    # ── cell access ─────────────────────────────────────────
    def _materialize(self, index: int) -> None:
        """Reset a stale cell to this frame's blank state, ready to draw on."""
        self._st[index] = self._frame
        self._ch[index] = ' '
        self._fg[index] = self._fg_blank
        self._bg[index] = self._bg_blank
        self._at[index] = 0
        self._gw[index] = 1
        self._fl[index] = 0

    def cell(self, x: int, y: int) -> Cell | None:
        """Read-only snapshot of one cell."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return None
        index = y * self.width + x
        if self._st[index] != self._frame:
            return Cell(' ', _unpack(self._fg_blank), _unpack(self._bg_blank))
        return Cell(self._ch[index], _unpack(self._fg[index]),
                    _unpack(self._bg[index]), self._at[index],
                    bool(self._fl[index] & _CONT))

    def _clear_wide_char(self, x: int, y: int, bg: int) -> None:
        """Repair a neighbouring wide glyph when overwriting half of it."""
        index = y * self.width + x
        if self._st[index] != self._frame:
            # Logically blank: these flags belong to the previous frame and
            # the row is being redrawn anyway, so nothing needs repairing.
            return
        if self._fl[index] & _CONT:
            left = index - 1
            if x > 0 and not (self._fl[left] & _CONT):
                self._materialize(left)
                self._bg[left] = bg
                self._touched[y] = 1
        elif self._gw[index] == 2:
            right = index + 1
            if x + 1 < self.width:
                self._materialize(right)
                self._bg[right] = bg
                self._touched[y] = 1

    def _blit(self, x: int, y: int, ch: str, fg: int | None, bg: int | None,
              attrs: int) -> None:
        if not self._visible(x, y):
            return
        index = y * self.width + x
        if self._st[index] != self._frame:
            self._materialize(index)
        # None means "keep whatever this cell holds".  It has to be resolved
        # here and not by the caller: the planes still contain the previous
        # frame until the cell is materialized, so a caller that reads them
        # first writes that old colour back and the frame never loses it --
        # which is how the status line kept a trail of its light.
        if fg is None:
            fg = self._fg[index]
        if bg is None:
            bg = self._bg[index]
        if is_control(ch):
            # A raw escape sequence inside tool output would otherwise be
            # written straight into the frame and corrupt the whole screen.
            ch = '·'
        width = char_width(ch)
        if width == 2 and not self._visible(x + 1, y):
            # A double-width glyph cannot be split; the terminal would wrap it
            # onto the next row and shift the whole frame.
            ch = ' '
            width = 1
        if (self._fl[index] & _CONT) or self._gw[index] == 2:
            self._clear_wide_char(x, y, bg)
        if (self._ch[index] != ch or self._fg[index] != fg
                or self._bg[index] != bg or self._at[index] != attrs
                or self._fl[index]):
            self._ch[index] = ch
            self._fg[index] = fg
            self._bg[index] = bg
            self._at[index] = attrs
            self._gw[index] = width
            self._fl[index] = 0
            self._touched[y] = 1
        if width == 2:
            self._clear_wide_char(x + 1, y, bg)
            right = index + 1
            if self._visible(x + 1, y):
                self._materialize(right)
                self._ch[right] = ''
                self._fg[right] = fg
                self._bg[right] = bg
                self._at[right] = attrs
                self._gw[right] = 0
                self._fl[right] = _CONT
                self._touched[y] = 1

    def put(self, x: int, y: int, ch: str, fg: RGB | None = None,
            bg: RGB | None = None, attrs: int = 0, alpha: int = 255) -> None:
        """Composite a single character over the cell at ``(x, y)``."""
        if alpha <= 0 or not self._visible(x, y):
            return
        index = y * self.width + x
        if self._st[index] != self._frame:
            self._materialize(index)
        dest_bg = self._bg[index]
        if bg is not None:
            new_bg = _pack(bg) if alpha >= 255 else _pack(
                _blend(bg, _unpack(dest_bg), alpha))
        else:
            new_bg = dest_bg
        base_fg = _pack(fg) if fg is not None else self._fg[index]
        new_fg = base_fg if alpha >= 255 else _pack(
            _blend(_unpack(base_fg), _unpack(new_bg), alpha))
        self._blit(x, y, ch, new_fg, new_bg, attrs)

    # ── primitives ──────────────────────────────────────────
    def _erase_glyphs(self, x0: int, x1: int, y: int) -> bool:
        """Blank the glyph plane over one row span.

        Half of a double-width glyph that straddles either edge is blanked
        too: an orphaned half would otherwise be drawn by the terminal over
        whatever sits next to the span.  Returns True when anything changed.
        """
        frame = self._frame
        start = y * self.width + x0
        end = start + (x1 - x0)
        span = x1 - x0
        erased = False
        if x0 > 0:
            left = start - 1
            if self._st[left] == frame and self._gw[left] == 2:
                self._ch[left] = ' '
                self._gw[left] = 1
                erased = True
        if x1 < self.width:
            right = end
            if self._st[right] == frame and self._fl[right] & _CONT:
                self._ch[right] = ' '
                self._fl[right] = 0
                self._gw[right] = 1
                erased = True
        blank_chars = [' '] * span
        if self._ch[start:end] != blank_chars:
            self._ch[start:end] = blank_chars
            erased = True
        if self._fl[start:end].count(0) != span:
            self._fl[start:end] = bytearray(span)
            erased = True
        if self._gw[start:end].count(1) != span:
            self._gw[start:end] = bytearray([1]) * span
            erased = True
        self._fg[start:end] = [self._fg_blank] * span
        self._at[start:end] = [0] * span
        return erased

    def fill_rect(self, x: int, y: int, width: int, height: int,
                  color: RGB, alpha: int = 255,
                  clear_text: bool = False) -> None:
        """Tint a rectangle.

        Opaque fills normally touch only the surface plane, so glyphs drawn
        earlier stay on top.  ``clear_text`` blanks those glyphs as well; an
        opaque panel uses it so the page underneath cannot show through its
        body.
        """
        if width <= 0 or height <= 0 or alpha <= 0:
            return
        x0 = max(x, self._cx0)
        y0 = max(y, self._cy0)
        x1 = min(x + width, self._cx1)
        y1 = min(y + height, self._cy1)
        if x0 >= x1 or y0 >= y1:
            return
        packed = _pack(color)
        span = x1 - x0
        line = self.width
        frame = self._frame
        stamps = self._st
        backgrounds = self._bg
        foregrounds = self._fg
        touched = self._touched
        if alpha >= 255:
            # C-level slice work: materialize stale cells, then overwrite the
            # surface plane with one assignment per row.
            target = [packed] * span
            frames = [frame] * span
            for row in range(y0, y1):
                base = row * line + x0
                end = base + span
                if clear_text and self._erase_glyphs(x0, x1, row):
                    touched[row] = 1
                if min(stamps[base:end]) != frame:
                    self._ch[base:end] = [' '] * span
                    foregrounds[base:end] = [self._fg_blank] * span
                    backgrounds[base:end] = [self._bg_blank] * span
                    self._at[base:end] = [0] * span
                    self._gw[base:end] = bytearray([1]) * span
                    self._fl[base:end] = bytearray(span)
                    stamps[base:end] = frames
                if backgrounds[base:end] != target:
                    backgrounds[base:end] = target
                    touched[row] = 1
            return
        for row in range(y0, y1):
            base = row * line + x0
            for offset in range(base, base + span):
                if stamps[offset] != frame:
                    self._materialize(offset)
                cell_bg = backgrounds[offset]
                blended = _pack(_blend(color, _unpack(cell_bg), alpha))
                if cell_bg != blended:
                    backgrounds[offset] = blended
                    # Keep existing glyphs but fade them into the new tint.
                    foregrounds[offset] = _pack(_blend(
                        _unpack(foregrounds[offset]), _unpack(blended), alpha))
                    touched[row] = 1

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
        limit = None if max_width is None else x + max_width
        if alpha < 255:
            # Translucent text has to composite against whatever is already in
            # the cell; the alpha-free fast lane below cannot express that.
            return self._draw_text_alpha(x, y, text, fg, bg, attrs, alpha, limit)
        if not (self._cy0 <= y < self._cy1):
            # Nothing is drawn, but the caller still gets the advance it asked
            # for, so layout keeps flowing.
            col = x
            for ch in text:
                if ch == '\n':
                    break
                if ch == '\r':
                    continue
                width = 4 - (col % 4) if ch == '\t' else char_width(ch)
                if limit is not None and col + width > limit:
                    break
                col += width
            return col - x

        fg_packed = None if fg is None else _pack(fg)
        bg_packed = None if bg is None else _pack(bg)
        line = self.width
        x0, x1 = self._cx0, self._cx1
        chs = self._ch
        fgs = self._fg
        bgs = self._bg
        ats = self._at
        gws = self._gw
        fls = self._fl
        stamps = self._st
        touched = self._touched
        frame = self._frame
        row_base = y * line
        col = x
        start = x
        for ch in text:
            code = ord(ch)
            if code < 128:
                if code == 10:                      # \n
                    break
                if code == 13:                      # \r
                    continue
                if code == 9:                       # \t
                    for _ in range(4 - (col % 4)):
                        if limit is not None and col >= limit:
                            break
                        if x0 <= col < x1:
                            index = row_base + col
                            if stamps[index] != frame:
                                self._materialize(index)
                            self._blit(
                                col, y, ' ',
                                fgs[index] if fg_packed is None else fg_packed,
                                bgs[index] if bg_packed is None else bg_packed,
                                attrs)
                        col += 1
                    continue
                if code < 32 or code == 0x7F:
                    continue
                if limit is not None and col >= limit:
                    break
                if not (x0 <= col < x1):
                    col += 1
                    continue
                index = row_base + col
                if stamps[index] != frame:
                    self._materialize(index)
                new_fg = fgs[index] if fg_packed is None else fg_packed
                new_bg = bgs[index] if bg_packed is None else bg_packed
                if not fls[index] and gws[index] != 2:
                    if (chs[index] != ch or fgs[index] != new_fg
                            or bgs[index] != new_bg or ats[index] != attrs):
                        chs[index] = ch
                        fgs[index] = new_fg
                        bgs[index] = new_bg
                        ats[index] = attrs
                        touched[y] = 1
                else:
                    self._blit(col, y, ch, new_fg, new_bg, attrs)
                col += 1
                continue
            if 0x80 <= code <= 0x9F:
                # Never attach a control byte to the previous cell: the frame
                # would then contain a raw escape sequence.
                continue
            width = char_width(ch)
            if width == 0:
                # Combining mark: append to the previous cell when possible.
                if col > start and x0 <= col - 1 < x1:
                    index = row_base + col - 1
                    if stamps[index] == frame and not (fls[index] & _CONT):
                        chs[index] = chs[index] + ch
                        touched[y] = 1
                continue
            if limit is not None and col + width > limit:
                break
            if not (x0 <= col < x1):
                # Past the right edge: the ascii lane skips the cell, the
                # wide one has to as well -- the buffer ends there.
                col += width
                continue
            self._blit(col, y, ch, fg_packed, bg_packed, attrs)
            col += width
        return col - start

    def _draw_text_alpha(self, x: int, y: int, text: str, fg: RGB | None,
                         bg: RGB | None, attrs: int, alpha: int,
                         limit: int | None) -> int:
        """Translucent text: composite each glyph through :meth:`put`."""
        col = x
        for ch in text:
            if ch == '\n':
                break
            if ch == '\r':
                continue
            if ch == '\t':
                for _ in range(4 - (col % 4)):
                    if limit is not None and col >= limit:
                        break
                    self.put(col, y, ' ', fg, bg, attrs, alpha)
                    col += 1
                continue
            if is_control(ch):
                continue
            width = char_width(ch)
            if width == 0:
                if col > x and self._cx0 <= col - 1 < self._cx1:
                    index = y * self.width + col - 1
                    if (self._st[index] == self._frame
                            and not (self._fl[index] & _CONT)):
                        self._ch[index] += ch
                        self._touched[y] = 1
                continue
            if limit is not None and col + width > limit:
                break
            self.put(col, y, ch, fg, bg, attrs, alpha)
            col += width
        return col - x

    def hline(self, x: int, y: int, width: int, color: RGB,
              alpha: int = 255, char: str = '─') -> None:
        """Draw a horizontal run.  Separators are drawn every frame, so this
        avoids the generic :meth:`put` path and packs the colour once."""
        if width <= 0 or not (self._cy0 <= y < self._cy1):
            return
        start = max(x, self._cx0)
        end = min(x + width, self._cx1)
        if start >= end:
            return
        self._glyph_run(start, y, end - start, 1, char, color, alpha)

    def vline(self, x: int, y: int, height: int, color: RGB,
              alpha: int = 255, char: str = '│') -> None:
        if height <= 0 or not (self._cx0 <= x < self._cx1):
            return
        start = max(y, self._cy0)
        end = min(y + height, self._cy1)
        if start >= end:
            return
        self._glyph_run(x, start, end - start, self.width, char, color, alpha)

    def _glyph_run(self, x: int, y: int, count: int, step: int, char: str,
                   color: RGB, alpha: int) -> None:
        """Repeat one glyph along a row (step 1) or down a column (step width).

        Only the foreground changes: a rule drawn over a panel keeps the panel
        surface underneath, exactly like ``put`` with ``bg=None``.
        """
        packed = _pack(color)
        glyph_width = char_width(char)
        line = self.width
        index = y * line + x
        chs = self._ch
        fgs = self._fg
        bgs = self._bg
        ats = self._at
        gws = self._gw
        fls = self._fl
        stamps = self._st
        frame = self._frame
        touched = self._touched
        row, col = y, x
        for _ in range(count):
            if stamps[index] != frame:
                self._materialize(index)
            dest_bg = bgs[index]
            fg = packed if alpha >= 255 else _pack(
                _blend(color, _unpack(dest_bg), alpha))
            if glyph_width == 1 and not fls[index] and gws[index] != 2:
                if chs[index] != char or fgs[index] != fg or ats[index] != 0:
                    chs[index] = char
                    fgs[index] = fg
                    ats[index] = 0
                    touched[row] = 1
            else:
                self._blit(col, row, char, fg, dest_bg, 0)
            index += step
            if step == 1:
                col += 1
            else:
                row += 1

    def box(self, x: int, y: int, width: int, height: int, border: RGB,
            bg: RGB | None = None, alpha: int = 255, title: str = '',
            title_color: RGB | None = None, title_attrs: int = BOLD) -> None:
        """Draw a rounded?  No - a plain square-cornered panel border."""
        if width < 2 or height < 2:
            return
        if bg is not None:
            self.fill_rect(x, y, width, height, bg, alpha, clear_text=True)
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
    def _sgr(self, fg: int, bg: int, attrs: int) -> str:
        """Cached SGR sequence for one style combination."""
        key = (fg, bg, attrs)
        cached = self._sgr_cache.get(key)
        if cached is not None:
            return cached
        codes = ['0']
        for bit, code in _ATTR_CODES:
            if attrs & bit:
                codes.append(code)
        if attrs & REVERSE:
            fg, bg = bg, fg
        codes.append(f'38;2;{(fg >> 16) & 0xFF};{(fg >> 8) & 0xFF};{fg & 0xFF}')
        codes.append(f'48;2;{(bg >> 16) & 0xFF};{(bg >> 8) & 0xFF};{bg & 0xFF}')
        sequence = '\x1b[' + ';'.join(codes) + 'm'
        self._sgr_cache[key] = sequence
        return sequence

    def _row_text(self, y: int, force_blank: bool = False) -> str:
        """Serialize one row, assuming the terminal starts from reset state."""
        blank = force_blank
        line = self.width
        base = 0 if blank else y * line
        chs = self._ch
        fgs = self._fg
        bgs = self._bg
        ats = self._at
        fls = self._fl
        stamps = self._st
        frame = self._frame
        fg_blank = self._fg_blank
        bg_blank = self._bg_blank
        out: list[str] = []
        append = out.append
        cur_fg = -1
        cur_bg = -1
        cur_attrs = -1
        for offset in range(line):
            index = base + offset
            if blank or stamps[index] != frame:
                fg, bg, attrs, glyph = fg_blank, bg_blank, 0, ' '
            else:
                if fls[index] & _CONT:
                    continue
                fg, bg, attrs, glyph = fgs[index], bgs[index], ats[index], chs[index]
            if fg != cur_fg or bg != cur_bg or attrs != cur_attrs:
                append(self._sgr(fg, bg, attrs))
                cur_fg, cur_bg, cur_attrs = fg, bg, attrs
            append(glyph)
        if cur_attrs != -1:
            append('\x1b[0m')
        return ''.join(out)

    def render_row(self, y: int) -> str:
        """Return the ANSI text for one row."""
        if not (0 <= y < self.height):
            return ''
        return self._row_text(y)

    def row_runs(self, y: int) -> list[tuple[str, int, int, int]]:
        """Serialize one row as style runs: (text, fg, bg, attrs).

        The ANSI serializer below is one consumer of the cell grid; a browser
        renderer is another, and re-parsing escape sequences back into cells
        would throw information away.  A run carries the state one SGR
        sequence encodes, so a renderer paints text and colours directly.
        REVERSE is folded into the two colours, exactly as _sgr does it, and
        continuation cells are skipped: the wide glyph that owns them is
        already in the run before them.
        """
        if not (0 <= y < self.height):
            return []
        line = self.width
        base = y * line
        chs, fgs, bgs, ats = self._ch, self._fg, self._bg, self._at
        fls, stamps = self._fl, self._st
        frame = self._frame
        fg_blank = self._fg_blank
        bg_blank = self._bg_blank
        runs: list[tuple[str, int, int, int]] = []
        text: list[str] = []
        cur_fg = cur_bg = cur_attrs = -1
        for offset in range(line):
            index = base + offset
            if stamps[index] != frame:
                fg, bg, attrs, glyph = fg_blank, bg_blank, 0, ' '
            else:
                if fls[index] & _CONT:
                    continue
                fg, bg, attrs, glyph = (
                    fgs[index], bgs[index], ats[index], chs[index])
                if attrs & REVERSE:
                    fg, bg = bg, fg
                    attrs &= ~REVERSE
            if fg != cur_fg or bg != cur_bg or attrs != cur_attrs:
                if text:
                    runs.append((''.join(text), cur_fg, cur_bg, cur_attrs))
                    text = []
                cur_fg, cur_bg, cur_attrs = fg, bg, attrs
            text.append(glyph)
        if text:
            runs.append((''.join(text), cur_fg, cur_bg, cur_attrs))
        return runs

    def render(self) -> str:
        """Serialize the whole canvas as one full-frame payload."""
        rows = []
        for y in range(self.height):
            text = self._row_text(y)
            self._rows[y] = text
            self._touched[y] = 0
            rows.append(text)
        return '\x1b[H' + '\r\n'.join(rows)

    def render_diff(self) -> str:
        """Payload for the rows that changed since the last flush.

        Each changed row is preceded by its absolute cursor position, so the
        terminal repaints exactly those rows and leaves the rest of the screen
        alone.  When every row changed (first frame, resize, full scroll) one
        home sequence plus a newline-joined block is emitted instead: exactly
        the bytes :meth:`render` produces, so a full repaint never pays a
        per-row positioning premium.  Returns an empty string when the frame is
        visually identical to the previous one - an idle UI writes nothing.
        """
        touched = self._touched
        rows = self._rows
        changed: list[tuple[int, str]] = []
        for y in range(self.height):
            if not touched[y] and rows[y] is not None:
                continue
            text = self._row_text(y)
            touched[y] = 0
            if rows[y] == text:
                continue
            rows[y] = text
            changed.append((y, text))
        if not changed:
            return ''
        if len(changed) == self.height:
            return '\x1b[H' + '\r\n'.join(
                text for _, text in changed)
        out: list[str] = []
        append = out.append
        for y, text in changed:
            append(f'\x1b[{y + 1};1H')
            append(text)
        return ''.join(out)
