"""Styled text spans and width-aware wrapping."""

from __future__ import annotations

from dataclasses import dataclass

from adcode.theme import RGB
from adcode.tui.canvas import char_width, is_control, text_width


@dataclass
class Span:
    text: str
    fg: str | RGB | None = None
    bg: str | RGB | None = None
    attrs: int = 0
    alpha: int = 255


Line = list[Span]

# A tool call row keeps its two-cell gutter but starts flush with the screen
# edge, so it declares how far it bleeds into the transcript's left margin.  The
# gutter cell is preserved: the bar sits in it, the text keeps its one-space
# separation, and nothing else gains a column.
ROW_BLEED = 1

# The bar is a left-edge block rather than a centred ``│``: both occupy exactly
# one cell, but this one paints its ink against the left border of that cell, so
# the rule reads as flush with the screen edge instead of floating mid-column.
ROW_BAR = '▏'


class Row(list):
    """A visual row that may draw to the left of the normal text margin."""

    __slots__ = ('bleed',)

    def __init__(self, line: Line | None = None, bleed: int = 0) -> None:
        super().__init__(line or [])
        self.bleed = bleed


def _same_style(left: Span, right: Span) -> bool:
    return (left.fg == right.fg and left.bg == right.bg
            and left.attrs == right.attrs and left.alpha == right.alpha)


def add_span(line: Line, span: Span) -> Line:
    if not span.text:
        return line
    if line and _same_style(line[-1], span):
        line[-1].text += span.text
    else:
        line.append(span)
    return line


def plain_text(line: Line) -> str:
    return ''.join(span.text for span in line)


def line_width(line: Line) -> int:
    return sum(text_width(span.text) for span in line)


def trim_line(line: Line) -> Line:
    """Drop leading/trailing whitespace spans without touching interior."""
    if not line:
        return line
    result: Line = [Span(span.text, span.fg, span.bg, span.attrs, span.alpha)
                    for span in line]
    while result:
        result[0].text = result[0].text.lstrip(' ')
        if result[0].text:
            break
        result.pop(0)
    while result:
        result[-1].text = result[-1].text.rstrip(' ')
        if result[-1].text:
            break
        result.pop()
    return result


def truncate_line(line: Line, width: int, ellipsis: str = '') -> Line:
    if width <= 0:
        return []
    total = line_width(line)
    if total <= width:
        return [Span(span.text, span.fg, span.bg, span.attrs, span.alpha)
                for span in line]
    budget = max(0, width - text_width(ellipsis))
    result: Line = []
    used = 0
    for span in line:
        if used >= budget:
            break
        source = span.text
        room = budget - used
        if source.isascii() and source.isprintable():
            # One cell per byte: the cut point is an index, not a scan.
            text = source[:room]
            used += len(text)
        else:
            text = ''
            for ch in source:
                size = char_width(ch)
                if used + size > budget:
                    break
                text += ch
                used += size
        if text:
            add_span(result, Span(text, span.fg, span.bg, span.attrs, span.alpha))
        if used >= budget:
            break
    if ellipsis and used + text_width(ellipsis) <= width:
        add_span(result, Span(ellipsis, 'muted'))
    return result


def wrap_line(line: Line, width: int, first_width: int | None = None) -> list[Line]:
    """Wrap a styled line into visual rows of at most ``width`` cells.

    Wrapping is cell-exact, not word-aware, so the cost is dominated by the
    number of glyphs.  Two paths keep that cost off the hot loop:

    * a printable-ASCII span is one cell per byte, so a whole cut is an index
      arithmetic step and produces at most one span per visual row instead of
      one span (and one ``add_span`` merge test) per character;
    * anything else -- CJK, emoji, combining marks, control characters -- runs
      the glyph-by-glyph path, but collects glyphs in a local buffer so the
      span is still appended once per row segment rather than once per glyph.
    """
    width = max(1, width)
    rows: list[Line] = []
    current: Line = []
    current_width = 0
    limit = width if first_width is None else max(1, first_width)

    for span in line:
        source = span.text
        if not source:
            continue
        fg, bg, attrs, alpha = span.fg, span.bg, span.attrs, span.alpha
        if source.isascii() and source.isprintable():
            # Fast path: every glyph is exactly one cell wide.
            text = source
            while text:
                room = limit - current_width
                if room <= 0:
                    rows.append(current)
                    current = []
                    current_width = 0
                    limit = width
                    continue
                if len(text) <= room:
                    add_span(current, Span(text, fg, bg, attrs, alpha))
                    current_width += len(text)
                    break
                add_span(current, Span(text[:room], fg, bg, attrs, alpha))
                text = text[room:]
                rows.append(current)
                current = []
                current_width = 0
                limit = width
            continue
        buffer = ''
        for ch in source:
            if is_control(ch):
                continue
            size = char_width(ch)
            if size == 0:
                # Combining marks attach to the last character.
                if buffer:
                    buffer += ch
                elif current and current[-1].text:
                    current[-1].text += ch
                elif rows and rows[-1]:
                    rows[-1][-1].text += ch
                continue
            if current_width and current_width + size > limit:
                if buffer:
                    add_span(current, Span(buffer, fg, bg, attrs, alpha))
                    buffer = ''
                rows.append(current)
                current = []
                current_width = 0
                limit = width
            if not current_width and size > limit:
                # A double-width glyph cannot fit here; place it anyway.
                if buffer:
                    add_span(current, Span(buffer, fg, bg, attrs, alpha))
                    buffer = ''
                add_span(current, Span(ch, fg, bg, attrs, alpha))
                current_width += size
                continue
            buffer += ch
            current_width += size
        if buffer:
            add_span(current, Span(buffer, fg, bg, attrs, alpha))
    if current or not rows:
        rows.append(current)
    return rows


def wrap_lines(lines: list[Line], width: int,
               first_width: int | None = None) -> list[Line]:
    result: list[Line] = []
    for index, line in enumerate(lines):
        result.extend(wrap_line(line, width, first_width if index == 0 else None))
    return result


def slice_visual(lines: list[Line], from_index: int, to_index: int) -> list[Line]:
    """Inclusive negative-index slice: (-5, -1) is the last five visual rows."""
    if not lines:
        return []
    total = len(lines)
    start = max(0, total + from_index) if from_index < 0 else min(from_index, total)
    end = max(0, total + to_index) if to_index < 0 else min(to_index, total - 1)
    if end < start:
        start, end = end, start
    return lines[start:end + 1]


def render_to_strings(lines: list[Line]) -> list[str]:
    return [plain_text(line) for line in lines]
