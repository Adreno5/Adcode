"""Incremental Markdown to styled spans.

Streaming output is fed chunk by chunk.  Complete lines are formatted
immediately; tables and fenced code blocks are buffered until the block ends.
The result is a list of logical :class:`Line` objects that the transcript view
wraps to the viewport width.
"""

from __future__ import annotations

import re

from adcode.tui.canvas import BOLD, DIM, ITALIC, STRIKE, UNDERLINE, char_width
from adcode.tui.text import Line, Span, add_span, line_width, plain_text

_HEADING_RE = re.compile(r'^(#{1,6})\s+(.*?)(?:\s+#+\s*)?$')
_UL_RE = re.compile(r'^(\s*)[-+*]\s+(.*)$')
_TASK_RE = re.compile(r'^(\s*)[-+*]\s+\[([ xX])\]\s+(.*)$')
_OL_RE = re.compile(r'^(\s*)(\d{1,3})[.)]\s+(.*)$')
_QUOTE_RE = re.compile(r'^(>\s?)+')
_HR_RE = re.compile(r'^\s*(?:-{3,}|\*{3,}|_{3,})\s*$')
_FENCE_RE = re.compile(r'^(`{3,})\s*([^`]*)$')
_LINK_RE = re.compile(r'!?\[([^\]]*)\]\(([^)\s]+)(?:\s+["\'][^"\']*["\'])?\)')
_REF_LINK_RE = re.compile(r'!?\[([^\]]+)\]\[([^\]]*)\]')
# Everything ``_parse_inline`` dispatches on.  Plain prose between two special
# characters is the overwhelming majority of a markdown stream, so the scanner
# jumps over it in one slice instead of walking it one character at a time.
_INLINE_SPECIAL_RE = re.compile(r'[\\`\[<!*_~=$]')
# First glyph of every block-level construct the renderer recognises: heading,
# table, fence, bullet/task, horizontal rule, quote, ordered list.  A line that
# starts with anything else is a plain paragraph.
_BLOCK_LEAD = frozenset('#|`-+*_>0123456789')


def _parse_inline(text: str, base: str | None = None) -> Line:
    """Parse emphasis, code, links, HTML-ish tags and math into styled spans."""
    spans: Line = []
    buffer = ''
    index = 0
    state = {'bold': False, 'italic': False, 'underline': False}

    def current_attrs() -> int:
        attrs = 0
        if state['bold']:
            attrs |= BOLD
        if state['italic']:
            attrs |= ITALIC
        if state['underline']:
            attrs |= UNDERLINE
        return attrs

    def flush() -> None:
        nonlocal buffer
        if buffer:
            add_span(spans, Span(buffer, base, attrs=current_attrs()))
            buffer = ''

    while index < len(text):
        # Jump straight to the next character that can start markup; anything
        # before it is literal text and can be taken in one slice.
        special = _INLINE_SPECIAL_RE.search(text, index)
        if special is None:
            buffer += text[index:]
            break
        if special.start() > index:
            buffer += text[index:special.start()]
            index = special.start()
        ch = text[index]
        if ch == '\\' and index + 1 < len(text):
            buffer += text[index + 1]
            index += 2
            continue

        if ch == '`':
            end = text.find('`', index + 1)
            if end != -1:
                flush()
                add_span(spans, Span(text[index + 1:end], 'subtext', 'panel'))
                index = end + 1
                continue

        if ch == '[' or (ch == '!' and index + 1 < len(text) and text[index + 1] == '['):
            match = _LINK_RE.match(text, index)
            if match:
                flush()
                label, target = match.group(1), match.group(2)
                if ch == '!':
                    add_span(spans, Span('[image] ', 'muted'))
                add_span(spans, Span(label, 'accent', attrs=UNDERLINE))
                add_span(spans, Span(f' <{target}>', 'muted'))
                index = match.end()
                continue
            match = _REF_LINK_RE.match(text, index)
            if match:
                flush()
                add_span(spans, Span(match.group(1), 'accent', attrs=UNDERLINE))
                if match.group(2):
                    add_span(spans, Span(f' [{match.group(2)}]', 'muted'))
                index = match.end()
                continue

        if ch == '<':
            end = text.find('>', index + 1)
            if end != -1:
                tag = text[index + 1:end].strip().lower()
                closing = tag.startswith('/')
                name = tag.lstrip('/').strip()
                if name in ('br', 'br/'):
                    flush()
                    buffer += ' '
                elif name in ('b', 'strong'):
                    flush()
                    state['bold'] = not closing
                elif name in ('i', 'em'):
                    flush()
                    state['italic'] = not closing
                elif name in ('u',):
                    flush()
                    state['underline'] = not closing
                index = end + 1
                continue

        if ch in '*_~=':
            triple = ch * 3
            if ch in '*_' and text.startswith(triple, index):
                end = text.find(triple, index + 3)
                if end != -1 and end > index + 3:
                    flush()
                    inner = text[index + 3:end]
                    for span in _parse_inline(inner, base):
                        span.attrs |= BOLD | ITALIC | current_attrs()
                        add_span(spans, span)
                    index = end + 3
                    continue
            marker = ch * 2
            if text.startswith(marker, index) and ch != '=':
                end = text.find(marker, index + 2)
                if end != -1 and end > index + 2:
                    flush()
                    attrs = STRIKE if ch == '~' else BOLD
                    inner = text[index + 2:end]
                    for span in _parse_inline(inner, base):
                        span.attrs |= attrs | current_attrs()
                        add_span(spans, span)
                    index = end + 2
                    continue
            if ch in '*_':
                opening_ok = index + 1 < len(text) and not text[index + 1].isspace()
                if ch == '_':
                    opening_ok = opening_ok and (
                        index == 0 or not text[index - 1].isalnum())
                if opening_ok:
                    end = text.find(ch, index + 1)
                    closing_ok = (
                        end > index + 1 and not text[end - 1].isspace()
                        and (ch != '_' or end + 1 >= len(text)
                             or not text[end + 1].isalnum())
                    )
                    if end != -1 and closing_ok:
                        flush()
                        inner = text[index + 1:end]
                        for span in _parse_inline(inner, base):
                            span.attrs |= ITALIC | current_attrs()
                            add_span(spans, span)
                        index = end + 1
                        continue

        if ch == '$':
            end = text.find('$', index + 1)
            if end != -1 and end > index + 1:
                flush()
                add_span(spans, Span(text[index + 1:end], 'accent'))
                index = end + 1
                continue

        buffer += ch
        index += 1
    flush()
    return spans


def _row_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not (stripped.startswith('|') and stripped.endswith('|')):
        return []
    return [cell.strip().replace('\\|', '|') for cell in re.split(r'(?<!\\)\|', stripped[1:-1])]


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r':?-{3,}:?', cell) for cell in cells)


class MarkdownRenderer:
    def __init__(self, width: int = 80) -> None:
        self.width = max(20, width)
        self._lines: list[Line] = []
        self._partial = ''
        self._mode = 'normal'          # normal | fence | table
        self._fence_ticks = 0
        self._fence_lang = ''
        self._fence_lines: list[str] = []
        self._table_lines: list[str] = []
        self._finished = False

    # ── feeding ─────────────────────────────────────────────
    def feed(self, chunk: str) -> None:
        if not chunk:
            return
        self._partial += chunk
        while '\n' in self._partial:
            line, self._partial = self._partial.split('\n', 1)
            self._process_line(line.rstrip('\r'))

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._partial:
            line = self._partial
            self._partial = ''
            if self._mode == 'fence':
                self._fence_lines.append(line)
            elif self._mode == 'table':
                self._table_lines.append(line)
                self._flush_table()
            else:
                self._process_line(line.rstrip('\r'), final=True)
        if self._mode == 'fence':
            self._flush_fence()
        if self._mode == 'table':
            self._flush_table()

    def lines(self) -> list[Line]:
        """Rendered lines including any in-progress block or partial line."""
        result = [list(line) for line in self._lines]
        if self._mode == 'fence':
            if self._fence_lang:
                result.append(self._fence_header(self._fence_lang))
            result.extend(self._render_code_lines(self._fence_lines))
            if self._partial:
                result.extend(self._render_code_lines([self._partial]))
        elif self._mode == 'table':
            table = list(self._table_lines)
            if self._partial:
                table.append(self._partial)
            result.extend(self._render_table(table))
        elif self._partial:
            result.extend(self._render_normal_line(self._partial, final=False))
        return result

    def text(self) -> str:
        return '\n'.join(plain_text(line) for line in self.lines())

    # ── line processing ─────────────────────────────────────
    def _process_line(self, line: str, final: bool = False) -> None:
        if self._mode == 'fence':
            fence = _FENCE_RE.match(line.strip())
            if fence and len(fence.group(1)) >= self._fence_ticks:
                self._flush_fence()
            else:
                self._fence_lines.append(line)
            return
        if self._mode == 'table':
            if _row_cells(line):
                self._table_lines.append(line)
                return
            self._flush_table()
            # fall through to normal processing of this line

        stripped = line.strip()
        lead = stripped[:1]
        # Only a fence or a table row can claim a line before the block pass,
        # and both announce themselves in the first glyph; skip the regexes
        # entirely for the ordinary prose that makes up most of a transcript.
        if lead == '`':
            fence = _FENCE_RE.match(stripped)
            if fence:
                self._mode = 'fence'
                self._fence_ticks = len(fence.group(1))
                self._fence_lang = fence.group(2).strip()
                self._fence_lines = []
                return
        elif lead == '|' and _row_cells(line):
            self._mode = 'table'
            self._table_lines = [line]
            return
        self._lines.extend(self._render_normal_line(line, final=final))

    def _flush_fence(self) -> None:
        if self._fence_lang:
            self._lines.append(self._fence_header(self._fence_lang))
        self._lines.extend(self._render_code_lines(self._fence_lines))
        self._lines.append([Span('└─', 'muted')])
        self._mode = 'normal'
        self._fence_lines = []
        self._fence_ticks = 0
        self._fence_lang = ''

    def _flush_table(self) -> None:
        self._lines.extend(self._render_table(self._table_lines))
        self._table_lines = []
        self._mode = 'normal'

    # ── block renderers ─────────────────────────────────────
    def _fence_header(self, language: str) -> Line:
        return [Span('┌─ ', 'muted'), Span(language or 'code', 'muted', attrs=DIM)]

    def _render_code_lines(self, lines: list[str]) -> list[Line]:
        result: list[Line] = []
        for raw in lines:
            result.append(self._code_line(raw))
        return result

    def _code_line(self, raw: str) -> Line:
        text = raw.expandtabs(4)
        available = max(1, self.width - 2)
        line: Line = [Span('│ ', 'muted')]
        if text.isascii() and text.isprintable():
            # One cell per byte: no glyph scan needed for the common case.
            chunk = text[:available]
            used = len(chunk)
        else:
            used = 0
            pieces: list[str] = []
            for ch in text:
                size = char_width(ch)
                if used + size > available:
                    break
                pieces.append(ch)
                used += size
            chunk = ''.join(pieces)
        add_span(line, Span(chunk, 'subtext', 'panel'))
        padding = self.width - 2 - used
        if padding > 0:
            add_span(line, Span(' ' * padding, 'subtext', 'panel'))
        return line

    def _render_normal_line(self, line: str, final: bool = True) -> list[Line]:
        if not line.strip():
            return [[]]
        lead = line.lstrip()[:1]
        if lead not in _BLOCK_LEAD:
            # Ordinary prose: no block marker can match, so go straight to the
            # inline scanner instead of walking six regexes.
            return [_parse_inline(line)]
        heading = _HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            markers = ('█', '▌', '▸', '›', '·', '·')
            attrs = BOLD if level <= 2 else 0
            role = 'text' if level <= 4 else 'subtext'
            result: Line = [Span(markers[level - 1] + ' ', 'accent' if level <= 3 else 'muted',
                                 attrs=attrs)]
            for span in _parse_inline(heading.group(2), role):
                span.attrs |= attrs
                add_span(result, span)
            return [result]
        if _HR_RE.match(line):
            return [[Span('─' * max(3, min(self.width, 72)), 'border')]]
        task = _TASK_RE.match(line)
        if task:
            box = 'x' if task.group(2).lower() == 'x' else ' '
            marker = Span(f'[{box}] ', 'accent' if box == 'x' else 'muted')
            result = [Span(task.group(1), None), marker]
            for span in _parse_inline(task.group(3)):
                add_span(result, span)
            return [result]
        unordered = _UL_RE.match(line)
        if unordered:
            result = [Span(unordered.group(1), None), Span('• ', 'accent')]
            for span in _parse_inline(unordered.group(2)):
                add_span(result, span)
            return [result]
        ordered = _OL_RE.match(line)
        if ordered:
            result = [Span(ordered.group(1), None),
                      Span(f'{ordered.group(2)}. ', 'accent')]
            for span in _parse_inline(ordered.group(3)):
                add_span(result, span)
            return [result]
        quote = _QUOTE_RE.match(line)
        if quote:
            depth = quote.group(0).count('>')
            content = line[quote.end():]
            result: Line = [Span('│ ' * depth, 'border')]
            for span in _parse_inline(content, 'subtext'):
                add_span(result, span)
            return [result]
        return [_parse_inline(line)]

    # ── tables ──────────────────────────────────────────────
    def _render_table(self, raw_lines: list[str]) -> list[Line]:
        if not raw_lines:
            return []
        rows: list[list[str]] = []
        alignments: list[str] = []
        separator_index = -1
        for index, raw in enumerate(raw_lines):
            cells = _row_cells(raw)
            if not cells:
                continue
            if index == 1 and _is_separator(cells):
                separator_index = len(rows) - 1
                alignments = [
                    'center' if cell.startswith(':') and cell.endswith(':')
                    else 'right' if cell.endswith(':')
                    else 'left'
                    for cell in cells
                ]
                continue
            rows.append(cells)
        if not rows:
            return [[Span(raw, 'text')] for raw in raw_lines]
        columns = max(len(row) for row in rows)
        for row in rows:
            row.extend([''] * (columns - len(row)))
        cell_spans = [[_parse_inline(cell) for cell in row] for row in rows]
        natural = [
            max(3, max(line_width(cell_spans[r][c]) for r in range(len(rows))))
            for c in range(columns)
        ]
        available = max(columns * 4, self.width - (3 * columns + 1))
        widths = list(natural)
        while sum(widths) > available:
            widest = max(range(columns), key=lambda index: widths[index])
            if widths[widest] <= 3:
                break
            widths[widest] -= 1
        output: list[Line] = []
        for row_index, row in enumerate(cell_spans):
            wrapped = [self._wrap_cell(row[col], widths[col]) for col in range(columns)]
            height = max(len(cell) for cell in wrapped)
            for visual in range(height):
                result: Line = [Span('│', 'border')]
                for col in range(columns):
                    cell_lines = wrapped[col]
                    content = cell_lines[visual] if visual < len(cell_lines) else []
                    used = line_width(content)
                    padding = max(0, widths[col] - used)
                    alignment = alignments[col] if col < len(alignments) else 'left'
                    if alignment == 'center':
                        left = padding // 2
                    elif alignment == 'right':
                        left = padding
                    else:
                        left = 0
                    right = padding - left
                    add_span(result, Span(' ' + ' ' * left, None))
                    for span in content:
                        add_span(result, Span(span.text, span.fg, span.bg,
                                              span.attrs, span.alpha))
                    add_span(result, Span(' ' * right + ' ', None))
                    add_span(result, Span('│', 'border'))
                output.append(result)
            if separator_index == row_index:
                rule: Line = [Span('├', 'border')]
                for col in range(columns):
                    rule.append(Span('─' * (widths[col] + 2), 'border'))
                    rule.append(Span('┼' if col < columns - 1 else '┤', 'border'))
                output.append(rule)
        return output

    @staticmethod
    def _wrap_cell(cell: Line, width: int) -> list[Line]:
        from adcode.tui.text import wrap_line
        if not cell:
            return [[]]
        return wrap_line(cell, width)


def render_markdown(text: str, width: int = 80) -> list[Line]:
    renderer = MarkdownRenderer(width)
    renderer.feed(text)
    renderer.finish()
    return renderer.lines()
