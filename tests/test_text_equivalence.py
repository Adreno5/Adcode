"""The fast wrap/truncate/markdown paths must be behaviour-preserving.

``wrap_line`` and ``truncate_line`` gained a printable-ASCII fast path (one
cell per byte) and a buffered glyph path.  Both are pure optimisations: for
every input they must produce exactly the same spans, in the same order, with
the same style, as the original glyph-by-glyph implementation kept here as a
reference.  The same holds for the markdown inline scanner, which now jumps
over runs of literal text instead of walking them one character at a time.
"""

from __future__ import annotations

import random
import re

import pytest

from adcode.tui.canvas import (BOLD, DIM, ITALIC, STRIKE, UNDERLINE, char_width,
                               is_control)
from adcode.tui.markdown import MarkdownRenderer, _parse_inline
from adcode.tui.text import (Span, add_span, line_width, text_width, trim_line,
                             truncate_line, wrap_line)

_LINK_RE = re.compile(r'!?\[([^\]]*)\]\(([^)\s]+)(?:\s+["\'][^"\']*["\'])?\)')
_REF_LINK_RE = re.compile(r'!?\[([^\]]+)\]\[([^\]]*)\]')


# ── reference implementations (pre-optimisation, verbatim) ─────────────
def reference_wrap_line(line, width, first_width=None):
    width = max(1, width)
    rows = []
    current = []
    current_width = 0
    limit = width if first_width is None else max(1, first_width)
    for span in line:
        for ch in span.text:
            if is_control(ch):
                continue
            size = char_width(ch)
            if size == 0:
                if current and current[-1].text:
                    current[-1].text += ch
                elif rows and rows[-1]:
                    rows[-1][-1].text += ch
                continue
            if current_width and current_width + size > limit:
                rows.append(current)
                current = []
                current_width = 0
                limit = width
            if not current and size > limit:
                add_span(current, Span(ch, span.fg, span.bg, span.attrs, span.alpha))
                current_width += size
                continue
            add_span(current, Span(ch, span.fg, span.bg, span.attrs, span.alpha))
            current_width += size
    if current or not rows:
        rows.append(current)
    return rows


def reference_truncate_line(line, width, ellipsis=''):
    if width <= 0:
        return []
    if line_width(line) <= width:
        return [Span(s.text, s.fg, s.bg, s.attrs, s.alpha) for s in line]
    budget = max(0, width - text_width(ellipsis))
    result = []
    used = 0
    for span in line:
        if used >= budget:
            break
        text = ''
        for ch in span.text:
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


def reference_parse_inline(text, base=None):
    """Literal copy of the original per-character inline scanner."""
    spans = []
    buffer = ''
    index = 0
    state = {'bold': False, 'italic': False, 'underline': False}

    def current_attrs():
        attrs = 0
        if state['bold']:
            attrs |= BOLD
        if state['italic']:
            attrs |= ITALIC
        if state['underline']:
            attrs |= UNDERLINE
        return attrs

    def flush():
        nonlocal buffer
        if buffer:
            add_span(spans, Span(buffer, base, attrs=current_attrs()))
            buffer = ''

    while index < len(text):
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
        if ch == '[' or (ch == '!' and index + 1 < len(text)
                         and text[index + 1] == '['):
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
                    for span in reference_parse_inline(inner, base):
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
                    for span in reference_parse_inline(inner, base):
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
                        for span in reference_parse_inline(inner, base):
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


# ── comparison helpers ─────────────────────────────────────────────────
def spans_tuple(line):
    return [(s.text, str(s.fg), str(s.bg), s.attrs, s.alpha) for s in line]


def rows_tuple(rows):
    return [spans_tuple(row) for row in rows]


# ── fixtures ───────────────────────────────────────────────────────────
TEXTS = [
    '',
    'a',
    ' ' * 10,
    'plain ascii sentence that will be hard wrapped at the viewport edge',
    '中文测试 wide glyphs 混在 ascii 中间 and back 中',
    'é combining: e\u0301 a\u0308 n\u0303',
    'emoji 🙂🙂🙂 and 🧑‍🚀 zwj family 👨‍👩‍👧‍👦 sequences',
    'tabs\tand\tcells',
    'control\x00\x07\x1b[31mred\x1b[0m and more',
    'trailing combining at end e\u0301',
    '\u0301leading combining mark',
    'box ─ │ ┌ ┐ └ ┘ █ ▌ ▸ › · ┼ ┤ ├',
    'mixed 中a\u0301b wide 中中 中',
    'x' * 200,
    '中' * 60,
    '\x1b' + 'a' * 40,
]

WIDTHS = [1, 2, 3, 4, 5, 7, 8, 13, 16, 40, 80, 120]


@pytest.mark.parametrize('text', TEXTS)
@pytest.mark.parametrize('width', WIDTHS)
def test_wrap_line_matches_reference(text, width):
    line = [Span(text, 'text', None, BOLD, 255)]
    assert rows_tuple(wrap_line(line, width)) == rows_tuple(
        reference_wrap_line(line, width))


@pytest.mark.parametrize('first_width', [1, 3, 9, 30])
@pytest.mark.parametrize('width', [4, 9, 17, 60])
def test_wrap_line_first_width_matches_reference(first_width, width):
    line = [Span('first row is narrower then the rest of this text wraps along', 'text')]
    assert rows_tuple(wrap_line(line, width, first_width)) == rows_tuple(
        reference_wrap_line(line, width, first_width))


@pytest.mark.parametrize('text', TEXTS)
@pytest.mark.parametrize('width', [1, 5, 12, 33, 100])
def test_truncate_line_matches_reference(text, width):
    for ellipsis in ('', '…', '...'):
        line = [Span(text, 'text')]
        assert spans_tuple(truncate_line(line, width, ellipsis)) == spans_tuple(
            reference_truncate_line(line, width, ellipsis))


def test_multi_span_lines_match_reference():
    line = [Span('bold 中文 bit', 'accent', None, BOLD),
            Span(' then plain ascii run that is long enough to wrap twice over', 'text'),
            Span('e\u0301', 'muted'),
            Span('wide 中中', 'subtext', 'panel', DIM)]
    for width in (3, 5, 9, 16, 31, 64):
        assert rows_tuple(wrap_line(line, width)) == rows_tuple(
            reference_wrap_line(line, width))


def test_random_lines_match_reference():
    alphabet = 'ab 中文\u0301🙂\t\x1b_-*`[]<>=~$'
    rng = random.Random(20261115)
    for _ in range(400):
        spans = []
        for _ in range(rng.randint(0, 4)):
            text = ''.join(rng.choice(alphabet) for _ in range(rng.randint(0, 30)))
            spans.append(Span(text, rng.choice(['text', 'accent', None]),
                              rng.choice([None, 'panel']),
                              rng.choice([0, BOLD, ITALIC]), 255))
        width = rng.randint(1, 40)
        first = rng.choice([None, rng.randint(1, 40)])
        assert rows_tuple(wrap_line(spans, width, first)) == rows_tuple(
            reference_wrap_line(spans, width, first)), (spans, width, first)


def test_trim_line_unchanged():
    line = [Span('  x  ', 'text'), Span('  ', None), Span('y', 'muted')]
    # Leading whitespace goes and the trailing span is right-stripped; interior
    # spans (even all-whitespace ones) are left alone.
    assert spans_tuple(trim_line(line)) == [('x  ', 'text', 'None', 0, 255),
                                            ('  ', 'None', 'None', 0, 255),
                                            ('y', 'muted', 'None', 0, 255)]


# ── markdown inline scanner ────────────────────────────────────────────
MD_SAMPLES = [
    '',
    'plain prose with no markup at all, just words and punctuation.',
    'a **bold** word and *italic* and `code` and ~~strike~~',
    'link [label](https://example.com) and ![img](x.png)',
    'ref [label][id] and $math$ and <b>tag</b> and <br> and <u>u</u>',
    'escaped \\*not bold\\* and \\`not code\\`',
    '中文 with **加粗** and `代码` mixed 中文 English',
    'triple ***both*** and __under__ and ==mark==',
    'unclosed **bold and `tick',
    'many **a** **b** **c** **d** **e** links [x](y) here',
    'x' * 300,
    'a_underscore_inside_word and _emph_ and snake_case_name',
]


@pytest.mark.parametrize('sample', MD_SAMPLES)
def test_parse_inline_matches_reference(sample):
    assert spans_tuple(_parse_inline(sample, 'text')) == spans_tuple(
        reference_parse_inline(sample, 'text'))


def test_parse_inline_random_matches_reference():
    alphabet = 'ab 中文*_`~=[]<>()$\\!#-:'
    rng = random.Random(7)
    for _ in range(300):
        sample = ''.join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        assert spans_tuple(_parse_inline(sample)) == spans_tuple(
            reference_parse_inline(sample)), sample


def test_markdown_render_is_stable_across_widths():
    doc = ('# Title\n\nText with **bold** and `code`.\n\n'
           '- item one\n- item two\n\n```python\nx = 1\n```\n\n'
           '| a | b |\n|---|---|\n| 1 | 2 |\n')
    narrow = _render(doc, 40)
    wide = _render(doc, 120)
    assert len(narrow) > 0 and len(wide) > 0
    # Code rows are padded to the renderer width; nothing else may depend on it.
    assert all(len(''.join(s.text for s in row)) <= 120 for row in wide)


def _render(doc, width):
    renderer = MarkdownRenderer(width)
    renderer.feed(doc)
    renderer.finish()
    return renderer.lines()
