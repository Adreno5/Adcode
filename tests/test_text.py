from adcode.tui.text import (
    Span, add_span, line_width, slice_visual, truncate_line, wrap_line,
    wrap_lines,
)


def test_add_span_merges_same_style():
    line = []
    add_span(line, Span('a'))
    add_span(line, Span('b'))
    assert len(line) == 1
    assert line[0].text == 'ab'
    add_span(line, Span('c', 'accent'))
    assert len(line) == 2


def test_wrap_line_widths():
    rows = wrap_line([Span('abcdefgh')], 3)
    assert [''.join(s.text for s in row) for row in rows] == ['abc', 'def', 'gh']


def test_wrap_line_cjk():
    rows = wrap_line([Span('中中中')], 4)
    assert [''.join(s.text for s in row) for row in rows] == ['中中', '中']


def test_wrap_first_width():
    rows = wrap_line([Span('abcdef')], 4, first_width=2)
    assert [''.join(s.text for s in row) for row in rows] == ['ab', 'cdef']


def test_truncate_line_keeps_styles():
    line = [Span('abc', 'text'), Span('def', 'accent')]
    result = truncate_line(line, 4, '…')
    assert ''.join(s.text for s in result) == 'abc…'


def test_slice_visual_negative_inclusive():
    lines = [[Span(str(index))] for index in range(10)]
    window = slice_visual(lines, -5, -1)
    assert [line[0].text for line in window] == ['5', '6', '7', '8', '9']
    window = slice_visual(lines, -6, -2)
    assert [line[0].text for line in window] == ['4', '5', '6', '7', '8']


def test_wrap_lines_multiline():
    rows = wrap_lines([[Span('ab')], [Span('cd')]], 10)
    assert len(rows) == 2
    assert line_width(rows[1]) == 2
