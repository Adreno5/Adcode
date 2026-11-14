from adcode.tui.app import PromptState, _format_tokens, _prompt_layout
from adcode.tui.text import plain_text


def layout(text, cursor, width, anchor=None):
    rows, row, col = _prompt_layout(text, cursor, anchor, width)
    return [plain_text(line) for line in rows], (row, col)


def test_layout_single_line():
    rows, caret = layout('hello', 0, 20)
    assert rows == ['> hello']
    assert caret == (0, 2)
    rows, caret = layout('hello', 5, 20)
    assert caret == (0, 7)


def test_layout_wraps_and_tracks_caret():
    text = 'hello world this is long'
    rows, caret = layout(text, len(text), 12)
    assert rows[0] == '> hello worl'
    assert rows[-1] == '  long'
    assert caret == (2, 6)


def test_layout_multiline():
    rows, caret = layout('a\nbb\nccc', 2, 20)
    assert rows == ['> a', '  bb', '  ccc']
    assert caret == (1, 2)
    rows, caret = layout('a\nbb\nccc', 5, 20)
    assert caret == (2, 2)


def test_layout_cjk():
    rows, caret = layout('中文', 2, 20)
    assert rows == ['> 中文']
    assert caret == (0, 6)


def test_layout_selection_spans():
    rows, _ = layout('hello', 5, 20, anchor=1)
    colors = [(span.text, span.bg) for span in rows_sources('hello', 5, 1)]
    assert any(bg == 'selection_bg' for _, bg in colors)


def rows_sources(text, cursor, anchor):
    from adcode.tui.app import _prompt_layout
    rows, _, _ = _prompt_layout(text, cursor, anchor, 20)
    return rows[0]


def test_prompt_state_insert_and_selection():
    state = PromptState()
    state.insert('abc')
    state.anchor = 1
    state.cursor = 2
    state.insert('X')
    assert state.text == 'aXc'
    assert state.selection() == (2, 2)


def test_prompt_state_delete():
    state = PromptState()
    state.insert('abc')
    state.delete_backward()
    assert state.text == 'ab'
    state.cursor = 0
    state.delete_forward()
    assert state.text == 'b'


def test_format_tokens():
    assert _format_tokens(999) == '999'
    assert _format_tokens(1_500) == '1.5k'
    assert _format_tokens(2_000_000) == '2.0M'


def test_prompt_state_cut_copy():
    state = PromptState()
    state.insert('hello')
    state.anchor = 1
    state.cursor = 4
    assert state.copy() == 'ell'
    assert state.cut() == 'ell'
    assert state.text == 'ho'
