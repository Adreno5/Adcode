from adcode.tui.markdown import MarkdownRenderer, render_markdown
from adcode.tui.text import plain_text


def _texts(text: str, width: int = 80) -> list[str]:
    return [plain_text(line) for line in render_markdown(text, width)]


def test_headings_and_paragraphs():
    lines = _texts('# Title\n\nHello **world**.')
    assert lines[0].startswith('█ Title')
    assert any('Hello world.' in line for line in lines)


def test_lists_and_tasks():
    lines = _texts('- one\n- two\n- [x] done')
    assert '• one' in lines
    assert '[x] done' in lines


def test_quote_and_rule():
    lines = _texts('> quoted text\n\n---')
    assert any(line.startswith('│ quoted text') for line in lines)
    assert any(set(line) <= {'─'} and len(line) > 3 for line in lines)


def test_fenced_code():
    lines = _texts('```python\nprint("hi")\n```')
    assert lines[0].startswith('┌─ python')
    assert any('print("hi")' in line for line in lines)
    assert lines[-1].startswith('└─')


def test_table_alignment(monkeypatch):
    text = '| name | value |\n|:-----|------:|\n| a | 1 |\n| bb | 22 |'
    lines = _texts(text, 40)
    assert any('name' in line and 'value' in line for line in lines)
    assert any('│' in line for line in lines)
    assert any('alpha' not in line for line in lines)


def test_links_and_inline_code():
    lines = _texts('Use `read` and [docs](https://example.com).')
    assert 'read' in lines[0]
    assert 'docs <https://example.com>' in lines[0]


def test_streaming_matches_oneshot():
    text = '# Title\n\nA **bold** line.\n\n- one\n- two\n'
    one = _texts(text)
    renderer = MarkdownRenderer(80)
    for chunk in ('# Ti', 'tle\n\nA **bo', 'ld** line.\n\n- on', 'e\n- two\n'):
        renderer.feed(chunk)
    renderer.finish()
    streamed = [plain_text(line) for line in renderer.lines()]
    assert streamed == one


def test_unfinished_fence_is_closed_on_finish():
    renderer = MarkdownRenderer(40)
    renderer.feed('```\ncode line\n')
    renderer.finish()
    lines = [plain_text(line) for line in renderer.lines()]
    assert any('code line' in line for line in lines)
    assert lines[-1].startswith('└─')


def test_escaped_and_html_tags():
    lines = _texts('a \\*not italic\\* <b>bold</b>')
    assert '*' in lines[0]
    assert 'bold' in lines[0]
    assert '<b>' not in lines[0]
