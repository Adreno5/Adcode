import time

from adcode.session import TranscriptItem
from adcode.tui.text import plain_text
from adcode.tui.transcript import LiveRender, TranscriptRenderer, _thinking_suffix


def render(config, items, width=80, live=None, version=0, now=None,
           palette=None):
    renderer = TranscriptRenderer()
    return renderer.render(items, live or LiveRender(), width, config,
                           version, now if now is not None else time.time(),
                           palette)


def texts(lines):
    return [plain_text(line) for line in lines]


def test_user_and_assistant(config):
    items = [
        TranscriptItem(kind='user', turn=1, text='Hello there'),
        TranscriptItem(kind='assistant', turn=1, text='Hi **you**.'),
    ]
    lines = texts(render(config, items))
    assert lines[0] == '> Hello there'
    assert any('Hi you.' in line for line in lines)


def test_thinking_modes(config):
    # A committed block renders the after-completion stage, so both stages
    # move together here -- that is what /thinking <mode> does as well.
    item = TranscriptItem(kind='thinking', turn=1, text='deep thought',
                          duration=1.5)
    config.thinking.mode = 'hidden'
    config.thinking.after_completion = 'hidden'
    assert texts(render(config, [item])) == []
    config.thinking.mode = 'label'
    config.thinking.after_completion = 'label'
    lines = texts(render(config, [item]))
    assert lines[0].startswith('Thinking')
    config.thinking.mode = 'full'
    config.thinking.after_completion = 'full'
    assert any('deep thought' in line for line in texts(render(config, [item])))
    config.thinking.mode = 'partial'
    config.thinking.after_completion = 'partial'
    assert any('deep thought' in line for line in texts(render(config, [item])))


def test_partial_window_ellipsis_and_gradient(config):
    config.thinking.mode = 'partial'
    config.thinking.after_completion = 'partial'
    config.thinking.from_index = -2
    config.thinking.to_index = -1
    config.thinking.style = 'gradient'
    config.thinking.gradient_top_alpha = 10
    config.thinking.ellipsis = 'label'
    config.thinking.ellipsis_label = '...'
    text = '\n'.join(f'line {index}' for index in range(10))
    item = TranscriptItem(kind='thinking', turn=1, text=text, duration=1.0)
    renderer = TranscriptRenderer()
    lines = renderer.render([item], LiveRender(), 40, config, 5, 0)
    joined = ' '.join(plain_text(line) for line in lines)
    assert 'line 9' in joined and 'line 8' in joined
    assert 'line 7' not in joined
    assert '...' in joined
    # The first drawn content row is the top of the window and therefore faded.
    content = [line for line in lines if line and plain_text(line).startswith('line')]
    assert content
    assert min(span.alpha for span in content[0]) < 255


def test_tool_display_modes(config):
    item = TranscriptItem(kind='tool', turn=1, name='bash', display='bash',
                          args={'command': 'ls -la'}, status='done',
                          output='one\ntwo\nthree', duration=0.2)
    config.tools.mode = 'hidden'
    assert texts(render(config, [item])) == []
    config.tools.mode = 'call'
    lines = texts(render(config, [item]))
    assert len(lines) == 1
    assert 'bash' in lines[0] and 'ls -la' in lines[0]
    config.tools.show_args = False
    assert 'ls -la' not in ' '.join(texts(render(config, [item])))
    config.tools.mode = 'result'
    lines = texts(render(config, [item]))
    assert not any('bash  ls' in line for line in lines)
    assert any('three' in line for line in lines)
    config.tools.mode = 'both'
    joined = ' '.join(texts(render(config, [item])))
    assert 'bash' in joined and 'three' in joined


def test_tool_call_lead_in_gradient(config, palette):
    from adcode.theme import lead_in_gradient
    from adcode.config import ToolDisplay
    item = TranscriptItem(kind='tool', turn=1, name='bash', display='bash',
                          args={'command': 'ls -la'}, status='done')
    config.tools = ToolDisplay(mode='call')

    def cell_backgrounds(line):
        # Tool call rows only use single-cell characters here, so one character
        # maps to one terminal column.
        return [span.bg for span in line for _ in span.text]

    lines = render(config, [item], palette=palette)
    assert len(lines) == 1
    colors = lead_in_gradient(palette)
    for column, bg in enumerate(cell_backgrounds(lines[0])[:len(colors)]):
        assert bg == colors[column]
    assert colors[0] != palette.panel
    # Past the glow the row is plain panel grey again.
    for bg in cell_backgrounds(lines[0])[len(colors):]:
        assert bg == 'panel'

    # Without a palette the row keeps the plain role-based background.
    plain = render(config, [item])
    assert all(span.bg == 'panel' for span in plain[0])


def test_tool_call_row_hugs_the_left_edge(config, palette):
    """The accent bar sits flush left; its gutter cell is still reserved."""
    from adcode.tui.canvas import char_width
    from adcode.tui.text import ROW_BAR, ROW_BLEED, line_width
    item = TranscriptItem(kind='tool', turn=1, name='bash', display='bash',
                          args={'command': 'ls -la'}, status='done')
    config.tools.mode = 'call'
    lines = render(config, [item], palette=palette)
    row = lines[0]
    # The row claims one cell of the transcript's left margin ...
    assert getattr(row, 'bleed', 0) == ROW_BLEED
    # ... but keeps the two-cell gutter, so the name keeps its separation.
    assert plain_text(row).startswith(f'{ROW_BAR} bash')
    assert row[0].fg == 'accent' and row[0].text == ROW_BAR
    # The bar paints against the left border of its cell, not its middle ...
    assert ROW_BAR == '▏'
    # ... and takes no more room than the old ``│`` did.
    assert char_width(ROW_BAR) == char_width('│') == 1
    # The row is one cell wider than wrapped text, so both edges line up.
    assert line_width(row) == 80 + ROW_BLEED

    # Every other row keeps the plain one-cell margin.
    others = render(config, [TranscriptItem(kind='user', turn=1, text='hi')])
    assert getattr(others[0], 'bleed', 0) == 0


def test_row_offset_for_bleeding_tool_rows():
    from adcode.tui.app import _row_offset
    from adcode.tui.text import ROW_BAR, ROW_BLEED, Row, Span
    assert _row_offset([Span('> hi', 'accent')]) == 1
    assert _row_offset(Row([Span(f'{ROW_BAR} bash', 'accent', 'panel')],
                           ROW_BLEED)) == 0


def test_tool_result_partial_and_full(config):
    item = TranscriptItem(kind='tool', turn=1, name='read', display='read',
                          args={}, status='done',
                          output='\n'.join(f'row {index}' for index in range(10)))
    config.tools.mode = 'result'
    config.tools.result_mode = 'partial'
    config.tools.from_index = -2
    config.tools.to_index = -1
    config.tools.ellipsis = 'count'
    lines = texts(render(config, [item]))
    assert any('row 9' in line for line in lines)
    assert all('row 7' not in line for line in lines)
    assert any('characters omitted' in line for line in lines)
    config.tools.result_mode = 'full'
    lines = texts(render(config, [item]))
    assert any('row 0' in line for line in lines)
    assert any('row 9' in line for line in lines)


def test_diff_rendering(config):
    diff = {
        'path': 'a.py', 'additions': 1, 'deletions': 1, 'omitted': False,
        'unified': '--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n-old\n+new\n same',
    }
    item = TranscriptItem(kind='tool', turn=1, name='edit', display='edit',
                          args={'path': 'a.py'}, status='done',
                          output='ok', diff=diff)
    config.tools.mode = 'both'
    lines = texts(render(config, [item]))
    joined = '\n'.join(lines)
    assert 'a.py' in joined
    assert '+new' in joined
    assert '-old' in joined
    assert '+1' in joined and '-1' in joined


def test_notice_aside_agent(config):
    items = [
        TranscriptItem(kind='notice', text='heads up', level='warning'),
        TranscriptItem(kind='aside', turn=1, display='why?', text='because'),
        TranscriptItem(kind='agent', turn=1, agent='child', text='reading files'),
    ]
    lines = texts(render(config, items))
    joined = '\n'.join(lines)
    assert '! heads up' in joined
    assert '? why?' in joined and 'because' in joined
    assert 'child' in joined and 'reading files' in joined


def test_live_thinking_label_animation(config):
    config.thinking.mode = 'label'
    config.thinking.animation = 'fixed'
    config.thinking.fixed_suffix = '...'
    live = LiveRender(phase='thinking')
    lines = texts(render(config, [], live=live, now=100.0))
    assert lines[0].startswith('Thinking ...')


def test_thinking_suffix_rotation():
    from adcode.config import ThinkingDisplay
    display = ThinkingDisplay(animation='rotating',
                              rotation=['a', 'b'], rotation_ms=[100, 100])
    assert _thinking_suffix(display, 0.0) == 'a'
    assert _thinking_suffix(display, 0.15) == 'b'
    assert _thinking_suffix(display, 0.25) == 'a'


def test_live_answer_and_compaction(config):
    live = LiveRender(phase='answering')
    from adcode.tui.markdown import MarkdownRenderer
    renderer = MarkdownRenderer(60)
    renderer.feed('streaming **answer**')
    live.answer_lines = renderer.lines()
    lines = texts(render(config, [], live=live))
    assert any('streaming answer' in line for line in lines)
    live = LiveRender(phase='compacting')
    renderer = MarkdownRenderer(60)
    renderer.feed('summary text')
    live.compaction_lines = renderer.lines()
    lines = texts(render(config, [], live=live))
    assert any('compacting' in line for line in lines)
    assert any('summary text' in line for line in lines)


def test_agent_items_spacing(config):
    items = [
        TranscriptItem(kind='assistant', turn=1, text='answer one'),
        TranscriptItem(kind='agent', turn=1, agent='child', text='working'),
        TranscriptItem(kind='assistant', turn=1, text='answer two'),
    ]
    lines = render(config, items)
    assert any(plain_text(line) == '' for line in lines)

def test_after_completion_is_a_stage_of_its_own(config):
    """The finished block follows its own mode, not the ongoing one."""
    item = TranscriptItem(kind='thinking', turn=1, text='deep thought',
                          duration=1.5)
    config.thinking.mode = 'full'
    config.thinking.after_completion = 'full'
    assert any('deep thought' in line for line in texts(render(config, [item])))
    # The live mode still says full; only the finished stage changed.
    config.thinking.after_completion = 'label'
    lines = [line for line in texts(render(config, [item])) if line.strip()]
    assert len(lines) == 1, lines
    assert lines[0].startswith('Thinking') and '1.5s' in lines[0], lines
    assert 'deep thought' not in lines[0], lines
    # Partial keeps the same window the ongoing block uses.
    config.thinking.after_completion = 'partial'
    config.thinking.from_index = -1
    config.thinking.to_index = -1
    assert 'deep thought' in ' '.join(texts(render(config, [item])))
    config.thinking.after_completion = 'hidden'
    assert texts(render(config, [item])) == []
    # 'keep' is the old "follow the ongoing mode", still read from old files
    # and used by the alt-held reveal.
    config.thinking.after_completion = 'keep'
    assert any('deep thought' in line for line in texts(render(config, [item])))

def test_after_completion_label_can_carry_the_token_count(config):
    item = TranscriptItem(kind='thinking', turn=1, text='word ' * 40,
                          duration=0.4)
    config.thinking.mode = 'full'
    config.thinking.after_completion = 'label'
    lines = [line for line in texts(render(config, [item])) if line.strip()]
    assert lines[0].startswith('Thinking'), lines
    assert 'tokens' not in lines[0], lines
    config.thinking.label_tokens = True
    lines = [line for line in texts(render(config, [item])) if line.strip()]
    assert 'tokens' in lines[0], lines
    assert 'word' not in lines[0], lines
