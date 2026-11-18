"""The work-step display: the merged tool chain and the thinking summary.

The thinking and the tool settings live in one group now, and ``/steps`` is a
preset over them:

- ``tools.chain = merged`` folds a chain of consecutive calls into one row
  with the tool names and their counts (``bash x2, grep, write``);
- ``thinking.mode = summary`` keeps the reasoning but only says how much
  there was (``thought of 412 tokens``).
"""

import time
from dataclasses import replace

from adcode.commands import dispatch
from adcode.session import TranscriptItem
from adcode.tui.text import ROW_BAR, plain_text
from adcode.tui.transcript import LiveRender, TranscriptRenderer


def render(config, items, width=80, live=None, version=0):
    renderer = TranscriptRenderer()
    return renderer.render(items, live or LiveRender(), width, config, version,
                           time.time())


def texts(lines):
    return [plain_text(line) for line in lines]


def chain_items() -> list[TranscriptItem]:
    return [
        TranscriptItem(kind='user', turn=1, text='fix the bug'),
        TranscriptItem(kind='tool', turn=1, name='bash', display='bash',
                      args={'command': 'ls -la'}, status='done',
                      output='a\nb\nc'),
        TranscriptItem(kind='tool', turn=1, name='bash', display='bash',
                      args={'command': 'cat x'}, status='done', output='body'),
        TranscriptItem(kind='tool', turn=1, name='grep', display='grep',
                      args={'pattern': 'cursor'}, status='done', output='hit'),
        TranscriptItem(kind='tool', turn=1, name='write', display='write',
                      args={'path': 'a.py'}, status='done', output='written'),
    ]


def test_a_merged_chain_is_one_row_with_the_names_and_the_counts(config):
    config.tools.chain = 'merged'
    lines = texts(render(config, chain_items()))
    tool_rows = [line for line in lines if line.strip().startswith('▏')]
    assert len(tool_rows) == 1, lines
    row = tool_rows[0]
    assert 'bash x2' in row, row
    assert 'grep' in row and 'write' in row, row
    # No details: neither the arguments nor the output.
    assert 'ls -la' not in row, row
    assert 'a\nb' not in row and 'body' not in row, row
    assert 'written' not in row, row


def test_a_separate_chain_keeps_every_call(config):
    config.tools.chain = 'separate'
    lines = texts(render(config, chain_items()))
    tool_rows = [line for line in lines if line.strip().startswith('▏')]
    assert len(tool_rows) == 4, lines
    assert any('ls -la' in row for row in tool_rows)


def test_a_merged_chain_breaks_at_the_next_turn(config):
    config.tools.chain = 'merged'
    items = chain_items() + [
        TranscriptItem(kind='assistant', turn=1, text='done'),
        TranscriptItem(kind='tool', turn=1, name='read', display='read',
                      args={'path': 'b.py'}, status='done', output='text'),
    ]
    lines = texts(render(config, items))
    tool_rows = [line for line in lines if line.strip().startswith('▏')]
    assert len(tool_rows) == 2, lines
    assert 'bash x2' in tool_rows[0], tool_rows
    assert 'read' in tool_rows[1], tool_rows


def test_the_merged_chain_hides_the_tool_output(config):
    config.tools.chain = 'merged'
    config.tools.mode = 'both'
    config.tools.result_mode = 'full'
    lines = texts(render(config, chain_items()))
    assert not any('written' in line for line in lines), lines


def test_the_label_can_carry_the_token_count(config):
    config.thinking.mode = 'label'
    item = TranscriptItem(kind='thinking', turn=1,
                          text='word ' * 400, duration=1.5)
    plain = [line for line in texts(render(config, [item])) if line.strip()]
    assert len(plain) == 1, plain
    assert plain[0].startswith('Thinking'), plain
    assert 'tokens' not in plain[0], plain

    config.thinking.label_tokens = True
    counted = [line for line in texts(render(config, [item])) if line.strip()]
    assert len(counted) == 1, counted
    assert counted[0].startswith('Thinking'), counted
    assert '1.5s' in counted[0] and 'tokens' in counted[0], counted
    # The reasoning itself is not spelled out.
    assert 'word' not in counted[0], counted


def test_the_label_mode_keeps_the_run_label(config):
    config.thinking.mode = 'label'
    config.thinking.label_tokens = True
    live = LiveRender(phase='thinking')
    lines = texts(render(config, [], live=live))
    assert lines and lines[0].startswith('Thinking'), lines
    # The count lands when the round does, not while it streams.
    assert 'token' not in lines[0], lines


def test_the_label_mode_shows_nothing_for_the_running_output(config):
    config.thinking.mode = 'label'
    config.tools.chain = 'merged'
    live = LiveRender(phase='tool', tool_output=['some output'])
    assert texts(render(config, [], live=live)) == []


def test_the_command_applies_a_preset(config):
    class FakeApp:
        def __init__(self) -> None:
            self.config = config
            self.notices: list[str] = []
            self.select = None

        def notify(self, text, level='info') -> None:
            self.notices.append(text)

        def config_changed(self) -> None:
            pass

        def open_select(self, *args, **kwargs) -> None:
            self.select = args

    app = FakeApp()
    dispatch(app, '/steps concise')
    assert config.thinking.mode == 'label'
    assert config.thinking.label_tokens is True
    assert config.tools.chain == 'merged'
    assert config.steps.preset == 'concise'
    dispatch(app, '/steps standard')
    assert config.thinking.mode == 'partial'
    assert config.thinking.label_tokens is False
    assert config.tools.chain == 'separate'
    assert config.steps.preset == 'standard'
    dispatch(app, '/steps detailed')
    assert config.thinking.mode == 'full'
    assert config.tools.result_mode == 'full'
    assert config.steps.preset == 'detailed'
    dispatch(app, '/steps')
    assert app.select is not None
    # ``custom`` is a settings-browser value only: the command never sets it.
    app.select = None
    dispatch(app, '/steps custom')
    assert app.select is not None
    assert config.steps.preset == 'detailed'


# ── collapse after completion ─────────────────────────────
def turn_items() -> list[TranscriptItem]:
    return [
        TranscriptItem(kind='user', turn=1, text='fix the bug'),
        TranscriptItem(kind='thinking', turn=1, text='x' * 400),
        TranscriptItem(kind='tool', turn=1, name='bash', display='bash',
                       args={'command': 'ls'}, status='done', output='a'),
        TranscriptItem(kind='thinking', turn=1, text='y' * 40),
        TranscriptItem(kind='tool', turn=1, name='bash', display='bash',
                       args={'command': 'cat x'}, status='done', output='b'),
        TranscriptItem(kind='tool', turn=1, name='edit', display='edit',
                       args={'path': 'a.py'}, status='done', output='c'),
        TranscriptItem(kind='assistant', turn=1, text='Fixed it.'),
    ]

def step_rows(lines):
    return [line for line in lines if line.strip().startswith(ROW_BAR)]

def test_a_finished_turn_folds_its_steps_into_one_row(config):
    config.steps.collapse_after_completion = True
    lines = texts(render(config, turn_items()))
    rows = step_rows(lines)
    assert len(rows) == 1, lines
    row = rows[0]
    # The thought is a token count, then every tool with how often it ran.
    assert 'Thought for 110 tokens' in row, row
    assert 'bash x2' in row and 'edit' in row, row
    # The row is the summary: neither the arguments nor the output ride along.
    for gone in ('ls', 'cat x', 'a.py', 'xxxx', 'yyyy'):
        assert gone not in row, (gone, row)
    # The summary sits where the work was: after the user message, before
    # the answer.
    assert 'Fixed it.' in lines, lines
    assert lines.index(row) < lines.index('Fixed it.'), lines

def test_the_collapse_is_off_by_default(config):
    assert config.steps.collapse_after_completion is False
    assert len(step_rows(texts(render(config, turn_items())))) > 1

def test_a_running_step_keeps_the_details_on_screen(config):
    config.steps.collapse_after_completion = True
    items = turn_items()
    items[3].status = 'running'      # a tool of the turn is still working
    lines = texts(render(config, items))
    assert len(step_rows(lines)) > 1, lines
    assert not any('Thought for' in line for line in lines), lines

def test_a_turn_that_did_not_answer_keeps_its_steps(config):
    config.steps.collapse_after_completion = True
    items = turn_items()
    items[-1] = TranscriptItem(kind='notice', turn=1, text='[interrupted]')
    lines = texts(render(config, items))
    assert not any('Thought for' in line for line in lines), lines
    assert step_rows(lines), lines

def test_each_turn_folds_on_its_own(config):
    config.steps.collapse_after_completion = True
    second = [
        TranscriptItem(kind='user', turn=2, text='and again'),
        TranscriptItem(kind='tool', turn=2, name='grep', display='grep',
                       status='done', output='hit'),
        TranscriptItem(kind='assistant', turn=2, text='Done.'),
    ]
    rows = step_rows(texts(render(config, turn_items() + second)))
    assert len(rows) == 2, rows
    assert 'bash x2' in rows[0] and 'grep' in rows[1], rows

def test_the_row_follows_the_display_policy(config):
    config.steps.collapse_after_completion = True
    config.tools.mode = 'hidden'
    rows = step_rows(texts(render(config, turn_items())))
    assert len(rows) == 1 and 'bash' not in rows[0], rows
    assert 'Thought for 110 tokens' in rows[0], rows
    config.tools.mode = 'both'
    config.thinking.after_completion = 'hidden'
    rows = step_rows(texts(render(config, turn_items())))
    assert len(rows) == 1 and 'Thought for' not in rows[0], rows
    assert 'bash x2' in rows[0], rows

def test_holding_alt_drops_the_collapse(config):
    """The alt-held reveal shows the steps again, like the other policies."""
    config.steps.collapse_after_completion = True
    config.steps = replace(config.steps, collapse_after_completion=False)
    assert len(step_rows(texts(render(config, turn_items())))) > 1

def test_the_thinking_command_knows_the_label_mode(config):
    class FakeApp:
        def __init__(self) -> None:
            self.config = config
            self.select = None

        def notify(self, text, level='info') -> None:
            pass

        def config_changed(self) -> None:
            pass

        def open_select(self, *args, **kwargs) -> None:
            self.select = args

    app = FakeApp()
    config.steps.preset = 'detailed'
    dispatch(app, '/thinking label')
    assert config.thinking.mode == 'label'
    # A bare mode is the whole display, so the finished thought follows it.
    assert config.thinking.after_completion == 'label'
    # The mode no longer matches the preset, so the settings count as the
    # user's own mix (which is what unlocks them in /config).
    assert config.steps.preset == 'custom'
    # One stage at a time: the other stage keeps its value, and the preset
    # is not touched by the second stage (no preset owns it).
    dispatch(app, '/thinking ongoing partial')
    assert config.thinking.mode == 'partial'
    assert config.thinking.after_completion == 'label'
    dispatch(app, '/thinking after full')
    assert config.thinking.after_completion == 'full'
    assert config.thinking.mode == 'partial'
    assert config.steps.preset == 'custom'
    # A stage without a mode opens that stage's picker.
    app.select = None
    dispatch(app, '/thinking after')
    assert app.select is not None
