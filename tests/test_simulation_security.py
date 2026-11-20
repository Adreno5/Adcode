"""The approval prompt, end to end.

In ``ask`` mode the effectful tools wait for a yes.  The dialog is opened by
the ui while the tool waits in its worker thread; the answer comes back as a
key press.
"""

from __future__ import annotations

import pytest

from bench.script import ScriptedResponder, ToolCall, Turn


def _ask_mode(config) -> None:
    config.security.mode = 'ask'


@pytest.fixture()
def strict_harness(harness_factory):
    def make(**kwargs):
        kwargs.setdefault('config_mutator', _ask_mode)
        return harness_factory(**kwargs)
    return make


def _bash_responder(command: str = 'echo hello'):
    return ScriptedResponder([
        Turn(tool_calls=[ToolCall(name='bash',
                                  args={'sub': 'execute', 'command': command},
                                  id='call_bash_1')]),
        Turn(text='the command ran'),
    ])


def test_an_effectful_call_waits_for_the_user(strict_harness):
    harness = strict_harness()
    harness.gateway.responder = _bash_responder()

    harness.type_text('run echo hello')
    harness.wait_prompt('run echo hello')
    harness.press('enter')

    # the prompt appears while the tool waits
    harness.wait_for_text('Approve tool call')
    frame = harness.screen.latest
    assert frame is not None
    assert 'bash' in frame.text(), frame.text()
    assert 'allow once' in frame.text(), frame.text()

    # allow it
    harness.type_text('y')
    harness.wait_until(lambda: len(harness.requests) >= 2, timeout=20.0,
                       description='the second model request')
    harness.wait_idle(20.0)
    assert not harness.screen.latest.vt_mismatch


def test_refusing_a_call_stops_it(strict_harness):
    harness = strict_harness()
    harness.gateway.responder = _bash_responder()

    harness.type_text('run echo hello')
    harness.wait_prompt('run echo hello')
    harness.press('enter')
    harness.wait_for_text('Approve tool call')
    harness.type_text('n')

    harness.wait_until(lambda: len(harness.requests) >= 2, timeout=20.0,
                       description='the second model request')
    harness.wait_idle(20.0)

    # the model is told the call was refused ...
    tools = [item for item in harness.transcript_items()
             if item.get('kind') == 'tool']
    assert tools, harness.transcript_items()
    assert '拒绝' in str(tools[0].get('output') or ''), tools[0]
    # ... and the user sees it in the transcript
    harness.wait_for_text('refused')


def test_always_allow_asks_only_once(strict_harness):
    harness = strict_harness()
    harness.gateway.responder = ScriptedResponder([
        Turn(tool_calls=[ToolCall(name='bash',
                                  args={'sub': 'execute', 'command': 'echo one'},
                                  id='call_bash_1')]),
        Turn(tool_calls=[ToolCall(name='bash',
                                  args={'sub': 'execute', 'command': 'echo two'},
                                  id='call_bash_2')]),
        Turn(text='both commands ran'),
    ])

    harness.type_text('run two commands')
    harness.wait_prompt('run two commands')
    harness.press('enter')
    harness.wait_for_text('Approve tool call')
    harness.type_text('a')      # always allow this tool
    harness.wait_until(lambda: len(harness.requests) >= 2, timeout=20.0,
                       description='the second model request')
    # the second call must not ask again
    harness.wait_until(lambda: len(harness.requests) >= 3, timeout=20.0,
                       description='the third model request')
    harness.wait_idle(20.0)
    frame = harness.screen.latest
    assert frame is not None
    assert 'Approve tool call' not in frame.text(), frame.text()
