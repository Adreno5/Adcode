"""The shell in the real app: no warning over the interface, no lost frame."""

from __future__ import annotations

from bench.script import ScriptedResponder, ToolCall, Turn


def _responder():
    return ScriptedResponder([
        Turn(tool_calls=[ToolCall(name='bash', args={
            'sub': 'execute', 'command': 'echo hello-from-the-shell'}, id='c1')]),
        Turn(text='the shell ran'),
    ])


def test_running_a_command_leaves_the_screen_intact(harness):
    harness.gateway.responder = _responder()
    harness.type_text('run a command')
    harness.wait_prompt('run a command')
    harness.press('enter')
    harness.wait_until(lambda: len(harness.requests) >= 2, timeout=20.0,
                       description='the second model request')
    harness.wait_idle(20.0)
    frame = harness.wait_for_text('hello-from-the-shell')
    assert frame is not None
    text = frame.text()
    assert 'hello-from-the-shell' in text
    # the sdk's "no sandbox backend" warning must not be on the screen
    assert '沙箱后端' not in text, text
    assert '未挂沙箱' not in text, text
    assert not frame.vt_mismatch, 'the frame must match the canvas'


def test_the_command_output_reaches_the_transcript(harness):
    harness.gateway.responder = _responder()
    harness.type_text('run a command')
    harness.wait_prompt('run a command')
    harness.press('enter')
    harness.wait_until(lambda: len(harness.requests) >= 2, timeout=20.0,
                       description='the second model request')
    harness.wait_idle(20.0)
    tools = [item for item in harness.transcript_items()
             if item.get('kind') == 'tool']
    assert tools, harness.transcript_items()
    assert 'hello-from-the-shell' in str(tools[0].get('output') or ''), tools[0]
