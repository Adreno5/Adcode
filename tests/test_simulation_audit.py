"""The audit trail as the user sees it."""

from __future__ import annotations

from bench.script import ScriptedResponder, ToolCall, Turn


def _responder():
    return ScriptedResponder([
        Turn(tool_calls=[ToolCall(name='read', args={'path': 'notes.txt'},
                                  id='call_read_1')]),
        Turn(tool_calls=[ToolCall(name='grep', args={'pattern': 'todo'},
                                  id='call_grep_1')]),
        Turn(text='looked around'),
    ])


def test_the_audit_command_lists_the_calls(harness):
    harness.gateway.responder = _responder()
    harness.type_text('look around')
    harness.wait_prompt('look around')
    harness.press('enter')
    harness.wait_until(lambda: len(harness.requests) >= 3, timeout=20.0,
                       description='three model requests')
    harness.wait_idle(20.0)

    harness.type_text('/audit')
    harness.wait_prompt('/audit')
    harness.press('enter')
    frame = harness.wait_for_text('Audit trail')
    assert frame is not None
    text = frame.text()
    assert 'read' in text and 'grep' in text, text


def test_the_audit_trail_survives_the_session(harness):
    harness.gateway.responder = _responder()
    harness.type_text('look around')
    harness.wait_prompt('look around')
    harness.press('enter')
    harness.wait_until(lambda: len(harness.requests) >= 3, timeout=20.0,
                       description='three model requests')
    harness.wait_idle(20.0)

    # the trail is in the database -- and in no file of the project
    events = harness.app.loop.audit.read_history(200)
    assert events, 'the trail is empty'
    assert any(event['tool'] == 'read' for event in events), events
    assert all(event['kind'].startswith(('audit.', 'approval'))
               for event in events), events
    assert not list(harness.workspace.project.glob('*.jsonl'))
    assert not list(harness.workspace.project.glob('audit*'))
