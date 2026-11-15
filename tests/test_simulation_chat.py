"""End-to-end simulated chat runs of the real TUI application.

The app is started on a thread with a fake screen and fake input; the model
gateway is a real local HTTP server.  These tests assert on what the terminal
would show, on the request the production client actually sent, and on the
files/session the production agent loop wrote.
"""

from __future__ import annotations

import threading

import pytest

from harness.script import ScriptedResponder, ToolCall, Turn
from harness.workspace import GLOBAL_MARKER, PROJECT_MARKER

WIRES = [
    'openai chat completions',
    'openai responses',
    'anthropic messages',
]


def test_typed_message_streams_answer_and_persists(harness):
    harness.gateway.responder = ScriptedResponder([Turn(
        thinking='Consider the request briefly.',
        text='Hello from the **gateway**.',
    )])

    harness.type_text('hi there')
    harness.wait_prompt('hi there')
    assert harness.screen.latest is not None
    assert '> hi there' in harness.screen_text()

    harness.press('enter')
    harness.wait_until(lambda: len(harness.requests) == 1, timeout=15.0,
                       description='the first model request')
    harness.wait_idle(20.0)
    harness.wait_for_text('Hello from the gateway.')

    items = harness.transcript_texts()
    assert items[0] == ('user', 'hi there')
    assert ('thinking', 'Consider the request briefly.') in items
    assert ('assistant', 'Hello from the **gateway**.') in items

    frame = harness.screen.latest
    assert frame is not None
    assert not frame.vt_mismatch, 'serialized ANSI must match the canvas'
    assert frame.vt_text == frame.text()

    request = harness.requests[0]
    assert 'hi there' in request.all_text
    assert request.model == 'harness-model'
    assert request.reasoning_effort in ('low', 'medium', 'high')
    assert PROJECT_MARKER in request.system_text
    assert GLOBAL_MARKER in request.system_text
    assert str(harness.workspace.project) in request.system_text
    assert {'read', 'write', 'edit', 'bash'}.issubset(set(request.tool_names))

    session = harness.workspace.session_payload()
    assert session, 'the agent loop must persist the session in the temp cwd'
    assert session['messages'][0] == {'role': 'user', 'content': 'hi there'}
    assert any(message.get('role') == 'assistant'
               for message in session['messages'])
    assert harness.workspace.session_files()

    # Everything happened against the temporary config, never the real one.
    assert harness.app is not None
    assert harness.app.store.path == harness.workspace.config_path
    assert harness.workspace.config_json()['selected_model'] == \
        'Harness::harness-model'
    assert str(harness.workspace.config_path).startswith(
        str(harness.workspace.root))


def test_tool_call_writes_into_the_temp_workspace(harness):
    harness.gateway.responder = ScriptedResponder([
        Turn(thinking='Write the file with the real tool registry.',
             tool_calls=[ToolCall(
                 name='write',
                 args={'path': 'notes.txt',
                       'content': 'from a scripted tool call\n'},
                 id='call_write_1',
             )]),
        Turn(text='Wrote notes.txt.'),
    ])

    harness.submit('create notes.txt', expect_requests=2, timeout=25.0)
    assert harness.workspace.project_file('notes.txt') == \
        'from a scripted tool call\n'
    assert not harness.screen.latest.vt_mismatch

    tool_items = [item for item in harness.transcript_items()
                  if item['kind'] == 'tool']
    assert len(tool_items) == 1
    tool = tool_items[0]
    assert tool['name'] == 'write'
    assert tool['status'] == 'done'
    assert 'notes.txt' in tool['output']
    assert tool['diff'] and tool['diff']['path'] == 'notes.txt'

    frame_text = harness.screen_text()
    assert 'write' in frame_text
    assert 'notes.txt' in frame_text
    assert 'Wrote notes.txt.' in frame_text

    # The second request carries the tool result back to the model.
    assert len(harness.requests) == 2
    follow_up = harness.requests[1]
    assert any(message.get('role') == 'tool'
               for message in follow_up.messages)
    assert 'notes.txt' in follow_up.all_text
    assert 'from a scripted tool call' in follow_up.all_text


def test_thinking_is_visible_before_the_answer_arrives(harness):
    hold = threading.Event()
    harness.gateway.responder = ScriptedResponder([Turn(
        thinking='HELD-REASONING: inspecting the project.',
        text='FINAL-ANSWER: the task is done.',
        hold=hold,
    )])

    harness.type_text('slow question')
    harness.press('enter')
    harness.wait_until(
        lambda: harness.phase == 'thinking'
        and 'HELD-REASONING' in harness.screen_text(),
        timeout=15.0, description='the streamed reasoning frame')

    frame_text = harness.screen_text()
    assert 'FINAL-ANSWER' not in frame_text
    assert len(harness.requests) == 1
    assert harness.requests[0].last_user_text() == 'slow question'

    hold.set()
    harness.wait_idle(20.0)
    harness.wait_for_text('FINAL-ANSWER')
    assert harness.phase == 'idle'


def test_escape_interrupts_an_in_flight_turn(harness):
    hold = threading.Event()
    harness.gateway.responder = ScriptedResponder([Turn(
        thinking='HELD-REASONING: this turn will be interrupted.',
        text='This answer must not be committed.',
        hold=hold,
    )])

    harness.type_text('start a slow turn')
    harness.press('enter')
    harness.wait_until(
        lambda: harness.phase == 'thinking'
        and 'HELD-REASONING' in harness.screen_text(),
        timeout=15.0, description='the held reasoning stream')

    harness.press('esc')
    hold.set()
    harness.wait_idle(20.0)
    harness.wait_for_text('[interrupted]')

    assert harness.phase == 'idle'
    assert any(kind == 'notice' and text == '[interrupted]'
               for kind, text in harness.transcript_texts())
    assert harness.app is not None and harness.app.loop is not None
    assert harness.app.loop.context.committed == []
    assert not harness.screen.latest.vt_mismatch


@pytest.mark.parametrize('wire', WIRES)
def test_app_runs_end_to_end_over_every_wire(harness_factory, wire):
    harness = harness_factory(
        wire=wire,
        responder=ScriptedResponder([Turn(
            thinking='Wire-specific plan.',
            text='Wire answer from the gateway.',
        )]),
    )
    harness.submit('hello over this wire')
    harness.wait_for_text('Wire answer from the gateway.')
    assert 'Wire answer from the gateway.' in harness.screen_text()
    assert harness.requests[0].wire == wire
    assert not harness.screen.latest.vt_mismatch
