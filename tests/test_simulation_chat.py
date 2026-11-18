"""End-to-end simulated chat runs of the real TUI application.

The app is started on a thread with a fake screen and fake input; the model
gateway is a real local HTTP server.  These tests assert on what the terminal
would show, on the request the production client actually sent, and on the
files/session the production agent loop wrote.
"""

from __future__ import annotations

import threading

import pytest

from adcode import images
from adcode.tui.text import ROW_BAR
from bench.script import ScriptedResponder, ToolCall, Turn
from bench.workspace import GLOBAL_MARKER, PROJECT_MARKER

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
    assert session, 'the agent loop must persist the session in the app state dir'
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


@pytest.mark.parametrize('wire', WIRES)
def test_alt_v_sends_a_native_image_part(harness_factory, wire):
    harness = harness_factory(wire=wire)
    harness.gateway.responder = ScriptedResponder([Turn(text='I see the image.')])
    harness.clipboard_image = images.encode_png(
        1, 1, bytes([255, 0, 0, 255]))

    harness.press('alt+v')
    harness.wait_until(lambda: '[image:' in harness.prompt_text, timeout=10.0,
                       description='the pasted image marker')
    harness.type_text(' what is this?')
    harness.press('enter')
    harness.wait_for_requests(1)
    harness.wait_idle(20.0)
    harness.wait_for_text('I see the image.')

    request = harness.requests[0]
    assert 'what is this?' in request.all_text
    if wire == 'openai chat completions':
        # That wire has no file id, so the image is inlined.
        assert harness.gateway.uploads == []
        assert 'data:image/png;base64,' in request.all_text
    else:
        # The image went through the Files API and the request only names it.
        assert len(harness.gateway.uploads) == 1
        upload = harness.gateway.uploads[0]
        assert upload['mime_type'] == 'image/png'
        assert upload['id'] in request.all_text
        assert 'base64' not in request.all_text
        kind = 'input_image' if wire == 'openai responses' else 'image'
        assert f'"type": "{kind}"' in request.all_text

    # The stored history keeps the small marker, never the encoded bytes.
    harness.wait_until(lambda: bool(harness.workspace.session_payload()),
                       timeout=10.0, description='the saved session')
    session = harness.workspace.session_payload()
    assert session
    content = session['messages'][0]['content']
    assert isinstance(content, str) and '[image: sessions/images/paste-' in content
    assert 'base64' not in content
    user_items = [text for kind, text in harness.transcript_texts()
                  if kind == 'user']
    assert user_items and 'what is this?' in user_items[0]


@pytest.mark.parametrize('wire', WIRES)
def test_the_model_can_look_at_an_image_by_itself(harness_factory, wire):
    harness = harness_factory(wire=wire)
    (harness.workspace.project / 'shot.png').write_bytes(
        images.encode_png(3, 3, bytes([10, 20, 30, 255] * 9)))
    harness.gateway.responder = ScriptedResponder([
        Turn(tool_calls=[ToolCall(name='view_image',
                                  args={'path': 'shot.png'},
                                  id='call_view_1')]),
        Turn(text='The screenshot is a solid dark square.'),
    ])

    harness.submit('what is in shot.png?', expect_requests=2, timeout=30.0)
    harness.wait_idle(20.0)
    harness.wait_for_text('solid dark square')

    request = harness.requests[1]
    kind = {'openai chat completions': 'image_url',
            'openai responses': 'input_image'}.get(wire, 'image')
    assert f'"type": "{kind}"' in request.all_text
    assert 'file_id' in request.all_text or 'base64' in request.all_text

    session = harness.workspace.session_payload()
    assert session
    contents = [message.get('content') for message in session['messages']]
    assert any(isinstance(text, str) and 'Loaded shot.png (3x3' in text
               for text in contents)
    assert any(isinstance(text, str) and text.startswith('<tool_images>')
               for text in contents)
    assert not any(isinstance(text, str) and 'base64' in text for text in contents)


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


def test_a_finished_turn_collapses_its_steps(harness_factory):
    """The real app folds a completed turn into one row, once the answer is in."""
    def collapse(config):
        config.steps.collapse_after_completion = True

    harness = harness_factory(config_mutator=collapse)
    harness.gateway.responder = ScriptedResponder([
        Turn(thinking='COLLAPSED-REASONING: read the file first.',
             tool_calls=[ToolCall(
                 name='write',
                 args={'path': 'notes.txt', 'content': 'hello\n'},
                 id='call_write_1')]),
        Turn(text='COLLAPSED-ANSWER: wrote the file.'),
    ])
    harness.submit('create notes.txt', expect_requests=2, timeout=25.0)
    harness.wait_idle(20.0)
    harness.wait_for_text('COLLAPSED-ANSWER')

    frame_text = harness.screen_text()
    # One row stands for the whole turn: the thought as a count, the tool
    # with how often it ran.
    rows = [line for line in frame_text.splitlines()
            if line.strip().startswith(ROW_BAR)]
    assert len(rows) == 1, frame_text
    row = rows[0]
    assert 'Thought for' in row and 'tokens' in row, row
    assert 'write' in row, row
    # The work itself is still in the transcript, just not on screen.
    assert 'COLLAPSED-REASONING' not in frame_text
    assert 'notes.txt' not in row
    kinds = [item['kind'] for item in harness.transcript_items()]
    assert kinds.count('thinking') == 1 and kinds.count('tool') == 1, kinds


def test_a_message_that_starts_with_a_path_reaches_the_model(harness):
    """Only a command name makes the input a command."""
    harness.gateway.responder = ScriptedResponder([Turn(text='reading it.')])
    message = '/home/darkstar/temp_workspace/CODE_STYLE.md summarize this'
    harness.submit(message)
    assert harness.requests[0].last_user_text() == message
    assert harness.transcript_texts()[0] == ('user', message)
    # nothing was swallowed by the command dispatcher
    assert not [item for item in harness.transcript_items()
                if item['kind'] == 'notice' and 'Unknown command' in item['text']]

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
    # The interrupted turn keeps its work: the request and the reasoning the
    # model had already streamed.  Dropping it made the next turn start over.
    committed = harness.app.loop.context.committed
    assert [message['role'] for message in committed] == ['user', 'assistant']
    assert committed[0]['content'] == 'start a slow turn'
    assert 'HELD-REASONING' in str(committed[1].get('content') or '')
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
