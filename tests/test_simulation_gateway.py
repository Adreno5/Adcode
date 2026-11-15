"""Protocol-level tests for the fake local LLM gateway.

These tests do not use the TUI.  They prove that the gateway is a real HTTP
service speaking all three wires, and that AdCode's production ``LLMClient``
(and the official SDKs beneath it) can stream text, reasoning, tool calls and
usage from it.
"""

from __future__ import annotations

import base64
import json
import urllib.request

import pytest

from adcode import images
from adcode.config import ModelConfig, ProviderConfig
from adcode.llm.client import LLMClient
from harness.gateway import FakeLLMGateway
from harness.script import ScriptedResponder, ToolCall, Turn

WIRES = [
    'openai chat completions',
    'openai responses',
    'anthropic messages',
]


def provider(base_url: str, wire: str) -> ProviderConfig:
    return ProviderConfig(
        name='Harness', base_url=base_url, api_key='harness-key', wire=wire,
        models=[ModelConfig(id='harness-model', context_window=100_000,
                            max_output_tokens=8_192, reasoning=True,
                            reasoning_style='effort',
                            reasoning_efforts=['low', 'medium', 'high'])],
    )


def tools(wire: str) -> list[dict]:
    if wire == 'anthropic messages':
        return [{'name': 'write', 'description': 'Write a file',
                 'input_schema': {'type': 'object',
                                  'properties': {'path': {'type': 'string'}}}}]
    return [{'type': 'function', 'function': {
        'name': 'write', 'description': 'Write a file',
        'parameters': {'type': 'object',
                       'properties': {'path': {'type': 'string'}}}}}]


def collect(client: LLMClient, wire: str, messages=None):
    configured = provider(client.provider.base_url, wire)
    return list(client.chat(
        messages or [{'role': 'user', 'content': 'hello from the test'}],
        configured.models[0], system='SYSTEM MARKER', max_tokens=2_048,
        tools=tools(wire), effort='high'))


@pytest.fixture()
def gateway():
    instance = FakeLLMGateway(ScriptedResponder([])).start()
    yield instance
    instance.stop()


@pytest.mark.parametrize('wire', WIRES)
def test_gateway_streams_thinking_text_and_usage(gateway, wire):
    gateway.responder = ScriptedResponder([
        Turn(thinking='Plan the answer. ', text='Hello from the gateway.'),
    ])
    client = LLMClient(provider(gateway.base_url, wire))
    events = collect(client, wire)

    thinking = ''.join(event.get('content', '') for event in events
                       if event.get('kind') == 'thinking')
    text = ''.join(event.get('content', '') for event in events
                   if event.get('kind') == 'text')
    usage = [event for event in events if event.get('kind') == 'usage']
    finish = [event.get('reason') for event in events
              if event.get('kind') == 'finish']

    assert thinking == 'Plan the answer. '
    assert text == 'Hello from the gateway.'
    assert usage, 'the gateway must send a usage event'
    assert finish and finish[-1] in ('stop', 'end_turn')
    assert len(gateway.requests) == 1
    request = gateway.requests[0]
    assert request.wire == wire
    assert request.model == 'harness-model'
    assert 'SYSTEM MARKER' in request.system_text
    assert 'write' in request.tool_names


@pytest.mark.parametrize('wire', WIRES)
def test_gateway_accumulates_tool_call_arguments(gateway, wire):
    gateway.responder = ScriptedResponder([
        Turn(thinking='Use the write tool.',
             tool_calls=[ToolCall(name='write',
                                  args={'path': 'notes.txt',
                                        'content': 'from the gateway\n'},
                                  id='call_write_1')]),
    ])
    client = LLMClient(provider(gateway.base_url, wire))
    events = collect(client, wire)
    calls = [event for event in events if event.get('kind') == 'tool_call']

    assert len(calls) == 1
    assert calls[0]['name'] == 'write'
    assert calls[0]['id'] == 'call_write_1'
    assert calls[0]['args'] == {'path': 'notes.txt',
                                'content': 'from the gateway\n'}


def test_gateway_sse_wire_format_is_literal_http(gateway):
    body = json.dumps({
        'model': 'harness-model',
        'messages': [{'role': 'user', 'content': 'raw http'}],
        'stream': True,
    }).encode()
    request = urllib.request.Request(
        gateway.base_url + '/chat/completions', data=body,
        headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=15) as response:
        assert response.headers['Content-Type'].startswith('text/event-stream')
        payload = response.read().decode('utf-8')
    assert 'data: [DONE]' in payload
    assert '"object":"chat.completion.chunk"' in payload
    assert len(gateway.requests) == 1
    assert gateway.requests[0].wire == 'openai chat completions'
    assert gateway.requests[0].body['messages'][0]['content'] == 'raw http'


@pytest.mark.parametrize('wire', WIRES)
def test_gateway_receives_a_native_image_part(gateway, wire, tmp_path):
    gateway.responder = ScriptedResponder([Turn(text='seen')])
    pixels = bytes([255, 0, 0, 255])
    (tmp_path / 'shot.png').write_bytes(images.encode_png(1, 1, pixels))
    client = LLMClient(provider(gateway.base_url, wire))
    client.base_dir = str(tmp_path)
    collect(client, wire, [{'role': 'user',
                            'content': 'look [image: shot.png] here'}])

    body = gateway.requests[0].body
    if wire == 'openai chat completions':
        parts = body['messages'][-1]['content']
        assert parts[0] == {'type': 'text', 'text': 'look '}
        assert parts[1]['type'] == 'image_url'
        url = parts[1]['image_url']['url']
        assert url.startswith('data:image/png;base64,')
        payload = base64.b64decode(url.split(',', 1)[1])
        assert payload == images.encode_png(1, 1, pixels)
    elif wire == 'openai responses':
        parts = body['input'][-1]['content']
        assert parts[0] == {'type': 'input_text', 'text': 'look '}
        assert parts[1]['type'] == 'input_image'
        assert parts[1]['image_url'].startswith('data:image/png;base64,')
    else:
        blocks = body['messages'][0]['content']
        assert blocks[0] == {'type': 'text', 'text': 'look '}
        assert blocks[1]['type'] == 'image'
        source = blocks[1]['source']
        assert source['type'] == 'base64'
        assert source['media_type'] == 'image/png'
        # The SDK read the file itself: the body carries base64, not a path.
        assert isinstance(source['data'], str)
        assert base64.b64decode(source['data']) == images.encode_png(1, 1, pixels)


@pytest.mark.parametrize('wire', WIRES)
def test_gateway_lists_models(gateway, wire):
    client = LLMClient(provider(gateway.base_url, wire))
    assert client.list_models() == ['harness-model', 'harness-model-mini']


def test_gateway_can_fail_a_request(gateway):
    # The OpenAI SDK retries once on a 400 before surfacing the error.
    gateway.fail_next = 5
    gateway.fail_status = 400
    client = LLMClient(provider(gateway.base_url, WIRES[0]))
    with pytest.raises(Exception, match='scripted gateway failure'):
        collect(client, WIRES[0])
    assert len(gateway.requests) >= 2


def test_gateway_can_drop_a_stream(gateway):
    gateway.responder = ScriptedResponder([
        Turn(text='x' * 400, chunk_size=20, disconnect_after=3),
    ])
    client = LLMClient(provider(gateway.base_url, WIRES[0]))
    collected = []
    try:
        collected = collect(client, WIRES[0])
    except Exception:
        pass
    text = ''.join(event.get('content', '') for event in collected
                   if event.get('kind') == 'text')
    reasons = [event.get('reason') for event in collected
               if event.get('kind') == 'finish']
    # The connection was closed mid-stream, so the response is incomplete.
    assert text != 'x' * 400
    assert not reasons or reasons[-1] == 'incomplete'
