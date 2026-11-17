"""The Responses wire must survive gateways that skip protocol events.

Compatible gateways frequently translate a chat completion into Responses
events and forget response.content_part.added.  The openai SDK's typed stream
used to index the accumulated content on the first text delta, so the missing
part surfaced as an IndexError from inside the SDK; AdCode answered that by
learning a chat-completions override for the endpoint.

aicf-core 0.6.1 reads the raw Responses stream instead, so that shape no longer
raises and the override is not needed for it.  These tests drive the real
client against such a gateway and pin the new behaviour: the stream simply
works, and the learned-wire cache still sends a later client straight to the
wire that was chosen.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import adcode.llm.client as client_module
from adcode import paths
from adcode.config import ModelConfig, ProviderConfig
from adcode.llm.client import LLMClient, is_responses_sequence_error

MODEL_ID = 'deepseek-v4.1-flash'


@pytest.fixture(autouse=True)
def wire_home(tmp_path, monkeypatch):
    """Keep the learned wire overrides out of the real ~/.adcode."""
    monkeypatch.setattr(client_module, '_wire_cache', None)
    paths.set_app_dir(tmp_path / 'adcode-home')
    try:
        yield
    finally:
        paths.set_app_dir(None)
        monkeypatch.setattr(client_module, '_wire_cache', None)


def _response_object(output, status='in_progress', usage=None):
    payload = {
        'id': 'resp_1', 'object': 'response', 'created_at': int(time.time()),
        'model': MODEL_ID, 'status': status, 'output': output,
        'parallel_tool_calls': True, 'tool_choice': 'auto', 'tools': [],
    }
    if usage is not None:
        payload['usage'] = usage
    return payload


def _responses_events(*, content_part: bool,
                      reasoning: bool) -> list[tuple[str, dict]]:
    """The event stream of a gateway that implements Responses partially."""
    events: list[tuple[str, dict]] = []
    sequence = [0]

    def emit(type_, **fields):
        sequence[0] += 1
        events.append((type_, {'type': type_, 'sequence_number': sequence[0],
                               **fields}))

    emit('response.created', response=_response_object([]))
    index = 0
    if reasoning:
        # DeepSeek-style reasoning arrives before the answer text, which is why
        # the failure lands mid-stream instead of before the first token.
        emit('response.output_item.added',
             item={'type': 'reasoning', 'id': 'rs_1'}, output_index=0)
        emit('response.reasoning_text.delta', delta='thinking',
             item_id='rs_1', output_index=0, content_index=0)
        index = 1
    emit('response.output_item.added',
         item={'type': 'message', 'id': 'msg_1', 'content': []},
         output_index=index)
    if content_part:
        emit('response.content_part.added', item_id='msg_1',
             output_index=index, content_index=0,
             part={'type': 'output_text', 'text': ''})
    emit('response.output_text.delta', delta='hello', item_id='msg_1',
         output_index=index, content_index=0, logprobs=[])
    output = [{'type': 'message', 'id': 'msg_1',
               'content': [{'type': 'output_text', 'text': 'hello'}]}]
    if reasoning:
        output.insert(0, {'type': 'reasoning', 'id': 'rs_1'})
    emit('response.completed',
         response=_response_object(output, 'completed',
                                   {'input_tokens': 1, 'output_tokens': 1,
                                    'total_tokens': 2}))
    return events


class _Gateway:
    """A real HTTP server answering both wires with scripted SSE."""

    def __init__(self, responses_events, chat_text='chat answer'):
        self.responses_events = responses_events
        self.chat_text = chat_text
        self.paths: list[str] = []
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args):   # noqa: A002 - base signature
                pass

            def do_POST(self):              # noqa: N802 - base signature
                length = int(self.headers.get('Content-Length') or 0)
                self.rfile.read(length)
                gateway.paths.append(self.path)
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.close_connection = True

                def write(name, payload):
                    self.wfile.write(
                        f'event: {name}\ndata: {json.dumps(payload)}\n\n'.encode())

                try:
                    if self.path.endswith('/responses'):
                        for name, payload in gateway.responses_events:
                            write(name, payload)
                    else:
                        for chunk in _chat_chunks(gateway.chat_text):
                            write('', chunk)
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)
        self._thread.start()
        self.base_url = (f'http://127.0.0.1:'
                         f'{self._server.server_address[1]}/v1')

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _chat_chunks(text: str):
    base = {'id': 'c1', 'object': 'chat.completion.chunk', 'created': 0,
            'model': MODEL_ID}
    yield {**base, 'choices': [{'index': 0, 'finish_reason': None,
                                'delta': {'role': 'assistant',
                                          'content': text}}]}
    yield {**base, 'choices': [{'index': 0, 'finish_reason': 'stop',
                                'delta': {}}]}


@pytest.fixture()
def gateway_factory():
    created: list[_Gateway] = []

    def make(responses_events, **kwargs) -> _Gateway:
        gateway = _Gateway(responses_events, **kwargs)
        created.append(gateway)
        return gateway

    yield make
    for gateway in created:
        gateway.close()


def _client(gateway: _Gateway, wire: str, warnings: list[str]
            ) -> tuple[LLMClient, ProviderConfig]:
    provider = ProviderConfig(
        name='LittleAPI', base_url=gateway.base_url, api_key='test', wire=wire,
        models=[ModelConfig(id=MODEL_ID)],
    )
    client = LLMClient(provider, provider.models[0], warn=warnings.append)
    return client, provider


def _collect(client: LLMClient) -> list:
    async def run():
        return [event async for event in
                client.stream([{'role': 'user', 'content': 'hi'}])]

    return asyncio.run(run())


def _text_of(events) -> str:
    return ''.join(event for event in events if isinstance(event, str))


def test_a_missing_content_part_no_longer_breaks_the_wire(gateway_factory):
    gateway = gateway_factory(_responses_events(content_part=False,
                                                reasoning=False))
    warnings: list[str] = []
    client, provider = _client(gateway, 'openai responses', warnings)
    try:
        events = _collect(client)
    finally:
        client.close()
    assert _text_of(events) == 'hello'
    assert [path for path in gateway.paths] == ['/v1/responses'], gateway.paths
    assert client.wire == 'openai responses'
    assert provider.wire == 'openai responses'
    assert warnings == []
    assert client_module.learned_wire(client._file_scope(),
                                     'openai responses') == ''


def test_reasoning_before_the_answer_survives_a_skipped_part(gateway_factory):
    """The hardest case used to fail mid-stream, after the reasoning shipped."""
    gateway = gateway_factory(_responses_events(content_part=False,
                                                reasoning=True))
    warnings: list[str] = []
    client, _provider = _client(gateway, 'openai responses', warnings)
    try:
        events = _collect(client)
    finally:
        client.close()
    assert {'reasoning': 'thinking'} in events, events
    assert _text_of(events) == 'hello'
    assert client.wire == 'openai responses'
    assert warnings == []
    assert all(path.endswith('/responses') for path in gateway.paths)


def test_the_learned_wire_is_used_by_the_next_client(gateway_factory):
    """The override an endpoint earned is remembered across clients."""
    gateway = gateway_factory(_responses_events(content_part=True,
                                                reasoning=False))
    client, _provider = _client(gateway, 'openai responses', [])
    try:
        client_module.remember_wire(client._file_scope(), 'openai responses',
                                    'openai chat completions')
    finally:
        client.close()
    gateway.paths.clear()
    again, _ = _client(gateway, 'openai responses', [])
    try:
        events = _collect(again)
    finally:
        again.close()
    assert _text_of(events) == 'chat answer'
    assert gateway.paths == ['/v1/chat/completions']
    assert again.wire == 'openai chat completions'


def test_a_complete_responses_stream_keeps_its_wire(gateway_factory):
    gateway = gateway_factory(_responses_events(content_part=True,
                                                reasoning=False))
    warnings: list[str] = []
    client, provider = _client(gateway, 'openai responses', warnings)
    try:
        events = _collect(client)
    finally:
        client.close()
    assert 'hello' in _text_of(events)
    assert client.wire == 'openai responses'
    assert provider.wire == 'openai responses'
    assert warnings == []
    assert all(path.endswith('/responses') for path in gateway.paths)
    assert client_module.learned_wire(client._file_scope(),
                                     'openai responses') == ''


def test_only_sequence_errors_from_the_sdk_are_recognised():
    try:
        raise IndexError('a plain bug')
    except IndexError as error:
        assert is_responses_sequence_error(error) is False
    try:
        raise ValueError('not a shape error')
    except ValueError as error:
        assert is_responses_sequence_error(error) is False


# -- the same gateway shape through the real application ------------------
def _skip_content_part(monkeypatch):
    from bench.gateway import FakeLLMGateway

    original = FakeLLMGateway._write_event

    def write_event(self, handler, event, payload, state):
        # The gateway LittleAPI/deepseek behaves like this: the message item is
        # announced and then text deltas arrive with no content part.
        if event == 'response.content_part.added':
            return
        return original(self, handler, event, payload, state)

    monkeypatch.setattr(FakeLLMGateway, '_write_event', write_event)


def test_an_app_turn_on_a_partial_gateway_streams_the_answer(
        harness_factory, monkeypatch):
    """The whole app: no failed turn, no wire switch, the answer arrives."""
    _skip_content_part(monkeypatch)

    harness = harness_factory(wire='openai responses')
    harness.type_text('hi there')
    harness.wait_prompt('hi there')
    harness.press('enter')
    harness.wait_until(lambda: len(harness.requests) >= 1, timeout=20.0,
                       description='the first model request')
    harness.wait_idle(30.0)

    items = harness.transcript_texts()
    assert any(kind == 'assistant' for kind, _text in items), items
    assert not any(kind == 'notice' and 'chat completions' in text
                   for kind, text in items), items
    assert not any(kind == 'notice' and 'Request failed' in text
                   for kind, text in items), items
    assert all(request.path.endswith('/responses')
               for request in harness.requests), \
        [request.path for request in harness.requests]
