"""A real local HTTP server that impersonates an LLM provider.

The gateway speaks all three wire protocols AdCode supports:

* ``openai chat completions`` -- ``POST /v1/chat/completions`` with SSE chunks
* ``openai responses``         -- ``POST /v1/responses`` with typed SSE events
* ``anthropic messages``       -- ``POST /v1/messages`` with Messages SSE events

It is deliberately a real socket server: the application's own ``LLMClient``
(and therefore the official ``openai`` / ``anthropic`` SDKs) performs the HTTP
request, parses the stream, accumulates tool-call deltas and normalizes usage.
Nothing in the model-access path is mocked.

The responder is a script: :class:`~harness.script.Turn` objects describe what
one model response should contain.  Every request is recorded so tests can
assert on the exact payload (system prompt, tool schemas, reasoning effort,
tool results) the application sent.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterable
from urllib.parse import urlsplit

from bench.script import (
    EchoResponder, ModelRequest, Responder, Turn,
)

_SSE_HEADERS = {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache',
    'Connection': 'close',
    'X-Accel-Buffering': 'no',
}


class _Disconnected(Exception):
    """The script asked to drop the connection mid-stream."""


def _multipart_file(raw: bytes, content_type: str) -> tuple[str, str, bytes]:
    """Pull the first file part out of a multipart/form-data body."""
    marker = 'boundary='
    index = content_type.find(marker)
    if index == -1 or not raw:
        return '', '', raw
    boundary = content_type[index + len(marker):].split(';')[0].strip().strip('"')
    if not boundary:
        return '', '', raw
    delimiter = b'--' + boundary.encode('ascii', 'replace')
    filename = ''
    mime = 'application/octet-stream'
    for part in raw.split(delimiter):
        head, separator, body = part.partition(b'\r\n\r\n')
        if not separator:
            continue
        if b'filename=' not in head:
            continue
        header_text = head.decode('utf-8', 'replace')
        for line in header_text.split('\r\n'):
            name, _, value = line.partition(':')
            name = name.strip().lower()
            value = value.strip()
            if name == 'content-type':
                mime = value or mime
            elif name == 'content-disposition':
                for token in value.split(';'):
                    token = token.strip()
                    if token.startswith('filename='):
                        filename = token[len('filename='):].strip().strip('"')
        if body.endswith(b'\r\n'):
            body = body[:-2]
        return filename, mime, body
    return filename, mime, b''


class _GatewayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], gateway: 'FakeLLMGateway') -> None:
        self.gateway = gateway
        super().__init__(address, _GatewayHandler)


class _GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'AdCodeFakeGateway/1.0'

    @property
    def gateway(self) -> 'FakeLLMGateway':
        return self.server.gateway  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Keep pytest output clean; the gateway records everything itself.
        pass

    # ── GET ────────────────────────────────────────────────
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path in ('/', '/health'):
            self._send_json(200, {'ok': True, 'gateway': 'adcode-harness'})
            return
        if path.rstrip('/').endswith('/models'):
            self._send_json(200, self.gateway.models_payload(path))
            return
        self._send_json(404, {'error': {'message': f'unknown path {path}'}})

    # ── POST ───────────────────────────────────────────────
    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path.rstrip('/').endswith('/files'):
            self.gateway.handle_upload(self)
            return
        wire = self.gateway.wire_for_path(path)
        if wire is None:
            self._send_json(404, {'error': {'message': f'unknown path {path}'}})
            return
        body = self._read_json()
        if body is None:
            self._send_json(400, {'error': {'message': 'invalid JSON body'}})
            return
        request = ModelRequest(
            index=0, wire=wire, path=path, body=body,
            headers={str(key): str(value) for key, value in self.headers.items()},
        )
        request = self.gateway.record(request)
        if self.gateway.fail_next > 0:
            self.gateway.fail_next -= 1
            self._send_json(self.gateway.fail_status,
                            {'error': {'message': 'scripted gateway failure',
                                       'type': 'gateway_error'}})
            return
        try:
            turn = self.gateway.responder.respond(request)
        except Exception as error:  # a bad script must fail visibly, not hang
            self._send_json(500, {'error': {'message': str(error),
                                            'type': 'responder_error'}})
            return
        self.send_response(200)
        for name, value in _SSE_HEADERS.items():
            self.send_header(name, value)
        self.end_headers()
        self.close_connection = True
        try:
            self.gateway.stream(self, wire, turn, model=request.model)
        except (_Disconnected, BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            # A malformed script should not take the server thread down.
            pass

    # ── helpers ────────────────────────────────────────────
    def _read_json(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get('Content-Length') or 0)
        except ValueError:
            length = 0
        try:
            raw = self.rfile.read(length) if length else b''
        except OSError:
            return None
        try:
            value = json.loads(raw or b'{}')
        except (ValueError, UnicodeDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(data)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class FakeLLMGateway:
    """A scripted, in-process model provider reachable over HTTP."""

    def __init__(self, responder: Responder | None = None,
                 host: str = '127.0.0.1', port: int = 0,
                 models: Iterable[str] | None = None) -> None:
        self.responder: Responder = responder or EchoResponder()
        self.host = host
        self.port = port
        self.model_ids = list(models or ['harness-model', 'harness-model-mini'])
        self.requests: list[ModelRequest] = []
        self.uploads: list[dict[str, Any]] = []
        self.upload_attempts = 0
        self.files_supported = True
        self.fail_next = 0
        self.fail_status = 500
        self._lock = threading.RLock()
        self._holds: list[threading.Event] = []
        self._release_pending = False
        self._server: _GatewayServer | None = None
        self._thread: threading.Thread | None = None
        self._next_index = 0
        self._upload_index = 0

    # ── lifecycle ──────────────────────────────────────────
    def start(self) -> 'FakeLLMGateway':
        if self._server is not None:
            return self
        self._server = _GatewayServer((self.host, self.port), self)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name='adcode-fake-gateway', daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.release_all()
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=3)

    close = stop

    def __enter__(self) -> 'FakeLLMGateway':
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    @property
    def base_url(self) -> str:
        return f'http://{self.host}:{self.port}/v1'

    # ── recording / control ────────────────────────────────
    def record(self, request: ModelRequest) -> ModelRequest:
        with self._lock:
            self._next_index += 1
            request.index = self._next_index
            self.requests.append(request)
        return request

    def reset(self) -> None:
        with self._lock:
            self.requests.clear()
            self.uploads.clear()
            self.upload_attempts = 0
            self._next_index = 0
            self._upload_index = 0
            self._release_pending = False
        if hasattr(self.responder, 'served'):
            self.responder.served = 0  # type: ignore[attr-defined]

    def register_hold(self, event: threading.Event | None) -> None:
        if event is None:
            return
        with self._lock:
            if event not in self._holds:
                self._holds.append(event)
            if self._release_pending:
                event.set()

    def release_all(self) -> None:
        with self._lock:
            holds = list(self._holds)
            self._holds.clear()
            self._release_pending = True
        for event in holds:
            event.set()

    def wait_for_requests(self, count: int, timeout: float = 15.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(self.requests) >= count:
                return True
            time.sleep(0.01)
        return False

    # ── URLs / protocol mapping ─────────────────────────────
    @staticmethod
    def wire_for_path(path: str) -> str | None:
        if path.endswith('/chat/completions'):
            return 'openai chat completions'
        if path.endswith('/responses'):
            return 'openai responses'
        if path.endswith('/messages'):
            return 'anthropic messages'
        return None

    # ── Files API ───────────────────────────────────────────
    def handle_upload(self, handler: '_GatewayHandler') -> None:
        """Accept a multipart upload and answer in the caller's file shape."""
        try:
            length = int(handler.headers.get('Content-Length') or 0)
        except ValueError:
            length = 0
        with self._lock:
            self.upload_attempts += 1
        try:
            # Read the body even when refusing, or the client sees a broken
            # connection instead of the status code.
            raw = handler.rfile.read(length) if length else b''
        except OSError:
            return
        if not self.files_supported:
            handler._send_json(404, {'error': {'message': 'files api disabled',
                                               'type': 'not_found'}})
            return
        filename, mime, body = _multipart_file(raw,
                                               handler.headers.get('Content-Type') or '')
        with self._lock:
            self._upload_index += 1
            file_id = f'file-harness-{self._upload_index}'
            self.uploads.append({
                'id': file_id, 'filename': filename, 'mime_type': mime,
                'data': body, 'bytes': len(body),
            })
        anthropic_shape = bool(handler.headers.get('anthropic-version'))
        if anthropic_shape:
            handler._send_json(200, {
                'id': file_id, 'type': 'file', 'filename': filename,
                'mime_type': mime, 'size_bytes': len(body),
                'created_at': '2026-01-01T00:00:00Z', 'downloadable': False,
            })
        else:
            handler._send_json(200, {
                'id': file_id, 'object': 'file', 'bytes': len(body),
                'created_at': 1767225600, 'filename': filename,
                'purpose': 'vision',
            })

    def models_payload(self, path: str = '') -> dict[str, Any]:
        # A superset of the OpenAI and Anthropic list shapes: both SDKs read
        # ``data[].id`` and ignore the other protocol's extra fields.
        data = [{
            'id': model,
            'object': 'model',
            'created': 0,
            'owned_by': 'harness',
            'type': 'model',
            'display_name': model,
            'created_at': '2026-01-01T00:00:00Z',
        } for model in self.model_ids]
        return {
            'object': 'list',
            'data': data,
            'has_more': False,
            'first_id': self.model_ids[0] if self.model_ids else None,
            'last_id': self.model_ids[-1] if self.model_ids else None,
        }

    # ── streaming ──────────────────────────────────────────
    def stream(self, handler: _GatewayHandler, wire: str, turn: Turn,
               model: str = 'harness-model') -> None:
        if wire == 'openai chat completions':
            self._stream_chat(handler, turn, model)
        elif wire == 'openai responses':
            self._stream_responses(handler, turn, model)
        elif wire == 'anthropic messages':
            self._stream_anthropic(handler, turn, model)
        else:
            raise ValueError(f'unsupported wire: {wire}')

    def _write_event(self, handler: _GatewayHandler, event: str | None,
                     payload: dict[str, Any], state: dict[str, int]) -> None:
        prefix = f'event: {event}\n' if event else ''
        data = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        chunk = (prefix + f'data: {data}\n\n').encode('utf-8')
        handler.wfile.write(chunk)
        handler.wfile.flush()
        state['events'] += 1

    def _check_disconnect(self, handler: _GatewayHandler, turn: Turn,
                          state: dict[str, int]) -> None:
        if turn.disconnect_after is not None \
                and state['events'] >= turn.disconnect_after:
            try:
                handler.wfile.flush()
            except OSError:
                pass
            handler.close_connection = True
            try:
                handler.connection.close()
            except OSError:
                pass
            raise _Disconnected()

    @staticmethod
    def _chunks(text: str, size: int) -> list[str]:
        if not text:
            return []
        size = max(1, size)
        return [text[index:index + size] for index in range(0, len(text), size)]

    @staticmethod
    def _paced(delay: float) -> None:
        if delay > 0:
            time.sleep(delay)

    def _apply_gates(self, turn: Turn) -> None:
        """Apply the scripted pause between reasoning and answer text."""
        self.register_hold(turn.hold)
        if turn.hold is not None:
            turn.hold.wait(turn.hold_timeout)
        self._paced(turn.chunk_delay)

    def _call_ids(self, turn: Turn) -> list[str]:
        return [call.id or f'call_{index + 1}'
                for index, call in enumerate(turn.tool_calls)]

    # ── OpenAI chat completions ─────────────────────────────
    def _stream_chat(self, handler: _GatewayHandler, turn: Turn,
                     model: str = 'harness-model') -> None:
        state = {'events': 0}
        base = {
            'id': 'chatcmpl-harness', 'object': 'chat.completion.chunk',
            'created': int(time.time()), 'model': model,
        }

        def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
            return {**base, 'choices': [{
                'index': 0, 'delta': delta, 'finish_reason': finish,
            }]}

        self._write_event(handler, None,
                          chunk({'role': 'assistant'}), state)
        self._check_disconnect(handler, turn, state)
        for piece in self._chunks(turn.thinking, turn.chunk_size):
            self._write_event(handler, None,
                              chunk({'reasoning_content': piece}), state)
            self._check_disconnect(handler, turn, state)
            self._paced(turn.chunk_delay)
        self._apply_gates(turn)
        for piece in self._chunks(turn.text, turn.chunk_size):
            self._write_event(handler, None, chunk({'content': piece}), state)
            self._check_disconnect(handler, turn, state)
            self._paced(turn.chunk_delay)

        call_ids = self._call_ids(turn)
        for index, (call, call_id) in enumerate(zip(turn.tool_calls, call_ids)):
            self._write_event(handler, None, chunk({'tool_calls': [{
                'index': index, 'id': call_id, 'type': 'function',
                'function': {'name': call.name, 'arguments': ''},
            }]}), state)
            self._check_disconnect(handler, turn, state)
            arguments = call.arguments_json()
            pieces = self._chunks(arguments, max(8, len(arguments) // 2 + 1))
            for piece in pieces:
                self._write_event(handler, None, chunk({'tool_calls': [{
                    'index': index,
                    'function': {'arguments': piece},
                }]}), state)
                self._check_disconnect(handler, turn, state)
            self._paced(turn.chunk_delay)

        finish = 'tool_calls' if turn.tool_calls else turn.finish
        self._write_event(handler, None, chunk({}, finish), state)
        self._check_disconnect(handler, turn, state)
        usage = {
            'prompt_tokens': turn.usage.input_tokens,
            'completion_tokens': turn.usage.output_tokens,
            'total_tokens': turn.usage.total_tokens,
            'prompt_tokens_details': {
                'cached_tokens': turn.usage.cached_tokens,
            },
        }
        self._write_event(handler, None, {**base, 'choices': [], 'usage': usage},
                          state)
        # The SDK expects the literal [DONE] sentinel, not JSON.
        try:
            handler.wfile.write(b'data: [DONE]\n\n')
            handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            raise _Disconnected() from None

    # ── OpenAI responses ────────────────────────────────────
    def _stream_responses(self, handler: _GatewayHandler, turn: Turn,
                          model: str = 'harness-model') -> None:
        state = {'events': 0, 'sequence': 0}

        def emit(type_: str, **fields: Any) -> None:
            state['sequence'] += 1
            self._write_event(
                handler, type_,
                {'type': type_, 'sequence_number': state['sequence'], **fields},
                state)
            self._check_disconnect(handler, turn, state)

        created = self._response_object(model, [], 'in_progress')
        emit('response.created', response=created)
        # The openai sdk's streaming helper validates the event sequence: a
        # content delta must follow its output_item.added, mirroring the real
        # api (reasoning item at index 0, message item at index 1, then the
        # function calls).
        emit('response.output_item.added',
             item={'type': 'reasoning', 'id': 'rs_harness'}, output_index=0)
        for piece in self._chunks(turn.thinking, turn.chunk_size):
            emit('response.reasoning_text.delta', delta=piece,
                 item_id='rs_harness', output_index=0, content_index=0)
            self._paced(turn.chunk_delay)
        self._apply_gates(turn)
        emit('response.output_item.added',
             item={'type': 'message', 'id': 'msg_harness', 'content': []},
             output_index=1)
        emit('response.content_part.added',
             item_id='msg_harness', output_index=1, content_index=0,
             part={'type': 'output_text', 'text': ''})
        for piece in self._chunks(turn.text, turn.chunk_size):
            emit('response.output_text.delta', delta=piece,
                 item_id='msg_harness', output_index=1, content_index=0,
                 logprobs=[])
            self._paced(turn.chunk_delay)
        emit('response.content_part.done',
             item_id='msg_harness', output_index=1, content_index=0,
             part={'type': 'output_text', 'text': turn.text})

        output: list[dict[str, Any]] = [
            {'type': 'reasoning', 'id': 'rs_harness'},
            {'type': 'message', 'id': 'msg_harness',
             'content': [{'type': 'output_text', 'text': turn.text}]},
        ]
        for offset, (call, call_id) in enumerate(
                zip(turn.tool_calls, self._call_ids(turn))):
            index = offset + 2
            item_id = f'fc_harness_{offset + 1}'
            item = {
                'type': 'function_call', 'id': item_id, 'call_id': call_id,
                'name': call.name, 'arguments': '', 'status': 'in_progress',
            }
            output.append({**item, 'status': 'completed',
                           'arguments': call.arguments_json()})
            emit('response.output_item.added', item=item, output_index=index)
            arguments = call.arguments_json()
            pieces = self._chunks(arguments, max(8, len(arguments) // 2 + 1))
            for piece in pieces:
                emit('response.function_call_arguments.delta',
                     item_id=item_id, output_index=index, delta=piece)
            emit('response.function_call_arguments.done',
                 item_id=item_id, output_index=index, arguments=arguments)

        usage = {
            'input_tokens': turn.usage.input_tokens,
            'input_tokens_details': {'cached_tokens': turn.usage.cached_tokens},
            'output_tokens': turn.usage.output_tokens,
            'output_tokens_details': {'reasoning_tokens': 0},
            'total_tokens': turn.usage.total_tokens,
        }
        completed = self._response_object(model, output, 'completed', usage)
        emit('response.completed', response=completed)

    @staticmethod
    def _response_object(model: str, output: list[dict[str, Any]],
                         status: str,
                         usage: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'id': 'resp_harness',
            'object': 'response',
            'created_at': int(time.time()),
            'model': model,
            'status': status,
            'output': output,
            'parallel_tool_calls': True,
            'tool_choice': 'auto',
            'tools': [],
        }
        if usage is not None:
            payload['usage'] = usage
        return payload

    # ── Anthropic messages ──────────────────────────────────
    def _stream_anthropic(self, handler: _GatewayHandler, turn: Turn,
                          model: str = 'harness-model') -> None:
        state = {'events': 0}

        def emit(event: str, payload: dict[str, Any]) -> None:
            self._write_event(handler, event, payload, state)
            self._check_disconnect(handler, turn, state)

        message = {
            'id': 'msg_harness', 'type': 'message', 'role': 'assistant',
            'model': model, 'content': [], 'stop_reason': None,
            'stop_sequence': None,
            'usage': {
                'input_tokens': turn.usage.input_tokens,
                'output_tokens': 1,
                'cache_read_input_tokens': turn.usage.cached_tokens,
            },
        }
        emit('message_start', {'type': 'message_start', 'message': message})

        index = 0
        if turn.thinking:
            emit('content_block_start', {
                'type': 'content_block_start', 'index': index,
                'content_block': {
                    'type': 'thinking', 'thinking': '',
                    'signature': 'harness-signature',
                },
            })
            for piece in self._chunks(turn.thinking, turn.chunk_size):
                emit('content_block_delta', {
                    'type': 'content_block_delta', 'index': index,
                    'delta': {'type': 'thinking_delta', 'thinking': piece},
                })
                self._paced(turn.chunk_delay)
            emit('content_block_stop',
                 {'type': 'content_block_stop', 'index': index})
            index += 1

        self._apply_gates(turn)

        if turn.text:
            emit('content_block_start', {
                'type': 'content_block_start', 'index': index,
                'content_block': {'type': 'text', 'text': ''},
            })
            for piece in self._chunks(turn.text, turn.chunk_size):
                emit('content_block_delta', {
                    'type': 'content_block_delta', 'index': index,
                    'delta': {'type': 'text_delta', 'text': piece},
                })
                self._paced(turn.chunk_delay)
            emit('content_block_stop',
                 {'type': 'content_block_stop', 'index': index})
            index += 1

        for call, call_id in zip(turn.tool_calls, self._call_ids(turn)):
            emit('content_block_start', {
                'type': 'content_block_start', 'index': index,
                'content_block': {
                    'type': 'tool_use', 'id': call_id, 'name': call.name,
                    'input': {},
                },
            })
            arguments = call.arguments_json()
            for piece in self._chunks(arguments, max(8, len(arguments) // 2 + 1)):
                emit('content_block_delta', {
                    'type': 'content_block_delta', 'index': index,
                    'delta': {'type': 'input_json_delta', 'partial_json': piece},
                })
            emit('content_block_stop',
                 {'type': 'content_block_stop', 'index': index})
            index += 1

        stop_reason = 'tool_use' if turn.tool_calls else 'end_turn'
        emit('message_delta', {
            'type': 'message_delta',
            'delta': {'stop_reason': stop_reason, 'stop_sequence': None},
            'usage': {'output_tokens': turn.usage.output_tokens},
        })
        emit('message_stop', {'type': 'message_stop'})


# Convenience aliases used in tests and the manual harness.
FakeGateway = FakeLLMGateway
