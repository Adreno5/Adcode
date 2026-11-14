"""MCP clients and transports: stdio, Streamable HTTP, and legacy HTTP+SSE.

Each client performs the MCP handshake, scans the complete tool list
(including pagination cursors) and executes tool calls.  Servers are kept
alive for the lifetime of the session and torn down on exit.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

PROTOCOL_VERSION = '2025-06-18'
CLIENT_INFO = {'name': 'AdCode', 'version': '2.0.0'}


class MCPError(RuntimeError):
    pass


def _decode_sse_payload(text: str) -> dict[str, Any]:
    """Extract the last JSON data payload from an SSE response body."""
    payloads: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith('data:'):
            current.append(line[5:].strip())
        elif not line.strip() and current:
            payloads.append('\n'.join(current))
            current = []
    if current:
        payloads.append('\n'.join(current))
    for payload in reversed(payloads):
        if payload and payload != '[DONE]':
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                continue
    return {}


def _decode_http_body(body: bytes, content_type: str) -> dict[str, Any]:
    if not body:
        return {}
    text = body.decode('utf-8', errors='replace')
    if 'text/event-stream' in content_type:
        return _decode_sse_payload(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise MCPError(f'invalid JSON response: {error}; body={text[:200]!r}') from error


class MCPTransport:
    name = 'transport'

    def __init__(self) -> None:
        self.connected = False
        self.last_error = ''
        self.protocol_version = ''
        self.connected_at: float | None = None

    def request(self, method: str, params: dict[str, Any] | None = None,
                timeout: float = 60.0) -> dict[str, Any]:
        raise NotImplementedError

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def diagnostics(self) -> dict[str, Any]:
        return {
            'transport': self.name,
            'connected': self.connected,
            'protocol_version': self.protocol_version,
            'last_error': self.last_error,
        }


class StdioTransport(MCPTransport):
    name = 'stdio'

    def __init__(self, command: str, args: list[str] | None = None,
                 env: dict[str, str] | None = None) -> None:
        super().__init__()
        self.command = command
        self.args = args or []
        self.env = env or {}
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._pending: dict[int, threading.Event] = {}
        self._results: dict[int, dict[str, Any]] = {}
        self._next_id = 0
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self.stderr_tail: list[str] = []

    def start(self) -> None:
        merged_env = os.environ.copy()
        merged_env.setdefault('PYTHONUTF8', '1')
        merged_env.setdefault('PYTHONIOENCODING', 'utf-8')
        merged_env.update(self.env)
        creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        self._proc = subprocess.Popen(
            [self.command, *self.args],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=merged_env, creationflags=creationflags,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True,
                                        name='mcp-stdio-reader')
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._stderr_loop, daemon=True,
                                               name='mcp-stdio-stderr')
        self._stderr_reader.start()

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            for line in iter(self._proc.stdout.readline, b''):
                try:
                    message = json.loads(line.decode('utf-8', errors='replace'))
                except json.JSONDecodeError:
                    continue
                self._dispatch(message)
        except Exception as error:
            self.last_error = f'reader failed: {error}'
        finally:
            exit_code = self._proc.poll() if self._proc else None
            with self._lock:
                pending = list(self._pending.items())
                for request_id, _ in pending:
                    self._results[request_id] = {
                        'error': {'code': -1,
                                  'message': f'server closed stdout (exit={exit_code})'},
                    }
                self._pending.clear()
            for _, event in pending:
                event.set()

    def _dispatch(self, message: dict[str, Any]) -> None:
        request_id = message.get('id')
        if request_id is None:
            # Server notification (tools/list_changed etc.) - ignored here; the
            # manager rescans on demand.
            return
        if 'method' in message:
            # Server-initiated request: answer ping, reject the rest politely.
            method = message.get('method')
            if method == 'ping':
                self._write({'jsonrpc': '2.0', 'id': request_id, 'result': {}})
            else:
                self._write({
                    'jsonrpc': '2.0', 'id': request_id,
                    'error': {'code': -32601, 'message': f'unsupported: {method}'},
                })
            return
        with self._lock:
            self._results[request_id] = message
            event = self._pending.pop(request_id, None)
        if event is not None:
            event.set()

    def _stderr_loop(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        try:
            for line in iter(self._proc.stderr.readline, b''):
                text = line.decode('utf-8', errors='replace').strip()
                if text:
                    self.stderr_tail.append(text)
                    del self.stderr_tail[:-20]
        except Exception:
            pass

    def _write(self, message: dict[str, Any]) -> None:
        data = (json.dumps(message, ensure_ascii=False,
                           separators=(',', ':')) + '\n').encode('utf-8')
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise MCPError('stdio transport is not running')
        with self._lock:
            proc.stdin.write(data)
            proc.stdin.flush()

    def request(self, method: str, params: dict[str, Any] | None = None,
                timeout: float = 60.0) -> dict[str, Any]:
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            event = threading.Event()
            self._pending[request_id] = event
        self._write({'jsonrpc': '2.0', 'id': request_id, 'method': method,
                     'params': params or {}})
        if event.wait(timeout=timeout):
            with self._lock:
                return self._results.pop(request_id, {
                    'error': {'code': -1, 'message': 'no response'}})
        with self._lock:
            self._pending.pop(request_id, None)
        try:
            self.notify('notifications/cancelled', {
                'requestId': request_id, 'reason': f'{method} timed out',
            })
        except Exception:
            pass
        detail = self.stderr_tail[-1] if self.stderr_tail else ''
        message = f'timeout; stderr: {detail}' if detail else 'timeout'
        return {'error': {'code': -1, 'message': message}}

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._write({'jsonrpc': '2.0', 'method': method, 'params': params or {}})

    def close(self) -> None:
        proc = self._proc
        self.connected = False
        if proc is not None:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=3)
                except Exception:
                    pass
            for pipe in (proc.stdout, proc.stderr):
                try:
                    if pipe:
                        pipe.close()
                except Exception:
                    pass
            self._proc = None
            for thread in (self._reader, self._stderr_reader):
                if thread is not None and thread is not threading.current_thread():
                    thread.join(timeout=1)
        self.connected_at = None

    def diagnostics(self) -> dict[str, Any]:
        proc = self._proc
        details = super().diagnostics()
        details.update({
            'command': self.command,
            'args': self.args,
            'pid': proc.pid if proc and proc.poll() is None else None,
            'process_exit_code': proc.poll() if proc else None,
            'stderr_tail': self.stderr_tail[-5:],
        })
        return details


class HttpTransport(MCPTransport):
    """MCP Streamable HTTP: JSON-RPC over POST, optional SSE responses."""

    name = 'http'

    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url
        self.session_id = ''
        self._next_id = 0

    def _headers(self) -> dict[str, str]:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream',
        }
        if self.session_id:
            headers['Mcp-Session-Id'] = self.session_id
        return headers

    def _send(self, payload: dict[str, Any], timeout: float,
              expect_response: bool = True) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False,
                          separators=(',', ':')).encode('utf-8')
        request = urllib.request.Request(self.url, data=data,
                                         headers=self._headers(), method='POST')
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                session = response.headers.get('Mcp-Session-Id')
                if session:
                    self.session_id = session
                if not expect_response:
                    return {}
                body = response.read()
                return _decode_http_body(body, response.headers.get('Content-Type', ''))
        except urllib.error.HTTPError as error:
            try:
                return _decode_http_body(error.read(),
                                         error.headers.get('Content-Type', ''))
            except Exception:
                return {'error': {'code': error.code, 'message': str(error)}}
        except Exception as error:
            return {'error': {'code': -1, 'message': str(error)}}

    def request(self, method: str, params: dict[str, Any] | None = None,
                timeout: float = 60.0) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        return self._send({'jsonrpc': '2.0', 'id': request_id, 'method': method,
                           'params': params or {}}, timeout)

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._send({'jsonrpc': '2.0', 'method': method, 'params': params or {}},
                   10.0, expect_response=False)

    def close(self) -> None:
        if self.session_id:
            try:
                request = urllib.request.Request(
                    self.url, headers={'Mcp-Session-Id': self.session_id},
                    method='DELETE')
                urllib.request.urlopen(request, timeout=3).close()
            except Exception:
                pass
        self.connected = False
        self.connected_at = None
        self.session_id = ''

    def diagnostics(self) -> dict[str, Any]:
        details = super().diagnostics()
        details.update({'url': self.url, 'session': bool(self.session_id)})
        return details


class SseTransport(MCPTransport):
    """Legacy HTTP+SSE: a GET event stream plus a POST message endpoint."""

    name = 'sse'

    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url
        self.message_url = ''
        self._next_id = 0
        self._lock = threading.Lock()
        self._pending: dict[int, threading.Event] = {}
        self._results: dict[int, dict[str, Any]] = {}
        self._stream = None
        self._reader: threading.Thread | None = None
        self._endpoint_ready = threading.Event()
        self._sock = None

    def start(self) -> None:
        self._reader = threading.Thread(target=self._read_loop, daemon=True,
                                        name='mcp-sse-reader')
        self._reader.start()
        if not self._endpoint_ready.wait(timeout=15):
            raise MCPError('did not receive an SSE endpoint event')

    def _read_loop(self) -> None:
        import http.client
        from urllib.parse import urlsplit
        parts = urlsplit(self.url)
        connection_class = (http.client.HTTPSConnection
                            if parts.scheme == 'https' else http.client.HTTPConnection)
        port = parts.port or (443 if parts.scheme == 'https' else 80)
        path = parts.path or '/'
        if parts.query:
            path += '?' + parts.query
        connection = connection_class(parts.hostname or '', port, timeout=600)
        self._connection = connection
        try:
            connection.request('GET', path, headers={
                'Accept': 'text/event-stream', 'Cache-Control': 'no-cache',
            })
            self._stream = connection.getresponse()
            self._sock = connection.sock
            event = ''
            data_lines: list[str] = []
            while True:
                raw = self._stream.readline()
                if not raw:
                    break
                line = raw.decode('utf-8', errors='replace').rstrip('\r\n')
                if line.startswith('event:'):
                    event = line[6:].strip()
                elif line.startswith('data:'):
                    data_lines.append(line[5:].strip())
                elif not line:
                    if data_lines:
                        payload = '\n'.join(data_lines)
                        self._handle_event(event, payload)
                    event = ''
                    data_lines = []
        except Exception as error:
            self.last_error = f'sse stream failed: {error}'
        finally:
            with self._lock:
                pending = list(self._pending.items())
                for request_id, _ in pending:
                    self._results[request_id] = {
                        'error': {'code': -1, 'message': 'SSE stream closed'}}
                self._pending.clear()
            for _, waiter in pending:
                waiter.set()
            try:
                connection.close()
            except Exception:
                pass
            self._stream = None
            self._sock = None

    def _handle_event(self, event: str, payload: str) -> None:
        if event == 'endpoint':
            target = payload.strip()
            self.message_url = urllib.parse.urljoin(self.url, target)
            self._endpoint_ready.set()
            return
        try:
            message = json.loads(payload)
        except json.JSONDecodeError:
            return
        if not isinstance(message, dict):
            return
        request_id = message.get('id')
        if request_id is None:
            return
        if 'method' in message:
            if message.get('method') == 'ping':
                self._post({'jsonrpc': '2.0', 'id': request_id, 'result': {}})
            return
        with self._lock:
            self._results[request_id] = message
            waiter = self._pending.pop(request_id, None)
        if waiter is not None:
            waiter.set()

    def _post(self, payload: dict[str, Any], timeout: float = 30.0) -> None:
        if not self.message_url:
            raise MCPError('SSE message endpoint is not ready')
        data = json.dumps(payload, ensure_ascii=False,
                          separators=(',', ':')).encode('utf-8')
        request = urllib.request.Request(
            self.message_url, data=data,
            headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response.read()
        except urllib.error.HTTPError as error:
            if error.code >= 400:
                raise MCPError(f'POST failed: {error.code} {error.reason}') from error

    def request(self, method: str, params: dict[str, Any] | None = None,
                timeout: float = 60.0) -> dict[str, Any]:
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            waiter = threading.Event()
            self._pending[request_id] = waiter
        try:
            self._post({'jsonrpc': '2.0', 'id': request_id, 'method': method,
                        'params': params or {}}, timeout=10)
        except Exception as error:
            with self._lock:
                self._pending.pop(request_id, None)
            return {'error': {'code': -1, 'message': str(error)}}
        if waiter.wait(timeout=timeout):
            with self._lock:
                return self._results.pop(request_id, {
                    'error': {'code': -1, 'message': 'no response'}})
        with self._lock:
            self._pending.pop(request_id, None)
        return {'error': {'code': -1, 'message': 'timeout'}}

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        try:
            self._post({'jsonrpc': '2.0', 'method': method, 'params': params or {}})
        except Exception:
            pass

    def close(self) -> None:
        self.connected = False
        self.connected_at = None
        sock = self._sock
        if sock is not None:
            try:
                import socket
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
        # The reader thread owns the response object and closes the connection
        # when its read unblocks, so close() never blocks on a concurrent read.
        self._sock = None

    def diagnostics(self) -> dict[str, Any]:
        details = super().diagnostics()
        details.update({'url': self.url, 'endpoint': bool(self.message_url)})
        return details


class MCPClient:
    """Handshake, tool discovery and tool execution for one server."""

    def __init__(self, name: str, transport: MCPTransport,
                 init_timeout: float = 15.0, request_timeout: float = 60.0) -> None:
        self.name = name
        self.transport = transport
        self.init_timeout = init_timeout
        self.request_timeout = request_timeout
        self.tools: list[dict[str, Any]] = []
        self.connected = False

    @property
    def protocol_version(self) -> str:
        return self.transport.protocol_version

    def connect(self) -> bool:
        try:
            response = self.transport.request('initialize', {
                'protocolVersion': PROTOCOL_VERSION,
                'capabilities': {},
                'clientInfo': CLIENT_INFO,
            }, timeout=self.init_timeout)
            if 'error' in response:
                self.transport.last_error = f'initialize: {response["error"].get("message")}'
                return False
            result = response.get('result', {})
            self.transport.protocol_version = str(result.get('protocolVersion', ''))
            self.transport.notify('notifications/initialized')
            if not self.scan_tools():
                return False
            self.connected = True
            self.transport.connected = True
            self.transport.connected_at = time.time()
            return True
        except Exception as error:
            self.transport.last_error = f'connect failed: {error}'
            try:
                self.transport.close()
            except Exception:
                pass
            return False

    def scan_tools(self) -> bool:
        """Fetch every page of tools/list and cache the normalized definitions."""
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(50):
            params: dict[str, Any] = {}
            if cursor:
                params['cursor'] = cursor
            response = self.transport.request('tools/list', params,
                                              timeout=self.request_timeout)
            if 'error' in response:
                self.transport.last_error = (
                    f'tools/list: {response["error"].get("message")}')
                return False
            result = response.get('result') or {}
            for tool in result.get('tools') or []:
                if not isinstance(tool, dict) or not tool.get('name'):
                    continue
                tools.append({
                    'name': str(tool['name']),
                    'description': str(tool.get('description') or ''),
                    'input_schema': tool.get('inputSchema')
                                    or {'type': 'object', 'properties': {}},
                })
            cursor = result.get('nextCursor')
            if not cursor:
                break
        self.tools = tools
        return True

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if not self.connected:
            return f'Error: MCP server {self.name} is not connected'
        try:
            response = self.transport.request(
                'tools/call', {'name': tool_name, 'arguments': arguments},
                timeout=self.request_timeout)
            if 'error' in response:
                return f'Error: {response["error"].get("message")}'
            result = response.get('result') or {}
            content = result.get('content') or []
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    parts.append(str(block))
                elif block.get('type') == 'text':
                    parts.append(str(block.get('text', '')))
                elif block.get('type') == 'resource':
                    parts.append(f'[resource: {block.get("resource", {})}]')
                elif block.get('type') == 'image':
                    parts.append(f'[image: {block.get("mimeType", "unknown")}]')
                else:
                    parts.append(json.dumps(block, ensure_ascii=False))
            text = '\n'.join(parts) or '(empty result)'
            if result.get('isError'):
                return f'Error: {text}'
            return text
        except Exception as error:
            return f'Error calling {tool_name}: {error}'

    def close(self) -> None:
        self.connected = False
        try:
            self.transport.close()
        except Exception:
            pass

    def diagnostics(self) -> dict[str, Any]:
        details = self.transport.diagnostics()
        details.update({
            'name': self.name,
            'tool_count': len(self.tools),
            'init_timeout': self.init_timeout,
            'request_timeout': self.request_timeout,
        })
        return details


def build_transport(config: dict[str, Any]) -> MCPTransport:
    transport = str(config.get('transport', 'stdio')).lower()
    if transport in ('sse',):
        url = str(config.get('url', ''))
        if not url:
            raise MCPError('missing url')
        return SseTransport(url)
    if transport in ('http', 'streamable http', 'streamable-http'):
        url = str(config.get('url', ''))
        if not url:
            raise MCPError('missing url')
        return HttpTransport(url)
    command = str(config.get('command', ''))
    if not command:
        raise MCPError('missing command')
    args = config.get('args') or []
    if not isinstance(args, list):
        args = []
    env = config.get('env') if isinstance(config.get('env'), dict) else {}
    return StdioTransport(command, [str(item) for item in args],
                          {str(k): str(v) for k, v in env.items()})
