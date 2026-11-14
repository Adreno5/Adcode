import json
import queue
import sys
import threading

import pytest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from adcode.mcp.manager import MCPManager
from adcode.tools.registry import ToolRegistry


STDIO_SERVER = '''import json, sys
NL = chr(10)

def send(obj):
    sys.stdout.write(json.dumps(obj) + NL)
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    mid = msg.get('id')
    method = msg.get('method')
    if method == 'initialize':
        send({'jsonrpc': '2.0', 'id': mid, 'result': {
            'protocolVersion': '2025-06-18', 'capabilities': {'tools': {}},
            'serverInfo': {'name': 'fake', 'version': '1'}}})
    elif method == 'notifications/initialized':
        pass
    elif method == 'tools/list':
        params = msg.get('params') or {}
        if not params.get('cursor'):
            send({'jsonrpc': '2.0', 'id': mid, 'result': {'tools': [{
                'name': 'echo', 'description': 'Echo text',
                'inputSchema': {'type': 'object', 'properties': {
                    'text': {'type': 'string'}}}}], 'nextCursor': 'page2'}})
        else:
            send({'jsonrpc': '2.0', 'id': mid, 'result': {'tools': [{
                'name': 'add', 'description': 'Add numbers',
                'inputSchema': {'type': 'object', 'properties': {
                    'a': {'type': 'number'}, 'b': {'type': 'number'}}}}]}})
    elif method == 'tools/call':
        params = msg.get('params') or {}
        arguments = params.get('arguments') or {}
        if params.get('name') == 'echo':
            text = 'echo:' + str(arguments.get('text', ''))
        else:
            text = str(arguments.get('a', 0) + arguments.get('b', 0))
        send({'jsonrpc': '2.0', 'id': mid, 'result': {
            'content': [{'type': 'text', 'text': text}]}})
'''


@pytest.fixture()
def stdio_server(tmp_path):
    path = tmp_path / 'mcp_server.py'
    path.write_text(STDIO_SERVER, encoding='utf-8')
    return path


def test_stdio_transport_and_pagination(stdio_server):
    registry = ToolRegistry()
    manager = MCPManager(registry)
    config = {'name': 'fake', 'transport': 'stdio', 'command': sys.executable,
              'args': [str(stdio_server)]}
    assert manager.add_server(config)
    client = manager.get_client('fake')
    assert {tool['name'] for tool in client.tools} == {'echo', 'add'}
    names = {spec.name for spec in registry.specs()
             if spec.source.startswith('mcp:')}
    assert names == {'mcp__fake__echo', 'mcp__fake__add'}
    assert registry.execute('mcp__fake__echo', {'text': 'hi'}).text == 'echo:hi'
    assert registry.execute('mcp__fake__add', {'a': 2, 'b': 3}).text == '5'
    assert manager.refresh()['fake'] == 'ok'
    assert manager.diagnostics()[0]['tool_count'] == 2
    manager.shutdown()
    assert not [spec for spec in registry.specs()
                if spec.source.startswith('mcp:')]


class HTTPHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        message = json.loads(self.rfile.read(length))
        method = message.get('method')
        if method == 'notifications/initialized':
            self.send_response(202)
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if method == 'initialize':
            result = {'protocolVersion': '2025-06-18', 'capabilities': {'tools': {}},
                      'serverInfo': {'name': 'http', 'version': '1'}}
        elif method == 'tools/list':
            result = {'tools': [{'name': 'ping', 'description': 'Ping',
                                 'inputSchema': {'type': 'object', 'properties': {}}}]}
        elif method == 'tools/call':
            result = {'content': [{'type': 'text', 'text': 'pong'}]}
        else:
            result = {}
        body = json.dumps({'jsonrpc': '2.0', 'id': message.get('id'),
                           'result': result}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_http_transport():
    server = ThreadingHTTPServer(('127.0.0.1', 0), HTTPHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    registry = ToolRegistry()
    manager = MCPManager(registry)
    url = f'http://127.0.0.1:{server.server_address[1]}/mcp'
    try:
        assert manager.add_server({'name': 'http', 'transport': 'http', 'url': url})
        assert registry.execute('mcp__http__ping', {}).text == 'pong'
    finally:
        manager.shutdown()
        server.shutdown()
        server.server_close()


class SSEServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages = queue.Queue()
        self.stopping = threading.Event()


class SSEHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(b'event: endpoint\ndata: /messages\n\n')
        self.wfile.flush()
        while not self.server.stopping.is_set():
            try:
                message = self.server.messages.get(timeout=0.1)
            except queue.Empty:
                continue
            if message is None:
                break
            try:
                payload = json.dumps(message)
                self.wfile.write(f'event: message\ndata: {payload}\n\n'.encode())
                self.wfile.flush()
            except Exception:
                break

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        message = json.loads(self.rfile.read(length))
        if 'id' in message:
            method = message.get('method')
            if method == 'initialize':
                result = {'protocolVersion': '2025-06-18', 'capabilities': {},
                          'serverInfo': {'name': 'sse', 'version': '1'}}
            elif method == 'tools/list':
                result = {'tools': [{'name': 'sse_echo', 'description': 'Echo',
                                     'inputSchema': {'type': 'object',
                                                     'properties': {}}}]}
            elif method == 'tools/call':
                result = {'content': [{'type': 'text', 'text': 'sse-pong'}]}
            else:
                result = {}
            self.server.messages.put({'jsonrpc': '2.0', 'id': message['id'],
                                      'result': result})
        self.send_response(202)
        self.send_header('Content-Length', '0')
        self.end_headers()


def test_sse_transport():
    server = SSEServer(('127.0.0.1', 0), SSEHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    registry = ToolRegistry()
    manager = MCPManager(registry)
    url = f'http://127.0.0.1:{server.server_address[1]}/sse'
    try:
        assert manager.add_server({'name': 'sse', 'transport': 'sse', 'url': url})
        assert registry.execute('mcp__sse__sse_echo', {}).text == 'sse-pong'
    finally:
        manager.shutdown()
        server.stopping.set()
        server.messages.put(None)
        server.shutdown()
        server.server_close()
