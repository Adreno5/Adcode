"""End-to-end proxy tests against a real SOCKS5 server.

A minimal SOCKS5 CONNECT proxy is started in-process, so the requests really
travel through the SOCKS handshake instead of only checking that a client can
be constructed.
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from infra import net
from adcode.config import ModelConfig, ProviderConfig
from adcode.llm.client import LLMClient


class Socks5Server(threading.Thread):
    """A tiny SOCKS5 proxy: no authentication, CONNECT only."""

    def __init__(self) -> None:
        super().__init__(daemon=True, name='test-socks5')
        self._server = socket.socket()
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(('127.0.0.1', 0))
        self._server.listen(8)
        self.port = self._server.getsockname()[1]
        self.connections = 0
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                client, _ = self._server.accept()
            except OSError:
                return
            self.connections += 1
            threading.Thread(target=self._handle, args=(client,),
                             daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._server.close()
        except OSError:
            pass

    def _handle(self, client: socket.socket) -> None:
        upstream = None
        try:
            header = client.recv(2)
            if len(header) < 2:
                return
            client.recv(header[1])
            client.sendall(b'\x05\x00')          # no authentication
            request = client.recv(4)
            if len(request) < 4:
                return
            atyp = request[3]
            if atyp == 1:
                host = socket.inet_ntoa(client.recv(4))
            elif atyp == 3:
                host = client.recv(client.recv(1)[0]).decode()
            else:
                host = socket.inet_ntop(socket.AF_INET6, client.recv(16))
            port = int.from_bytes(client.recv(2), 'big')
            upstream = socket.create_connection((host, port), timeout=10)
            client.sendall(b'\x05\x00\x00\x01' + b'\x00' * 4 + b'\x00\x00')
            self._pipe(client, upstream)
        except OSError:
            pass
        finally:
            for sock in (client, upstream):
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass

    @staticmethod
    def _pipe(left: socket.socket, right: socket.socket) -> None:
        def copy(source: socket.socket, sink: socket.socket) -> None:
            try:
                while True:
                    data = source.recv(65536)
                    if not data:
                        break
                    sink.sendall(data)
            except OSError:
                pass
            finally:
                try:
                    sink.shutdown(socket.SHUT_WR)
                except OSError:
                    pass

        threads = [threading.Thread(target=copy, args=pair, daemon=True)
                   for pair in ((left, right), (right, left))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)


class EchoHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = b'hello through the proxy'
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        payload = json.loads(self.rfile.read(length) or b'{}')
        chunks = [
            {'id': '1', 'object': 'chat.completion.chunk', 'created': 1,
             'model': payload.get('model', 'm'),
             'choices': [{'index': 0, 'delta': {'content': 'proxied'},
                          'finish_reason': None}]},
            {'id': '1', 'object': 'chat.completion.chunk', 'created': 1,
             'model': payload.get('model', 'm'),
             'choices': [{'index': 0, 'delta': {},
                          'finish_reason': 'stop'}]},
        ]
        body = b''.join(
            f'data: {json.dumps(chunk)}\n\n'.encode() for chunk in chunks)
        body += b'data: [DONE]\n\n'
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def socks5():
    server = Socks5Server()
    server.start()
    yield server
    server.stop()


@pytest.fixture()
def http_server():
    server = ThreadingHTTPServer(('127.0.0.1', 0), EchoHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def clean_proxy_env(monkeypatch):
    for name in ('http_proxy', 'HTTP_PROXY', 'https_proxy', 'HTTPS_PROXY',
                 'all_proxy', 'ALL_PROXY', 'no_proxy', 'NO_PROXY'):
        monkeypatch.delenv(name, raising=False)


def test_a_request_travels_through_the_socks5_proxy(socks5, http_server,
                                                     monkeypatch):
    monkeypatch.setenv('ALL_PROXY', f'socks5://127.0.0.1:{socks5.port}')
    url = f'http://127.0.0.1:{http_server.server_address[1]}/hello'

    import urllib.request

    with net.urlopen(urllib.request.Request(url), timeout=10) as response:
        assert response.status == 200
        assert response.read() == b'hello through the proxy'
    assert socks5.connections >= 1


def test_the_model_api_travels_through_the_socks5_proxy(socks5, http_server,
                                                         monkeypatch):
    monkeypatch.setenv('ALL_PROXY', f'socks5://127.0.0.1:{socks5.port}')
    provider = ProviderConfig(
        name='P', base_url=f'http://127.0.0.1:{http_server.server_address[1]}/v1',
        api_key='k', models=[ModelConfig(id='m')])
    client = LLMClient(provider)
    try:
        events = list(client.chat([{'role': 'user', 'content': 'hi'}],
                                  provider.models[0], max_tokens=64))
    finally:
        client.close()
    text = ''.join(str(event.get('content', '')) for event in events
                   if event.get('kind') == 'text')
    assert text == 'proxied'
    assert socks5.connections >= 1
