"""The HTTP endpoint: static page, assets and the WebSocket upgrade.

One handler serves the three static files the page needs and hijacks the
connection on /ws.  A threaded HTTP server keeps a browser's frame stream from
blocking the requests of another device on the network.
"""

from __future__ import annotations

import secrets
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from adcode.web.session import WebSession
from adcode.web.websocket import WebSocketConnection, accept_key

_STATIC_DIR = Path(__file__).with_name('static')

_CONTENT_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.svg': 'image/svg+xml',
    '.png': 'image/png',
    '.ico': 'image/x-icon',
}


def _read_static(name: str) -> bytes | None:
    if not name or '/' in name or name.startswith('.'):
        return None
    try:
        return (_STATIC_DIR / name).read_bytes()
    except OSError:
        return None


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'AdCode'
    sys_version = ''

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """One request per frame would flood the console; stay quiet."""

    # -- routing ---------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - the base class API
        parsed = urlsplit(self.path)
        if parsed.path == '/ws':
            self._websocket(parse_qs(parsed.query))
            return
        self._static(parsed.path)

    def _static(self, path: str) -> None:
        name = 'index.html' if path in ('', '/') else path.lstrip('/')
        data = _read_static(name)
        if data is None:
            self.send_error(404, 'not found')
            return
        self.send_response(200)
        self.send_header('Content-Type', _CONTENT_TYPES.get(
            Path(name).suffix, 'application/octet-stream'))
        self.send_header('Content-Length', str(len(data)))
        # A stale key bar after an upgrade is worse than a re-download.
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(data)

    # -- websocket -------------------------------------------------
    def _websocket(self, query: dict[str, list[str]]) -> None:
        session: WebSession = self.server.session  # type: ignore[attr-defined]
        token = self.server.token  # type: ignore[attr-defined]
        given = (query.get('token') or [''])[0]
        if token and not secrets.compare_digest(str(given), str(token)):
            self.send_error(403, 'invalid token')
            return
        key = self.headers.get('Sec-WebSocket-Key')
        upgrade = (self.headers.get('Upgrade') or '').lower()
        if not key or upgrade != 'websocket':
            self.send_error(400, 'websocket handshake expected')
            return
        self.send_response(101, 'Switching Protocols')
        self.send_header('Upgrade', 'websocket')
        self.send_header('Connection', 'Upgrade')
        self.send_header('Sec-WebSocket-Accept', accept_key(key))
        self.end_headers()
        self.close_connection = True
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        session.attach(WebSocketConnection(self.rfile, self.wfile))


class WebServer:
    """Serves a WebSession to browsers."""

    def __init__(self, session: WebSession, host: str = '0.0.0.0',
                 port: int = 8765, token: str | None = None) -> None:
        self.session = session
        self.host = host
        self.port = port
        self.token = token
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> 'WebServer':
        httpd = ThreadingHTTPServer((self.host, self.port), _Handler)
        httpd.daemon_threads = True
        httpd.session = self.session  # type: ignore[attr-defined]
        httpd.token = self.token  # type: ignore[attr-defined]
        self._httpd = httpd
        self.port = httpd.server_address[1]
        self._thread = threading.Thread(
            target=httpd.serve_forever, name='adcode-web-http', daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        httpd = self._httpd
        self._httpd = None
        if httpd is None:
            return
        try:
            httpd.shutdown()
        except Exception:
            pass
        try:
            httpd.server_close()
        except Exception:
            pass

    def url(self, address: str = '127.0.0.1') -> str:
        suffix = f'?token={self.token}' if self.token else ''
        return f'http://{address}:{self.port}/{suffix}'


def local_addresses() -> list[str]:
    """The addresses another device on the LAN can reach this machine on."""
    addresses: list[str] = []
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Connecting a UDP socket sends nothing; it only picks the route.
            probe.connect(('192.168.1.1', 9))
            addresses.append(probe.getsockname()[0])
        finally:
            probe.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            address = info[4][0]
            if not address.startswith('127.') and address not in addresses:
                addresses.append(address)
    except OSError:
        pass
    return addresses


def run_web(host: str = '0.0.0.0', port: int = 8765,
            token: str | None = None, cwd: str | None = None,
            resume: bool = True) -> int:
    """Serve the TUI until the app exits or Ctrl+C is pressed."""
    session = WebSession(cwd=cwd, resume=resume).start()
    server = WebServer(session, host=host, port=port, token=token).start()
    addresses = local_addresses()
    if host in ('127.0.0.1', 'localhost'):
        addresses = ['127.0.0.1']
    print('AdCode is serving its terminal UI to browsers:', flush=True)
    for address in addresses or ['127.0.0.1']:
        print(f'  {server.url(address)}', flush=True)
    if host not in ('127.0.0.1', 'localhost'):
        print('  (bind 127.0.0.1 with --host to keep it on this machine)',
              flush=True)
    if not token:
        print('  WARNING: no token, anyone who can reach this port can drive '
              'the session, including the shell tools.', flush=True)
    print('Press Ctrl+C to stop.', flush=True)
    try:
        while not session.finished.wait(0.5):
            if not session.running:
                break
    except KeyboardInterrupt:
        print('', flush=True)
    finally:
        server.close()
        session.close()
        session.finished.wait(5.0)
    error = session.error
    if error is not None:
        import traceback
        traceback.print_exception(type(error), error, error.__traceback__)
        return 1
    return 0
