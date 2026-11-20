"""Tests for the browser bridge: key sequences, frame runs, and the server."""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import time

import pytest

import adcode.config as config_module
import adcode.paths as paths_module
from adcode.tui.canvas import Canvas
from adcode.tui.input import KeyParser, key_sequence, split_key
from adcode.theme import build_palette
from adcode.web.server import WebServer, local_addresses
from adcode.web.session import WebSession
from adcode.web.websocket import accept_key, frame_bytes
from bench.workspace import Workspace, build_config


# ── helpers ────────────────────────────────────────────────
class RawClient:
    """A tiny WebSocket client: enough to speak to the bridge."""

    def __init__(self, host: str, port: int, path: str = '/ws') -> None:
        self.sock = socket.create_connection((host, port), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode('ascii')
        request = (
            f'GET {path} HTTP/1.1\r\n'
            f'Host: {host}:{port}\r\n'
            'Upgrade: websocket\r\n'
            'Connection: Upgrade\r\n'
            f'Sec-WebSocket-Key: {key}\r\n'
            'Sec-WebSocket-Version: 13\r\n\r\n'
        )
        self.sock.sendall(request.encode('ascii'))
        self._buffer = b''
        while b'\r\n\r\n' not in self._buffer:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise AssertionError('handshake failed')
            self._buffer += chunk
        head, _, rest = self._buffer.partition(b'\r\n\r\n')
        self._buffer = rest
        self.status_line = head.split(b'\r\n')[0].decode('ascii')
        self.messages: list[dict] = []

    # -- protocol -------------------------------------------
    def send(self, payload: dict) -> None:
        data = json.dumps(payload).encode('utf-8')
        mask = os.urandom(4)
        header = bytearray([0x81])
        length = len(data)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header += struct.pack('!H', length)
        else:
            header.append(0x80 | 127)
            header += struct.pack('!Q', length)
        header += mask
        masked = bytes(byte ^ mask[index & 3]
                       for index, byte in enumerate(data))
        self.sock.sendall(bytes(header) + masked)

    def _read_exact(self, count: int) -> bytes:
        while len(self._buffer) < count:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise AssertionError('connection closed')
            self._buffer += chunk
        data, self._buffer = self._buffer[:count], self._buffer[count:]
        return data

    def receive(self, timeout: float = 10.0) -> dict:
        deadline = time.monotonic() + timeout
        self.sock.settimeout(max(0.05, deadline - time.monotonic()))
        first = self._read_exact(2)
        opcode = first[0] & 0x0F
        length = first[1] & 0x7F
        if length == 126:
            length = struct.unpack('!H', self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack('!Q', self._read_exact(8))[0]
        payload = self._read_exact(length) if length else b''
        if opcode == 0x8:
            raise AssertionError('server closed the connection')
        message = json.loads(payload.decode('utf-8'))
        self.messages.append(message)
        return message

    def wait_for(self, predicate, timeout: float = 15.0, what: str = 'message'):
        deadline = time.monotonic() + timeout
        seen: list[dict] = []
        while time.monotonic() < deadline:
            message = self.receive(timeout=max(0.05, deadline - time.monotonic()))
            seen.append(message)
            if predicate(message):
                return message
        raise AssertionError(f'timed out waiting for {what}: {seen[-3:]}')

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def screen_text(messages: list[dict], rows: int = 30) -> str:
    """Fold the frame messages a client received into a text screen."""
    grid = [''] * rows
    for message in messages:
        for line in message.get('lines') or []:
            y = line[0]
            if 0 <= y < rows:
                grid[y] = ''.join(run[0] for run in line[1])
    return '\n'.join(grid)


# ── key sequences ──────────────────────────────────────────
@pytest.mark.parametrize('name', [
    'ctrl+c', 'alt+up', 'shift+tab', 'tab', 'enter', 'esc', 'backspace',
    'up', 'down', 'left', 'right', 'home', 'end', 'pgup', 'pgdn', 'insert',
    'delete', 'f1', 'f5', 'f12', 'ctrl+,', 'alt+enter', 'shift+enter',
    'ctrl+shift+left', 'ctrl+home', 'alt+v', 'ctrl+j', 'ctrl+space',
    'shift+home', 'alt+pgup', 'ctrl+alt+right',
])
def test_key_sequence_round_trips_through_the_parser(name: str) -> None:
    parser = KeyParser()
    events = parser.feed(key_sequence(name)) + parser.flush_pending()
    assert [event.key for event in events] == [name]
    assert events[0].kind == 'key'


def test_key_sequence_keeps_the_parser_flags() -> None:
    parser = KeyParser()
    events = parser.feed(key_sequence('ctrl+shift+left'))
    assert events[0].ctrl and events[0].shift and not events[0].alt


def test_plain_characters_are_text_not_keys() -> None:
    parser = KeyParser()
    events = parser.feed(key_sequence('a'))
    assert [(event.kind, event.text) for event in events] == [('text', 'a')]


def test_split_key_rejects_unknown_modifiers() -> None:
    with pytest.raises(KeyError):
        split_key('hyper+x')
    with pytest.raises(KeyError):
        key_sequence('')


# ── canvas runs ────────────────────────────────────────────
def test_row_runs_cover_every_cell() -> None:
    palette = build_palette('dark', '#8fa7d6')
    canvas = Canvas(20, 2, palette)
    canvas.clear()
    canvas.draw_text(0, 0, 'hi there', palette.text, palette.background, 1)
    runs = canvas.row_runs(0)
    assert sum(len(run[0]) for run in runs) == 20
    assert runs[0][0] == 'hi there'
    assert runs[0][3] & 1        # BOLD


def test_row_runs_folds_reverse_and_skips_wide_continuations() -> None:
    from adcode.tui.canvas import REVERSE
    palette = build_palette('dark', '#8fa7d6')
    canvas = Canvas(6, 1, palette)
    canvas.clear()
    canvas.put(0, 0, 'A', palette.text, palette.background, REVERSE)
    canvas.draw_text(1, 0, '漢', palette.text, palette.background)
    runs = canvas.row_runs(0)
    text = ''.join(run[0] for run in runs)
    assert text.startswith('A漢')
    assert not runs[0][3] & REVERSE
    # The reversed cell swapped its colours; the plain one after it did not.
    background = canvas.palette.background
    packed_background = (background[0] << 16) | (background[1] << 8) | background[2]
    assert runs[0][1] == packed_background
    # The wide glyph covers two cells in the browser as well, so the row still
    # measures the full width even though one continuation cell was skipped.
    from adcode.tui.canvas import text_width
    assert text_width(text) == 6


def test_row_runs_are_blank_for_an_untouched_row() -> None:
    palette = build_palette('dark', '#8fa7d6')
    canvas = Canvas(5, 2, palette)
    canvas.clear()
    runs = canvas.row_runs(1)
    assert [run[0] for run in runs] == ['     ']


# ── websocket plumbing ─────────────────────────────────────
def test_accept_key_matches_the_spec() -> None:
    assert accept_key('dGhlIHNhbXBsZSBub25jZQ==') == (
        's3pPLMBiTxaQ9kYGzzhZRbK+xOo=')


def test_frame_bytes_encodes_lengths() -> None:
    small = frame_bytes(0x1, b'hi')
    assert small[0] == 0x81 and small[1] == 2 and small[2:] == b'hi'
    medium = frame_bytes(0x1, b'x' * 200)
    assert medium[1] == 126 and struct.unpack('!H', medium[2:4])[0] == 200
    large = frame_bytes(0x1, b'x' * 70000)
    assert large[1] == 127 and struct.unpack('!Q', large[2:10])[0] == 70000


# ── the session and the served page ────────────────────────
@pytest.fixture()
def web_session(monkeypatch):
    workspace = Workspace()
    monkeypatch.setattr(config_module, 'encrypt_secret', lambda value: value)
    monkeypatch.setattr(config_module, 'decrypt_secret', lambda value: value)
    monkeypatch.setattr(paths_module, 'APP_DIR',
                        workspace.root / 'adcode-home')
    workspace.write_config(build_config('http://127.0.0.1:9/v1'))
    session = WebSession(cwd=str(workspace.project), resume=False,
                         config_path=str(workspace.config_path),
                         size=(80, 24))
    session.start()
    try:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            assert session.error is None, repr(session.error)
            if json.loads(session.snapshot())['lines']:
                break
            time.sleep(0.02)
        yield session
    finally:
        session.close()
        workspace.cleanup()


def test_web_input_delivers_terminal_events() -> None:
    from adcode.web.input import WebInput

    import queue
    input_layer = WebInput(queue.Queue())
    delivered: list = []
    original_put = input_layer.events.put

    def capture(event, *args, **kwargs):
        delivered.append(event)
        return original_put(event, *args, **kwargs)

    input_layer.events.put = capture
    input_layer.type_text('hi')
    input_layer.press('ctrl+c')
    input_layer.press('up')
    input_layer.press('shift+enter')
    input_layer.paste('line1\nline2')
    input_layer.mouse('wheel_up', 3, 4)
    kinds = [event.kind for event in delivered]
    assert kinds[:2] == ['text', 'text']
    assert delivered[2].key == 'ctrl+c' and delivered[2].ctrl
    assert delivered[3].key == 'up'
    assert delivered[4].key == 'shift+enter' and delivered[4].shift
    paste = [event for event in delivered if event.kind == 'paste']
    assert paste and paste[0].text == 'line1\nline2'
    mouse = [event for event in delivered if event.kind == 'mouse']
    assert mouse and mouse[0].action == 'wheel_up'


def test_web_modifier_press_and_release() -> None:
    from adcode.web.input import WebInput

    import queue
    input_layer = WebInput(queue.Queue())
    delivered: list = []
    original_put = input_layer.events.put

    def capture(event, *args, **kwargs):
        delivered.append(event)
        return original_put(event, *args, **kwargs)

    input_layer.events.put = capture
    input_layer.modifier('alt', 'press')
    input_layer.modifier('alt', 'release')
    assert [event.kind for event in delivered] == ['modifier', 'modifier']
    assert [event.action for event in delivered] == ['press', 'release']
    assert all(event.key == 'alt' for event in delivered)


def test_static_page_is_served(web_session) -> None:
    server = WebServer(web_session, host='127.0.0.1', port=0).start()
    try:
        import http.client
        connection = http.client.HTTPConnection('127.0.0.1', server.port,
                                                timeout=10)
        connection.request('GET', '/')
        response = connection.getresponse()
        assert response.status == 200
        body = response.read().decode('utf-8')
        assert 'id="keybar"' in body and 'app.js' in body
        connection.request('GET', '/app.js')
        script = connection.getresponse()
        assert script.status == 200
        assert 'wheel_up' in script.read().decode('utf-8')
        connection.request('GET', '/nope.js')
        assert connection.getresponse().status == 404
        connection.close()
        addresses = local_addresses()
        assert isinstance(addresses, list)
        assert all(not address.startswith('127.') for address in addresses)
    finally:
        server.close()


def test_token_is_required(web_session) -> None:
    server = WebServer(web_session, host='127.0.0.1', port=0,
                       token='secret').start()
    try:
        import http.client
        connection = http.client.HTTPConnection('127.0.0.1', server.port,
                                                timeout=10)
        connection.request('GET', '/ws')
        assert connection.getresponse().status == 403
        connection.close()
        # The page itself stays readable: it carries the token in its URL.
        connection = http.client.HTTPConnection('127.0.0.1', server.port,
                                                timeout=10)
        connection.request('GET', '/?token=secret')
        assert connection.getresponse().status == 200
        connection.close()
        denied = RawClient('127.0.0.1', server.port)
        assert '403' in denied.status_line
        denied.close()
        client = RawClient('127.0.0.1', server.port, path='/ws?token=secret')
        assert client.receive()['t'] == 'full'
        client.close()
    finally:
        server.close()


def test_browser_types_and_the_prompt_shows_it(web_session) -> None:
    server = WebServer(web_session, host='127.0.0.1', port=0).start()
    client = RawClient('127.0.0.1', server.port)
    try:
        first = client.receive()
        assert first['t'] == 'full'
        assert first['w'] > 0 and first['h'] > 0

        client.send({'t': 'text', 'v': 'hello bridge'})
        client.wait_for(
            lambda message: 'hello bridge' in screen_text(client.messages),
            what='the typed text to reach a frame')

        client.send({'t': 'key', 'k': 'backspace'})
        client.wait_for(
            lambda message: ('> hello bridg' in screen_text(client.messages)
                             and 'hello bridge' not in screen_text(client.messages)),
            what='the backspace to reach a frame')

        client.send({'t': 'size', 'cols': 60, 'rows': 18})
        size = client.wait_for(lambda message: message['t'] == 'size',
                               what='the grid to follow the window')
        assert (size['w'], size['h']) == (60, 18)
        full = client.wait_for(
            lambda message: message['t'] == 'full' and message['w'] == 60,
            what='a full frame at the new size')
        assert full['h'] == 18
        assert web_session.grid() == (60, 18)
    finally:
        client.close()
        server.close()


def test_the_wheel_scrolls_the_transcript(web_session) -> None:
    server = WebServer(web_session, host='127.0.0.1', port=0).start()
    client = RawClient('127.0.0.1', server.port)
    try:
        client.receive()
        app = web_session.app
        assert app is not None
        app.scroll = 0
        client.send({'t': 'mouse', 'a': 'wheel_up', 'x': 1, 'y': 1})
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and app.follow:
            time.sleep(0.02)
        assert app.follow is False
    finally:
        client.close()
        server.close()


def test_holding_alt_in_the_browser_reveals_details(web_session) -> None:
    server = WebServer(web_session, host='127.0.0.1', port=0).start()
    client = RawClient('127.0.0.1', server.port)
    try:
        client.receive()
        app = web_session.app
        assert app is not None
        app._alt_reveal_seconds = 0.2      # keep the test quick
        client.send({'t': 'modifier', 'm': 'alt', 'a': 'press'})
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not app._alt_details_shown:
            time.sleep(0.02)
        assert app._alt_details_shown is True
        client.send({'t': 'modifier', 'm': 'alt', 'a': 'release'})
        while time.monotonic() < deadline and app._alt_details_shown:
            time.sleep(0.02)
        assert app._alt_details_shown is False
    finally:
        client.close()
        server.close()


def test_second_browser_joins_the_same_session(web_session) -> None:
    server = WebServer(web_session, host='127.0.0.1', port=0).start()
    first = RawClient('127.0.0.1', server.port)
    second = None
    try:
        first.receive()
        first.send({'t': 'size', 'cols': 90, 'rows': 30})
        first.wait_for(lambda message: message['t'] == 'size')
        second = RawClient('127.0.0.1', server.port)
        snapshot = second.receive()
        assert snapshot['t'] == 'full'
        # The newcomer asks for less room; both browsers get the smaller grid
        # so neither has to scroll to see the whole screen.
        second.send({'t': 'size', 'cols': 50, 'rows': 20})
        second.wait_for(lambda message: message['t'] == 'size')
        first.wait_for(lambda message: message['t'] == 'size'
                       and message['w'] == 50)
        assert web_session.grid() == (50, 20)
    finally:
        first.close()
        if second is not None:
            second.close()
        server.close()
