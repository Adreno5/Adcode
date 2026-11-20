"""A minimal server-side WebSocket, sized for the browser bridge.

Only what the web UI needs: a text-only writer, and a reader that understands
continuation frames, ping/pong and close.  Keeping it here means the project
keeps its "standard library only" property instead of pulling a framework in
for one endpoint.
"""

from __future__ import annotations

import base64
import hashlib
import struct
import threading
from typing import Iterator

_GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'

# Refuse absurd frames rather than allocating whatever a client announces.
_MAX_PAYLOAD = 8 * 1024 * 1024

_OP_CONTINUATION = 0x0
_OP_TEXT = 0x1
_OP_BINARY = 0x2
_OP_CLOSE = 0x8
_OP_PING = 0x9
_OP_PONG = 0xA


def accept_key(client_key: str) -> str:
    """The Sec-WebSocket-Accept value for a client's Sec-WebSocket-Key."""
    digest = hashlib.sha1((client_key + _GUID).encode('ascii')).digest()
    return base64.b64encode(digest).decode('ascii')


def frame_bytes(opcode: int, payload: bytes) -> bytes:
    """Encode one frame.  Server frames are never masked."""
    header = bytearray([0x80 | opcode])
    length = len(payload)
    if length < 126:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header += struct.pack('!H', length)
    else:
        header.append(127)
        header += struct.pack('!Q', length)
    return bytes(header) + payload


class WebSocketClosed(Exception):
    """The peer went away; the connection is no longer usable."""


class WebSocketConnection:
    """One upgraded HTTP connection, speaking WebSocket frames."""

    def __init__(self, rfile, wfile) -> None:
        self._rfile = rfile
        self._wfile = wfile
        self._write_lock = threading.Lock()
        self._closed = False
        self._last_frame_final = True

    # -- sending -------------------------------------------------
    def send_text(self, text: str) -> None:
        self._send(_OP_TEXT, text.encode('utf-8'))

    def send_close(self, code: int = 1000) -> None:
        try:
            self._send(_OP_CLOSE, struct.pack('!H', code))
        except WebSocketClosed:
            pass

    def _send(self, opcode: int, payload: bytes) -> None:
        data = frame_bytes(opcode, payload)
        with self._write_lock:
            if self._closed:
                raise WebSocketClosed('connection closed')
            try:
                self._wfile.write(data)
                self._wfile.flush()
            except (OSError, ValueError) as error:
                self._closed = True
                raise WebSocketClosed(str(error)) from error

    def close(self) -> None:
        """Stop writing; the read loop ends when the socket does."""
        with self._write_lock:
            already = self._closed
            self._closed = True
        if not already:
            try:
                self._wfile.write(frame_bytes(_OP_CLOSE, struct.pack('!H', 1000)))
                self._wfile.flush()
            except Exception:
                pass

    # -- receiving -----------------------------------------------
    def messages(self) -> Iterator[str]:
        """Yield decoded text messages until the peer closes."""
        buffer = bytearray()
        while True:
            opcode, payload = self._read_frame()
            if opcode is None:
                return
            if opcode == _OP_CLOSE:
                self.close()
                return
            if opcode == _OP_PING:
                try:
                    self._send(_OP_PONG, payload)
                except WebSocketClosed:
                    return
                continue
            if opcode == _OP_PONG:
                continue
            if opcode not in (_OP_TEXT, _OP_BINARY, _OP_CONTINUATION):
                continue
            buffer += payload
            if len(buffer) > _MAX_PAYLOAD:
                self.send_close(1009)
                return
            if not self._last_frame_final:
                continue
            message = bytes(buffer)
            buffer = bytearray()
            try:
                yield message.decode('utf-8')
            except UnicodeDecodeError:
                continue

    def _read_exact(self, count: int) -> bytes:
        data = self._rfile.read(count) if count else b''
        if data is None or len(data) < count:
            raise WebSocketClosed('connection closed')
        return data

    def _read_frame(self) -> tuple[int | None, bytes]:
        """Read one frame; returns (None, b'') when the connection ended."""
        try:
            header = self._read_exact(2)
            first, second = header[0], header[1]
            opcode = first & 0x0F
            final = bool(first & 0x80)
            masked = bool(second & 0x80)
            length = second & 0x7F
            if length == 126:
                length = struct.unpack('!H', self._read_exact(2))[0]
            elif length == 127:
                length = struct.unpack('!Q', self._read_exact(8))[0]
            if length > _MAX_PAYLOAD:
                return None, b''
            mask = self._read_exact(4) if masked else b''
            payload = self._read_exact(length) if length else b''
        except (WebSocketClosed, OSError, ValueError):
            return None, b''
        if masked and payload:
            payload = bytes(byte ^ mask[index & 3]
                            for index, byte in enumerate(payload))
        self._last_frame_final = final
        return opcode, payload
