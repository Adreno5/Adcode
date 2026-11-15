"""Pasted images: markers, on-disk storage and clipboard capture.

Past images are kept as files next to the session store and referenced from the
conversation with a tiny ``[image: path]`` marker.  The wire adapters turn a
marker into a native image part only while a request is being built, so base64
never enters the message history, the session JSON or the token estimates.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The marker is the only thing stored in a message.  It survives sessions,
# rewind, fork and compaction because it is plain text.
MARKER_RE = re.compile(r'\[image:\s*([^\[\]\r\n]+?)\s*\]')

# Providers bill an image by its area, not by the marker's character count.
# A fixed estimate keeps the automatic compaction honest for typical screenshots.
IMAGE_TOKEN_ESTIMATE = 1_200
# The Messages API accepts at most 5 MB per image, so a file that passes this
# check is safe for every wire.
MAX_IMAGE_BYTES = 5 * 1024 * 1024
SUPPORTED_MEDIA = ('image/png', 'image/jpeg', 'image/gif', 'image/webp')
_MEDIA_EXTENSIONS = {
    'image/png': '.png', 'image/jpeg': '.jpg',
    'image/gif': '.gif', 'image/webp': '.webp',
}


def image_marker(reference: str) -> str:
    return f'[image: {str(reference).replace(os.sep, "/")}]'


def has_image_marker(text: str) -> bool:
    return bool(MARKER_RE.search(text))


def count_image_markers(text: str) -> int:
    return len(MARKER_RE.findall(text))


def strip_image_markers(text: str, replacement: str = '') -> str:
    return MARKER_RE.sub(replacement, text)


def summarize_image_markers(text: str) -> str:
    """Shorten markers for display: only the file name stays readable."""
    def shorten(match: re.Match[str]) -> str:
        reference = match.group(1).strip()
        if reference.startswith(('http://', 'https://')):
            return f'[image: {reference}]'
        return f'[image: {reference.rsplit("/", 1)[-1]}]'

    return MARKER_RE.sub(shorten, text)


def iter_segments(text: str) -> list[tuple[str, str]]:
    """Split a message into ('text', chunk) and ('image', reference) parts."""
    segments: list[tuple[str, str]] = []
    position = 0
    for match in MARKER_RE.finditer(text):
        if match.start() > position:
            segments.append(('text', text[position:match.start()]))
        segments.append(('image', match.group(1).strip()))
        position = match.end()
    if position < len(text):
        segments.append(('text', text[position:]))
    return segments


def is_image_url(reference: str) -> bool:
    """True for an http(s) URL that names an image file."""
    value = reference.strip().strip('"\'').lower()
    if not value.startswith(('http://', 'https://')):
        return False
    path = value.split('?', 1)[0].split('#', 1)[0]
    return path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))


def file_uri_to_path(reference: str) -> str:
    """Turn a ``file://`` URI into a local path; other values pass through.

    Drag-and-drop and file managers hand out URIs on some systems, while every
    other clipboard shape is already a path.
    """
    value = reference.strip().strip('"\'')
    if not value.lower().startswith('file://'):
        return value
    from urllib.parse import unquote, urlparse
    parsed = urlparse(value)
    path = unquote(parsed.path)
    if len(path) > 2 and path[0] == '/' and path[2] == ':':
        path = path[1:]            # file:///C:/shots/a.png
    if parsed.netloc and parsed.netloc != 'localhost':
        path = f'//{parsed.netloc}{path}'
    return path or value


def sniff_media_type(data: bytes) -> str | None:
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if data.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return 'image/gif'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'
    return None


def image_info(data: bytes) -> tuple[int, int] | None:
    """Return ``(width, height)`` read from the header, or None."""
    media_type = sniff_media_type(data)
    try:
        if media_type == 'image/png' and len(data) >= 24:
            return struct.unpack('>II', data[16:24])
        if media_type == 'image/gif' and len(data) >= 10:
            return struct.unpack('<HH', data[6:10])
        if media_type == 'image/jpeg':
            return _jpeg_size(data)
        if media_type == 'image/webp':
            return _webp_size(data)
    except (struct.error, IndexError):
        return None
    return None


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    index, length = 2, len(data)
    while index + 9 < length:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        size = struct.unpack('>H', data[index + 2:index + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack('>HH', data[index + 5:index + 9])
            return (width, height)
        index += 2 + size
    return None


def _webp_size(data: bytes) -> tuple[int, int] | None:
    if len(data) < 25:
        return None
    kind = data[12:16]
    if kind == b'VP8 ':
        if len(data) < 30 or data[23:26] != b'\x9d\x01\x2a':
            return None
        width, height = struct.unpack('<HH', data[26:30])
        return (width & 0x3FFF, height & 0x3FFF)
    if kind == b'VP8L':
        if data[20] != 0x2F:
            return None
        bits = int.from_bytes(data[21:25], 'little')
        return ((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
    if kind == b'VP8X' and len(data) >= 30:
        return (int.from_bytes(data[24:27], 'little') + 1,
                int.from_bytes(data[27:30], 'little') + 1)
    return None


@dataclass(frozen=True)
class ImageRef:
    """One resolved image: a local file or a remote URL."""

    reference: str
    path: Path | None = None
    url: str = ''
    media_type: str = ''
    size: int = 0


def resolve_image(reference: str, base_dir: str = '') -> ImageRef | None:
    """Resolve a marker path against the workspace, or accept an http(s) URL."""
    raw = reference.strip().strip('"\'')
    if not raw:
        return None
    if raw.startswith(('http://', 'https://')):
        return ImageRef(reference=raw, url=raw)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(base_dir or os.getcwd()) / path
    try:
        if not path.is_file():
            return None
        size = path.stat().st_size
        with path.open('rb') as stream:
            head = stream.read(16)
    except OSError:
        return None
    media_type = sniff_media_type(head)
    if media_type not in SUPPORTED_MEDIA or size > MAX_IMAGE_BYTES:
        return None
    return ImageRef(reference=reference, path=path, media_type=media_type, size=size)


def data_url(ref: ImageRef) -> str:
    """Inline image URL, built at request time and thrown away afterwards."""
    if ref.url:
        return ref.url
    data = ref.path.read_bytes() if ref.path is not None else b''
    encoded = base64.b64encode(data).decode('ascii')
    return f'data:{ref.media_type};base64,{encoded}'


def anthropic_source(ref: ImageRef, file_id: str = '') -> dict[str, object]:
    """Anthropic image source.

    A file uploaded to the Files API is referenced by id, so the request body
    stays tiny.  Otherwise the SDK is handed the ``Path`` and reads and encodes
    the bytes while serializing, so AdCode never holds base64 itself.
    """
    if ref.url:
        return {'type': 'url', 'url': ref.url}
    if file_id:
        return {'type': 'file', 'file_id': file_id}
    return {
        'type': 'base64',
        'media_type': ref.media_type,
        'data': ref.path,
    }


class FileIdCache:
    """Provider-scoped file ids issued by a provider's Files API.

    Uploading is a network round trip, so ids are remembered on disk keyed by
    the local image and the provider endpoint.  A provider that rejects the
    upload is remembered too, so every later request does not retry it.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def _entry(self, scope: str) -> dict[str, Any]:
        entry = self._data.get(scope)
        if not isinstance(entry, dict):
            entry = {}
            self._data[scope] = entry
        files = entry.get('files')
        if not isinstance(files, dict):
            entry['files'] = {}
        return entry

    def get_id(self, scope: str, key: str) -> str:
        with self._lock:
            value = self._entry(scope)['files'].get(key)
            return str(value) if value else ''

    def set_id(self, scope: str, key: str, file_id: str) -> None:
        if not file_id:
            return
        with self._lock:
            self._entry(scope)['files'][key] = file_id
            self._save()

    def unsupported(self, scope: str) -> bool:
        with self._lock:
            return bool(self._entry(scope).get('unsupported'))

    def mark_unsupported(self, scope: str, reason: str = '') -> None:
        with self._lock:
            entry = self._entry(scope)
            if entry.get('unsupported'):
                return
            entry['unsupported'] = True
            if reason:
                entry['reason'] = reason[:200]
            self._save()

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_name(f'.{self.path.name}.tmp')
            temp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2),
                            encoding='utf-8')
            os.replace(temp, self.path)
        except OSError:
            pass


class ImageStore:
    """Content-addressed image files under the session directory."""

    def __init__(self, directory: str | os.PathLike[str],
                 base_dir: str | os.PathLike[str] | None = None) -> None:
        self.directory = Path(directory)
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()

    def save(self, data: bytes, prefix: str = 'paste') -> str:
        media_type = sniff_media_type(data)
        if media_type is None:
            raise ValueError('the clipboard does not contain a supported image')
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError(
                f'image is {len(data) / 1_048_576:.1f} MB; the limit is '
                f'{MAX_IMAGE_BYTES // 1_048_576} MB')
        digest = hashlib.sha256(data).hexdigest()[:12]
        path = self.directory / f'{prefix}-{digest}{_MEDIA_EXTENSIONS[media_type]}'
        if not path.exists():
            self.directory.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(f'.{path.name}.tmp')
            try:
                temp.write_bytes(data)
                os.replace(temp, path)
            finally:
                if temp.exists():
                    temp.unlink()
        return self._relative(path)

    def save_path(self, source: str | os.PathLike[str]) -> str:
        path = Path(source).expanduser()
        if not path.is_absolute():
            path = self.base_dir / path
        return self.save(path.read_bytes())

    def _relative(self, path: Path) -> str:
        try:
            relative = os.path.relpath(path, self.base_dir)
        except ValueError:
            relative = str(path)
        return relative.replace(os.sep, '/')


# ── PNG encoding for the Windows clipboard ─────────────────


def encode_png(width: int, height: int, rgba: bytes | bytearray) -> bytes:
    """Encode straight RGBA rows as a PNG (filter 0 on every row)."""
    if width <= 0 or height <= 0:
        raise ValueError('image dimensions must be positive')
    row_bytes = width * 4
    if len(rgba) < row_bytes * height:
        raise ValueError('pixel buffer is too short')
    raw = bytearray(height * (row_bytes + 1))
    for y in range(height):
        start = y * (row_bytes + 1)
        raw[start] = 0
        raw[start + 1:start + 1 + row_bytes] = rgba[y * row_bytes:(y + 1) * row_bytes]
    return (
        b'\x89PNG\r\n\x1a\n'
        + _png_chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0))
        + _png_chunk(b'IDAT', zlib.compress(bytes(raw), 6))
        + _png_chunk(b'IEND', b'')
    )


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack('>I', len(payload)) + tag + payload + struct.pack('>I', crc)


def dib_to_png(data: bytes) -> bytes | None:
    """Convert a Windows CF_DIB / CF_DIBV5 payload into PNG bytes."""
    if len(data) < 40:
        return None
    header_size = struct.unpack_from('<I', data, 0)[0]
    if header_size < 40 or header_size > len(data):
        return None
    width, signed_height = struct.unpack_from('<ii', data, 4)
    planes, bits = struct.unpack_from('<HH', data, 12)
    compression = struct.unpack_from('<I', data, 16)[0]
    colors_used = struct.unpack_from('<I', data, 32)[0]

    if compression == 5:            # BI_PNG: the payload already is a PNG
        payload = data[header_size:]
        return payload if sniff_media_type(payload) == 'image/png' else None
    if compression == 4:            # BI_JPEG: no decoder available
        return None
    if planes != 1 or width <= 0 or signed_height == 0:
        return None
    if width > 20_000 or abs(signed_height) > 20_000:
        return None
    if width * abs(signed_height) > 40_000_000:
        return None
    if bits not in (8, 24, 32) or compression not in (0, 3):
        return None
    if compression == 3 and bits != 32:
        return None

    height = abs(signed_height)
    top_down = signed_height < 0
    stride = ((width * bits + 31) // 32) * 4
    offset = header_size
    palette = b''
    if bits == 8:
        entries = colors_used or 256
        palette = data[offset:offset + entries * 4]
        offset += entries * 4
    elif compression == 3 and header_size < 108:
        # BITMAPINFOHEADER + three colour masks (BITMAPV4/V5 keep them inside).
        offset += 12
    if offset + stride * height > len(data):
        return None

    rgba = bytearray(width * height * 4)
    any_alpha = False
    for y in range(height):
        source_y = y if top_down else height - 1 - y
        row = data[offset + source_y * stride:offset + source_y * stride + stride]
        target = y * width * 4
        if bits == 32:
            pixels = bytearray(row[:width * 4])
            pixels[0::4], pixels[2::4] = pixels[2::4], pixels[0::4]
            any_alpha = any_alpha or any(pixels[3::4])
            rgba[target:target + width * 4] = pixels
        elif bits == 24:
            pixels = bytearray(width * 4)
            pixels[0::4] = row[2:width * 3:3]
            pixels[1::4] = row[1:width * 3:3]
            pixels[2::4] = row[0:width * 3:3]
            pixels[3::4] = b'\xff' * width
            rgba[target:target + width * 4] = pixels
        else:
            for x in range(width):
                index = row[x] * 4
                entry = palette[index:index + 4]
                if len(entry) < 4:
                    return None
                pixels_at = target + x * 4
                rgba[pixels_at] = entry[2]
                rgba[pixels_at + 1] = entry[1]
                rgba[pixels_at + 2] = entry[0]
                rgba[pixels_at + 3] = 255
    if bits == 32 and not any_alpha:
        # Most Windows applications leave the alpha channel at zero.
        rgba[3::4] = b'\xff' * (width * height)
    try:
        return encode_png(width, height, rgba)
    except (ValueError, MemoryError):
        return None


# ── clipboard capture ──────────────────────────────────────


def read_clipboard_image() -> bytes | None:
    """Return the clipboard image as PNG/JPEG bytes, or None."""
    try:
        if os.name == 'nt':
            return _read_windows_image()
        if sys.platform == 'darwin':
            return _read_macos_image()
        return _read_linux_image()
    except Exception:
        return None


def _read_windows_image() -> bytes | None:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    user32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
    user32.RegisterClipboardFormatW.restype = wintypes.UINT
    kernel32.GlobalLock.argtypes = [wintypes.HANDLE]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HANDLE]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    kernel32.GlobalSize.argtypes = [wintypes.HANDLE]
    kernel32.GlobalSize.restype = ctypes.c_size_t

    opened = False
    for _ in range(20):
        if user32.OpenClipboard(None):
            opened = True
            break
        time.sleep(0.01)
    if not opened:
        return None
    try:
        formats = []
        for name in ('PNG', 'image/png'):
            fmt = int(user32.RegisterClipboardFormatW(name))
            if fmt:
                formats.append(fmt)
        for fmt in formats:
            data = _read_clipboard_bytes(user32, kernel32, fmt)
            if data and sniff_media_type(data) == 'image/png':
                return data
        for fmt in (8, 17):         # CF_DIB, CF_DIBV5
            data = _read_clipboard_bytes(user32, kernel32, fmt)
            if data:
                png = dib_to_png(data)
                if png:
                    return png
        return None
    finally:
        user32.CloseClipboard()


def _read_clipboard_bytes(user32, kernel32, fmt: int) -> bytes | None:
    import ctypes

    handle = user32.GetClipboardData(fmt)
    if not handle:
        return None
    size = int(kernel32.GlobalSize(handle))
    if not size:
        return None
    pointer = kernel32.GlobalLock(handle)
    if not pointer:
        return None
    try:
        return ctypes.string_at(pointer, size)
    finally:
        kernel32.GlobalUnlock(handle)


def _read_macos_image() -> bytes | None:
    try:
        data = subprocess.check_output(
            ['pngpaste', '-'], stderr=subprocess.DEVNULL, timeout=5)
        if sniff_media_type(data):
            return data
    except Exception:
        pass
    name = ''
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as stream:
            name = stream.name
        script = (
            'set f to open for access (POSIX file "%s") with write permission\n'
            'set eof f to 0\n'
            'write (the clipboard as \xabclass PNGf\xbb) to f\n'
            'close access f'
        ) % name.replace('\\', '\\\\').replace('"', '\\"')
        subprocess.run(['osascript', '-e', script], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=8)
        data = Path(name).read_bytes()
        return data if sniff_media_type(data) == 'image/png' else None
    except Exception:
        return None
    finally:
        if name:
            try:
                Path(name).unlink()
            except OSError:
                pass


def _read_linux_image() -> bytes | None:
    commands = (
        ['wl-paste', '--type', 'image/png', '--no-newline'],
        ['xclip', '-selection', 'clipboard', '-t', 'image/png', '-o'],
    )
    for command in commands:
        try:
            data = subprocess.check_output(
                command, stderr=subprocess.DEVNULL, timeout=5)
        except Exception:
            continue
        if sniff_media_type(data):
            return data
    return None
