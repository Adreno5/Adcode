"""Pasted images: markers, on-disk storage and clipboard capture.

The marker vocabulary, the on-disk store and the wire helpers live in the
engine sdk (``harness.images``); this module keeps what stays
application-shaped here:

- ``file_uri_to_path`` - drag-and-drop and file managers hand out URIs;
- clipboard capture - the raw bytes behind ``alt+v``, read per platform;
- the PNG encoders the Windows clipboard needs (a CF_DIB arrives when no
  application put a PNG on the clipboard).
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

# The shared image vocabulary: one implementation, re-exported so the
# historical import path keeps working.
from harness.images import (  # noqa: F401 - re-exported
    FileIdCache, ImageRef, ImageStore, IMAGE_TOKEN_ESTIMATE, MAX_IMAGE_BYTES,
    MARKER_RE, SUPPORTED_MEDIA, count_image_markers, data_url,
    has_image_marker, image_info, image_marker, is_image_url, iter_segments,
    resolve_image, sniff_media_type, strip_image_markers,
    summarize_image_markers,
)

__all__ = [
    'FileIdCache', 'ImageRef', 'ImageStore', 'IMAGE_TOKEN_ESTIMATE',
    'MAX_IMAGE_BYTES', 'MARKER_RE', 'SUPPORTED_MEDIA', 'count_image_markers',
    'data_url', 'file_uri_to_path', 'has_image_marker', 'image_info',
    'image_marker', 'is_image_url', 'iter_segments', 'read_clipboard_image',
    'resolve_image', 'sniff_media_type', 'strip_image_markers',
    'summarize_image_markers',
]


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
