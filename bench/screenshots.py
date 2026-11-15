"""Rasterize harness cell dumps to SVG (no dependencies) or PNG (Pillow).

The dump format is what :meth:`harness.screen.FrameSnapshot.to_payload`
produces:

    {"label": "...", "width": 100, "height": 30,
     "rows": [[[ch, [r,g,b], [r,g,b], attrs, cont], ...], ...]}

SVG always works.  PNG needs Pillow; when the app's virtualenv does not have it,
the manual harness can invoke this file with another interpreter that does:

    python harness/screenshots.py frame.json frame.png
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

FONT_CANDIDATES = (
    'C:/Windows/Fonts/consola.ttf',
    'C:/Windows/Fonts/consolab.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',
    '/System/Library/Fonts/Menlo.ttc',
)
BOLD_FONT_CANDIDATES = (
    'C:/Windows/Fonts/consolab.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf',
)


def _color(value: Any, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    try:
        return (int(value[0]), int(value[1]), int(value[2]))
    except (TypeError, ValueError, IndexError):
        return fallback


def _hex(color: tuple[int, int, int]) -> str:
    return '#{:02x}{:02x}{:02x}'.format(*color)


def svg_from_payload(payload: dict[str, Any], path: str | Path,
                     label: str | None = None, *,
                     font_size: int = 15, cell_width: float | None = None,
                     cell_height: int | None = None) -> Path:
    """Write a dependency-free SVG "screenshot" of the cell grid."""
    width = int(payload.get('width') or 0)
    height = int(payload.get('height') or 0)
    rows = payload.get('rows') or []
    label = label if label is not None else str(payload.get('label') or '')
    cell_width = cell_width or round(font_size * 0.603, 2)
    cell_height = cell_height or (font_size + 6)
    pad = 10
    title_h = 28
    image_w = int(width * cell_width + pad * 2)
    image_h = int(height * cell_height + pad * 2 + title_h)
    parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (f'<svg xmlns="http://www.w3.org/2000/svg" width="{image_w}" '
         f'height="{image_h}" viewBox="0 0 {image_w} {image_h}">'),
        '<rect width="100%" height="100%" fill="#111318"/>',
        (f'<text x="{pad}" y="20" fill="#8f97a8" font-family="Consolas, '
         f'Menlo, monospace" font-size="13">{escape(label)}'
         f'  {width}x{height}</text>'),
    ]
    for y, row in enumerate(rows[:height]):
        top = pad + title_h + y * cell_height
        # Merge adjacent cells with the same background to keep the SVG small.
        run_start = 0
        run_color: tuple[int, int, int] | None = None
        for x in range(width + 1):
            color = (_color(row[x][2], (0, 0, 0)) if x < width and x < len(row)
                     else None)
            if color != run_color:
                if run_color is not None and x > run_start:
                    parts.append(
                        f'<rect x="{pad + run_start * cell_width:.2f}" '
                        f'y="{top}" width="{(x - run_start) * cell_width:.2f}" '
                        f'height="{cell_height}" fill="{_hex(run_color)}"/>')
                run_start = x
                run_color = color
        for x in range(min(width, len(row))):
            cell = row[x]
            ch = str(cell[0] or '')
            if not ch or cell[4]:
                continue
            fg = _hex(_color(cell[1], (255, 255, 255)))
            attrs = int(cell[3] or 0)
            weight = ' font-weight="bold"' if attrs & 1 else ''
            opacity = ' opacity="0.65"' if attrs & 2 else ''
            parts.append(
                f'<text x="{pad + x * cell_width:.2f}" '
                f'y="{top + font_size + 1}" fill="{fg}"{weight}{opacity} '
                f'font-family="Consolas, Menlo, monospace" '
                f'font-size="{font_size}" xml:space="preserve">{escape(ch)}</text>')
    parts.append('</svg>')
    path = Path(path)
    path.write_text('\n'.join(parts), encoding='utf-8')
    return path


def _load_font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates = BOLD_FONT_CANDIDATES if bold else FONT_CANDIDATES
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def png_from_payload(payload: dict[str, Any], path: str | Path,
                     label: str | None = None, *,
                     font_size: int = 16, scale: float = 1.4) -> Path:
    """Write a PNG screenshot.  Requires Pillow."""
    from PIL import Image, ImageDraw

    width = int(payload.get('width') or 0)
    height = int(payload.get('height') or 0)
    rows = payload.get('rows') or []
    label = label if label is not None else str(payload.get('label') or '')
    font = _load_font(font_size)
    bold_font = _load_font(font_size, bold=True)
    small = _load_font(12)
    cell_width = float(font.getlength('M')) or font_size * 0.6
    cell_height = font_size + 6
    pad = 8
    title_h = 26
    size = (int(width * cell_width) + pad * 2,
            int(height * cell_height) + pad * 2 + title_h)
    image = Image.new('RGB', size, (17, 19, 24))
    draw = ImageDraw.Draw(image)
    draw.text((pad, 5), f'{label}  {width}x{height}', font=small,
              fill=(143, 151, 168))
    for y in range(min(height, len(rows))):
        row = rows[y]
        top = pad + title_h + y * cell_height
        for x in range(min(width, len(row))):
            cell = row[x]
            bg = _color(cell[2], (0, 0, 0))
            left = pad + x * cell_width
            draw.rectangle([left, top, left + cell_width, top + cell_height],
                           fill=bg)
        for x in range(min(width, len(row))):
            cell = row[x]
            ch = str(cell[0] or '')
            if not ch or cell[4]:
                continue
            fg = _color(cell[1], (255, 255, 255))
            attrs = int(cell[3] or 0)
            draw.text((pad + x * cell_width, top + 2), ch,
                      font=bold_font if attrs & 1 else font, fill=fg)
    if scale != 1.0:
        image = image.resize(
            (int(size[0] * scale), int(size[1] * scale)),
            Image.LANCZOS)
    path = Path(path)
    image.save(path)
    return path


def find_raster_python(explicit: str | None = None) -> str | None:
    """Return an interpreter that can import Pillow, if one is available."""
    candidates: list[str] = []
    for candidate in (explicit, os.environ.get('ADCODE_RASTER_PYTHON'),
                      sys.executable, 'python', 'python3', 'py'):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        try:
            probe = subprocess.run(
                [candidate, '-c', 'import PIL'],
                capture_output=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0:
            return candidate
    return None


def render_png(dump_path: str | Path, png_path: str | Path,
               raster_python: str | None = None,
               label: str | None = None) -> Path:
    """Render a dump to PNG, in-process or by spawning a Pillow interpreter."""
    dump_path = Path(dump_path)
    payload = json.loads(dump_path.read_text(encoding='utf-8'))
    try:
        return png_from_payload(payload, png_path, label)
    except ImportError:
        pass
    interpreter = find_raster_python(raster_python)
    if interpreter is None:
        raise RuntimeError(
            'Pillow is not installed in this interpreter and no raster '
            'interpreter was found; set ADCODE_RASTER_PYTHON or install Pillow')
    subprocess.run(
        [interpreter, str(Path(__file__).resolve()), str(dump_path),
         str(png_path), label or ''],
        check=True, timeout=120)
    return Path(png_path)


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print('usage: screenshots.py <frame.json> <out.png> [label]',
              file=sys.stderr)
        return 2
    dump_path = Path(argv[1])
    png_path = Path(argv[2])
    label = argv[3] if len(argv) > 3 else None
    payload = json.loads(dump_path.read_text(encoding='utf-8'))
    png_from_payload(payload, png_path, label)
    print(png_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
