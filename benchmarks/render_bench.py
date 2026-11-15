"""Render-engine benchmark for the AdCode TUI canvas.

Three scenarios, each mirroring what the real app does per frame at 120x40:

``full``   clear the canvas and redraw every widget row (what ``App._render``
           does on every frame), then serialize the frame.
``idle``   nothing changes; the compositor still calls render.  The interesting
           number here is *bytes written to the terminal*, which should be zero
           for a renderer that only emits changed rows.
``tick``   only one transcript row changed (the common streaming case: the model
           appends a character to the live answer).

Usage::

    python benchmarks/render_bench.py                 # current canvas module
    python benchmarks/render_bench.py --legacy        # pre-refactor snapshot
    python benchmarks/render_bench.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adcode.theme import build_palette  # noqa: E402

WIDTH, HEIGHT = 120, 40
CYCLES = 300

SAMPLE_LINES = [
    ('| ', 'border', 255, 'Ran ', 'muted', 255, 'uv run pytest -q', 'text', 255),
    ('| ', 'border', 255, '+ ', 'success', 255, '12 passed in 3.41s', 'text', 255),
    ('', None, 255, 'The renderer composites every widget onto a single cell grid. ', 'text', 255),
    ('', None, 255, 'Wide glyphs 中文测试 must not be split by a dirty rect.', 'text', 255),
    ('> ', 'accent', 255, 'normalize the frame diff before the flush', 'text', 255),
    ('', None, 255, '· ', 'muted', 160, 'thought for 2.1s', 'muted', 160),
]
HEADERS = ['AdCode', '~ 12.4k tokens', 'deepseek-v4', 'idle']


def _color(palette, role: str | None):
    if role is None:
        return palette.text
    return getattr(palette, role, palette.text)


def build_scene(Canvas, palette) -> object:
    """Draw one realistic frame: header, transcript viewport, prompt, footer."""
    canvas = Canvas(WIDTH, HEIGHT, palette)
    canvas.clear()
    # header bar
    canvas.fill_rect(0, 0, WIDTH, 1, palette.panel)
    x = 1
    for index, title in enumerate(HEADERS):
        x += canvas.draw_text(x, 0, title, palette.text if index == 0 else palette.muted)
        x += 3
    canvas.fill_rect(0, 1, WIDTH, 1, palette.border, alpha=90)
    # transcript viewport with a panel behind it
    canvas.fill_rect(0, 2, WIDTH - 1, HEIGHT - 6, palette.background)
    row = 2
    while row < HEIGHT - 5:
        for line in SAMPLE_LINES:
            if row >= HEIGHT - 5:
                break
            column = 2
            for index in range(0, len(line), 3):
                text, role, alpha = line[index], line[index + 1], line[index + 2]
                if not text:
                    column += 1
                    continue
                column += canvas.draw_text(
                    column, row, text, _color(palette, role), None, 0, alpha,
                    max_width=WIDTH - column - 1)
            row += 1
    canvas.box(0, 2, WIDTH, HEIGHT - 3, palette.border, title='transcript')
    # scrollbar
    canvas.fill_rect(WIDTH - 1, 3, 1, HEIGHT - 8, palette.panel_alt)
    # separator + prompt + footer
    canvas.hline(0, HEIGHT - 4, WIDTH, palette.border, alpha=110)
    canvas.fill_rect(0, HEIGHT - 3, WIDTH, 2, palette.panel)
    canvas.draw_text(2, HEIGHT - 3, '> refactor the renderer', palette.text)
    canvas.fill_rect(0, HEIGHT - 1, WIDTH, 1, palette.panel_alt)
    canvas.draw_text(1, HEIGHT - 1, 'enter send   shift+enter newline   esc cancel',
                     palette.muted)
    return canvas


def _frames(impl: str) -> list[str]:
    """Render ``full`` -> ``idle`` -> ``tick`` and return the three payloads."""
    if impl == 'legacy':
        from legacy_canvas import Canvas
        render = lambda canvas: canvas.render()          # noqa: E731
        draw_tick = lambda canvas, palette: canvas.draw_text(  # noqa: E731
            4, HEIGHT - 3, 'x', palette.text)
    else:
        from adcode.tui.canvas import Canvas
        render = lambda canvas: canvas.render_diff()     # noqa: E731
        draw_tick = lambda canvas, palette: canvas.draw_text(  # noqa: E731
            4, HEIGHT - 3, 'x', palette.text)
    palette = build_palette('dark', '#8fa7d6')
    canvas = build_scene(Canvas, palette)
    if impl == 'legacy':
        # Legacy has no diff: a frame always re-serializes the whole grid, so
        # "idle" and "tick" measure the same full payload by construction.
        return [render(canvas), render(canvas), render(canvas)]
    full = render(canvas)
    idle = render(canvas)
    draw_tick(canvas, palette)
    tick = render(canvas)
    return [full, idle, tick]


def measure(impl: str, cycles: int = CYCLES) -> dict:
    from legacy_canvas import Canvas as LegacyCanvas  # noqa: F401
    if impl == 'legacy':
        from legacy_canvas import Canvas
    else:
        from adcode.tui.canvas import Canvas  # noqa: F401
    palette = build_palette('dark', '#8fa7d6')

    results: dict[str, dict[str, float | int]] = {}
    for name in ('full', 'idle', 'tick'):
        # warm up
        for _ in range(5):
            canvas = build_scene(Canvas, palette)
            render = (canvas.render if impl == 'legacy'
                      else canvas.render_diff)
            if name == 'full':
                render()
            elif name == 'idle':
                render()
                render()
            else:
                render()
                canvas.draw_text(4, HEIGHT - 3, 'x', palette.text)
                render()
        samples: list[float] = []
        payload = 0
        for _ in range(cycles):
            canvas = build_scene(Canvas, palette)
            render = (canvas.render if impl == 'legacy'
                      else canvas.render_diff)
            if name == 'full':
                started = time.perf_counter()
                text = render()
                samples.append(time.perf_counter() - started)
            elif name == 'idle':
                render()                     # baseline frame, not measured
                started = time.perf_counter()
                text = render()
                samples.append(time.perf_counter() - started)
            else:
                render()                     # baseline frame
                canvas.draw_text(4, HEIGHT - 3, 'x', palette.text)
                started = time.perf_counter()
                text = render()
                samples.append(time.perf_counter() - started)
            payload = len(text)
        results[name] = {
            'median_ms': round(statistics.median(samples) * 1000, 3),
            'min_ms': round(min(samples) * 1000, 3),
            'bytes': payload,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy', action='store_true',
                        help='benchmark the pre-refactor canvas snapshot')
    parser.add_argument('--cycles', type=int, default=CYCLES)
    parser.add_argument('--json', type=Path, default=None)
    args = parser.parse_args()

    impl = 'legacy' if args.legacy else 'current'
    results = measure(impl, args.cycles)
    header = f'{impl}  {WIDTH}x{HEIGHT}  n={args.cycles}'
    print(header)
    print('-' * len(header))
    for name, row in results.items():
        print(f'{name:<6} median {row["median_ms"]:>7.3f} ms   '
              f'min {row["min_ms"]:>7.3f} ms   bytes {row["bytes"]:>6}')
    if args.json:
        args.json.write_text(json.dumps({'impl': impl, 'results': results},
                                        indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
