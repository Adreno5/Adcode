"""Per-frame cost of the live streaming answer path (answer grows every frame)."""
from __future__ import annotations
import statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adcode.config import AppConfig
from adcode.theme import build_palette
from adcode.tui.markdown import MarkdownRenderer
from adcode.tui.transcript import TranscriptRenderer, LiveRender

pal = build_palette('dark')
cfg = AppConfig()
PARA = ('Paragraph with **bold**, `code` and a link [x](https://example.com) plus '
        'enough words to wrap past the viewport width.\n\n')

def measure(total_chars: int, width=120, step=24):
    r = TranscriptRenderer()
    live = LiveRender(phase='answering')
    md = MarkdownRenderer(width)
    times = []
    source = (PARA * (total_chars // len(PARA) + 2))[:total_chars]
    fed = 0
    while fed < len(source):
        chunk = source[fed:fed + step]
        md.feed(chunk)
        fed += len(chunk)
        live.answer_lines = md.lines()
        t0 = time.perf_counter()
        r._live_lines(live, width, cfg, time.time())
        times.append((time.perf_counter() - t0) * 1000)
    times = times[len(times) // 2:]  # steady state, mid-stream
    return fed, len(live.answer_lines), statistics.median(times), max(times)

print(f'{"answer chars":>13} {"md lines":>9} {"median":>9} {"max":>8}')
for n in (2000, 8000, 20000, 40000):
    chars, lines, med, mx = measure(n)
    print(f'{chars:13d} {lines:9d} {med:8.3f}m {mx:7.3f}m')
