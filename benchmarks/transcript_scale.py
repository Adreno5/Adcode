"""Transcript-layer costs the canvas refactor does not touch.

* cold render (config/theme change, session load): full markdown + wrap
* resize: every item's width changes -> cache miss for the whole transcript
* long scrollback: per-frame cost scaling with item count

``--legacy`` swaps the pre-optimisation wrap/truncate/inline scanner back in so
both numbers come from the same script, on the same machine, in one run:

    python benchmarks/transcript_scale.py            # current
    python benchmarks/transcript_scale.py --legacy   # glyph-by-glyph path
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adcode.config import AppConfig
from adcode.session import TranscriptItem
from adcode.theme import build_palette
from adcode.tui import markdown as markdown_module
from adcode.tui import text as text_module
from adcode.tui import transcript as transcript_module
from adcode.tui.transcript import LiveRender, TranscriptRenderer

ANS = ('## Heading\n\nSome **bold** text and `code` here with prose that wraps.\n'
       '- bullet one with a few words\n- bullet two `x = 1`\n\n'
       '```python\ndef f(a, b):\n    return a + b\n```\n\n'
       'Trailing paragraph with [a link](https://example.com) and *emphasis*.\n')


def items(n):
    out = []
    for i in range(n):
        k = ('user', 'assistant', 'thinking', 'tool', 'notice')[i % 5]
        if k == 'assistant':
            out.append(TranscriptItem(kind=k, text=ANS + f'\npara {i}\n'))
        elif k == 'tool':
            out.append(TranscriptItem(kind=k, name='bash', display='bash(pytest)',
                                      args={'command': 'pytest -q'},
                                      output='\n'.join(f'line {j} of output'
                                                       for j in range(15)),
                                      status='done'))
        elif k == 'thinking':
            out.append(TranscriptItem(kind=k, text=f'thinking {i} ' * 6, duration=1.2))
        else:
            out.append(TranscriptItem(kind=k, text=f'item {i} ' * 5, level='info'))
    return out


def install_legacy() -> None:
    """Restore the pre-optimisation implementations used as the baseline."""
    sys.path.insert(0, str(ROOT / 'tests'))
    from test_text_equivalence import (reference_parse_inline,
                                       reference_truncate_line,
                                       reference_wrap_line)
    for module in (text_module, transcript_module):
        module.wrap_line = reference_wrap_line
        module.truncate_line = reference_truncate_line
    markdown_module._parse_inline = reference_parse_inline
    print('legacy (glyph-by-glyph) implementations installed\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy', action='store_true')
    parser.add_argument('--frames', type=int, default=30)
    args = parser.parse_args()
    if args.legacy:
        install_legacy()

    pal = build_palette('dark')
    cfg = AppConfig()
    live = LiveRender(phase='idle')
    label = 'legacy' if args.legacy else 'current'
    print(f'{label}:   cold = first render after a width/config change;'
          f' warm = same width, live idle')
    print(f'{"items":>6} {"cold":>9} {"warm/idle":>10} {"resize":>9} {"lines":>7}')
    for n in (50, 200, 500, 1000):
        renderer = TranscriptRenderer()
        its = items(n)
        now = time.time()
        t0 = time.perf_counter()
        lines = renderer.render(its, live, 120, cfg, 0, now, pal)
        cold = (time.perf_counter() - t0) * 1000
        warm = []
        for f in range(args.frames):
            t0 = time.perf_counter()
            renderer.render(its, live, 120, cfg, 0, now + f * 0.03, pal)
            warm.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        renderer.render(its, live, 118, cfg, 0, now, pal)
        resize = (time.perf_counter() - t0) * 1000
        print(f'{n:6d} {cold:8.2f}m {statistics.median(warm):9.3f}m '
              f'{resize:8.2f}m {len(lines):7d}')


if __name__ == '__main__':
    main()
