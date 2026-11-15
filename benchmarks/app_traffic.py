"""Terminal traffic of a real scripted chat run, old renderer vs new.

This drives the production app on the simulated terminal with a scripted model
reply and counts every byte of frame payload the app flushed.  Two runs of the
same interaction are compared: one forced onto the pre-refactor canvas
snapshot, one on the current canvas.

    python benchmarks/app_traffic.py [--legacy] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'benchmarks'))

import adcode.tui.app as app_module  # noqa: E402
from harness.app import AppHarness  # noqa: E402
from harness.script import ScriptedResponder, Turn  # noqa: E402
import legacy_canvas  # noqa: E402


class LegacyCanvas(legacy_canvas.Canvas):
    """Legacy renderer with the new flush API bolted on."""

    def render_diff(self) -> str:
        return self.render()

ANSWER = (
    'Here is the plan.\n\n'
    '1. Split the canvas into a glyph plane and a surface plane.\n'
    '2. Track a frame stamp so clearing is a counter bump.\n'
    '3. Flush only the rows whose serialized text changed.\n\n'
    'Wide glyphs like 中文测试 stay intact because a dirty row is always '
    'repainted from its first cell.\n'
)


def run(*, legacy: bool) -> dict:
    original = app_module.Canvas
    if legacy:
        app_module.Canvas = LegacyCanvas
    harness = AppHarness(
        responder=ScriptedResponder([Turn(
            thinking='Plan the refactor in a few steps.',
            text=ANSWER,
            chunk_size=12,
            chunk_delay=0.02,
        )]),
        width=120,
        height=40,
    )
    try:
        harness.start()
        harness.wait_rendered()
        startup = sum(len(frame.raw) for frame in harness.screen.frames)
        startup_frames = len(harness.screen.frames)

        harness.submit('refactor the renderer')
        harness.wait_idle(30.0)
        harness.wait_for_text('Here is the plan.')

        total = sum(len(frame.raw) for frame in harness.screen.frames)
        frames = len(harness.screen.frames)
        return {
            'impl': 'legacy' if legacy else 'current',
            'startup_bytes': startup,
            'startup_frames': startup_frames,
            'total_bytes': total,
            'frames': frames,
            'bytes_per_frame': round(total / max(1, frames), 1),
        }
    finally:
        harness.close()
        app_module.Canvas = original


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy', action='store_true')
    parser.add_argument('--json', type=Path, default=None)
    args = parser.parse_args()
    result = run(legacy=args.legacy)
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
