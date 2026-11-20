"""The recorded tour of AdCode that sits at the top of the README.

The demo is not a mock-up: it drives the real application against the fake
local gateway, exactly like the other harness scenarios.  A sampler thread
reads the composited cell grid at a fixed rate while the scripted story plays,
so the result is a real screen recording of the TUI -- streaming, panels,
dialogs and all.

Generate the assets with::

    python -m bench demo --out docs

That writes ``docs/demo.gif``, ``docs/demo.webm`` and a poster frame.  The
GIF and the video are assembled with ffmpeg; Pillow renders the cell dumps.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from bench.app import AppHarness
from bench.script import ModelRequest, ToolCall, Turn, TurnUsage

# ── the story ──────────────────────────────────────────────

FPS = 8
WIDTH = 104
HEIGHT = 32

GREET_SOURCE = '''"""greet.py -- a tiny greeting CLI."""

from __future__ import annotations

import argparse


def greeting(name: str, *, excited: bool = False) -> str:
    text = f"Hello, {name}!"
    return text + " Nice to meet you." if excited else text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Say hello")
    parser.add_argument("name", nargs="?", default="world")
    parser.add_argument("-e", "--excited", action="store_true")
    args = parser.parse_args(argv)
    print(greeting(args.name, excited=args.excited))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

TEST_RUNNER = '''"""A fake test runner: prints a climbing percentage for the footer."""

import time

for pct in (8, 19, 31, 44, 58, 69, 81, 92, 100):
    print(f"running the suite ... {pct}%", flush=True)
    time.sleep(0.32)
print("42 passed")
'''


def _todo(action: str, **args: Any) -> ToolCall:
    return ToolCall(name='todo', args={'action': action, **args},
                    id=f'call_todo_{action}_{len(args)}')


PLAN = [
    {'title': 'Inspect the project layout'},
    {'title': 'Write greet.py with a small CLI'},
    {'title': 'Run the CLI and check the output'},
    {'title': 'Ask scout to review the change'},
]


class DemoResponder:
    """Root and child turns for the tour, routed by role.

    A small state machine keeps the script readable: each root request advances
    one step, while the child answers every request with a held-open review so
    the sub-agent strip stays on screen long enough to be inspected.
    """

    def __init__(self, hold: threading.Event) -> None:
        self.hold = hold
        self.parent_step = 0

    # ── root ────────────────────────────────────────────────
    def _parent(self, request: ModelRequest) -> Turn:
        step = self.parent_step
        self.parent_step += 1
        if step == 0:
            return Turn(
                thinking=('The user wants a small, real change.  I should plan '
                          'it first, then look at the project before writing '
                          'anything.  A review by a sub-agent is cheap, so I '
                          'will delegate that after the tests pass.'),
                tool_calls=[_todo('create', items=PLAN)],
                usage=TurnUsage(1_284, 96),
            )
        if step == 1:
            return Turn(
                thinking='Let me see what is already here.',
                tool_calls=[
                    ToolCall(name='glob', args={'pattern': '**/*.py'},
                             id='call_glob_1'),
                    ToolCall(name='read', args={'path': 'README.md'},
                             id='call_read_1'),
                ],
                usage=TurnUsage(1_402, 84),
            )
        if step == 2:
            return Turn(
                thinking=('The project is empty apart from the README, so the '
                          'CLI is a new file.  I will keep it dependency-free '
                          'and add an argparse entry point.'),
                tool_calls=[
                    _todo('update', id='item-00', status='completed'),
                    _todo('update', id='item-01', status='in_progress'),
                    ToolCall(name='write',
                             args={'path': 'greet.py', 'content': GREET_SOURCE},
                             id='call_write_1'),
                ],
                usage=TurnUsage(1_610, 120),
            )
        if step == 3:
            return Turn(
                thinking='Now run it both ways so the output is checked.',
                tool_calls=[
                    _todo('update', id='item-01', status='completed'),
                    _todo('update', id='item-02', status='in_progress'),
                    ToolCall(name='bash',
                             args={'command': 'python greet.py Ada && '
                                              'python greet.py Ada --excited'},
                             id='call_bash_1'),
                ],
                usage=TurnUsage(1_830, 70),
            )
        if step == 4:
            return Turn(
                thinking=('The output is right.  Hand the file to scout for a '
                          'review while I finish the plan.'),
                tool_calls=[
                    _todo('update', id='item-02', status='completed'),
                    _todo('update', id='item-03', status='in_progress'),
                    ToolCall(name='spawn_agent',
                             args={'task_name': 'scout',
                                   'message': 'Review greet.py for correctness '
                                              'and style.  Report anything '
                                              'that would surprise a user.'},
                             id='call_spawn_1'),
                ],
                usage=TurnUsage(2_040, 88),
            )
        if step == 5:
            return Turn(
                text=('I added `greet.py`: a small argparse CLI with a pure '
                      '`greeting()` helper, so the formatting is testable.\n\n'
                      '```text\n$ python greet.py Ada\nHello, Ada!\n```\n\n'
                      'It handles the default name and `--excited` too.  '
                      '`scout` is reviewing the file now; the plan and the '
                      'sub-agent strip above the prompt show the state.'),
                usage=TurnUsage(2_196, 140),
            )
        if step == 6:
            return Turn(
                thinking=('scout came back clean.  Close the review item and '
                          'the plan, then report.'),
                tool_calls=[
                    # completing the last item closes the plan and the panel
                    _todo('update', id='item-03', status='completed',
                          result='scout: nothing blocking'),
                ],
                usage=TurnUsage(2_220, 58),
            )
        if step == 7:
            return Turn(
                text=('The plan is closed: all four items are done and scout '
                      'reported nothing blocking.'),
                usage=TurnUsage(2_230, 34),
            )
        if step == 8:
            return Turn(
                thinking='A progress report makes the footer easy to see.',
                tool_calls=[ToolCall(
                    name='bash',
                    args={'command': 'python run_tests.py'},
                    id='call_bash_2')],
                usage=TurnUsage(2_240, 40),
            )
        if step == 9:
            return Turn(
                text=('The suite is green: **42 passed**.  The footer turned '
                      'into a progress bar while the command ran, then into '
                      'the finished state.'),
                usage=TurnUsage(2_310, 46),
            )
        if step == 10:
            return Turn(
                thinking='The attachment is a small chart; describe what is '
                         'visible and keep it short.',
                text=('The screenshot is a small bar chart: five grey bars '
                      'with the tallest one highlighted in the accent colour.'),
                usage=TurnUsage(2_520, 38),
            )
        if step == 11:
            return Turn(
                thinking=('This turn is deliberately long so the recording can '
                          'show the reasoning stream and then interrupt it.'),
                text=('This answer should never be committed: the user is '
                      'about to press Esc.'),
                hold=self.hold,
                usage=TurnUsage(2_600, 40),
            )
        return Turn(text='Demo complete.')

    # ── child ───────────────────────────────────────────────
    def _child(self, request: ModelRequest) -> Turn:
        return Turn(
            thinking=('Reading greet.py and checking the edges: empty name, '
                      'exit code, the excited branch.  The helper is pure, '
                      'which is the right shape.'),
            text=('scout: greet.py looks good.  `greeting()` is pure and the '
                  'CLI exits 0 on both paths.  One nit: the docstring could '
                  'name the default name.  Nothing blocking.'),
            hold=self.hold,
            usage=TurnUsage(1_120, 64),
        )

    def respond(self, request: ModelRequest) -> Turn:
        if 'role: sub-agent' in request.system_text:
            return self._child(request)
        return self._parent(request)


# ── driving the app ────────────────────────────────────────

def beat(seconds: float) -> None:
    time.sleep(seconds)


def type_slow(harness: AppHarness, text: str, delay: float = 0.035) -> None:
    for char in text:
        harness.type_text(char)
        time.sleep(delay)


def configure_demo(config: Any) -> None:
    """The settings the tour uses: agents on, details visible, dark theme."""
    config.agents.enabled = True
    config.agents.team_enabled = True
    config.agents.max_concurrent = 6
    config.thinking.mode = 'full'
    config.thinking.style = 'normal'
    config.thinking.after_completion = 'keep'
    config.tools.mode = 'both'
    config.tools.chain = 'separate'
    config.tools.result_mode = 'partial'
    config.tools.show_args = True
    config.tools.show_diff = True
    config.ui.theme = 'dark'
    config.select(config.providers[0].name, config.providers[0].models[0].id)
    config.providers[0].models[0].vision = True


def _git_init(harness: AppHarness) -> None:
    project = harness.workspace.project
    try:
        subprocess.run(['git', 'init', '-q'], cwd=project, check=True,
                       capture_output=True)
        subprocess.run(['git', 'add', '-A'], cwd=project, check=True,
                       capture_output=True)
        subprocess.run(['git', '-c', 'user.email=demo@adcode',
                        '-c', 'user.name=AdCode', 'commit', '-qm', 'initial'],
                       cwd=project, check=True, capture_output=True)
    except (OSError, subprocess.SubprocessError):
        pass


def _make_image(path: Path) -> None:
    """A tiny PNG for the image-attachment scene (no Pillow needed)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    image = Image.new('RGB', (420, 260), (24, 26, 32))
    draw = ImageDraw.Draw(image)
    for index, height in enumerate((70, 120, 180, 96, 140)):
        left = 40 + index * 70
        draw.rectangle([left, 230 - height, left + 42, 230],
                       fill=(232, 234, 240) if index != 2 else (110, 168, 254))
    draw.text((40, 20), 'requests / day', fill=(160, 168, 184))
    image.save(path)


def _wait_text(harness: AppHarness, needle: str, timeout: float = 20.0) -> None:
    harness.wait_for_text(needle, timeout=timeout)


def record(out_dir: str | Path, *, fps: int = FPS) -> Path:
    """Play the tour and write the raw cell frames; return the frame directory."""
    out_dir = Path(out_dir)
    frames_dir = out_dir / 'demo-frames'
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    hold = threading.Event()   # clear: the child review stays mid-stream
    responder = DemoResponder(hold)
    harness = AppHarness(width=WIDTH, height=HEIGHT, responder=responder,
                         config_mutator=configure_demo,
                         project_files={
                             'README.md':
                                 '# Greeter\n\nA tiny greeting CLI.\n',
                             'run_tests.py': TEST_RUNNER,
                         })
    harness.start()
    _make_image(harness.workspace.project / 'shot.png')
    _git_init(harness)

    stop = threading.Event()

    def sampler() -> None:
        # A fixed-rate sampler, not a paint hook: the app only paints dirty
        # frames, but a recording needs a constant frame rate so the holds in
        # the story keep their real duration.
        index = 0
        while not stop.is_set():
            frame = harness.screen.latest
            if frame is not None:
                payload = frame.to_payload('demo')
                (frames_dir / f'{index:05d}.json').write_text(
                    json.dumps(payload, separators=(',', ':')),
                    encoding='utf-8')
                index += 1
            time.sleep(1.0 / fps)

    thread = threading.Thread(target=sampler, name='adcode-demo-sampler',
                              daemon=True)
    thread.start()
    try:
        _script(harness, responder, hold)
    finally:
        beat(0.6)
        stop.set()
        thread.join(timeout=10.0)
        harness.close()
    return frames_dir


def _script(harness: AppHarness, responder: DemoResponder,
            hold: threading.Event) -> None:
    harness.wait_idle(20.0)
    beat(0.8)

    # 1. plan, write, run -------------------------------------------------
    type_slow(harness, 'Add a small greeting CLI and have scout review it.',
              delay=0.028)
    beat(0.5)
    harness.press('enter')
    harness.wait_until(lambda: responder.parent_step >= 2, timeout=20.0,
                       description='the plan and the first tool round')
    beat(1.2)

    harness.wait_until(lambda: responder.parent_step >= 4, timeout=30.0,
                       description='the write and the run')
    beat(1.4)
    harness.wait_until(lambda: responder.parent_step >= 5, timeout=30.0,
                       description='the scout to be spawned')
    beat(1.8)
    harness.wait_idle(30.0)
    _wait_text(harness, 'Hello, Ada!')
    beat(1.4)

    # 2. release the review, then read its full output --------------------
    hold.set()
    harness.wait_until(
        lambda: harness.app is not None and harness.app.loop is not None and
        all(row.get('status') not in ('running', 'pending_init')
            for row in harness.app.loop.agent_view()
            if row.get('name') != 'root'),
        timeout=20.0, description='scout to report back')
    harness.wait_idle(20.0)
    beat(0.8)
    harness.press('ctrl+g')
    beat(1.2)
    harness.press('enter')
    beat(1.8)
    harness.press('esc')
    beat(0.7)

    # 3. close the plan ---------------------------------------------------
    harness.submit('Mark the review done and wrap up.', expect_requests=1,
                   timeout=30.0)
    harness.wait_until(lambda: responder.parent_step >= 8, timeout=30.0,
                       description='the closed plan')
    harness.wait_idle(30.0)
    _wait_text(harness, 'plan is closed')
    beat(1.2)

    # 4. /agents list -----------------------------------------------------
    harness.run_command('/agents list')
    beat(1.4)
    harness.press('esc')
    beat(0.5)

    # 5. the diff and the scrollback --------------------------------------
    harness.run_command('/diff')
    beat(1.8)
    harness.wheel('wheel_up')
    harness.wheel('wheel_up')
    beat(1.0)
    harness.wheel('wheel_down')
    harness.wheel('wheel_down')
    beat(0.8)

    # 6. the command palette, the settings browser and a theme change -----
    type_slow(harness, '/th', delay=0.06)
    beat(0.9)
    harness.press('tab')
    type_slow(harness, 'full', delay=0.05)
    beat(0.5)
    harness.press('enter')
    beat(1.2)
    harness.run_command('/config')
    beat(1.2)
    harness.press('down')
    harness.press('enter')
    harness.press('down', 'down')
    beat(0.6)
    harness.press('enter')
    beat(1.6)
    harness.press('ctrl+s')
    beat(1.0)

    # 7. sessions ---------------------------------------------------------
    harness.run_command('/sessions')
    beat(1.4)
    harness.press('esc')
    beat(0.5)

    # 8. the footer progress bar and the done state -----------------------
    type_slow(harness, 'Run the test suite and report the result.', delay=0.02)
    beat(0.4)
    harness.press('enter')
    harness.wait_until(lambda: responder.parent_step >= 9, timeout=40.0,
                       description='the progress command')
    beat(0.9)
    harness.wait_idle(30.0)
    _wait_text(harness, '42 passed')
    beat(1.6)

    # 9. an attached image ------------------------------------------------
    harness.run_command('/image shot.png')
    beat(1.0)
    type_slow(harness, 'Describe the attached screenshot.', delay=0.025)
    beat(0.4)
    harness.press('enter')
    harness.wait_until(lambda: responder.parent_step >= 11, timeout=20.0,
                       description='the image answer')
    harness.wait_idle(20.0)
    beat(1.4)

    # 10. interrupt a held stream ----------------------------------------
    hold.clear()
    type_slow(harness, 'Analyse the whole repository.', delay=0.025)
    beat(0.4)
    harness.press('enter')
    _wait_text(harness, 'deliberately long')
    beat(1.7)
    harness.press('esc')
    beat(1.0)
    hold.set()
    harness.wait_idle(20.0)
    _wait_text(harness, '[interrupted]')
    beat(1.8)


# ── turning frames into a recording ────────────────────────

def _raster_python(explicit: str | None = None) -> str:
    from bench.screenshots import find_raster_python
    interpreter = find_raster_python(explicit)
    if interpreter is None:
        raise RuntimeError('no interpreter with Pillow found; set '
                           'ADCODE_RASTER_PYTHON')
    return interpreter


RENDERER = r'''
import json, sys
from pathlib import Path
from bench.screenshots import png_from_payload

source, target = Path(sys.argv[1]), Path(sys.argv[2])
width, height = int(sys.argv[3]), int(sys.argv[4])
blank = [' ', [255, 255, 255], [17, 19, 24], 0, False]


def pad(payload):
    """A resize scene changes the grid; every PNG must have one size."""
    rows = [list(row)[:width] + [blank] * max(0, width - len(row))
            for row in payload.get('rows', [])[:height]]
    rows += [[blank] * width for _ in range(height - len(rows))]
    return {**payload, 'width': width, 'height': height, 'rows': rows}


target.mkdir(parents=True, exist_ok=True)
for index, dump in enumerate(sorted(source.glob('*.json'))):
    payload = json.loads(dump.read_text(encoding='utf-8'))
    png_from_payload(pad(payload), target / f'{index:05d}.png', 'AdCode',
                     font_size=14, scale=1.0)
print(len(list(target.glob('*.png'))))
'''


def render_frames(frames_dir: Path, images_dir: Path,
                  raster_python: str | None = None) -> int:
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    dumps = sorted(frames_dir.glob('*.json'))
    if not dumps:
        raise RuntimeError(f'no frames were recorded in {frames_dir}')
    sizes = []
    for dump in dumps:
        payload = json.loads(dump.read_text(encoding='utf-8'))
        sizes.append((int(payload.get('width') or 0),
                      int(payload.get('height') or 0)))
    width = max(size[0] for size in sizes)
    height = max(size[1] for size in sizes)
    python = _raster_python(raster_python)
    result = subprocess.run(
        [python, '-c', RENDERER, str(frames_dir), str(images_dir),
         str(width), str(height)],
        check=True, capture_output=True, text=True)
    return int(result.stdout.strip() or 0)


def _ffmpeg() -> str:
    executable = shutil.which('ffmpeg')
    if executable is None:
        raise RuntimeError('ffmpeg is required to assemble the demo')
    return executable


def assemble(images_dir: Path, out_path: Path, *, fps: int = FPS,
             width: int = 960) -> Path:
    """Build the GIF in two passes: palette first, then dither with it.

    The single-graph ``split``+``palettegen``+``paletteuse`` recipe loses
    frames on some ffmpeg builds; the two-pass form is deterministic and
    still small.
    """
    ffmpeg = _ffmpeg()
    palette = images_dir / 'palette.png'
    scaled = f'fps={fps},scale={width}:-1:flags=lanczos'
    subprocess.run(
        [ffmpeg, '-y', '-loglevel', 'error', '-framerate', str(fps),
         '-i', str(images_dir / '%05d.png'),
         '-vf', f'{scaled},palettegen=max_colors=96:stats_mode=diff',
         str(palette)], check=True)
    try:
        subprocess.run(
            [ffmpeg, '-y', '-loglevel', 'error', '-framerate', str(fps),
             '-i', str(images_dir / '%05d.png'), '-i', str(palette),
             '-lavfi',
             f'{scaled}[x];[x][1:v]paletteuse=dither=sierra2_4a:'
             'diff_mode=rectangle',
             '-loop', '0', str(out_path)], check=True)
    finally:
        palette.unlink(missing_ok=True)
    return out_path


def assemble_video(images_dir: Path, out_path: Path, *, fps: int = FPS,
                   width: int = 1280) -> Path:
    """A small VP9/WebM copy for browsers (libx264 is not always available)."""
    ffmpeg = _ffmpeg()
    subprocess.run(
        [ffmpeg, '-y', '-loglevel', 'error', '-framerate', str(fps),
         '-i', str(images_dir / '%05d.png'),
         '-vf', f'fps={fps},scale={width}:-1:flags=lanczos',
         '-c:v', 'libvpx-vp9', '-b:v', '0', '-crf', '34',
         '-pix_fmt', 'yuv420p', str(out_path)], check=True)
    return out_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog='python -m bench demo')
    parser.add_argument('--out', default='docs', help='asset directory')
    parser.add_argument('--fps', type=int, default=FPS)
    parser.add_argument('--keep-frames', action='store_true')
    parser.add_argument('--raster-python', default=None)
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = record(out_dir, fps=args.fps)
    images = out_dir / 'demo-images'
    count = render_frames(frames, images, args.raster_python)
    print(f'{count} frames rendered')
    gif = assemble(images, out_dir / 'demo.gif', fps=args.fps)
    print(f'wrote {gif} ({gif.stat().st_size / 1e6:.1f} MB)')
    poster = images / '00000.png'
    if poster.exists():
        shutil.copyfile(poster, out_dir / 'demo-poster.png')
    if not args.no_video:
        video = assemble_video(images, out_dir / 'demo.webm', fps=args.fps)
        print(f'wrote {video} ({video.stat().st_size / 1e6:.1f} MB)')
    if not args.keep_frames:
        shutil.rmtree(frames, ignore_errors=True)
        shutil.rmtree(images, ignore_errors=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
