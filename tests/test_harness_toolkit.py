"""Tests for the harness toolkit itself (VT replay, input, screenshots, CLI)."""

from __future__ import annotations

import json
import queue

import pytest

from bench.artifacts import ArtifactWriter
from bench.cli import main as cli_main
from bench.input import ScriptedInput, key_sequence, mouse_sequence
from bench.screenshots import render_png, svg_from_payload
from bench.vt import VirtualScreen


def drain(events: 'queue.Queue') -> list:
    items = []
    while True:
        try:
            items.append(events.get_nowait())
        except queue.Empty:
            return items


def test_input_driver_builds_real_terminal_sequences():
    events: 'queue.Queue' = queue.Queue()
    driver = ScriptedInput(events)

    driver.type_text('ab')
    driver.press('left')
    driver.press('ctrl+c')
    driver.press('alt+enter')
    driver.press('shift+enter')
    driver.wheel('wheel_up', 7, 9)
    driver.paste('one\ntwo')

    delivered = drain(events)
    assert [event.kind for event in delivered[:2]] == ['text', 'text']
    assert delivered[2].key == 'left'
    assert delivered[3].key == 'ctrl+c'
    assert delivered[4].key == 'alt+enter' and delivered[4].alt
    assert delivered[5].key == 'shift+enter' and delivered[5].shift
    mouse = next(event for event in delivered if event.kind == 'mouse')
    assert mouse.action == 'wheel_up' and (mouse.x, mouse.y) == (7, 9)
    paste = next(event for event in delivered if event.kind == 'paste')
    assert paste.text == 'one\ntwo'


def test_key_and_mouse_sequence_builders():
    assert key_sequence('ctrl+a') == '\x01'
    assert key_sequence('up') == '\x1b[A'
    assert key_sequence('ctrl+home') == '\x1b[1;5H'
    # A chord on punctuation has no control byte; it uses the kitty
    # keyboard protocol.
    assert key_sequence('ctrl+,') == '\x1b[44;5u'
    assert mouse_sequence('wheel_down', 3, 4) == '\x1b[<65;3;4M'
    assert mouse_sequence('release', 3, 4) == '\x1b[<0;3;4m'


def test_virtual_screen_replays_the_serialized_frames(harness):
    harness.type_text('screen replay')
    harness.wait_prompt('screen replay')
    assert len(harness.screen.frames) >= 2
    for frame in harness.screen.frames:
        assert not frame.vt_mismatch, (
            'the ANSI frame and the canvas disagree at frame '
            f'{frame.index}')
        assert frame.vt_text == frame.text()


def test_vt_screen_cursor_and_wide_glyph_handling():
    screen = VirtualScreen(10, 3)
    screen.feed('\x1b[Hhello\r\n\x1b[38;2;1;2;3m世界\x1b[0m')
    assert screen.line(0) == 'hello'
    assert screen.line(1) == '世界'
    assert screen.cursor() == (4, 1)


def test_screenshot_svg_and_png(tmp_path, harness):
    harness.type_text('screenshot')
    harness.wait_prompt('screenshot')
    frame = harness.screen.latest
    assert frame is not None
    payload = frame.to_payload('unit-test')

    svg_path = svg_from_payload(payload, tmp_path / 'frame.svg', 'unit-test')
    svg = svg_path.read_text(encoding='utf-8')
    assert svg.startswith('<?xml')
    assert '<svg' in svg and '</svg>' in svg
    assert 'font-family="Consolas' in svg
    # Text is emitted one cell at a time, so assert on the frame payload.
    assert payload['text'].splitlines()[0].startswith(' AdCode')

    dump = tmp_path / 'frame.json'
    dump.write_text(json.dumps(payload), encoding='utf-8')
    try:
        png_path = render_png(dump, tmp_path / 'frame.png', label='unit-test')
    except RuntimeError as error:
        pytest.skip(str(error))
    assert png_path.exists()
    assert png_path.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'


def test_png_renderer_reports_missing_pillow(monkeypatch, tmp_path):
    def missing(*args, **kwargs):
        raise ImportError('no pillow')

    monkeypatch.setattr('bench.screenshots.png_from_payload', missing)
    monkeypatch.setattr('bench.screenshots.find_raster_python',
                        lambda explicit=None: None)
    dump = tmp_path / 'frame.json'
    dump.write_text(json.dumps({'width': 1, 'height': 1, 'rows': [
        [[' ', [0, 0, 0], [0, 0, 0], 0, False]],
    ]}), encoding='utf-8')
    with pytest.raises(RuntimeError, match='Pillow'):
        render_png(dump, tmp_path / 'frame.png')


def test_artifact_writer_writes_frames_and_report(tmp_path, harness):
    harness.type_text('artifact')
    harness.wait_prompt('artifact')
    frame = harness.screen.latest
    assert frame is not None

    writer = ArtifactWriter(tmp_path / 'run', png=False)
    row = writer.frame('boot', frame)
    assert (tmp_path / 'run' / row['json']).exists()
    assert (tmp_path / 'run' / row['svg']).exists()
    assert (tmp_path / 'run' / row['text']).read_text(encoding='utf-8')

    writer.finish(harness, 'artifact-unit', 'unit summary')
    manifest = json.loads((tmp_path / 'run' / 'manifest.json').read_text(
        encoding='utf-8'))
    assert manifest['scenario'] == 'artifact-unit'
    assert manifest['frames'][0]['label'] == 'boot'
    assert manifest['state']['phase'] == 'idle'
    assert (tmp_path / 'run' / 'transcript.json').exists()
    assert (tmp_path / 'run' / 'gateway_requests.json').exists()
    assert (tmp_path / 'run' / 'workspace' / 'AGENTS.md').exists()
    report = (tmp_path / 'run' / 'report.md').read_text(encoding='utf-8')
    assert '# AdCode harness' in report
    assert 'Screenshots' in report


def test_manual_cli_runs_a_scenario(tmp_path):
    out = tmp_path / 'cli-run'
    code = cli_main([
        'run', 'chat', '--out', str(out), '--no-png',
        '--width', '90', '--height', '26',
    ])
    assert code == 0
    assert (out / 'report.md').exists()
    assert (out / 'manifest.json').exists()
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['scenario'] == 'chat'
    assert manifest['width'] == 90 and manifest['height'] == 26
    assert manifest['gateway_requests'] >= 1
    assert len(manifest['frames']) >= 3
    assert (out / 'screens').is_dir()
    assert (out / 'workspace' / 'AGENTS.md').exists()
