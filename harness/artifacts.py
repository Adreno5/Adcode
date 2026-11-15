"""Artifact collection for manual harness runs.

A run writes a self-contained directory:

    out/
      manifest.json
      report.md
      transcript.json
      gateway_requests.json
      config.json
      session.json
      screens/
        001-idle.json      full cell dump
        001-idle.svg       dependency-free vector screenshot
        001-idle.png       raster screenshot (when Pillow is available)
        001-idle.txt       plain text as a terminal would show it
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.screen import FrameSnapshot
from harness.screenshots import render_png, svg_from_payload

_SLUG = re.compile(r'[^A-Za-z0-9_.-]+')


def _slug(value: str) -> str:
    cleaned = _SLUG.sub('-', value.strip()).strip('-').lower()
    return cleaned[:48] or 'frame'


class ArtifactWriter:
    def __init__(self, out_dir: str | Path, *, png: bool = True,
                 raster_python: str | None = None) -> None:
        self.out_dir = Path(out_dir)
        self.screens_dir = self.out_dir / 'screens'
        self.screens_dir.mkdir(parents=True, exist_ok=True)
        self.png = png
        self.raster_python = raster_python
        self.frames: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self.started_at = datetime.now(timezone.utc).isoformat(
            timespec='seconds')
        self.scenario = ''
        self.metadata: dict[str, Any] = {}

    # ── frames ─────────────────────────────────────────────
    def frame(self, label: str, snapshot: FrameSnapshot) -> dict[str, Any]:
        index = len(self.frames) + 1
        base = f'{index:03d}-{_slug(label)}'
        payload = snapshot.to_payload(label)
        json_path = self.screens_dir / f'{base}.json'
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        text_path = self.screens_dir / f'{base}.txt'
        text_path.write_text(snapshot.text() + '\n', encoding='utf-8')
        svg_path = self.screens_dir / f'{base}.svg'
        svg_from_payload(payload, svg_path, label)
        png_path: Path | None = None
        if self.png:
            try:
                png_path = render_png(json_path, self.screens_dir / f'{base}.png',
                                      self.raster_python, label)
            except Exception as error:
                message = f'PNG rasterization skipped for {base}: {error}'
                if message not in self.warnings:
                    self.warnings.append(message)
        row = {
            'index': index,
            'label': label,
            'width': snapshot.width,
            'height': snapshot.height,
            'cursor': list(snapshot.cursor) if snapshot.cursor else None,
            'json': json_path.relative_to(self.out_dir).as_posix(),
            'text': text_path.relative_to(self.out_dir).as_posix(),
            'svg': svg_path.relative_to(self.out_dir).as_posix(),
            'png': (png_path.relative_to(self.out_dir).as_posix()
                    if png_path is not None else None),
            'nonempty_lines': len(snapshot.nonempty_lines()),
            'vt_mismatch': snapshot.vt_mismatch,
        }
        self.frames.append(row)
        return row

    # ── data files ─────────────────────────────────────────
    def json(self, name: str, payload: Any) -> Path:
        path = self.out_dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding='utf-8')
        return path

    def text(self, name: str, content: str) -> Path:
        path = self.out_dir / name
        path.write_text(content, encoding='utf-8')
        return path

    # ── report ─────────────────────────────────────────────
    def finish(self, harness: Any, scenario: str,
               summary: str = '') -> Path:
        self.scenario = scenario
        payload = harness.artifact_payload()
        self.json('transcript.json', payload['transcript'])
        self.json('gateway_requests.json', payload['gateway_requests'])
        self.json('config.json', payload['config'])
        self.json('session.json', payload['session'])
        workspace_artifact = self.out_dir / 'workspace'
        try:
            shutil.copytree(harness.workspace.project, workspace_artifact,
                            dirs_exist_ok=True)
        except OSError as error:
            message = f'workspace copy skipped: {error}'
            if message not in self.warnings:
                self.warnings.append(message)
        manifest = {
            'scenario': scenario,
            'summary': summary,
            'started_at': self.started_at,
            'finished_at': datetime.now(timezone.utc).isoformat(
                timespec='seconds'),
            'python': sys.version,
            'executable': sys.executable,
            'platform': platform.platform(),
            'cwd': os.getcwd(),
            'width': harness.screen.columns,
            'height': harness.screen.rows_count,
            'wire': harness.wire,
            'model': harness.model_id,
            'state': payload['scenario_state'],
            'frames': self.frames,
            'gateway_requests': len(payload['gateway_requests']),
            'transcript_items': len(payload['transcript']),
            'warnings': self.warnings,
            'workspace_root': str(harness.workspace.root),
            'workspace_artifact': 'workspace',
            'config_path': str(harness.workspace.config_path),
        }
        self.json('manifest.json', manifest)
        self.text('report.md', self._report(manifest, payload))
        return self.out_dir

    def _report(self, manifest: dict[str, Any], payload: dict[str, Any]) -> str:
        state = payload.get('scenario_state') or {}
        lines = [
            f'# AdCode harness · {manifest["scenario"]}',
            '',
            manifest.get('summary') or '',
            '',
            '## Run',
            '',
            f'- wire: `{manifest["wire"]}`',
            f'- model: `{manifest["model"]}`',
            f'- screen: {manifest["width"]}x{manifest["height"]}',
            f'- gateway requests: {manifest["gateway_requests"]}',
            f'- transcript items: {manifest["transcript_items"]}',
            f'- phase at end: `{state.get("phase", "")}`',
            f'- session: `{state.get("session", "")}`',
            f'- temp workspace: `{manifest["workspace_root"]}`',
            f'- workspace copy: `{manifest.get("workspace_artifact", "")}`',
            '',
            '## Screenshots',
            '',
        ]
        for row in self.frames:
            image = row.get('png') or row.get('svg')
            lines.append(
                f'### {row["index"]:03d} · {row["label"]} '
                f'({row["width"]}x{row["height"]})')
            lines.append('')
            if image:
                lines.append(f'![{row["label"]}]({image})')
                lines.append('')
            lines.append(f'Text: `{row["text"]}` · cells: `{row["json"]}`')
            lines.append('')
        lines.extend(['## Gateway requests', ''])
        for request in payload.get('gateway_requests') or []:
            tools = ', '.join(request.get('tools') or [])[:400]
            lines.append(
                f'- `#{request["index"]}` {request["wire"]} · '
                f'model `{request["model"]}` · effort '
                f'`{request.get("reasoning_effort") or "-"}`')
            lines.append(f'  - tools: {tools}')
            first = (request.get('system') or '').splitlines()[0:2]
            if first:
                lines.append(f'  - system: {" / ".join(first)[:240]}')
        lines.extend(['', '## Transcript', ''])
        for item in payload.get('transcript') or []:
            kind = item.get('kind', '')
            text = str(item.get('text') or item.get('display')
                       or item.get('name') or '').replace('\n', ' ')[:220]
            status = item.get('status') or item.get('level') or ''
            lines.append(f'- `{kind}` {status} · {text}')
        if self.warnings:
            lines.extend(['', '## Warnings', ''])
            lines.extend(f'- {warning}' for warning in self.warnings)
        lines.append('')
        return '\n'.join(lines)
