"""Isolated on-disk workspace for a simulated AdCode run.

Every run gets a private temporary directory with:

* ``project/`` -- the working directory AdCode's tools operate on and the
  ``sessions/`` store lives under;
* ``home/.adcode/config.json`` -- the temporary config the app loads;
* ``home/.adcode/AGENTS.md`` -- isolated global instructions;
* project ``AGENTS.md`` -- a marker the gateway request can be checked against.

The real ``~/.adcode/config.json`` is never read or written.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from adcode.config import (
    CONFIG_VERSION, AppConfig, ModelConfig, ProviderConfig,
)

DEFAULT_MODEL = 'harness-model'
GLOBAL_MARKER = 'HARNESS-GLOBAL-INSTRUCTIONS'
PROJECT_MARKER = 'HARNESS-PROJECT-INSTRUCTIONS'


def build_config(base_url: str, wire: str = 'openai chat completions', *,
                 model_id: str = DEFAULT_MODEL, reasoning: bool = True) -> AppConfig:
    """A deterministic config pointed at the fake gateway."""
    config = AppConfig()
    model = ModelConfig(
        id=model_id,
        name='Harness Model',
        context_window=200_000,
        max_output_tokens=16_384,
        reasoning=reasoning,
        reasoning_style='effort' if reasoning else 'none',
        reasoning_efforts=['low', 'medium', 'high'] if reasoning else [],
        tool_call=True,
        source='manual',
    )
    provider = ProviderConfig(
        name='Harness',
        base_url=base_url,
        api_key='harness-key',
        wire=wire,
        models=[model],
    )
    config.providers = [provider]
    config.select(provider.name, model.id)
    config.skills.enabled = False
    config.agents.enabled = False
    config.agents.team_enabled = False
    config.context.auto_compact = False
    config.context.reserved_output = 4_096
    config.max_output_tokens = 4_096
    config.thinking.mode = 'full'
    config.thinking.style = 'normal'
    config.thinking.after_completion = 'keep'
    config.tools.mode = 'both'
    config.tools.result_mode = 'full'
    config.tools.show_args = True
    config.tools.show_diff = True
    config.ui.theme = 'dark'
    config.ui.show_header = True
    config.ui.show_token_status = True
    config.ui.mouse_scroll_lines = 3
    return config


class Workspace:
    """Owns the temporary project/home and the config JSON."""

    def __init__(self, *, root: str | Path | None = None, keep: bool = False,
                 model_id: str = DEFAULT_MODEL) -> None:
        self.keep = keep
        self.root = (Path(root) if root is not None
                     else Path(tempfile.mkdtemp(prefix='adcode-sim-')))
        self.root.mkdir(parents=True, exist_ok=True)
        self.project = self.root / 'project'
        self.home = self.root / 'home'
        self.config_dir = self.home / '.adcode'
        self.project.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.config_dir / 'config.json'
        self.global_instructions_path = self.config_dir / 'AGENTS.md'
        self.model_id = model_id
        self.write_defaults()

    # ── files ──────────────────────────────────────────────
    def write_defaults(self) -> None:
        self.write_project_file(
            'AGENTS.md',
            '# Harness project\n\n'
            f'{PROJECT_MARKER}: the project instructions belong to this run.\n')
        self.write_project_file(
            'README.md',
            '# Simulated project\n\n'
            'This directory belongs to an AdCode harness run.\n')
        self.write_global_instructions(
            f'{GLOBAL_MARKER}: global instructions for the simulated home.\n')

    def write_project_file(self, relative: str, content: str) -> Path:
        path = self.project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return path

    def write_global_instructions(self, content: str) -> Path:
        self.global_instructions_path.write_text(content, encoding='utf-8')
        return self.global_instructions_path

    def write_config(self, config: AppConfig) -> Path:
        payload = asdict(config)
        payload['version'] = CONFIG_VERSION
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        self.config_path.write_text(text, encoding='utf-8')
        return self.config_path

    # ── reads ──────────────────────────────────────────────
    def config_json(self) -> dict[str, Any]:
        try:
            value = json.loads(self.config_path.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    def project_file(self, relative: str) -> str:
        return (self.project / relative).read_text(encoding='utf-8')

    def project_path(self, relative: str) -> Path:
        return self.project / relative

    def session_store(self):
        # the app redirects its state dir into the workspace (see the bench
        # isolation), so the database lives under adcode-home
        from harness.transcript import DbSessionStore

        return DbSessionStore(
            db_path=self.root / 'adcode-home' / 'sessions' / 'adcode.db')

    def session_payload(self, name: str = '__last__') -> dict[str, Any]:
        return self.session_store().load(name).to_json() \
            if self.session_store().load(name) is not None else {}

    def session_files(self) -> list[str]:
        return sorted(row['name'] for row in self.session_store().list())

    # ── lifecycle ──────────────────────────────────────────
    def cleanup(self) -> None:
        if not self.keep:
            shutil.rmtree(self.root, ignore_errors=True)

    def __fspath__(self) -> str:
        return str(self.root)
