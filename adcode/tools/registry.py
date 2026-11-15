"""Tool registry, schema generation and execution."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

from adcode.diff import FileDiff, format_diff_for_model, make_file_diff, snapshot_file
from adcode.tools.builtin import builtin_tools
from adcode.tools.runtime import resolve_path


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]
    read_only: bool = False
    source: str = 'builtin'


@dataclass(frozen=True)
class ToolResult:
    text: str
    file_diff: FileDiff | None = None
    error: bool = False


class ToolRegistry:
    """Owns the live tool set: built-ins plus MCP, skills and agent tools."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._lock = threading.RLock()
        for raw in builtin_tools():
            self.register(ToolSpec(
                name=raw['name'],
                description=raw['description'],
                parameters=raw['parameters'],
                handler=raw['func'],
                read_only=bool(raw.get('read_only')),
                source='builtin',
            ))

    # ── registration ────────────────────────────────────────
    def register(self, spec: ToolSpec) -> None:
        with self._lock:
            self._specs[spec.name] = spec

    def register_many(self, specs: list[ToolSpec]) -> None:
        with self._lock:
            for spec in specs:
                self._specs[spec.name] = spec

    def unregister_source(self, source: str) -> None:
        with self._lock:
            self._specs = {
                name: spec for name, spec in self._specs.items()
                if spec.source != source
            }

    def unregister_source_prefix(self, prefix: str) -> None:
        with self._lock:
            self._specs = {
                name: spec for name, spec in self._specs.items()
                if not spec.source.startswith(prefix)
            }

    def get(self, name: str) -> ToolSpec | None:
        with self._lock:
            return self._specs.get(name)

    def specs(self, allowed: set[str] | None = None) -> list[ToolSpec]:
        with self._lock:
            values = list(self._specs.values())
        if allowed is not None:
            values = [spec for spec in values if spec.name in allowed]
        return sorted(values, key=lambda spec: spec.name)

    def names(self) -> list[str]:
        return [spec.name for spec in self.specs()]

    # ── schemas ─────────────────────────────────────────────
    def schemas(self, wire: str, allowed: set[str] | None = None) -> list[dict[str, Any]]:
        specs = self.specs(allowed)
        if wire == 'anthropic messages':
            return [{
                'name': spec.name,
                'description': spec.description,
                'input_schema': spec.parameters,
            } for spec in specs]
        return [{
            'type': 'function',
            'function': {
                'name': spec.name,
                'description': spec.description,
                'parameters': spec.parameters,
            },
        } for spec in specs]

    # ── execution ───────────────────────────────────────────
    def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        spec = self.get(name)
        if spec is None:
            return ToolResult(f'Error: unknown tool: {name}', error=True)
        # Snapshot the file the tool will actually touch: paths resolve against
        # the tool context's working directory, not the process cwd.  The diff
        # keeps the path the model asked for so prompts stay readable.
        raw_path = str(args.get('path', '')) if name in {'write', 'edit'} else ''
        path = resolve_path(raw_path) if raw_path else ''
        before = snapshot_file(path) if path else None
        try:
            text = spec.handler(args)
        except Exception as error:
            return ToolResult(f'Error executing {name}: {error}', error=True)
        file_diff = None
        if before is not None and path:
            after = snapshot_file(path)
            file_diff = make_file_diff(raw_path, before, after)
        if file_diff is not None and not text.startswith('Error:'):
            text += format_diff_for_model(file_diff)
        return ToolResult(text, file_diff, text.startswith('Error:'))
