"""Skill discovery and loading.

A skill is a directory containing ``SKILL.md`` with YAML frontmatter
(``name`` and ``description``) followed by instructions.  Skills are scanned in
``~/.agents/skills``, the project's ``.agents/skills``, ``.adcode/skills`` and
any configured extra paths.  The model sees a compact preview in its system
prompt and loads the full instructions through the ``skill_load`` tool only
when a skill is relevant.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from adcode.config import SkillsSettings
from adcode.tools.registry import ToolSpec


@dataclass
class Skill:
    name: str
    description: str
    path: Path
    root: Path
    body: str = ''
    metadata: dict[str, str] = field(default_factory=dict)

    def file_tree(self, limit: int = 40) -> list[str]:
        entries: list[str] = []
        try:
            for dirpath, dirnames, filenames in os.walk(self.root):
                dirnames[:] = [name for name in sorted(dirnames)
                               if name not in ('__pycache__', '.git')]
                for filename in sorted(filenames):
                    full = Path(dirpath) / filename
                    relative = full.relative_to(self.root)
                    entries.append(str(relative).replace('\\', '/'))
                    if len(entries) >= limit:
                        return entries
        except OSError:
            pass
        return entries


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse a minimal YAML frontmatter block without external dependencies."""
    if text.startswith('﻿'):
        text = text.lstrip('﻿')
    stripped = text.lstrip('\r\n ')
    if not stripped.startswith('---'):
        return {}, text
    lines = stripped.splitlines()
    if lines[0].strip() != '---':
        return {}, text
    metadata: dict[str, str] = {}
    index = 1
    key = ''
    block_indent = 0
    block_style = ''
    block_lines: list[str] = []
    while index < len(lines):
        line = lines[index]
        if line.strip() == '---':
            index += 1
            break
        indent = len(line) - len(line.lstrip(' '))
        if key and block_style:
            if line.strip() and indent >= block_indent:
                block_lines.append(line[block_indent:])
                index += 1
                continue
            separator = '\n' if block_style == '|' else ' '
            metadata[key] = separator.join(block_lines).strip()
            key = ''
            block_style = ''
            block_lines = []
            continue
        if ':' in line and not line.lstrip().startswith('#'):
            raw_key, raw_value = line.split(':', 1)
            key = raw_key.strip()
            value = raw_value.strip()
            if value in ('|', '>'):
                block_style = value
                block_indent = indent + 2
                block_lines = []
                index += 1
                continue
            metadata[key] = _strip_quotes(value)
            key = ''
        elif key and line.startswith(' '):
            # Plain multi-line continuation.
            metadata[key] = (metadata[key] + ' ' + line.strip()).strip()
        index += 1
    if key and block_style:
        separator = '\n' if block_style == '|' else ' '
        metadata[key] = separator.join(block_lines).strip()
    body = '\n'.join(lines[index:]).lstrip('\r\n')
    return metadata, body


def load_skill(file_path: Path, root: Path) -> Skill | None:
    try:
        text = file_path.read_text(encoding='utf-8-sig', errors='replace')
    except OSError:
        return None
    metadata, body = parse_frontmatter(text)
    name = str(metadata.get('name') or root.name).strip()
    description = str(metadata.get('description') or '').strip()
    if not description:
        for line in body.splitlines():
            candidate = line.strip().lstrip('#').strip()
            if candidate:
                description = candidate[:200]
                break
    if not name:
        return None
    return Skill(name=name, description=description, path=file_path, root=root,
                 body=body, metadata=metadata)


def default_skill_roots(settings: SkillsSettings, cwd: str | None = None) -> list[Path]:
    roots: list[Path] = [Path.home() / '.agents' / 'skills']
    if settings.include_project:
        base = Path(cwd or os.getcwd())
        roots.extend([base / '.agents' / 'skills', base / '.adcode' / 'skills'])
    roots.extend(Path(path).expanduser() for path in settings.extra_paths)
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def _iter_skill_files(root: Path, max_depth: int = 3) -> Iterable[tuple[Path, Path]]:
    if not root.is_dir():
        return
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        if depth > max_depth:
            continue
        try:
            children = sorted(directory.iterdir())
        except OSError:
            continue
        found = False
        for child in children:
            if child.is_file() and child.name.lower() == 'skill.md':
                yield child, directory
                found = True
                break
        if found:
            continue
        for child in children:
            if child.is_dir() and not child.name.startswith('.'):
                stack.append((child, depth + 1))


class SkillRegistry:
    def __init__(self, settings: SkillsSettings, cwd: str | None = None) -> None:
        self.settings = settings
        self.cwd = cwd
        self.skills: dict[str, Skill] = {}
        self.roots: list[Path] = []

    def scan(self) -> 'SkillRegistry':
        self.skills.clear()
        self.roots = default_skill_roots(self.settings, self.cwd)
        if not self.settings.enabled:
            return self
        for root in self.roots:
            for file_path, skill_root in _iter_skill_files(root):
                skill = load_skill(file_path, skill_root)
                if skill is None:
                    continue
                # First scan wins; earlier roots have higher priority.
                self.skills.setdefault(skill.name, skill)
        return self

    def list(self) -> list[Skill]:
        return sorted(self.skills.values(), key=lambda skill: skill.name.lower())

    def get(self, name: str) -> Skill | None:
        direct = self.skills.get(name)
        if direct is not None:
            return direct
        lowered = name.strip().lower()
        for skill in self.skills.values():
            if skill.name.lower() == lowered:
                return skill
        return None

    # ── model-facing surfaces ───────────────────────────────
    def preview_prompt(self, limit: int = 60) -> str:
        if not self.settings.enabled or not self.skills:
            return ''
        lines = [
            '<available_skills>',
            'Skills are reusable instruction packs. Read the description; when a '
            'skill is relevant, call skill_load(name) to read its full instructions '
            'and follow them for the current task.',
        ]
        for skill in self.list()[:limit]:
            description = ' '.join(skill.description.split())
            lines.append(f'- {skill.name}: {description}')
        lines.append('</available_skills>')
        return '\n'.join(lines)

    def tool_specs(self) -> list[ToolSpec]:
        if not self.settings.enabled:
            return []

        def skill_list(args: dict[str, Any]) -> str:
            query = str(args.get('query', '') or '').lower()
            results = []
            for skill in self.list():
                haystack = f'{skill.name} {skill.description}'.lower()
                if query and query not in haystack:
                    continue
                results.append(f'{skill.name} - {skill.description}')
            return '\n'.join(results) or '(no matching skills)'

        def skill_load(args: dict[str, Any]) -> str:
            skill = self.get(str(args.get('name', '')))
            if skill is None:
                return f'Error: unknown skill: {args.get("name", "")}'
            tree = skill.file_tree()
            tree_text = '\n'.join(f'  {entry}' for entry in tree)
            return (
                f'<skill name="{skill.name}" path="{skill.path}">\n'
                f'{skill.body.strip()}\n'
                f'</skill>\n'
                + (f'<skill_files>\n{tree_text}\n</skill_files>\n' if tree_text else '')
                + 'Read supporting files with the read tool relative to the skill path.'
            )

        return [
            ToolSpec(
                'skill_list',
                'List installed skills, optionally filtered by a query.',
                {'type': 'object', 'properties': {
                    'query': {'type': 'string', 'description': 'Optional filter text'},
                }, 'required': []},
                skill_list, read_only=True, source='skill',
            ),
            ToolSpec(
                'skill_load',
                'Load the full instructions of an installed skill by name.',
                {'type': 'object', 'properties': {
                    'name': {'type': 'string', 'description': 'Skill name'},
                }, 'required': ['name']},
                skill_load, read_only=True, source='skill',
            ),
        ]
