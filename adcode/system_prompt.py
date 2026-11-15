"""System prompt assembly: global/project instructions, environment, goals and skills."""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path

from adcode.config import CONFIG_DIR, AppConfig
from adcode.goals import GoalStore
from adcode.skills import SkillRegistry

BASE_SUFFIX = (
    'You are AdCode, a concise and capable terminal coding agent. '
    'Use the available tools to inspect, change, and verify the project. '
    'Keep working until the task is complete. Prefer reading before editing. '
    'Answer in the language the user writes in. '
    'Do not use emoji characters in replies. '
    'All model operations, including compaction and sub-agents, must use streaming APIs.'
)


GLOBAL_INSTRUCTIONS_PATH = CONFIG_DIR / 'AGENTS.md'
PROJECT_INSTRUCTION_NAMES = ('AGENTS.md', 'prompt')


def _read_instructions(path: Path) -> str:
    if not path.is_file():
        return ''
    try:
        return path.read_text(encoding='utf-8-sig').strip()
    except OSError:
        return ''


def global_instructions() -> str:
    """User-level instructions shared by every project (``~/.adcode/AGENTS.md``)."""
    return _read_instructions(GLOBAL_INSTRUCTIONS_PATH)


def project_instructions(cwd: str | None = None) -> str:
    base = Path(cwd or os.getcwd())
    for name in PROJECT_INSTRUCTION_NAMES:
        text = _read_instructions(base / name)
        if text:
            return text
    return ''


def instructions_block(cwd: str | None = None) -> str:
    """Global and project instructions as tagged sections, global first."""
    blocks: list[str] = []
    global_text = global_instructions()
    if global_text:
        blocks.append(
            '<global_instructions>\n' + global_text + '\n</global_instructions>')
    project_text = project_instructions(cwd)
    if project_text:
        blocks.append(
            '<project_instructions>\n' + project_text + '\n</project_instructions>')
    if not blocks:
        return ''
    if len(blocks) == 2:
        blocks.append(
            'Global instructions apply everywhere; when they conflict with '
            'project instructions, follow the project instructions.')
    return '\n\n'.join(blocks)


def environment_block(cwd: str | None = None) -> str:
    return (
        '<environment>\n'
        f'cwd: {cwd or os.getcwd()}\n'
        f'platform: {platform.system()} {platform.release()}\n'
        f'date: {time.strftime("%Y-%m-%d")}\n'
        '</environment>'
    )


def build_system_prompt(config: AppConfig, goals: GoalStore | None,
                        skills: SkillRegistry | None, cwd: str | None = None,
                        vision: bool = False) -> str:
    parts = [instructions_block(cwd), environment_block(cwd), BASE_SUFFIX]
    if vision:
        parts.append(
            'You can see images. A user message that carries one shows it to you '
            'directly; call view_image to look at an image file or URL yourself '
            'instead of guessing what it contains.'
        )
    goal_prompt = GoalStore.prompt(goals.active()) if goals is not None else ''
    if goal_prompt:
        parts.append(goal_prompt)
    if skills is not None:
        preview = skills.preview_prompt()
        if preview:
            parts.append(preview)
    if config.agents.enabled:
        parts.append(
            'You can delegate independent work to sub-agents with spawn_agent or '
            'spawn_team. Give each agent a concrete, self-contained task and wait '
            'for the results when you need them.'
        )
    return '\n\n'.join(part for part in parts if part)
