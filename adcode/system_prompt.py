"""System prompt assembly: project instructions, environment, goals and skills."""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path

from adcode.config import AppConfig
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


def project_instructions(cwd: str | None = None) -> str:
    base = Path(cwd or os.getcwd())
    for name in ('AGENTS.md', 'prompt'):
        path = base / name
        if path.is_file():
            try:
                return path.read_text(encoding='utf-8-sig').strip()
            except OSError:
                continue
    return ''


def environment_block(cwd: str | None = None) -> str:
    return (
        '<environment>\n'
        f'cwd: {cwd or os.getcwd()}\n'
        f'platform: {platform.system()} {platform.release()}\n'
        f'date: {time.strftime("%Y-%m-%d")}\n'
        '</environment>'
    )


def build_system_prompt(config: AppConfig, goals: GoalStore | None,
                        skills: SkillRegistry | None, cwd: str | None = None) -> str:
    parts = [project_instructions(cwd), environment_block(cwd), BASE_SUFFIX]
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
