"""System prompt assembly: global and project instructions."""

from __future__ import annotations

from pathlib import Path

from adcode import system_prompt
from adcode.config import AppConfig


def _write(path: Path, text: str, encoding: str = 'utf-8') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding)


def _project(tmp_path: Path) -> Path:
    project = tmp_path / 'project'
    project.mkdir()
    return project


def test_global_instructions_alone(tmp_path, monkeypatch):
    global_path = tmp_path / 'home' / '.adcode' / 'AGENTS.md'
    monkeypatch.setattr(system_prompt, 'GLOBAL_INSTRUCTIONS_PATH', global_path)
    _write(global_path, 'Always answer in Chinese.')

    block = system_prompt.instructions_block(str(_project(tmp_path)))

    assert '<global_instructions>' in block
    assert 'Always answer in Chinese.' in block
    assert '<project_instructions>' not in block
    assert 'conflict' not in block


def test_global_and_project_instructions_together(tmp_path, monkeypatch):
    global_path = tmp_path / 'home' / '.adcode' / 'AGENTS.md'
    monkeypatch.setattr(system_prompt, 'GLOBAL_INSTRUCTIONS_PATH', global_path)
    _write(global_path, 'Global rule.')
    project = _project(tmp_path)
    _write(project / 'AGENTS.md', 'Project rule.')

    block = system_prompt.instructions_block(str(project))

    assert block.index('Global rule.') < block.index('Project rule.')
    assert '<global_instructions>' in block
    assert '<project_instructions>' in block
    assert 'follow the project instructions' in block


def test_project_instructions_fall_back_to_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        system_prompt, 'GLOBAL_INSTRUCTIONS_PATH', tmp_path / 'missing.md')
    project = _project(tmp_path)
    _write(project / 'AGENTS.md', '')
    _write(project / 'prompt', 'Use pytest.')

    assert system_prompt.project_instructions(str(project)) == 'Use pytest.'


def test_missing_instruction_files_are_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(
        system_prompt, 'GLOBAL_INSTRUCTIONS_PATH', tmp_path / 'missing.md')

    assert system_prompt.global_instructions() == ''
    assert system_prompt.instructions_block(str(_project(tmp_path))) == ''


def test_global_file_bom_is_stripped(tmp_path, monkeypatch):
    global_path = tmp_path / 'home' / '.adcode' / 'AGENTS.md'
    monkeypatch.setattr(system_prompt, 'GLOBAL_INSTRUCTIONS_PATH', global_path)
    _write(global_path, '\ufeffGlobal BOM rule\n')

    assert system_prompt.global_instructions() == 'Global BOM rule'


def test_build_system_prompt_injects_both(tmp_path, monkeypatch):
    global_path = tmp_path / 'home' / '.adcode' / 'AGENTS.md'
    monkeypatch.setattr(system_prompt, 'GLOBAL_INSTRUCTIONS_PATH', global_path)
    _write(global_path, 'Global build rule.')
    project = _project(tmp_path)
    _write(project / 'AGENTS.md', 'Project build rule.')
    config = AppConfig()
    config.agents.enabled = False

    prompt = system_prompt.build_system_prompt(config, None, None, str(project))

    assert 'Global build rule.' in prompt
    assert 'Project build rule.' in prompt
    assert prompt.index('Global build rule.') < prompt.index('Project build rule.')
