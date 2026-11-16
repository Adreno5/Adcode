"""System prompt assembly: global and project instructions."""

from __future__ import annotations

from pathlib import Path

from adcode import system_prompt
from adcode.config import AppConfig, ModelConfig, ProviderConfig


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


def _config_with_model(model_id: str, provider: str = 'demo') -> AppConfig:
    config = AppConfig()
    provider_config = ProviderConfig(
        name=provider, base_url='https://example.com/v1',
        models=[ModelConfig(id=model_id)])
    config.providers = [provider_config]
    config.select(provider, model_id)
    return config


def test_identity_block_states_the_model_id_and_runtime():
    config = _config_with_model('gpt-5-codex')

    prompt = system_prompt.build_system_prompt(config, None, None, '/tmp')

    assert '<identity>' in prompt
    assert 'model id: gpt-5-codex' in prompt
    assert 'AdCode' in prompt
    assert 'AICF' in prompt
    assert 'harness' in prompt


def test_identity_block_uses_the_explicit_model():
    config = _config_with_model('gpt-5-codex')

    prompt = system_prompt.build_system_prompt(
        config, None, None, '/tmp', model_id='other-model', provider='other')

    assert 'model id: other-model' in prompt
    assert 'gpt-5-codex' not in prompt


def test_identity_block_without_a_model_says_unknown():
    prompt = system_prompt.build_system_prompt(AppConfig(), None, None, '/tmp')

    assert 'model id: unknown' in prompt


def test_replace_identity_swaps_the_model_id():
    prompt = ('<environment>\ncwd: /tmp\n</environment>\n\n'
              + system_prompt.identity_block('parent-model', 'demo'))

    replaced = system_prompt.replace_identity(
        prompt, 'child-model', 'demo', role='sub-agent')

    assert 'model id: child-model' in replaced
    assert 'parent-model' not in replaced
    assert 'role: sub-agent' in replaced
    assert replaced.startswith('<environment>')


def test_replace_identity_leaves_a_prompt_without_identity_alone():
    prompt = 'no identity here'

    assert system_prompt.replace_identity(prompt, 'x') == prompt
