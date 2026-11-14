import json

import pytest

from adcode import config as config_module
from adcode.config import (
    AgentTypeConfig, AppConfig, ConfigStore, ModelConfig, ProviderConfig,
    config_from_dict, get_by_path, set_by_path,
)


@pytest.fixture(autouse=True)
def _identity_crypto(monkeypatch):
    monkeypatch.setattr(config_module, 'encrypt_secret', lambda value: f'enc:{value}')
    monkeypatch.setattr(config_module, 'decrypt_secret',
                        lambda value: value[4:] if value.startswith('enc:') else value)


def test_defaults():
    cfg = AppConfig()
    assert cfg.thinking.mode == 'label'
    assert cfg.tools.mode == 'both'
    assert cfg.agents.max_concurrent == 10
    assert 'general' in cfg.agent_types
    assert 'build' in cfg.agent_teams


def test_migration_from_legacy():
    legacy = {
        'providers': [{
            'name': 'P', 'base_url': 'https://api.example.com/v1',
            'api_key': 'secret', 'wire': 'openai chat completions',
            'models': ['m1', 'm2'],
        }],
        'selected_model': 'P::m2',
        'theme': '\x1b[36m',
        'verbose_mode': 'mini',
        'reasoning_effort': 'high',
        'max_subagents': 4,
        'max_agent_depth': 2,
        'agent_wait_timeout_ms': 5000,
        'max_output_tokens': 4096,
        'mcp_servers': [{'name': 's', 'transport': 'stdio', 'command': 'x'}],
    }
    cfg = config_from_dict(legacy)
    assert cfg.selected_model == 'P::m2'
    assert cfg.providers[0].models[1].id == 'm2'
    assert cfg.thinking.mode == 'label'
    assert cfg.tools.mode == 'call'
    assert cfg.reasoning.effort == 'high'
    assert cfg.agents.max_concurrent == 4
    assert cfg.context.reserved_output == 4096


def test_roundtrip(tmp_path):
    path = tmp_path / 'config.json'
    store = ConfigStore(path)
    cfg = store.load()
    provider = ProviderConfig(
        name='P', base_url='http://x/v1', api_key='k',
        models=[ModelConfig(id='m', context_window=1234)])
    cfg.providers = [provider]
    cfg.select('P', 'm')
    cfg.thinking.mode = 'partial'
    cfg.ui.accent = '#ff0000'
    store.save()
    raw = json.loads(path.read_text(encoding='utf-8'))
    assert raw['version'] == 2
    assert raw['providers'][0]['api_key'] == 'enc:k'
    reloaded = ConfigStore(path).load()
    assert reloaded.providers[0].api_key == 'k'
    assert reloaded.thinking.mode == 'partial'
    assert reloaded.selected()[1].id == 'm'


def test_legacy_config_moves_to_home(tmp_path, monkeypatch):
    legacy = tmp_path / 'checkout' / 'config.json'
    legacy.parent.mkdir()
    legacy.write_text(json.dumps({
        'version': 2,
        'providers': [],
        'thinking': {'mode': 'partial'},
    }), encoding='utf-8')
    target = tmp_path / 'home' / '.adcode' / 'config.json'
    monkeypatch.setattr(config_module, 'CONFIG_PATH', target)
    monkeypatch.setattr(config_module, 'LEGACY_CONFIG_PATH', legacy)

    cfg = ConfigStore().load()

    assert cfg.thinking.mode == 'partial'
    assert target.exists()
    assert not legacy.exists()


def test_model_metadata():
    model = ModelConfig(id='x')
    model.apply_metadata({
        'vision': True, 'context_window': 1_000_000,
        'max_output_tokens': 64_000, 'reasoning': True,
        'reasoning_style': 'effort', 'reasoning_efforts': ['low', 'high'],
        'tool_call': True, 'source': 'models.dev',
    })
    assert model.vision
    assert model.context_window == 1_000_000
    assert model.reasoning_efforts == ['low', 'high']
    assert model.source == 'models.dev'


def test_selection_and_removal():
    cfg = AppConfig()
    cfg.providers = [
        ProviderConfig(name='A', base_url='http://a', models=[ModelConfig(id='a')]),
        ProviderConfig(name='B', base_url='http://b', models=[ModelConfig(id='b')]),
    ]
    cfg.select('B', 'b')
    assert cfg.selected()[0].name == 'B'
    cfg.remove_provider('B')
    assert cfg.selected()[0].name == 'A'


def test_dotted_paths():
    cfg = AppConfig()
    assert get_by_path(cfg, 'thinking.label') == 'Thinking'
    set_by_path(cfg, 'thinking.label', 'Reasoning')
    assert cfg.thinking.label == 'Reasoning'


def test_agent_type_from_dict():
    agent = AgentTypeConfig.from_dict('x', {'description': 'd', 'tools': ['read'],
                                             'max_turns': 3})
    assert agent.tools == ['read']
    assert agent.max_turns == 3
