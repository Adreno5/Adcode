"""Shared test fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adcode.config import (  # noqa: E402
    AppConfig, ModelConfig, ProviderConfig,
)
from adcode.theme import build_palette  # noqa: E402


@pytest.fixture()
def config() -> AppConfig:
    cfg = AppConfig()
    provider = ProviderConfig(
        name='Test', base_url='http://localhost:1234/v1', api_key='test-key',
        wire='openai chat completions',
        models=[ModelConfig(id='test-model', context_window=100_000,
                            max_output_tokens=8_192)],
    )
    cfg.providers = [provider]
    cfg.select('Test', 'test-model')
    cfg.skills.enabled = False
    return cfg


@pytest.fixture()
def palette():
    return build_palette('dark', '#8fa7d6')
