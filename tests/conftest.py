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
from bench.app import AppHarness  # noqa: E402


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


@pytest.fixture()
def harness_factory():
    """Start real simulated app runs and close them afterwards."""
    created: list[AppHarness] = []

    def make(**kwargs) -> AppHarness:
        harness = AppHarness(**kwargs)
        harness.start()
        created.append(harness)
        return harness

    yield make

    errors: list[BaseException] = []
    for harness in reversed(created):
        try:
            harness.close()
        except BaseException as error:  # report after every run is closed
            errors.append(error)
    if errors:
        raise errors[0]


@pytest.fixture()
def harness(harness_factory):
    return harness_factory()
