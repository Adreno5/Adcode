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


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Every test gets its own state directory.

    The state -- the database, the documents, the audit trail -- lives in
    one place now, and a test must neither read nor write the real one.
    The directory sits *next to* the test's workspace, not inside it: the
    rule "state never lands in the project" applies to the tests too.
    """
    from adcode import paths
    from adcode.state import reset_state_store

    state_dir = tmp_path.parent / f'{tmp_path.name}-state'
    monkeypatch.setenv('ADCODE_HOME', str(state_dir))
    paths.set_app_dir(None)
    paths.set_project_root('')
    reset_state_store()
    yield
    reset_state_store()
    paths.set_app_dir(None)
    paths.set_project_root('')


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
