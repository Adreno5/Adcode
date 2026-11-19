"""The three access levels of a project folder.

``readonly``  the project can be read; nothing can be changed
``workspace`` files inside the project may be changed
``full``      no boundary at all

The level belongs to the folder: it is remembered per project root, so a
trusted project can be opened up while the default stays strict.
"""

from __future__ import annotations

from adcode.access import (ACCESS_LEVELS, declared_sandbox, describe,
                           effective_access, write_allowed)
from adcode.config import AppConfig, SecuritySettings
from adcode.security import AuditSink, build_guard, build_policy
from adcode.tools.builtin import install_coding_tools
from adcode.tools.registry import ManagedToolRegistry
from adcode.tools.specs import attach_registry_specs, set_sandbox_level


def _config(access: str = 'full') -> AppConfig:
    config = AppConfig()
    config.security = SecuritySettings(access=access)
    return config


def test_there_are_three_levels():
    assert ACCESS_LEVELS == ('readonly', 'workspace', 'full')


def test_the_default_level_applies_to_every_project(tmp_path):
    config = _config('readonly')
    assert effective_access(config, str(tmp_path)) == 'readonly'


def test_a_project_can_have_a_level_of_its_own(tmp_path):
    config = _config('readonly')
    root = str(tmp_path.resolve())
    config.security.access_by_project = {root: 'full'}
    assert effective_access(config, str(tmp_path)) == 'full'
    # another project keeps the default
    other = tmp_path / 'other'
    other.mkdir()
    assert effective_access(config, str(other)) == 'readonly'


def test_each_level_declares_its_own_sandbox():
    assert declared_sandbox('readonly') == 'readonly'
    assert declared_sandbox('workspace') == 'workspace'
    assert declared_sandbox('full') == 'none'


def test_the_write_boundary_follows_the_level(tmp_path):
    inside = str(tmp_path / 'file.py')
    outside = '/etc/passwd'
    assert write_allowed(inside, str(tmp_path), 'full')[0] is True
    assert write_allowed(inside, str(tmp_path), 'workspace')[0] is True
    assert write_allowed(outside, str(tmp_path), 'workspace')[0] is False
    assert write_allowed(inside, str(tmp_path), 'readonly')[0] is False
    assert write_allowed(outside, str(tmp_path), 'readonly')[0] is False


def test_the_reason_is_a_sentence_for_the_model(tmp_path):
    allowed, reason = write_allowed('/etc/passwd', str(tmp_path), 'workspace')
    assert allowed is False
    assert reason and 'project' in reason


def test_read_only_refuses_the_writing_tools_by_policy():
    policy = build_policy(SecuritySettings(access='readonly'))
    from adcode.tools.specs import BUILTIN_SPECS
    for name in ('write', 'edit'):
        decision = policy.admit(name, spec=BUILTIN_SPECS[name])
        assert decision.allowed is False, name
        assert decision.reason == 'policy'


def test_workspace_access_lets_the_writing_tools_through():
    policy = build_policy(SecuritySettings(access='workspace'))
    from adcode.tools.specs import BUILTIN_SPECS
    decision = policy.admit('write', spec=BUILTIN_SPECS['write'])
    assert decision.allowed is True


def test_the_declared_level_follows_the_access_level():
    registry = ManagedToolRegistry()
    install_coding_tools(registry)
    attach_registry_specs(registry)
    set_sandbox_level(registry, 'readonly')
    bash = next(tool for tool in registry.tools()
                if getattr(tool, 'name', '') == 'bash')
    assert bash.spec.sandbox == 'readonly'
    set_sandbox_level(registry, 'workspace')
    assert bash.spec.sandbox == 'workspace'
    set_sandbox_level(registry, 'full')
    assert bash.spec.sandbox == 'none'


def test_the_shell_may_not_write_outside_the_project(tmp_path):
    """The workspace level fences the shell to the project."""
    from harness.tools.context import ToolContext, pop_context, push_context
    from harness.tools.shell import ShellSessions, session_shell_tool

    registry = ManagedToolRegistry()
    registry.register(session_shell_tool(name='bash', sessions=ShellSessions()),
                      replace=True, source='builtin')
    attach_registry_specs(registry)
    registry.attach_guard(build_guard(
        SecuritySettings(mode='off', access='workspace'),
        sink=AuditSink(), workspace=str(tmp_path)))

    token = push_context(ToolContext(cwd=str(tmp_path)))
    try:
        inside, _, error = registry.execute(
            'bash', {'sub': 'execute', 'command': 'echo hi > inside.txt'})
        assert not error, inside
        assert (tmp_path / 'inside.txt').exists()

        outside, _, error = registry.execute(
            'bash', {'sub': 'execute',
                     'command': 'echo hi > /tmp/adcode-access-probe.txt'})
        assert error is True, outside
        assert not (tmp_path / 'outside.txt').exists()
    finally:
        pop_context(token)


def test_read_only_may_not_write_anywhere(tmp_path):
    """The read-only level stops the shell from writing at all."""
    from harness.tools.context import ToolContext, pop_context, push_context
    from harness.tools.shell import ShellSessions, session_shell_tool

    registry = ManagedToolRegistry()
    registry.register(session_shell_tool(name='bash', sessions=ShellSessions()),
                      replace=True, source='builtin')
    attach_registry_specs(registry)
    set_sandbox_level(registry, 'readonly')
    registry.attach_guard(build_guard(
        SecuritySettings(mode='off', access='readonly'),
        sink=AuditSink(), workspace=str(tmp_path)))

    token = push_context(ToolContext(cwd=str(tmp_path)))
    try:
        text, _, error = registry.execute(
            'bash', {'sub': 'execute', 'command': 'echo hi > inside.txt'})
        assert error is True, text
        assert not (tmp_path / 'inside.txt').exists()
    finally:
        pop_context(token)


def test_the_file_tools_refuse_a_write_outside_the_project(tmp_path,
                                                           monkeypatch):
    """The write tool is python, not a shell: it checks the level itself."""
    from adcode.config import set_current_config
    from adcode.tools.builtin import _write

    config = _config('workspace')
    set_current_config(config)
    monkeypatch.setattr('adcode.tools.builtin.current_context',
                        lambda: type('Ctx', (), {'cwd': str(tmp_path)})())
    text = _write({'path': '/tmp/adcode-write-probe.txt', 'content': 'x'})
    assert text.startswith('Error:'), text
    assert 'outside the project' in text, text


def test_each_level_has_a_sentence_for_the_interface():
    for level in ACCESS_LEVELS:
        assert describe(level)


def test_the_level_is_applied_at_once(tmp_path):
    """A change in the settings rebuilds the guard, not just the record."""
    from adcode.agent_loop import AgentLoop
    from adcode.config import AppConfig
    from adcode.goals import GoalStore
    from adcode.session import ArchivedSessionStore
    from adcode.skills import SkillRegistry
    from adcode.tools.registry import ManagedToolRegistry

    from adcode.config import ModelConfig, ProviderConfig

    config = AppConfig()
    config.security.access = 'full'
    provider = ProviderConfig(name='Test', base_url='http://localhost:1/v1',
                              api_key='k', wire='openai chat completions',
                              models=[ModelConfig(id='m')])
    config.providers = [provider]
    config.select('Test', 'm')
    registry = ManagedToolRegistry()
    loop = AgentLoop(
        config, registry,
        ArchivedSessionStore(db_path=str(tmp_path / 'sessions' / 'adcode.db')),
        GoalStore(tmp_path / 'goals.json'),
        SkillRegistry(config.skills, str(tmp_path)),
        cwd=str(tmp_path),
    )
    try:
        assert loop.access == 'full'
        first = loop.guard
        config.security.access = 'readonly'
        assert loop.apply_access() == 'readonly'
        assert loop.guard is not first, 'the guard must be rebuilt'
        # the writing tools are refused by the new policy
        from adcode.tools.specs import BUILTIN_SPECS
        decision = loop.guard.policy.admit('write', spec=BUILTIN_SPECS['write'])
        assert decision.allowed is False
    finally:
        loop.shutdown()
