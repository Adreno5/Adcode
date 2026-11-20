"""The guard in front of the tools: specs, policy and the audit trail."""

from __future__ import annotations

import json
import pathlib

import pytest

from adcode.config import SecuritySettings
from adcode.security import (AuditSink, build_guard, build_policy,
                             format_audit_event)
from adcode.tools.builtin import install_coding_tools
from adcode.tools.registry import ManagedToolRegistry
from adcode.tools.specs import (BUILTIN_SPECS, COLLAB_SPECS, attach_spec,
                                attach_registry_specs, dynamic_spec)
from adcode.tools.todo import install_todo_tool


def _registry() -> ManagedToolRegistry:
    registry = ManagedToolRegistry()
    install_coding_tools(registry)
    install_todo_tool(registry)
    attach_registry_specs(registry)
    return registry


def test_every_builtin_tool_declares_what_it_does():
    registry = _registry()
    for name in ('read', 'write', 'edit', 'glob', 'grep', 'bash', 'fetch',
                 'diff', 'ls', 'view_image', 'todo'):
        tool = registry.get(name)
        assert tool is not None, name
        assert tool.spec is not None, f'{name} has no spec'
        assert tool.spec.name == name
        assert tool.spec.effect in ('pure', 'idempotent', 'effectful')


def test_reads_are_pure_and_writes_are_idempotent():
    registry = _registry()
    assert registry.get('read').spec.effect == 'pure'
    assert registry.get('read').spec.sandbox == 'readonly'
    assert registry.get('write').spec.effect == 'idempotent'
    assert registry.get('write').spec.sandbox == 'workspace'
    # the shell and the network are the effectful ones
    assert registry.get('bash').spec.effect == 'effectful'
    assert registry.get('fetch').spec.effect == 'effectful'
    assert registry.get('fetch').spec.network == 'allowlist'


def test_collab_tools_are_effectful():
    for name, spec in COLLAB_SPECS.items():
        assert spec.effect == 'effectful', name
        assert spec.group == 'collab'


def test_a_runtime_tool_gets_the_middle_grade():
    class Fake:
        name = 'mcp__thing'
        description = 'a runtime tool'
        read_only = False
    spec = dynamic_spec('mcp__thing', Fake())
    assert spec.effect == 'effectful'
    assert spec.group == 'dynamic'

    class ReadOnly:
        name = 'mcp__read'
        description = 'a runtime reader'
        read_only = True
    assert dynamic_spec('mcp__read', ReadOnly()).effect == 'pure'


def test_a_tool_that_declares_nothing_keeps_its_own_spec():
    from harness.tools.spec import ToolSpec
    from adcode.tools.registry import AnnotatedTool

    tool = AnnotatedTool(name='custom', description='mine',
                         parameters={'type': 'object', 'properties': {}},
                         handler=lambda: 'x')
    own = ToolSpec(name='custom', effect='pure', sandbox='readonly')
    tool.spec = own
    attach_spec(tool)
    assert tool.spec is own


def test_the_policy_allows_everything_when_nobody_has_to_approve():
    policy = build_policy(SecuritySettings(mode='off'))
    decision = policy.admit('bash', spec=BUILTIN_SPECS['bash'])
    assert decision.allowed is True, decision.to_dict()


def test_the_policy_asks_for_the_effectful_tools_in_ask_mode():
    policy = build_policy(SecuritySettings(mode='ask'))
    shell = policy.admit('bash', spec=BUILTIN_SPECS['bash'])
    assert shell.allowed is False and shell.needs_approval is True
    # a pure read still goes through
    read = policy.admit('read', spec=BUILTIN_SPECS['read'])
    assert read.allowed is True, read.to_dict()


def test_a_denied_tool_is_refused_by_the_policy():
    policy = build_policy(SecuritySettings(mode='ask', deny_tools=['bash']))
    decision = policy.admit('bash', spec=BUILTIN_SPECS['bash'])
    assert decision.allowed is False
    assert decision.reason == 'policy'


def test_the_audit_trail_is_written_to_the_database():
    sink = AuditSink()
    sink.emit('audit.tool_call', {'tool': 'read', 'decision': 'auto',
                                  'outcome': 'ok'})
    sink.emit('audit.denied', {'tool': 'bash', 'reason': 'policy',
                               'detail': 'nope'})
    sink.emit('tool_intent', {'tool': 'read'})      # not an audit event
    events = sink.read_history(50)
    assert len(events) == 2, events
    first = events[0]
    assert first['kind'] == 'audit.tool_call' and first['tool'] == 'read'
    assert first['ts'] > 0
    assert len(sink.recent()) == 2


def test_the_audit_trail_is_no_file_in_the_project(tmp_path):
    """The trail goes to the database; the project stays clean."""
    from adcode import paths
    sink = AuditSink()
    sink.emit('audit.tool_call', {'tool': 'read', 'decision': 'auto'})
    assert not list(tmp_path.glob('*.jsonl'))
    assert not list(tmp_path.glob('audit*'))
    assert paths.db_path().name.endswith('.db')


def test_the_audit_ring_is_bounded():
    sink = AuditSink(limit=5)
    for index in range(20):
        sink.emit('audit.tool_call', {'tool': f't{index}'})
    assert len(sink.recent(50)) == 5


def test_an_audit_event_reads_as_one_line():
    line = format_audit_event({'kind': 'audit.tool_call', 'tool': 'write',
                                'decision': 'auto', 'outcome': 'ok',
                                'ts': 0})
    assert 'write' in line and 'ok' in line
    denied = format_audit_event({'kind': 'audit.denied', 'tool': 'bash',
                                  'reason': 'approval', 'detail': 'refused',
                                  'ts': 0})
    assert 'refused' in denied and 'approval' in denied


def test_the_guard_is_built_for_the_current_settings():
    sink = AuditSink()
    guard = build_guard(SecuritySettings(mode='ask'), sink=sink,
                        ask=lambda request: None)
    assert guard.policy.auto_approve_risk == 'low'
    assert guard.approval is not None
    off = build_guard(SecuritySettings(mode='off'), sink=sink)
    assert off.approval is None
    assert off.policy.auto_approve_risk == 'high'


def test_the_workspace_sandbox_fences_the_shell(tmp_path):
    """With workspace access, the shell may not write outside the project."""
    from harness.tools.context import ToolContext, pop_context, push_context
    from harness.tools.shell import ShellSessions, session_shell_tool

    registry = ManagedToolRegistry()
    registry.register(session_shell_tool(name='bash', sessions=ShellSessions()),
                      replace=True, source='builtin')
    attach_registry_specs(registry)
    sink = AuditSink()
    registry.attach_guard(build_guard(
        SecuritySettings(mode='off', access='workspace'), sink=sink,
        workspace=str(tmp_path)))

    token = push_context(ToolContext(cwd=str(tmp_path)))
    try:
        inside, _, error = registry.execute(
            'bash', {'sub': 'execute', 'command': 'echo hi > inside.txt'})
        assert not error, inside
        assert (tmp_path / 'inside.txt').exists()

        outside, _, error = registry.execute(
            'bash', {'sub': 'execute',
                     'command': 'echo hi > /tmp/adcode-escape-probe.txt'})
        assert error is True
        assert '沙箱拒绝' in outside or 'denied' in outside, outside
        assert not pathlib.Path('/tmp/adcode-escape-probe.txt').exists()
    finally:
        pop_context(token)


def test_with_full_access_the_shell_writes_wherever_it_wants(tmp_path):
    from harness.tools.context import ToolContext, pop_context, push_context
    from harness.tools.shell import ShellSessions, session_shell_tool

    registry = ManagedToolRegistry()
    registry.register(session_shell_tool(name='bash', sessions=ShellSessions()),
                      replace=True, source='builtin')
    attach_registry_specs(registry)
    registry.attach_guard(build_guard(
        SecuritySettings(mode='off', access='full'),
        sink=AuditSink()))

    target = tmp_path / 'probe.txt'
    token = push_context(ToolContext(cwd=str(tmp_path)))
    try:
        text, _, error = registry.execute(
            'bash', {'sub': 'execute', 'command': f'echo hi > {target}'})
        assert not error, text
        assert target.exists()
    finally:
        pop_context(token)


def test_the_fetch_tool_honours_the_host_allowlist():
    from adcode.tools.builtin import _host_allowed

    assert _host_allowed('https://example.com/x', []) is True, 'empty = no limit'
    assert _host_allowed('https://example.com/x', ['example.com']) is True
    assert _host_allowed('https://api.example.com/x', ['example.com']) is True
    assert _host_allowed('https://evil.test/x', ['example.com']) is False
    assert _host_allowed('not a url', ['example.com']) is False


def test_the_allowlist_comes_from_the_published_config():
    from adcode.config import AppConfig, set_current_config
    from adcode.tools.builtin import _network_allowlist

    config = AppConfig()
    config.security.network_allowlist = ['example.com']
    set_current_config(config)
    try:
        assert _network_allowlist() == ['example.com']
    finally:
        set_current_config(None)
    assert _network_allowlist() == []


def test_the_shell_does_not_warn_when_the_sandbox_is_off(tmp_path, caplog):
    """No backend is not the same as a chosen backend.

    The sdk warns loudly when the shell has no sandbox backend.  With the
    sandbox turned *off* on purpose the command still runs -- but it must not
    warn: a warning that lands on the terminal destroys the full-screen ui.
    """
    import logging

    from harness.tools.context import ToolContext, pop_context, push_context
    from harness.tools.shell import ShellSessions, session_shell_tool

    registry = ManagedToolRegistry()
    registry.register(session_shell_tool(name='bash', sessions=ShellSessions()),
                      replace=True, source='builtin')
    attach_registry_specs(registry)
    registry.attach_guard(build_guard(
        SecuritySettings(mode='off', access='full'),
        sink=AuditSink()))

    with caplog.at_level(logging.WARNING):
        token = push_context(ToolContext(cwd=str(tmp_path)))
        try:
            text, _, error = registry.execute(
                'bash', {'sub': 'execute', 'command': 'echo hi'})
        finally:
            pop_context(token)
    assert not error, text
    assert 'hi' in text, text
    warnings = [record for record in caplog.records
                if '沙箱后端' in record.getMessage()]
    assert not warnings, [record.getMessage() for record in warnings]


def test_the_logs_go_to_the_file_and_not_to_the_terminal(tmp_path, capsys):
    """A stray warning must not be printed over the interface."""
    import logging

    import adcode.paths as paths
    from adcode.logging_setup import configure_logging

    previous = paths.APP_DIR
    paths.set_app_dir(tmp_path / 'home')
    try:
        target = configure_logging()
        logging.getLogger('harness.tools.shell').warning('a stray warning')
        captured = capsys.readouterr()
        assert 'a stray warning' not in captured.out
        assert 'a stray warning' not in captured.err
        assert target.exists()
        assert 'a stray warning' in target.read_text(encoding='utf-8')
    finally:
        paths.set_app_dir(previous)


def test_a_logger_without_its_own_handler_still_stays_off_the_terminal(tmp_path, capsys):
    """The fallback handler writes to stderr -- that is the last leak."""
    import logging

    import adcode.paths as paths
    from adcode.logging_setup import configure_logging

    previous = paths.APP_DIR
    paths.set_app_dir(tmp_path / 'home')
    try:
        target = configure_logging()
        logging.getLogger('lonely.logger').warning('no handler of its own')
        captured = capsys.readouterr()
        assert 'no handler of its own' not in captured.err
        assert 'no handler of its own' in target.read_text(encoding='utf-8')
    finally:
        paths.set_app_dir(previous)
