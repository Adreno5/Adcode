"""The guard around every tool call.

The sdk (aicf-core 1.0.0rc0) puts a single door in front of tool execution:
``ToolGuard``.  Behind it sit the policy engine (what is allowed), the
approval broker (who says yes), the sandbox (where the shell may write) and
the audit trail (what actually happened).  AdCode assembles those pieces here
and hands the guard to the tool registry, so *every* tool -- built-in, skill,
MCP or collaboration -- goes through the same door.

Two modes:

* ``off``  -- nothing waits for a human, but every call is still audited and
  the sandbox still fences the shell when it is asked to.
* ``ask``  -- effectful tools (the shell, the network, spawning a worker)
  wait for a yes.  The answer can be remembered for the session, so the
  question is asked once per tool rather than once per call.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Optional

from harness.approval.base import ApprovalDecision, ApprovalRequest
from harness.approval.broker import LocalApprovalBroker
from harness.runtime_client import NullRuntimeClient
from harness.tools.guard import ToolGuard
from harness.tools.sandbox.paths import PathPolicy
from harness.tools.sandbox.restricted import RestrictedSandbox
from infra.policy.base import PolicyRule
from infra.policy.engine import RulePolicyEngine

from adcode.access import WRITE_TOOLS
from adcode.config import SecuritySettings

#: The events that go into the audit trail.  The guard emits a wider stream
#: (tool_intent, tool_end, ...); the audit face is what a reviewer reads.
AUDIT_KINDS = ('audit.tool_call', 'audit.denied', 'approval_request',
               'approval_decision', 'approval_timeout', 'tool_blocked')

#: How many events the in-memory ring keeps for ``/audit``.
RING_LIMIT = 200


class _OpenBackend:
    """A backend that fences nothing -- on purpose.

    The sdk treats a missing backend as a degradation and says so loudly
    (the shell tool warns that the command ran with the user's own
    permissions).  That warning is right for a *missing* backend and wrong
    for a *chosen* one, so this backend is a real one: it answers with an
    execution plan that runs the command exactly as it would have run
    before -- the whole environment, no path checks, the platform's own
    shell.
    """

    name = 'open'

    def default_network_level(self) -> str:
        return 'full'

    def degraded(self) -> bool:
        return False

    def prepare(self, request: Any) -> Any:
        from harness.tools.sandbox.restricted import PreparedCommand
        command = request.cmd[0] if request.cmd else ''
        if request.shell:
            if os.name == 'nt':
                argv = [os.environ.get('COMSPEC', 'cmd.exe'), '/c', command]
            else:
                argv = ['/bin/sh', '-c', command]
        else:
            argv = list(request.cmd)
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'}
        return PreparedCommand(argv=argv, env=env,
                               cwd=request.cwd or os.getcwd(), degraded=False)


class AuditSink:
    """The telemetry port: the audit trail in the database, plus a ring.

    The sdk's ``AuditStore`` keeps records in memory only, and AdCode wants
    them to outlive the session -- so the records go into the state
    database (an append-only table).  They used to be a jsonl file next to
    the project; a repository is for code, and the trail is not code.

    Writes are best-effort: a failed audit write must never break the tool
    call that was being audited.
    """

    def __init__(self, store: Any = None, *,
                 session_id: Any = '',
                 limit: int = RING_LIMIT,
                 on_event: Optional[Callable[[str, dict], None]] = None) -> None:
        self._store = store
        # The session name is not known when the sink is built (the engine
        # starts later), so it may be a callable.
        self._session_id = session_id
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._limit = limit
        self._on_event = on_event

    # ── the telemetry port ──────────────────────────────────
    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        if kind not in AUDIT_KINDS:
            return
        event = {'kind': kind, 'ts': time.time(),
                 'session_id': self._session_name(), **dict(payload)}
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._limit:
                del self._events[:len(self._events) - self._limit]
        self._append(event)
        if self._on_event is not None:
            try:
                self._on_event(kind, event)
            except Exception:      # noqa: BLE001 - a listener must not break audit
                pass

    def _append(self, event: dict[str, Any]) -> None:
        store = self._store
        if store is None:
            try:
                from adcode.state import state_store
                store = state_store()
                self._store = store
            except Exception:      # noqa: BLE001 - no database, no trail
                return
        try:
            store.append_audit(event)
        except Exception:      # noqa: BLE001 - the trail is best effort
            pass

    def _session_name(self) -> str:
        source = self._session_id
        if callable(source):
            try:
                return str(source() or '')
            except Exception:      # noqa: BLE001 - a bad source is no name
                return ''
        return str(source or '')

    # ── reading ─────────────────────────────────────────────
    def recent(self, count: int = 50) -> list[dict[str, Any]]:
        """The newest events, newest last."""
        with self._lock:
            events = list(self._events)
        return events[-count:]

    def read_history(self, count: int = 200) -> list[dict[str, Any]]:
        """The tail of the trail (across sessions, out of the database)."""
        store = self._store
        if store is None:
            try:
                from adcode.state import state_store
                store = state_store()
                self._store = store
            except Exception:      # noqa: BLE001
                return []
        try:
            return store.audit_tail(count)
        except Exception:      # noqa: BLE001
            return []

    def read_file(self, count: int = 200) -> list[dict[str, Any]]:
        """The old name of :meth:`read_history`."""
        return self.read_history(count)


def build_policy(settings: SecuritySettings, *,
                 access: str = '') -> RulePolicyEngine:
    """The admission rules.

    ``off``  -- everything passes the risk threshold, nothing waits.
    ``ask``  -- only what the rules allow passes; the rest needs a human.

    The access level adds one rule of its own: in ``readonly`` the tools
    that change files are refused outright.  The sandbox cannot do that --
    it fences the *shell*, while these tools write with python -- so the
    refusal has to happen here, where every tool passes anyway.
    """
    access = access or getattr(settings, 'access', '') or 'full'
    rules = [PolicyRule(action='deny', tool=name,
                        id=f'deny-{name}', reason='refused in the configuration')
             for name in settings.deny_tools]
    if access == 'readonly':
        rules.extend(
            PolicyRule(action='deny', tool=name,
                       id=f'readonly-{name}',
                       reason='the project is read-only: this session may '
                              'read files but not change them')
            for name in WRITE_TOOLS)
    return RulePolicyEngine(
        rules=rules,
        # 'off': every declared effect clears the threshold, so no call waits.
        # 'ask': only the pure ones do; the effectful ones need a yes.
        auto_approve_risk='high' if settings.mode == 'off' else 'low',
        require_approval_effects=('effectful',),
        allow_readonly=True,
        deny_unknown_tools=True,
    )


def build_sandbox(settings: SecuritySettings, access: str = '',
                  workspace: str = '') -> Any:
    """The backend that fences the shell, per access level.

    ``full``      no boundary -- the user's own rights, as before.
    ``workspace`` the shell may write inside the project and nowhere else.
    ``readonly``  the project is readable and nothing is writable.

    The backend is always a *real* one: a missing backend would run every
    command with the user's permissions while the audit trail still
    claimed a fence.
    """
    access = access or getattr(settings, 'access', '') or 'full'
    allowlist = list(settings.network_allowlist)
    if access == 'full':
        return _OpenBackend()
    root = workspace or os.getcwd()
    if access == 'readonly':
        policy = PathPolicy(readonly=(root,), writable=(), cwd_writable=False)
    else:
        policy = PathPolicy(readonly=(), writable=(root,), cwd_writable=False)
    return RestrictedSandbox(network_allowlist=allowlist, policy=policy)


def build_guard(settings: SecuritySettings, *,
                sink: AuditSink,
                ask: Optional[Callable[[ApprovalRequest], ApprovalDecision]] = None,
                session_id: str = '',
                access: str = '',
                workspace: str = '') -> ToolGuard:
    """Assemble the guard for the current settings.

    ``access`` and ``workspace`` may be left out: the level comes from the
    settings, and the project from the working directory.
    """
    access = access or getattr(settings, 'access', '') or 'full'
    policy = build_policy(settings, access=access)
    approval = None
    if settings.mode == 'ask':
        broker = LocalApprovalBroker(
            resolver=ask, timeout=max(5.0, settings.approval_timeout),
            session_id=session_id)
        approval = broker
    sandbox = build_sandbox(settings, access=access, workspace=workspace)
    return ToolGuard(
        policy=policy,
        approval=approval,
        sandbox=sandbox,
        telemetry=sink,
        runtime=NullRuntimeClient(),
        session_id=session_id,
        approval_timeout=max(5.0, settings.approval_timeout),
    )


def format_audit_event(event: dict[str, Any]) -> str:
    """One readable line per audit event (for ``/audit`` and the transcript)."""
    kind = str(event.get('kind') or '')
    tool = str(event.get('tool') or '')
    stamp = time.strftime('%H:%M:%S', time.localtime(event.get('ts') or 0))
    if kind == 'audit.tool_call':
        decision = str(event.get('decision') or '')
        outcome = event.get('outcome') or ''
        parts = [f'{tool}']
        if decision:
            parts.append(decision)
        if outcome:
            parts.append(str(outcome))
        return f'{stamp}  ' + '  '.join(parts)
    if kind == 'audit.denied':
        reason = str(event.get('reason') or '')
        detail = str(event.get('detail') or '')
        return f'{stamp}  {tool}  refused ({reason})  {detail}'.rstrip()
    if kind == 'approval_request':
        return f'{stamp}  asked about {tool}'
    if kind == 'approval_decision':
        return (f'{stamp}  {tool}  {event.get("decision") or ""} '
                f'({event.get("decided_by") or ""})').rstrip()
    if kind == 'approval_timeout':
        return f'{stamp}  {tool}  approval timed out'
    if kind == 'tool_blocked':
        return (f'{stamp}  {tool}  blocked ({event.get("reason") or ""}) '
                f'{event.get("violation") or ""}').rstrip()
    return f'{stamp}  {kind}  {tool}'.rstrip()
