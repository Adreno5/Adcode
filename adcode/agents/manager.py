"""Sub-agent and agent-team runtime.

Each spawned agent runs a fully streamed ReAct loop on its own thread, shares
the workspace and tool registry, exchanges mailbox messages with its parent and
children, and notifies the UI through the AgentEvent queue.  Agent types and
teams are configuration-driven so behaviour stays adjustable from ``/config``
and ``config.json``.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import json
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator

from adcode.config import AgentTypeConfig, AppConfig, ModelConfig, ProviderConfig
from adcode.diff import FileDiff
from adcode.goals import GoalStore
from adcode.images import (
    MARKER_RE as IMAGE_MARKER_RE, FileIdCache, image_marker,
)
from adcode.llm.client import LLMClient
from adcode.llm.context import (
    ConversationContext, StablePrefix, stream_compact,
)
from adcode.llm.usage import (
    TokenBudget, effective_reserved, estimate_tokens, merge_usage,
    normalize_tool_result, normalize_usage,
)
from adcode.tools.registry import ToolRegistry, ToolResult, ToolSpec
from adcode.tools.runtime import ToolContext, pop_context, push_context

ROOT_AGENT = 'root'
_TASK_NAME_RE = re.compile(r'^[a-z0-9_]+$')
_FINAL_STATUSES = {'interrupted', 'completed', 'errored', 'shutdown'}
MAX_CONTINUATIONS = 3


class AgentStatus(str, Enum):
    PENDING_INIT = 'pending_init'
    RUNNING = 'running'
    INTERRUPTED = 'interrupted'
    COMPLETED = 'completed'
    ERRORED = 'errored'
    SHUTDOWN = 'shutdown'


@dataclass(frozen=True)
class AgentEvent:
    agent_name: str
    kind: str
    content: str = ''
    tool_name: str = ''
    tool_args: dict[str, Any] = field(default_factory=dict)
    status: str = ''
    file_diff: FileDiff | None = None
    created_at: float = field(default_factory=time.time)


@dataclass
class AgentScope:
    name: str
    parent: str | None
    depth: int
    agent_type: AgentTypeConfig
    provider: ProviderConfig
    model: ModelConfig
    system: str
    context: ConversationContext
    allowed_tools: set[str] | None = None


@dataclass
class AgentRecord:
    scope: AgentScope
    status: AgentStatus = AgentStatus.PENDING_INIT
    final_message: str = ''
    error: str = ''
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    interrupt: threading.Event = field(default_factory=threading.Event)
    inbox: deque[str] = field(default_factory=deque)
    pending_turns: deque[str] = field(default_factory=deque)
    restart_after_interrupt: bool = False
    thread: threading.Thread | None = None


_CURRENT_SCOPE: contextvars.ContextVar[AgentScope | None] = contextvars.ContextVar(
    'adcode_current_agent_scope', default=None)

SUBAGENT_SYSTEM = """
You are an AdCode sub-agent controlled by a parent agent. Work only on the delegated task.
Use the available tools in a ReAct loop: inspect evidence, act, verify, then return a concise
final result with concrete files, commands, and unresolved risks. You may spawn children when
the task splits into genuinely independent work. Do not ask the end user questions; report a
blocking dependency to the parent instead. Do not use emoji.
""".strip()


class AgentManager:
    def __init__(self, config: AppConfig, provider: ProviderConfig, model: ModelConfig,
                 system: str, root_context: ConversationContext,
                 registry: ToolRegistry, *, goal_store: GoalStore | None = None,
                 event_sink: Callable[[AgentEvent], None] | None = None,
                 cwd: str = '', file_cache: FileIdCache | None = None) -> None:
        self.config = config
        self.goal_store = goal_store
        self.registry = registry
        self.event_sink = event_sink
        self.cwd = cwd or '.'
        self.file_cache = file_cache
        self._lock = threading.RLock()
        self._activity = threading.Condition(self._lock)
        self._events: 'queue.SimpleQueue[AgentEvent]' = queue.SimpleQueue()
        self._records: dict[str, AgentRecord] = {}
        self._wait_snapshots: dict[str, tuple[dict[str, float], float]] = {}
        self._shutdown = False
        root_scope = AgentScope(
            ROOT_AGENT, None, 0,
            AgentTypeConfig(name='root'), copy.deepcopy(provider),
            copy.deepcopy(model), system, root_context,
        )
        self._records[ROOT_AGENT] = AgentRecord(root_scope, status=AgentStatus.RUNNING)
        self._wait_snapshots[ROOT_AGENT] = ({}, 0.0)

    # ── limits ──────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        return self.config.agents.enabled

    @property
    def max_subagents(self) -> int:
        return max(1, int(self.config.agents.max_concurrent))

    @property
    def max_depth(self) -> int:
        return max(1, int(self.config.agents.max_depth))

    def update_root(self, context: ConversationContext, provider: ProviderConfig,
                    model: ModelConfig, system: str) -> None:
        with self._lock:
            scope = self._records[ROOT_AGENT].scope
            scope.context = context
            scope.provider = provider
            scope.model = model
            scope.system = system

    @contextlib.contextmanager
    def root_scope(self) -> Iterator[AgentScope]:
        with self._lock:
            scope = self._records[ROOT_AGENT].scope
        token = _CURRENT_SCOPE.set(scope)
        try:
            yield scope
        finally:
            _CURRENT_SCOPE.reset(token)

    @contextlib.contextmanager
    def scope(self, scope: AgentScope) -> Iterator[AgentScope]:
        token = _CURRENT_SCOPE.set(scope)
        try:
            yield scope
        finally:
            _CURRENT_SCOPE.reset(token)

    def current_scope(self) -> AgentScope:
        return _CURRENT_SCOPE.get() or self._records[ROOT_AGENT].scope

    # ── tools ───────────────────────────────────────────────
    def handles_tool(self, name: str) -> bool:
        return name in {spec.name for spec in self.tool_specs()}

    def tool_specs(self) -> list[ToolSpec]:
        specs = [
            ToolSpec(
                'spawn_agent',
                ('Spawn a background sub-agent with its own streaming ReAct loop. '
                 'The child shares the workspace and may recursively spawn children '
                 'within the configured limits. fork_turns is none, all, or a '
                 'positive number of recent turns.'),
                {'type': 'object', 'properties': {
                    'task_name': {'type': 'string',
                                  'description': 'Lowercase letters, digits, underscores.'},
                    'message': {'type': 'string', 'description': 'Concrete delegated task.'},
                    'agent_type': {'type': 'string',
                                   'description': 'One of the configured agent types.'},
                    'fork_turns': {'type': 'string',
                                   'description': 'none, all, or a positive integer.'},
                }, 'required': ['task_name', 'message']},
                self._tool_spawn_agent, source='agent',
            ),
            ToolSpec(
                'spawn_team',
                ('Spawn every member of a configured agent team on the same task. '
                 'Each member has its own agent type and returns a result to you.'),
                {'type': 'object', 'properties': {
                    'team_name': {'type': 'string'},
                    'message': {'type': 'string', 'description': 'Task for every member.'},
                    'fork_turns': {'type': 'string'},
                }, 'required': ['team_name', 'message']},
                self._tool_spawn_team, source='agent',
            ),
            ToolSpec(
                'send_message',
                'Queue a message for an existing agent without starting a new turn.',
                {'type': 'object', 'properties': {
                    'target': {'type': 'string'}, 'message': {'type': 'string'},
                }, 'required': ['target', 'message']},
                self._tool_send_message, source='agent',
            ),
            ToolSpec(
                'followup_task',
                'Send a follow-up task and start a turn when the target is idle.',
                {'type': 'object', 'properties': {
                    'target': {'type': 'string'}, 'message': {'type': 'string'},
                }, 'required': ['target', 'message']},
                self._tool_followup_task, source='agent',
            ),
            ToolSpec(
                'list_agents',
                'List the live agent tree and the last known status of each agent.',
                {'type': 'object', 'properties': {
                    'path_prefix': {'type': 'string'},
                }, 'required': []},
                self._tool_list_agents, read_only=True, source='agent',
            ),
            ToolSpec(
                'wait_agent',
                'Wait for an agent status or mailbox update.',
                {'type': 'object', 'properties': {
                    'timeout_ms': {'type': 'integer', 'minimum': 0, 'maximum': 3_600_000},
                }, 'required': []},
                self._tool_wait_agent, source='agent',
            ),
            ToolSpec(
                'interrupt_agent',
                'Interrupt a spawned agent current turn.',
                {'type': 'object', 'properties': {'target': {'type': 'string'}},
                 'required': ['target']},
                self._tool_interrupt_agent, source='agent',
            ),
            ToolSpec(
                'update_goal',
                'Update the active durable goal after it is achieved or blocked.',
                {'type': 'object', 'properties': {
                    'status': {'type': 'string',
                               'enum': ['active', 'paused', 'completed', 'blocked',
                                        'cancelled']},
                    'result': {'type': 'string'},
                }, 'required': ['status']},
                self._tool_update_goal, source='agent',
            ),
        ]
        if not self.config.agents.team_enabled:
            specs = [spec for spec in specs if spec.name != 'spawn_team']
        return specs

    # ── tool handlers ───────────────────────────────────────
    def _tool_spawn_agent(self, args: dict[str, Any]) -> str:
        record = self.spawn_agent(
            str(args.get('task_name', '')),
            str(args.get('message', '')),
            str(args.get('agent_type', '') or self.config.agents.default_type),
            str(args.get('fork_turns', '') or self.config.agents.default_fork_turns),
        )
        return _json({'task_name': record.scope.name, 'status': record.status.value,
                      'agent_type': record.scope.agent_type.name})

    def _tool_spawn_team(self, args: dict[str, Any]) -> str:
        records = self.spawn_team(
            str(args.get('team_name', '')),
            str(args.get('message', '')),
            str(args.get('fork_turns', '') or self.config.agents.default_fork_turns),
        )
        return _json({'members': [
            {'task_name': record.scope.name, 'agent_type': record.scope.agent_type.name}
            for record in records
        ]})

    def _tool_send_message(self, args: dict[str, Any]) -> str:
        self.send_message(str(args.get('target', '')), str(args.get('message', '')))
        return _json({'delivered': True})

    def _tool_followup_task(self, args: dict[str, Any]) -> str:
        self.send_message(str(args.get('target', '')), str(args.get('message', '')),
                          trigger=True)
        return _json({'delivered': True, 'turn_triggered': True})

    def _tool_list_agents(self, args: dict[str, Any]) -> str:
        return _json({'agents': self.list_agents(args.get('path_prefix'))})

    def _tool_wait_agent(self, args: dict[str, Any]) -> str:
        timeout = int(args.get('timeout_ms', self.config.agents.wait_timeout_ms))
        return _json(self.wait_for_activity(timeout))

    def _tool_interrupt_agent(self, args: dict[str, Any]) -> str:
        previous = self.interrupt_agent(str(args.get('target', '')))
        return _json({'previous_status': previous})

    def _tool_update_goal(self, args: dict[str, Any]) -> str:
        if self.goal_store is None:
            return 'Error: goal mode is not initialized'
        goal = self.goal_store.active()
        if goal is None:
            return 'Error: there is no active goal'
        status = str(args.get('status', '')).strip()
        if status not in {'active', 'paused', 'completed', 'blocked', 'cancelled'}:
            return 'Error: invalid goal status'
        result = str(args.get('result', '')).strip()
        self.goal_store.update(goal, status=status,  # type: ignore[arg-type]
                               result=result or None)
        return _json({'goal_id': goal.id, 'status': goal.status, 'result': goal.result})

    # ── spawning ────────────────────────────────────────────
    def _resolve_type(self, name: str) -> AgentTypeConfig:
        agent_type = self.config.agent_types.get(name)
        if agent_type is None:
            fallback = self.config.agents.default_type
            agent_type = self.config.agent_types.get(fallback)
        if agent_type is None:
            agent_type = AgentTypeConfig(name=name)
        return agent_type

    def _resolve_provider_model(self, caller: AgentScope,
                                agent_type: AgentTypeConfig) -> tuple[ProviderConfig, ModelConfig]:
        override = (agent_type.model or '').strip()
        if not override:
            return copy.deepcopy(caller.provider), copy.deepcopy(caller.model)
        if '::' in override:
            provider_name, model_id = override.split('::', 1)
            provider = self.config.provider_by_name(provider_name)
            if provider is not None:
                model = provider.model(model_id)
                if model is not None:
                    return copy.deepcopy(provider), copy.deepcopy(model)
        model = caller.provider.model(override)
        if model is not None:
            return copy.deepcopy(caller.provider), copy.deepcopy(model)
        return copy.deepcopy(caller.provider), copy.deepcopy(caller.model)

    def spawn_agent(self, task_name: str, message: str, agent_type_name: str = 'general',
                    fork_turns: str = 'all') -> AgentRecord:
        if not self.enabled:
            raise RuntimeError('sub-agents are disabled in the configuration')
        caller = self.current_scope()
        task_name = task_name.strip()
        message = message.strip()
        if not _TASK_NAME_RE.fullmatch(task_name):
            raise ValueError('task_name must contain only lowercase letters, digits, '
                             'and underscores')
        if not message:
            raise ValueError('message must not be empty')
        depth = caller.depth + 1
        if depth > self.max_depth:
            raise RuntimeError(f'agent depth limit reached ({self.max_depth})')
        canonical = f'{caller.name}/{task_name}'
        agent_type = self._resolve_type(agent_type_name or self.config.agents.default_type)
        with self._lock:
            if self._shutdown:
                raise RuntimeError('agent manager is shutting down')
            if canonical in self._records:
                raise ValueError(f'agent task already exists: {canonical}')
            active = sum(
                record.thread is not None
                or record.status in {AgentStatus.PENDING_INIT, AgentStatus.RUNNING}
                for name, record in self._records.items() if name != ROOT_AGENT
            )
            if active >= self.max_subagents:
                raise RuntimeError(f'sub-agent limit reached ({self.max_subagents})')
            context, system, allowed = self._fork_context(caller, agent_type, fork_turns)
            provider, model = self._resolve_provider_model(caller, agent_type)
            scope = AgentScope(canonical, caller.name, depth, agent_type, provider,
                               model, system, context, allowed)
            record = AgentRecord(scope)
            self._records[canonical] = record
            self._wait_snapshots[canonical] = (
                {name: candidate.updated_at
                 for name, candidate in self._records.items() if name != canonical},
                record.updated_at,
            )
            self._start_record(record, message)
        return record

    def spawn_team(self, team_name: str, message: str,
                   fork_turns: str = 'all') -> list[AgentRecord]:
        if not self.config.agents.team_enabled:
            raise RuntimeError('agent teams are disabled in the configuration')
        members = self.config.agent_teams.get(team_name)
        if not members:
            raise ValueError(f'unknown agent team: {team_name}')
        records: list[AgentRecord] = []
        failures: list[str] = []
        with self._lock:
            self._team_spawn_count = getattr(self, '_team_spawn_count', 0) + 1
            spawn_index = self._team_spawn_count
        for index, member_type in enumerate(members, 1):
            base_name = f'{_sanitize_task(team_name)}_{spawn_index}_{index}'
            try:
                records.append(self.spawn_agent(base_name, message, member_type, fork_turns))
            except Exception as error:
                failures.append(f'{member_type}: {error}')
        if not records:
            raise RuntimeError('team spawn failed: ' + '; '.join(failures))
        return records

    def _fork_context(self, parent: AgentScope, agent_type: AgentTypeConfig,
                      fork_turns: str) -> tuple[ConversationContext, str, set[str]]:
        fork_turns = (fork_turns or self.config.agents.default_fork_turns).strip().lower()
        history: list[dict[str, Any]] = []
        # Include the turn that is still in flight: turns commit atomically, so
        # committed alone would be empty while the parent is working.
        parent_messages = parent.context.request_messages()
        if fork_turns == 'all':
            history = _safe_history(parent_messages)
        elif fork_turns == 'none':
            history = []
        else:
            try:
                count = int(fork_turns)
            except ValueError as error:
                raise ValueError('fork_turns must be none, all, or a positive integer') \
                    from error
            if count <= 0:
                raise ValueError('fork_turns must be none, all, or a positive integer')
            groups = _conversation_groups(parent_messages)
            history = _safe_history(
                [item for group in groups[-count:] for item in group])
        parts = [parent.system, SUBAGENT_SYSTEM]
        if agent_type.system:
            parts.append(agent_type.system)
        if self.config.agents.system_suffix:
            parts.append(self.config.agents.system_suffix)
        child_system = '\n\n'.join(part for part in parts if part)
        allowed: set[str] | None = None
        if agent_type.tools:
            allowed = set(agent_type.tools)
        wire = parent.provider.wire
        tools = self.registry.schemas(wire, allowed, vision=parent.model.vision)
        child_prefix = StablePrefix.create(
            provider=parent.provider.name, endpoint=parent.provider.base_url,
            model=parent.model.id, system=child_system, tools=tools,
            generation=1, wire=wire,
        )
        return ConversationContext(child_prefix, history), child_system, allowed

    # ── control plane ───────────────────────────────────────
    def list_agents(self, path_prefix: Any = None) -> list[dict[str, Any]]:
        prefix = str(path_prefix or '').strip().strip('/')
        with self._lock:
            result = []
            for name in sorted(self._records):
                if prefix and not name.startswith(prefix):
                    continue
                record = self._records[name]
                status: Any = record.status.value
                if record.status == AgentStatus.COMPLETED:
                    status = {'completed': record.final_message or None}
                elif record.status == AgentStatus.ERRORED:
                    status = {'errored': record.error}
                result.append({
                    'agent_name': name,
                    'agent_type': record.scope.agent_type.name,
                    'agent_status': status,
                })
            return result

    def send_message(self, target: str, message: str, trigger: bool = False) -> None:
        message = message.strip()
        if not message:
            raise ValueError('message must not be empty')
        caller = self.current_scope()
        record = self._resolve(target, caller)
        if trigger and record.scope.name == ROOT_AGENT:
            raise ValueError('follow-up tasks cannot target the root agent')
        start_new_turn = False
        with self._lock:
            record.updated_at = time.time()
            if trigger:
                record.pending_turns.append(message)
                if record.status == AgentStatus.INTERRUPTED:
                    record.restart_after_interrupt = True
                start_new_turn = record.status not in {
                    AgentStatus.PENDING_INIT, AgentStatus.RUNNING}
            else:
                record.inbox.append(f'From {caller.name}: {message}')
            self._activity.notify_all()
        self._emit(AgentEvent(record.scope.name, 'message', content=message))
        if start_new_turn:
            self._start_record(record, None)

    def interrupt_agent(self, target: str) -> str:
        caller = self.current_scope()
        record = self._resolve(target, caller)
        if record.scope.name == ROOT_AGENT:
            raise ValueError('root is not a spawned agent')
        if record.scope.name == caller.name:
            raise ValueError('an agent cannot interrupt itself')
        with self._lock:
            previous = record.status.value
            if record.status not in {AgentStatus.PENDING_INIT, AgentStatus.RUNNING}:
                return previous
            if record.status == AgentStatus.PENDING_INIT and record.pending_turns:
                record.pending_turns.popleft()
            record.restart_after_interrupt = bool(record.pending_turns)
            record.interrupt.set()
            record.status = AgentStatus.INTERRUPTED
            record.updated_at = time.time()
            self._activity.notify_all()
        self._emit(AgentEvent(record.scope.name, 'status',
                              status=AgentStatus.INTERRUPTED.value))
        return previous

    def wait_for_activity(self, timeout_ms: int = 30_000) -> dict[str, Any]:
        timeout_ms = max(0, min(3_600_000, int(timeout_ms)))
        deadline = time.monotonic() + timeout_ms / 1000
        caller_name = self.current_scope().name
        with self._activity:
            while True:
                caller = self._records.get(caller_name)
                current = {name: record.updated_at
                           for name, record in self._records.items()
                           if name != caller_name}
                previous, seen_mailbox = self._wait_snapshots.get(
                    caller_name, (current, caller.updated_at if caller else 0))
                changed = [name for name, updated_at in current.items()
                           if updated_at != previous.get(name)]
                mailbox_changed = bool(
                    caller and caller.inbox and caller.updated_at != seen_mailbox)
                if changed or mailbox_changed:
                    self._wait_snapshots[caller_name] = (
                        current, caller.updated_at if caller else seen_mailbox)
                    if mailbox_changed and not changed:
                        changed = [caller_name]
                    return {'timed_out': False, 'updated_agents': changed}
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._wait_snapshots[caller_name] = (
                        current, caller.updated_at if caller else seen_mailbox)
                    return {'timed_out': True, 'updated_agents': []}
                self._activity.wait(remaining)

    def drain_events(self, limit: int = 1_000) -> list[AgentEvent]:
        events: list[AgentEvent] = []
        while len(events) < limit:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                break
        return events

    def consume_mailbox(self, target: str = ROOT_AGENT) -> list[str]:
        with self._lock:
            record = self._records.get(target)
            if record is None:
                return []
            messages = list(record.inbox)
            record.inbox.clear()
            return messages

    def requeue_mailbox(self, messages: list[str], target: str = ROOT_AGENT) -> None:
        """Put consumed updates back when the receiving turn was rolled back."""
        if not messages:
            return
        with self._lock:
            record = self._records.get(target)
            if record is None:
                return
            for message in reversed(messages):
                record.inbox.appendleft(message)
            record.updated_at = time.time()
            self._activity.notify_all()

    def shutdown(self) -> None:
        changed: list[AgentRecord] = []
        with self._lock:
            self._shutdown = True
            records = [record for name, record in self._records.items()
                       if name != ROOT_AGENT]
            for record in records:
                record.interrupt.set()
                if record.thread is not None:
                    record.status = AgentStatus.SHUTDOWN
                    record.updated_at = time.time()
                    changed.append(record)
            self._activity.notify_all()
        for record in changed:
            self._emit(AgentEvent(record.scope.name, 'status',
                                  status=AgentStatus.SHUTDOWN.value))
        for record in records:
            if record.thread and record.thread.is_alive():
                record.thread.join(timeout=1)

    # ── worker lifecycle ────────────────────────────────────
    def _start_record(self, record: AgentRecord, message: str | None) -> None:
        with self._lock:
            if self._shutdown:
                raise RuntimeError('agent manager is shutting down')
            if record.thread is not None:
                return
            record.interrupt.clear()
            record.restart_after_interrupt = False
            record.status = AgentStatus.PENDING_INIT
            if message:
                record.pending_turns.appendleft(message)
            record.thread = threading.Thread(
                target=self._agent_worker, args=(record,),
                name=f'adcode-agent-{record.scope.name.replace("/", "-")}',
                daemon=True,
            )
            self._emit(AgentEvent(record.scope.name, 'status',
                                  status=AgentStatus.PENDING_INIT.value))
            try:
                record.thread.start()
            except Exception:
                record.thread = None
                record.status = AgentStatus.ERRORED
                record.updated_at = time.time()
                self._activity.notify_all()
                raise

    def _agent_worker(self, record: AgentRecord) -> None:
        current_thread = threading.current_thread()
        try:
            with self.scope(record.scope):
                while True:
                    terminal_status: AgentStatus | None = None
                    task = ''
                    with self._lock:
                        if self._shutdown:
                            terminal_status = AgentStatus.SHUTDOWN
                        elif record.interrupt.is_set():
                            if record.restart_after_interrupt and record.pending_turns:
                                # A follow-up arrived while interrupting the
                                # pending task; keep the worker alive for it.
                                record.interrupt.clear()
                                record.restart_after_interrupt = False
                            else:
                                terminal_status = AgentStatus.INTERRUPTED
                        if terminal_status is None:
                            if not record.pending_turns:
                                if record.thread is current_thread:
                                    record.thread = None
                                record.updated_at = time.time()
                                self._activity.notify_all()
                                return
                            task = record.pending_turns.popleft()
                            queued = list(record.inbox)
                            record.inbox.clear()
                            if queued:
                                task = (f'{task}\n\n<agent_mailbox>\n'
                                        + '\n'.join(queued) + '\n</agent_mailbox>')
                            record.status = AgentStatus.RUNNING
                            record.updated_at = time.time()
                            self._activity.notify_all()
                        else:
                            record.status = terminal_status
                            record.updated_at = time.time()
                            self._activity.notify_all()
                    if terminal_status is not None:
                        self._emit(AgentEvent(record.scope.name, 'status',
                                              status=terminal_status.value))
                        if terminal_status == AgentStatus.INTERRUPTED:
                            self._notify_parent(record, record.final_message,
                                                terminal_status)
                        self._clear_thread(record, current_thread)
                        return

                    self._emit(AgentEvent(record.scope.name, 'status',
                                          status=AgentStatus.RUNNING.value))
                    try:
                        result = self._run_react_turn(record, task)
                        if self._shutdown:
                            status = AgentStatus.SHUTDOWN
                        elif record.interrupt.is_set():
                            status = AgentStatus.INTERRUPTED
                        else:
                            status = AgentStatus.COMPLETED
                        error_message = ''
                    except Exception as error:
                        record.scope.context.rollback()
                        result = str(error)
                        if self._shutdown:
                            status = AgentStatus.SHUTDOWN
                        elif record.interrupt.is_set():
                            status = AgentStatus.INTERRUPTED
                        else:
                            status = AgentStatus.ERRORED
                        error_message = str(error)
                    with self._lock:
                        record.status = status
                        if status != AgentStatus.ERRORED:
                            record.final_message = result
                        record.error = error_message
                        record.updated_at = time.time()
                        self._activity.notify_all()
                    self._emit(AgentEvent(record.scope.name, 'status',
                                          content=result, status=status.value))
                    if status != AgentStatus.SHUTDOWN:
                        self._notify_parent(record, result, status)

                    with self._lock:
                        if (status == AgentStatus.INTERRUPTED
                                and record.restart_after_interrupt
                                and record.pending_turns and not self._shutdown):
                            record.interrupt.clear()
                            record.restart_after_interrupt = False
                        should_continue = (
                            not self._shutdown
                            and not record.interrupt.is_set()
                            and bool(record.pending_turns)
                        )
                        if not should_continue and record.thread is current_thread:
                            record.thread = None
                        record.updated_at = time.time()
                        self._activity.notify_all()
                    if not should_continue:
                        return
        finally:
            restart = False
            with self._lock:
                if record.thread is current_thread:
                    record.thread = None
                    restart = (not self._shutdown and not record.interrupt.is_set()
                               and bool(record.pending_turns))
                    record.updated_at = time.time()
                    self._activity.notify_all()
            if restart:
                self._start_record(record, None)

    def _clear_thread(self, record: AgentRecord, thread: threading.Thread) -> None:
        with self._lock:
            if record.thread is thread:
                record.thread = None
            record.updated_at = time.time()
            self._activity.notify_all()

    # ── the ReAct loop ──────────────────────────────────────
    def _run_react_turn(self, record: AgentRecord, task: str) -> str:
        scope = record.scope
        context = scope.context
        context.stage({'role': 'user', 'content': task})
        client = LLMClient(scope.provider)
        client.base_dir = self.cwd
        client.file_cache = self.file_cache
        last_answer = ''
        answer = ''
        turns = 0
        continuations = 0
        tool_context = ToolContext(
            cwd=self.cwd, interrupt=record.interrupt, agent=scope.name)
        token = push_context(tool_context)
        try:
            while True:
                if record.interrupt.is_set():
                    context.rollback()
                    return last_answer
                with self._lock:
                    mailbox = list(record.inbox)
                    record.inbox.clear()
                if mailbox:
                    context.stage({
                        'role': 'user',
                        'content': '<agent_mailbox>\n' + '\n'.join(mailbox)
                                   + '\n</agent_mailbox>',
                    })
                turns += 1
                max_turns = scope.agent_type.max_turns
                if max_turns and turns > max_turns:
                    context.rollback()
                    return last_answer or f'agent stopped after {max_turns} turns'

                tools = self.registry.schemas(scope.provider.wire, scope.allowed_tools,
                                              vision=scope.model.vision)
                candidate = StablePrefix.create(
                    provider=scope.provider.name, endpoint=scope.provider.base_url,
                    model=scope.model.id, system=scope.system, tools=tools,
                    generation=context.stable_prefix.generation,
                    wire=scope.provider.wire,
                )
                if candidate.fingerprint != context.stable_prefix.fingerprint:
                    context.stable_prefix = StablePrefix.create(
                        provider=scope.provider.name, endpoint=scope.provider.base_url,
                        model=scope.model.id, system=scope.system, tools=tools,
                        generation=context.stable_prefix.generation + 1,
                        wire=scope.provider.wire,
                    )
                prefix = context.stable_prefix
                budget = TokenBudget(
                    context_window=scope.model.context_window,
                    reserved_output=effective_reserved(
                        scope.model.context_window,
                        self.config.context.reserved_output,
                        scope.model.max_output_tokens),
                    stable_tokens=estimate_tokens(
                        {'system': prefix.system, 'tools': prefix.tools}),
                )
                if (self.config.context.auto_compact
                        and context.needs_compaction(
                            budget, self.config.context.compact_threshold)):
                    self._emit(AgentEvent(scope.name, 'compaction_start'))

                    def compact_stream(messages: list[dict[str, Any]],
                                       **kwargs: Any):
                        for event in client.chat(
                            messages, scope.model,
                            effort=self.config.context.compaction_reasoning_effort,
                            **kwargs,
                        ):
                            if record.interrupt.is_set() or self._shutdown:
                                raise RuntimeError('compaction interrupted')
                            yield event

                    stream_compact(
                        context, budget.usable() - estimate_tokens(context.draft),
                        compact_stream,
                        lambda delta: self._emit(
                            AgentEvent(scope.name, 'compaction_delta', delta)),
                        self.config.context.keep_recent_turns,
                    )
                    self._emit(AgentEvent(scope.name, 'compaction_done'))
                context.enforce_budget(budget, self.config.context.keep_recent_turns)
                prefix = context.stable_prefix

                thinking = ''
                full = ''
                tool_calls: list[dict[str, Any]] = []
                usage = None
                finish_reason = ''
                for event in client.chat(
                    context.request_messages(), scope.model,
                    system=scope.system,
                    max_tokens=max(256, min(self.config.max_output_tokens,
                                            scope.model.max_output_tokens,
                                            budget.reserved_output)),
                    tools=tools,
                    prompt_cache_key=prefix.fingerprint,
                    effort=self.config.reasoning.effort,
                    force_reasoning=self.config.reasoning.force,
                ):
                    if record.interrupt.is_set():
                        context.rollback()
                        return (answer + full) or last_answer
                    kind = event.get('kind', '')
                    if kind == 'thinking':
                        delta = str(event.get('content', ''))
                        thinking += delta
                        self._emit(AgentEvent(scope.name, 'thinking_delta', delta))
                    elif kind == 'text':
                        delta = str(event.get('content', ''))
                        full += delta
                        self._emit(AgentEvent(scope.name, 'text_delta', delta))
                    elif kind == 'tool_call':
                        tool_calls.append(event)
                    elif kind == 'usage':
                        normalized = normalize_usage(event.get('usage'))
                        if (normalized.get('input_tokens') is not None
                                or normalized.get('output_tokens') is not None):
                            usage = merge_usage(usage, normalized)
                            self._emit(AgentEvent(scope.name, 'usage',
                                                  content=_json(usage)))
                    elif kind == 'finish':
                        finish_reason = str(event.get('reason', ''))
                if tool_calls:
                    stamp = int(time.time() * 1000)
                    call_ids = [
                        str(call.get('id') or f'call_{stamp}_{index}')
                        for index, call in enumerate(tool_calls)
                    ]
                    image_markers: list[str] = []
                    draft: list[dict[str, Any]] = [{
                        'role': 'assistant',
                        'content': full or None,
                        'reasoning_content': thinking,
                        'tool_calls': [{
                            'id': call_ids[index], 'type': 'function',
                            'function': {
                                'name': call.get('name', ''),
                                'arguments': _json(call.get('args', {})),
                            },
                        } for index, call in enumerate(tool_calls)],
                    }]
                    for index, call in enumerate(tool_calls):
                        if record.interrupt.is_set():
                            context.rollback()
                            return last_answer
                        name = str(call.get('name', ''))
                        args = call.get('args', {}) if isinstance(call.get('args'), dict) else {}
                        self._emit(AgentEvent(scope.name, 'tool_start',
                                              tool_name=name, tool_args=args))
                        tool_context.live_sink = (
                            lambda line, agent_name=scope.name: self._emit(
                                AgentEvent(agent_name, 'tool_output_line', content=line)))
                        try:
                            if self.handles_tool(name):
                                text = self._execute_collaboration(name, args)
                                detailed = ToolResult(text)
                            else:
                                detailed = self.registry.execute(name, args)
                        finally:
                            tool_context.live_sink = None
                        normalized = normalize_tool_result(name, detailed.text)['text']
                        self._emit(AgentEvent(
                            scope.name, 'tool_result', content=normalized,
                            tool_name=name, tool_args=args,
                            file_diff=detailed.file_diff))
                        draft.append({
                            'role': 'tool', 'tool_call_id': call_ids[index],
                            'content': normalized,
                        })
                        for reference in IMAGE_MARKER_RE.findall(normalized):
                            marker = image_marker(reference)
                            if marker not in image_markers:
                                image_markers.append(marker)
                    if image_markers:
                        draft.append({
                            'role': 'user',
                            'content': '<tool_images>\n'
                                       + '\n'.join(image_markers)
                                       + '\n</tool_images>',
                        })
                    context.stage(*draft)
                    last_answer = full or last_answer
                    answer = ''
                    continue

                if finish_reason == 'length' and continuations < MAX_CONTINUATIONS:
                    continuation = ''
                    if thinking:
                        continuation += f'<reasoning>\n{thinking}\n</reasoning>\n'
                    if full:
                        continuation += f'<output>\n{full}\n</output>\n'
                    continuation += (
                        '\n[The previous response was cut off by the output token '
                        'limit. Continue exactly where it stopped. Do not repeat '
                        'anything above.]')
                    context.stage({'role': 'assistant', 'content': continuation})
                    answer += full
                    continuations += 1
                    continue

                answer += full
                if full:
                    context.stage({
                        'role': 'assistant', 'content': full,
                        'reasoning_content': thinking,
                    })
                context.commit()
                return answer or last_answer or '(no response)'
        finally:
            pop_context(token)

    def _execute_collaboration(self, name: str, args: dict[str, Any]) -> str:
        spec = next((item for item in self.tool_specs() if item.name == name), None)
        if spec is None:
            return f'Error: unknown collaboration tool: {name}'
        try:
            return spec.handler(args)
        except Exception as error:
            return f'Error executing {name}: {error}'

    # ── parent notifications ────────────────────────────────
    def _notify_parent(self, record: AgentRecord, result: str,
                       status: AgentStatus) -> None:
        parent_name = record.scope.parent
        if not parent_name:
            return
        message = (
            f'<subagent_update agent="{record.scope.name}" '
            f'status="{status.value}">\n{result}\n</subagent_update>'
        )
        with self._lock:
            parent = self._records.get(parent_name)
            if parent is not None:
                parent.inbox.append(message)
                parent.updated_at = time.time()
                self._activity.notify_all()

    def _resolve(self, target: str, caller: AgentScope) -> AgentRecord:
        normalized = target.strip().strip('/')
        with self._lock:
            if normalized in self._records:
                return self._records[normalized]
            relative = f'{caller.name}/{normalized}'
            if relative in self._records:
                return self._records[relative]
            matches = [record for name, record in self._records.items()
                       if name.endswith('/' + normalized)]
            if len(matches) == 1:
                return matches[0]
        raise ValueError(f'agent not found: {target}')

    def _emit(self, event: AgentEvent) -> None:
        self._events.put(event)
        if self.event_sink is not None:
            try:
                self.event_sink(event)
            except Exception:
                pass


def _safe_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe: list[dict[str, Any]] = []
    for message in messages:
        role = message.get('role')
        if role == 'user':
            safe.append(copy.deepcopy(message))
        elif role == 'assistant' and not message.get('tool_calls'):
            safe.append(copy.deepcopy(message))
    return safe


def _conversation_groups(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for message in messages:
        if message.get('role') == 'user' or not groups:
            groups.append([])
        groups[-1].append(message)
    return groups


def _sanitize_task(name: str) -> str:
    cleaned = re.sub(r'[^a-z0-9_]+', '_', name.strip().lower()).strip('_')
    return cleaned or 'team'


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'))
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)
