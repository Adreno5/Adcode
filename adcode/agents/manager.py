"""Sub-agent collaboration on the engine sdk.

Delegation, mailboxes and waiting come from ``harness.agent.collab``; this
module adds what stays application-shaped: adcode's agent types (per-type
model, system and tool allowlist), the ``update_goal`` tool backed by the
GoalStore, and the ``AgentEvent`` stream the TUI pumps.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from harness.agent.collab import (
    ACTIVE_STATES,
    AgentManager as SdkAgentManager,
    AgentRecord,
    ROOT_AGENT,
    SUBAGENT_SYSTEM,
)
from harness.agent.engine import ConversationEngine
from harness.agent.engine_config import AgentSettings, EngineConfig, ModelInfo

from adcode.config import AgentsSettings, AppConfig, ModelConfig, ProviderConfig
from adcode.tools.registry import AnnotatedTool, ManagedToolRegistry


class AgentStatus:
    """Sub-agent lifecycle names (the transcript vocabulary)."""

    PENDING_INIT = 'pending_init'
    RUNNING = 'running'
    INTERRUPTED = 'interrupted'
    COMPLETED = 'completed'
    ERRORED = 'errored'
    SHUTDOWN = 'shutdown'


@dataclass
class AgentEvent:
    """One child-agent activity item, consumed by ``pump_agent_events``."""

    agent_name: str
    kind: str
    content: str = ''
    tool_name: str = ''
    tool_args: dict[str, Any] = field(default_factory=dict)
    status: str = ''
    file_diff: Any = None
    created_at: float = field(default_factory=time.time)


def _child_sink(engine: ConversationEngine,
                sink_into: Callable[[tuple[str, dict[str, Any]]], None]) -> Any:
    from harness.agent.events import EventSink

    def emit(kind: str, payload: dict[str, Any]) -> None:
        sink_into((kind, {'agent': engine.agent_name, **payload}))

    return EventSink(emit=emit)


def _model_info(provider: ProviderConfig, model: ModelConfig) -> ModelInfo:
    if model is None:
        model = ModelConfig()
    if provider is None:
        provider = ProviderConfig(name='', base_url='', api_key='', wire='',
                                  models=[])
    wire = ('anthropic' if provider.wire == 'anthropic messages'
            else 'responses' if provider.wire == 'openai responses' else 'chat')
    return ModelInfo(
        id=model.id, provider=provider.name, endpoint=provider.base_url,
        wire=wire, context_window=model.context_window,
        max_output_tokens=model.max_output_tokens, vision=model.vision,
        reasoning_style=model.reasoning_style,
        reasoning_efforts=tuple(model.reasoning_efforts or ()),
        reasoning=bool(model.reasoning),
    )


class AgentManager(SdkAgentManager):
    """Delegation and mailboxes with adcode's agent types and goal tool.

    The root engine is attached with ``attach``; children are spawned through
    the ``spawn_agent`` tool.  Each child is a full ``ConversationEngine``
    sharing the registry; its model / system / tool allowlist come from the
    configured agent type.
    """

    def __init__(self, config: AppConfig, settings: AgentsSettings,
                 registry: ManagedToolRegistry, client: Any,
                 model_info: ModelInfo, *, goal_store=None,
                 client_factory: Optional[
                     Callable[[ProviderConfig, ModelConfig], Any]] = None,
                 cwd: str = '', store=None) -> None:
        super().__init__(
            client=client, registry=registry, model=model_info,
            config=AgentSettings(
                max_concurrent=settings.max_concurrent,
                max_depth=settings.max_depth,
                wait_timeout_ms=settings.wait_timeout_ms,
                default_fork_turns=settings.default_fork_turns,
                show_activity=settings.show_activity,
                show_output=settings.show_output,
                default_type=settings.default_type,
                system_suffix=settings.system_suffix,
            ),
            cwd=cwd, store=store,
        )
        self.app_config = config
        self.settings = settings
        self.goal_store = goal_store
        self._client_factory = client_factory
        self._shared_client = client
        self._provider: ProviderConfig | None = None
        self._model_config: ModelConfig | None = None
        # Child activity for the facade's pump.  The engine-side pump is
        # deliberately inert (drain_events returns empty) so exactly one
        # consumer - the facade - turns these into AgentEvents.
        self._agent_events: deque[tuple[str, dict[str, Any]]] = deque(maxlen=2_000)

    # ── wiring ──────────────────────────────────────────────
    def attach(self, engine: ConversationEngine) -> None:
        # The facade creates the loop before run_forever and injects it here;
        # a bare get_running_loop only works when attach runs inside the loop.
        if self._loop is None or self._condition is None:
            self._loop = asyncio.get_running_loop()
            self._condition = asyncio.Condition()
        self._agents[ROOT_AGENT] = AgentRecord(
            name=ROOT_AGENT, parent=None, depth=0, engine=engine,
            status=AgentStatus.RUNNING)
        engine.collab = self
        for tool in self._collab_tools() + self._goal_tools():
            self._registry.register(tool, replace=True, source='agent')

    def reinstall_tools(self) -> None:
        """Re-register the collaboration tools after a registry refresh."""
        for tool in self._collab_tools() + self._goal_tools():
            self._registry.register(tool, replace=True, source='agent')

    def set_model(self, provider: ProviderConfig, model: ModelConfig,
                  model_info: ModelInfo, client: Any) -> None:
        """Future children spawn with the new model; the client is shared."""
        self._provider = provider
        self._model_config = model
        self._model = model_info
        self._shared_client = client

    # ── event surfaces ──────────────────────────────────────
    def drain_events(self, limit: int = 1_000) -> list[tuple[str, dict[str, Any]]]:
        # The engine's per-round pump must not consume child activity: the
        # facade is its only reader (see pump_agent_events).
        return []

    def drain_agent_events(self, limit: int = 1_000) -> list[AgentEvent]:
        events: list[AgentEvent] = []
        while len(events) < limit and self._agent_events:
            kind, payload = self._agent_events.popleft()
            events.append(self._as_agent_event(kind, payload))
        return events

    def _as_agent_event(self, kind: str, payload: dict[str, Any]) -> AgentEvent:
        agent = str(payload.get('agent', ''))
        if kind == 'usage':
            usage = {key: payload.get(key) for key in
                     ('input_tokens', 'cached_input_tokens', 'output_tokens')}
            return AgentEvent(agent, 'usage', content=json.dumps(usage))
        if kind == 'status':
            return AgentEvent(agent, 'status',
                              status=str(payload.get('status', '')))
        if kind == 'tool_start':
            return AgentEvent(agent, 'tool_start',
                              tool_name=str(payload.get('name', '')),
                              tool_args=dict(payload.get('args') or {}))
        if kind == 'tool_output':
            return AgentEvent(agent, 'tool_output_line',
                              content=str(payload.get('text', '')))
        if kind == 'tool_end':
            return AgentEvent(agent, 'tool_result',
                              content=str(payload.get('output', ''))[:500])
        if kind == 'agent_event':
            return AgentEvent(agent, 'status',
                              status=str(payload.get('status', '')),
                              content=str(payload.get('content', '')))
        if kind in ('text_delta', 'thinking_delta', 'compaction_delta'):
            return AgentEvent(agent, kind, content=str(payload.get('text', '')))
        return AgentEvent(agent, 'message', content=str(payload))

    # ── event delivery ──────────────────────────────────────
    def _deliver_to_parent(self, record: AgentRecord) -> None:
        """Mailbox delivery + lifecycle event, into THIS manager's queue.

        The base class feeds its own ``_events`` deque; the facade consumes
        ``_agent_events`` (drain_events stays inert on purpose), so the
        lifecycle announcement lands there too.
        """
        parent = self._agents.get(record.parent) if record.parent else None
        if parent is None:
            return
        body = record.error or record.final_message
        parent.inbox.append(
            f'<subagent_update agent="{record.name}" '
            f'status="{record.status}">\n{body}\n</subagent_update>')
        self._agent_events.append(('agent_event', {
            'agent': record.name, 'status': record.status, 'content': body}))
        self._agent_events.append(('agents_refresh', {}))
        self._touch(record)

    # ── spawning with agent types ───────────────────────────
    async def spawn(self, name: str, message: str, *,
                    agent_type: str = 'general',
                    fork_turns: str | int = 'all',
                    system: str = '') -> AgentRecord:
        if not re.match(r'^[a-z0-9_]+$', str(name)):
            raise ValueError('task_name must match ^[a-z0-9_]+$')
        name = str(name).strip().lower()
        if not message.strip():
            raise ValueError('message must not be empty')
        parent = self._current_name()
        parent_record = self._agents.get(parent)
        depth = (parent_record.depth + 1) if parent_record else 0
        if depth > self.config.max_depth:
            raise ValueError(f'max depth {self.config.max_depth} exceeded')
        if name in self._agents and self._agents[name].status in ACTIVE_STATES:
            raise ValueError(f'agent {name} is already active')
        active = [r for r in list(self._agents.values())
                  if r.status in ACTIVE_STATES]
        if len(active) >= max(1, self.config.max_concurrent):
            raise ValueError('too many concurrent agents')

        type_config = self._resolve_type(agent_type)
        provider, model_config, child_client = self._resolve_client(
            parent_record, type_config)
        model_info = _model_info(provider, model_config)
        allowed_tools = set(type_config.tools) if type_config.tools else None

        child = ConversationEngine(
            client=child_client, registry=self._registry, store=self._store,
            cwd=self._cwd, model=model_info,
            config=EngineConfig(agents=self.config),
            system_provider=lambda: self._child_system(parent_record,
                                                       type_config.system),
            allowed_tools=allowed_tools, agent_name=name)
        child.sink = _child_sink(child, self._agent_events.append)
        child.conversation.committed.extend(
            self._fork_history(parent_record, fork_turns))
        record = AgentRecord(name=name, parent=parent, depth=depth, engine=child)
        self._agents[name] = record
        self._touch(record)
        record.task = self._loop.create_task(self._run_child(record, message))
        return record

    def _resolve_type(self, name: str):
        agent_type = self.app_config.agent_types.get(name)
        if agent_type is None:
            agent_type = self.app_config.agent_types.get(self.settings.default_type)
        if agent_type is None:
            from adcode.config import AgentTypeConfig
            agent_type = AgentTypeConfig(name=name)
        return agent_type

    def _resolve_client(self, parent_record, type_config):
        """Resolve (provider, model, client) for a child's agent type."""
        override = (type_config.model or '').strip()
        if not override:
            return self._provider, self._model_config, self._shared_client
        if '::' in override:
            provider_name, model_id = override.split('::', 1)
            provider = self.app_config.provider_by_name(provider_name)
            model = None
            if provider is not None:
                model = provider.model(model_id)
            if model is None:
                model = ModelConfig(id=model_id)
            if provider is None:
                provider = self._provider
        else:
            provider = self._provider
            model = (provider.model(override) if provider is not None else None)
            if model is None:
                model = ModelConfig(id=override)
        if (provider is self._provider and model is not None
                and self._model_config is not None
                and model.id == self._model_config.id):
            return provider, model, self._shared_client
        if self._client_factory is None:
            return provider, model, self._shared_client
        return provider, model, self._client_factory(provider, model)

    def _fork_history(self, parent_record,
                      fork_turns: str | int) -> list[dict[str, Any]]:
        """Safe history for a child: user messages and plain assistant text.

        Tool-call exchanges belong to the turn that made them; copying them
        into a fresh child leaves dangling tool results no provider accepts.
        """
        if not parent_record or fork_turns in ('none', 'None', 0):
            return []
        safe = [copy.deepcopy(m) for m in parent_record.engine.request_messages()
                if m.get('role') == 'user'
                or (m.get('role') == 'assistant' and not m.get('tool_calls'))]
        if str(fork_turns).lower() == 'all':
            return safe
        try:
            keep = int(fork_turns)
        except (TypeError, ValueError):
            return safe
        starts = [i for i, m in enumerate(safe) if m.get('role') == 'user']
        if 0 < keep < len(starts):
            safe = safe[starts[-keep]:]
        return safe

    def _child_system(self, parent_record, type_system: str) -> str:
        parts = []
        if parent_record is not None:
            parts.append(parent_record.engine.system_prompt())
        parts.append(SUBAGENT_SYSTEM)
        if type_system:
            parts.append(type_system)
        if self.settings.system_suffix:
            parts.append(self.settings.system_suffix)
        return '\n\n'.join(p for p in parts if p)

    # ── goal tool ───────────────────────────────────────────
    def _goal_tools(self) -> list[AnnotatedTool]:
        def update_goal(status: str = '', result: str = '') -> str:
            if self.goal_store is None:
                return 'Error: goal mode is not initialized'
            goal = self.goal_store.active()
            if goal is None:
                return 'Error: there is no active goal'
            status = str(status).strip()
            if status not in {'active', 'paused', 'completed', 'blocked',
                              'cancelled'}:
                return 'Error: invalid goal status'
            result = str(result).strip()
            self.goal_store.update(goal, status=status, result=result or None)
            return json.dumps({'goal_id': goal.id, 'status': goal.status,
                               'result': goal.result})

        return [
            AnnotatedTool(
                name='update_goal',
                description=('Update the active durable goal after it is achieved '
                             'or blocked.'),
                parameters={'type': 'object', 'properties': {
                    'status': {'type': 'string',
                               'enum': ['active', 'paused', 'completed', 'blocked',
                                        'cancelled']},
                    'result': {'type': 'string'},
                }, 'required': ['status']},
                handler=update_goal,
            ),
        ]

    # ── compatibility shims ─────────────────────────────────
    def root_scope(self) -> contextlib.AbstractContextManager:
        """Identity scope: per-agent isolation comes from ToolContext.agent."""
        return contextlib.nullcontext()

    def scope(self, scope: Any) -> contextlib.AbstractContextManager:
        return contextlib.nullcontext()

    def current_scope(self) -> str:
        return self._current_name()

    @property
    def enabled(self) -> bool:
        return self.settings.max_concurrent > 0

    def records(self) -> dict[str, AgentRecord]:
        return self._agents
