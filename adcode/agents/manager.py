"""Sub-agent collaboration on the engine sdk.

Delegation, mailboxes, agent types and waiting come from
``harness.agent.collab``; this module keeps what stays application-shaped:
the agent types the config defines (mapped onto the sdk's ``AgentTypeSpec``
registry), the model resolver behind ``provider::model``, the identity swap
in a child's system prompt, the ``update_goal`` tool and the ``AgentEvent``
stream the TUI pumps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from harness.agent.collab import (
    AgentManager as SdkAgentManager,
    AgentRecord,
    AgentTypeSpec,
)
from harness.agent.engine import ConversationEngine
from harness.agent.engine_config import AgentSettings, ModelInfo

from adcode.config import (
    AgentTypeConfig, AgentsSettings, AppConfig, ModelConfig, ProviderConfig,
)
from adcode.system_prompt import RUNTIME_NAME, replace_identity
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
    registered agent type.
    """

    def __init__(self, config: AppConfig, settings: AgentsSettings,
                 registry: ManagedToolRegistry, client: Any,
                 model_info: ModelInfo, *,
                 provider: ProviderConfig | None = None,
                 model_config: ModelConfig | None = None,
                 goal_store=None,
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
            cwd=cwd, store=store, model_resolver=self._resolve_model,
        )
        self.app_config = config
        self.settings = settings
        self.goal_store = goal_store
        self._client_factory = client_factory
        self._shared_client = client
        self._provider = provider
        self._model_config = model_config
        for name, agent_type in config.agent_types.items():
            self.register_agent_type(AgentTypeSpec(
                name=name, system=agent_type.system, model=agent_type.model,
                tools=list(agent_type.tools)))

    def set_model(self, provider: ProviderConfig, model: ModelConfig,
                  model_info: ModelInfo, client: Any) -> None:
        """Future children spawn with the new model; the client is shared."""
        self._provider = provider
        self._model_config = model
        super().set_model(model_info, client)

    # ── the sdk's model hook ────────────────────────────────
    def _resolve_model(self, override: str) -> tuple[Any, ModelInfo] | None:
        """Resolve an agent type's model into (client, ModelInfo).

        The override is ``provider::model`` or a bare model id looked up in
        the current provider.  ``None`` keeps the parent's model.
        """
        override = (override or '').strip()
        if not override:
            return None
        if '::' in override:
            provider_name, model_id = override.split('::', 1)
            provider = self.app_config.provider_by_name(provider_name)
            model = provider.model(model_id) if provider is not None else None
            if model is None:
                model = ModelConfig(id=model_id)
            if provider is None:
                provider = self._provider
        else:
            provider = self._provider
            model = (provider.model(override) if provider is not None
                     else None)
            if model is None:
                model = ModelConfig(id=override)
        if (provider is self._provider and model is not None
                and self._model_config is not None
                and model.id == self._model_config.id):
            return self._shared_client, self._model
        if self._client_factory is None:
            return self._shared_client, _model_info(provider, model)
        client = self._client_factory(provider, model)
        return client, _model_info(provider, model)

    # ── the child's system prompt ───────────────────────────
    def _parent_system(self, parent_record: AgentRecord,
                       model: ModelInfo | None = None) -> str:
        """The parent prompt a child starts from, with the identity swapped.

        The child may run a different model; without the swap it would claim
        to be the parent's model.
        """
        prompt = parent_record.engine.system_prompt()
        if model is not None and model.id:
            prompt = replace_identity(
                prompt, model.id, model.provider,
                agent_name=RUNTIME_NAME, role='sub-agent')
        return prompt

    def extra_tools(self) -> list[AnnotatedTool]:
        """The goal tool rides along with the collaboration suite."""
        return self._goal_tools()

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

    # ── the activity stream, in transcript vocabulary ───────
    def drain_agent_events(self, limit: int = 1_000) -> list[AgentEvent]:
        """Convert the sdk activity stream into the transcript vocabulary."""
        events: list[AgentEvent] = []
        while len(events) < limit:
            batch = self.drain_activity(limit - len(events))
            if not batch:
                break
            events.extend(self._as_agent_event(kind, payload)
                          for kind, payload in batch)
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

    # ── compatibility ───────────────────────────────────────
    def records(self) -> dict[str, AgentRecord]:
        """The live agent records, keyed by name."""
        return self._agents
