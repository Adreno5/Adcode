"""The conversation engine facade.

The engine itself (turn queue, status machine, compaction, sessions,
collaboration) lives in the sdk: ``harness.agent.ConversationEngine``.  This
module is the adapter the TUI talks to.  It keeps the historical contract:

- the engine runs on a worker thread with its own event loop; the UI thread
  calls the control plane and drains ``(kind, payload)`` events, whose
  vocabulary is unchanged;
- ``submit``/``request_compact`` are thread-safe; loop-affine operations
  (sessions, model changes, tool refreshes) hop onto the engine loop and
  surface their exceptions to the caller;
- sub-agent activity reaches the UI through ``pump_agent_events`` exactly as
  before, converted from the collab manager's event stream.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
import time
from collections import deque
from typing import Any, Callable

from harness.agent import ConversationEngine, EngineConfig, EngineSnapshot
from harness.agent.engine_config import (
    AgentSettings, ContextSettings, HoldSettings, ModelInfo,
    ReasoningSettings, ThinkingSettings,
)
from harness.agent.events import EventSink
from harness.images import FileIdCache
from harness.transcript import SessionPayload, TranscriptItem
from infra.orchestrator.failover import FailoverChat, FailoverEndpoint

from adcode.agents.manager import AgentManager, AgentStatus
from adcode.config import AppConfig, FailoverSettings, ModelConfig, ProviderConfig
from adcode.llm.client import LLMClient
from adcode import paths
from adcode.system_prompt import build_system_prompt
from adcode.tools.builtin import install_coding_tools, shutdown_shell_sessions

# The snapshot the TUI renders is the engine's own dataclass; the fields are
# identical (status, turn, token counters, context window, session identity,
# queued depth, agent rows).
AgentSnapshot = EngineSnapshot

_MODEL_WIRES = {
    'openai chat completions': 'chat',
    'openai responses': 'responses',
    'anthropic messages': 'anthropic',
}


class _WorkQueue:
    """A minimal FIFO with exact ``qsize`` semantics for the bench.

    An item stays queued until the pump has actually handed it to the engine
    *and* the engine turned busy, so ``qsize()`` never drops a pending turn
    from under ``wait_idle``.
    """

    def __init__(self) -> None:
        self._items: deque[tuple[str, str]] = deque()
        self._condition = threading.Condition()
        self._stop_requested = False

    def put(self, item: tuple[str, str]) -> None:
        with self._condition:
            self._items.append(item)
            self._condition.notify_all()

    def qsize(self) -> int:
        with self._condition:
            return len(self._items)

    def empty(self) -> bool:
        return self.qsize() == 0

    def peek(self) -> tuple[str, str] | None:
        with self._condition:
            return self._items[0] if self._items else None

    def pop(self) -> None:
        with self._condition:
            if self._items:
                self._items.popleft()

    def get(self, timeout: float = 0.2) -> tuple[str, str] | None:
        """Blocking pop; None on timeout or shutdown."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._items:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stop_requested:
                    return None
                self._condition.wait(remaining)
            return self._items.popleft()


class AgentLoop:
    """Owns the sdk engine behind a thread-safe, event-driven facade."""

    def __init__(self, config: AppConfig, registry, store: Any,
                 goals, skills, emit: Callable[[str, dict[str, Any]], None] | None = None,
                 mcp_manager: Any = None, cwd: str = '') -> None:
        self.config = config
        self.mcp = mcp_manager
        self.cwd = cwd
        self.goals = goals
        self.skills = skills
        self._registry = registry
        self._store = store
        self._wake = emit or (lambda kind, payload: None)
        self._stop = False
        self._work = _WorkQueue()
        self._sink = EventSink(emit=self._on_engine_event)

        self._selected = config.selected()
        if self._selected is None:
            raise RuntimeError('no provider/model configured')
        self.provider, self.model = self._selected
        self._file_cache = FileIdCache(str(paths.image_file_ids()))
        # The app hands us an empty registry; the built-in suite is installed
        # here, and the collab / skill sources follow the engine's lifecycle.
        install_coding_tools(registry)
        registry.unregister_source('collab')
        registry.unregister_source('skill')
        # The engine talks to one IChatClient; with failover it is a chain
        # that tries the other providers when the selected one is down.
        self._endpoints: list[LLMClient] = []
        self._rebuild_clients(config.failover)

        self.agent_live: dict[str, dict[str, str]] = {}
        self._engine: ConversationEngine | None = None
        self.agents: AgentManager | None = None
        self._thread: threading.Thread | None = None
        self._pump: threading.Thread | None = None
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    def _rebuild_clients(self, failover: FailoverSettings) -> None:
        """Build the selected client and, when enabled, the fallback chain.

        A fallback is the same model id on another provider when it has one,
        its first model otherwise; at most three extra endpoints are built.
        """
        primary = self._make_client(self.provider, self.model)
        self._endpoints = [primary]
        if failover.enabled:
            for provider in self.config.providers:
                if provider is self.provider or not provider.models:
                    continue
                if len(self._endpoints) >= 4:
                    break
                model = provider.model(self.model.id) or provider.models[0]
                self._endpoints.append(self._make_client(provider, model))
        if len(self._endpoints) == 1:
            self.client = primary
            return
        self.client = FailoverChat([
            FailoverEndpoint(f'{client.provider.name}::{client.model}',
                             client)
            for client in self._endpoints
        ])

    # ── construction helpers ────────────────────────────────
    def _on_engine_event(self, kind: str, payload: dict[str, Any]) -> None:
        """Every engine event passes through here before the UI sees it."""
        try:
            self._wake(kind, payload)
        except Exception:
            pass

    def _make_client(self, provider: ProviderConfig, model: ModelConfig) -> LLMClient:
        return LLMClient(provider, model, base_dir=self.cwd or '.',
                         file_cache=self._file_cache, warn=self._warn)

    def _warn(self, text: str) -> None:
        engine = self._engine
        if engine is not None:
            engine.add_notice(text, 'warning')

    def _model_info(self, provider: ProviderConfig, model: ModelConfig) -> ModelInfo:
        wire = _MODEL_WIRES.get(provider.wire, 'chat')
        return ModelInfo(
            id=model.id, provider=provider.name, endpoint=provider.base_url,
            wire=wire, context_window=model.context_window,
            max_output_tokens=model.max_output_tokens, vision=model.vision,
            reasoning_style=model.reasoning_style,
            reasoning_efforts=tuple(model.reasoning_efforts or ()),
            reasoning=bool(model.reasoning),
        )

    def _engine_config(self, config: AppConfig) -> EngineConfig:
        show_completed = (config.thinking.show_completed
                          and config.thinking.mode != 'hidden')
        return EngineConfig(
            context=ContextSettings(
                auto_compact=config.context.auto_compact,
                compact_threshold=config.context.compact_threshold,
                keep_recent_turns=config.context.keep_recent_turns,
                reserved_output=config.context.reserved_output,
                compaction_reasoning_effort=config.context.compaction_reasoning_effort,
            ),
            reasoning=ReasoningSettings(effort=config.reasoning.effort,
                                        force=config.reasoning.force),
            thinking=ThinkingSettings(show_completed=show_completed),
            agents=AgentSettings(
                max_concurrent=config.agents.max_concurrent,
                max_depth=config.agents.max_depth,
                wait_timeout_ms=config.agents.wait_timeout_ms,
                default_fork_turns=config.agents.default_fork_turns,
                show_activity=config.agents.show_activity,
                show_output=config.agents.show_output,
                default_type=config.agents.default_type,
                system_suffix=config.agents.system_suffix,
            ),
            hold=HoldSettings(enabled=config.hold.enabled,
                              max_retries=config.hold.max_retries,
                              delay=config.hold.delay),
            max_output_tokens=config.max_output_tokens,
        )

    def apply_hold(self) -> None:
        """Push the ``/hold`` settings into the engine."""
        engine = self._engine
        if engine is None:
            return
        hold = self.config.hold
        engine.configure_hold(hold.enabled, hold.max_retries, hold.delay)

    def _system_provider(self) -> str:
        # The model id is the live one: a model change must be reflected in
        # the next request's system prompt.
        model = self.model
        provider = self.provider
        return build_system_prompt(self.config, self.goals, self.skills,
                                   cwd=self.cwd or '.',
                                   vision=bool(model.vision),
                                   model_id=model.id, provider=provider.name)

    # ── lifecycle ───────────────────────────────────────────
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop_thread,
                                        name='adcode-engine-loop', daemon=True)
        self._pump = threading.Thread(target=self._pump_thread,
                                      name='adcode-agent-pump', daemon=True)
        self._thread.start()
        self._ready.wait(timeout=15)
        self._pump.start()

    def _loop_thread(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        engine = ConversationEngine(
            client=self.client, registry=self._registry,
            store=self._store,
            sink=self._sink, system_provider=self._system_provider,
            cwd=self.cwd or '.', model=self._model_info(self.provider, self.model),
            config=self._engine_config(self.config))
        self._engine = engine
        # Every endpoint stops its stream at the next chunk when the turn is
        # interrupted, so an interrupt is not stuck behind a slow response.
        for endpoint in self._endpoints:
            endpoint.interrupt = engine.interrupt_event
        self.apply_hold()
        if self.config.agents.enabled:
            self.agents = AgentManager(
                self.config, self.config.agents, self._registry,
                client=self.client,
                model_info=self._model_info(self.provider, self.model),
                provider=self.provider, model_config=self.model,
                goal_store=self.goals,
                client_factory=self._client_for,
                cwd=self.cwd or '.', store=self._store)

        def boot() -> None:
            # attach and engine.start need a *running* loop, so they run from
            # the loop's own callback queue right after run_forever begins
            if self.agents is not None:
                self.agents.attach(engine)
            engine.start()
            self._ready.set()

        loop.call_soon_threadsafe(boot)
        try:
            loop.run_forever()
        finally:
            loop.close()

    def _client_for(self, provider: ProviderConfig, model: ModelConfig) -> LLMClient:
        """Build a dedicated client for a child agent's provider::model."""
        return LLMClient(provider, model, base_dir=self.cwd or '.',
                         file_cache=self._file_cache, warn=self._warn)

    def _sync_config(self, engine: ConversationEngine) -> None:
        """Copy the live config knobs into the engine before a turn.

        The engine snapshots settings at construction; commands like /effort
        mutate the AppConfig afterwards and expect the next request to honor
        them, exactly like the previous loop read them per request.
        """
        engine.config.reasoning.effort = self.config.reasoning.effort
        engine.config.reasoning.force = self.config.reasoning.force
        engine.config.max_output_tokens = self.config.max_output_tokens
        context = engine.config.context
        context.auto_compact = self.config.context.auto_compact
        context.compact_threshold = self.config.context.compact_threshold
        context.keep_recent_turns = self.config.context.keep_recent_turns
        context.reserved_output = self.config.context.reserved_output
        context.compaction_reasoning_effort = \
            self.config.context.compaction_reasoning_effort

    def _submit_now(self, mode: str, text: str) -> None:
        # The queued event was emitted with the facade-side queue position
        # already, so the engine's own copy is suppressed.
        engine = self._engine
        if engine is None:
            return
        self._sync_config(engine)
        engine.submit(text, mode, announce=False)

    def _pump_thread(self) -> None:
        """Hand queued turns to the engine, in order.

        The engine's own work queue serializes turns (its _run loop pops one
        turn at a time), so the pump only ferries items across the thread
        boundary; qsize() is the honest not-yet-submitted count.
        """
        while not self._stop:
            item = self._work.get(timeout=0.2)
            if item is None or self._stop:
                continue
            mode, text = item
            try:
                self._call(lambda: self._submit_now(mode, text))
            except RuntimeError:
                continue

    def _call(self, fn: Callable[[], Any]) -> Any:
        """Run a callable on the engine loop and wait for it."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return fn()
        future: concurrent.futures.Future = concurrent.futures.Future()

        def runner() -> None:
            try:
                future.set_result(fn())
            except BaseException as error:    # noqa: BLE001 - propagate to caller
                future.set_exception(error)

        loop.call_soon_threadsafe(runner)
        return future.result(timeout=30)

    # ── control plane (thread-safe) ─────────────────────────
    def submit(self, text: str, mode: str = 'chat') -> bool:
        if not text.strip():
            return False
        engine = self._engine
        if engine is not None and not engine.busy():
            engine.clear_interrupt()
        self._work.put((mode, text))
        self._emit('queued', text=text, mode=mode,
                   position=self._work.qsize())
        return True

    def interrupt(self) -> None:
        engine = self._engine
        if engine is not None:
            engine.interrupt()

    stop_turn = interrupt

    def request_compact(self) -> None:
        engine = self._engine
        if engine is not None and not engine.busy():
            engine.clear_interrupt()
        self._work.put(('compact', ''))

    def add_notice(self, text: str, level: str = 'info') -> None:
        engine = self._engine
        if engine is not None:
            engine.add_notice(text, level)
        else:
            self._emit('notice', text=text, level=level)

    def add_diff_item(self, display: str, file_diff: Any | None) -> None:
        engine = self._engine
        if engine is not None:
            engine.add_diff_item(display, file_diff)
        else:
            self._emit('notice', text='No working-tree changes.')

    def set_model(self, provider: ProviderConfig, model: ModelConfig) -> None:
        self._require_idle('model change')
        model_info = self._model_info(provider, model)
        self.provider = provider
        self.model = model
        # The chain is rebuilt around the new selection: the engine and the
        # agent manager only ever see the primary client.
        primary = self._endpoints[0]
        if (self._endpoints and primary.provider is provider
                and primary.model == model.id):
            client = self.client
        else:
            self._rebuild_clients(self.config.failover)
            client = self._endpoints[0]
        self._call(lambda: self._apply_model(provider, model, model_info,
                                             client))

    def _apply_model(self, provider, model, model_info, client) -> None:
        engine = self._engine
        if engine is not None:
            engine.set_client(self.client, model_info)
            for endpoint in self._endpoints:
                endpoint.interrupt = engine.interrupt_event
        if self.agents is not None:
            self.agents.set_model(provider, model, model_info, client)

    def refresh_tools(self) -> None:
        self._require_idle('tool refresh')
        registry = self._registry
        registry.unregister_source('collab')
        registry.unregister_source('skill')
        if self.agents is not None:
            self.agents.reinstall_tools()
        specs = self.skills.tool_specs() if self.skills is not None else []
        for tool in specs:
            registry.register(tool, replace=True, source='skill')
        self._call(lambda: self._engine.refresh_tools())

    def shutdown(self) -> None:
        self._stop = True
        engine = self._engine
        if engine is not None:
            engine.interrupt()
        self._drain_work()
        if self._loop is not None and not self._loop.is_closed():
            try:
                if self.agents is not None:
                    asyncio.run_coroutine_threadsafe(self.agents.shutdown(),
                                                     self._loop).result(timeout=5)
                if engine is not None:
                    asyncio.run_coroutine_threadsafe(engine.shutdown(),
                                                     self._loop).result(timeout=5)
            except concurrent.futures.TimeoutError:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._pump is not None:
            self._pump.join(timeout=2)
        shutdown_shell_sessions()

    def _drain_work(self) -> None:
        self._work._stop_requested = True
        with self._work._condition:
            self._work._condition.notify_all()
        self._work._items.clear()

    # ── snapshots ───────────────────────────────────────────
    @property
    def busy(self) -> bool:
        engine = self._engine
        return engine.busy() if engine is not None else False

    @property
    def status(self) -> str:
        engine = self._engine
        return engine.status if engine is not None else 'idle'

    @property
    def items(self) -> list[TranscriptItem]:
        engine = self._engine
        return engine.items if engine is not None else []

    @property
    def context(self) -> Any:
        return self._engine.conversation

    @property
    def usage_records(self) -> list[dict[str, Any]]:
        engine = self._engine
        return engine.usage_records() if engine is not None else []

    @property
    def session_name(self) -> str:
        engine = self._engine
        return engine.session_name if engine is not None else ''

    @property
    def session_title(self) -> str:
        engine = self._engine
        return engine.session_title if engine is not None else ''

    def snapshot(self) -> AgentSnapshot:
        engine = self._engine
        if engine is None:
            return AgentSnapshot(provider=self.provider.name,
                                 model=self.model.id,
                                 effort=self.config.reasoning.effort)
        snap = engine.snapshot()
        snap.queued = self._work.qsize()
        return snap

    def items_snapshot(self) -> list[TranscriptItem]:
        engine = self._engine
        return engine.items_snapshot() if engine is not None else []

    def drain_ui_events(self, limit: int = 2_000) -> list[tuple[str, dict]]:
        return self._sink.drain(limit)

    # ── sessions ────────────────────────────────────────────
    def _require_idle(self, action: str) -> None:
        if self.busy:
            raise RuntimeError(f'{action} while a turn is still running')

    def payload(self) -> SessionPayload:
        return self._sync_call(lambda: self._engine.payload())

    def save_active(self) -> None:
        self._sync_call(lambda: self._engine.save_active())

    def save_as(self, name: str) -> None:
        self._require_idle('session rename')
        self._sync_call(lambda: self._engine.save_as(name))

    def load_session(self, name: str) -> bool:
        self._require_idle('session load')
        return self._sync_call(lambda: self._engine.load_session(name))

    def apply_payload(self, payload: SessionPayload) -> None:
        self._require_idle('session load')
        self._sync_call(lambda: self._engine.apply_payload(payload))

    def clear(self) -> None:
        self._require_idle('session clear')

        def reset() -> None:
            self._engine.clear()
            # The pane is empty again; the header bar starts clean too.
            self._engine.session_title = ''

        self._sync_call(reset)

    def forget_session(self) -> None:
        """Drop the session identity after its row was deleted.

        The engine names a session on its first save and keeps that name; a
        deleted session must not be resurrected by the next message under the
        old name.
        """
        self._require_idle('session delete')

        def reset() -> None:
            self._engine.session_name = ''
            self._engine.session_title = ''

        self._sync_call(reset)

    def rewind(self, turn: int) -> str:
        self._require_idle('rewind')
        return self._sync_call(lambda: self._engine.rewind(turn))

    def fork(self, name: str = '') -> str:
        self._require_idle('fork')
        return self._sync_call(lambda: self._engine.fork(name))

    def _sync_call(self, fn: Callable[[], Any]) -> Any:
        """Run an engine method; on the loop when it is alive, direct if not."""
        if self._loop is not None and not self._loop.is_closed():
            return self._call(fn)
        return fn()

    # ── sub-agent pump ──────────────────────────────────────
    def pump_agent_events(self) -> None:
        """Fold child-agent activity into the transcript and live view."""
        manager = self.agents
        if manager is None:
            return
        dirty = False
        show_activity = self.config.agents.show_activity
        show_output = self.config.agents.show_output
        for event in manager.drain_agent_events():
            name = event.agent_name
            if event.kind in ('thinking_delta', 'text_delta', 'compaction_delta'):
                mode = {'thinking_delta': 'thinking', 'text_delta': 'answer',
                        'compaction_delta': 'compacting'}[event.kind]
                live = self.agent_live.setdefault(
                    name, {'mode': mode, 'text': ''})
                live['mode'] = mode
                live['text'] = (live['text'] + event.content)[-4_000:]
                dirty = True
            elif event.kind == 'usage':
                try:
                    usage = json.loads(event.content)
                except ValueError:
                    usage = {}
                self._emit('usage', **usage)
            elif event.kind == 'status':
                status = event.status
                if status in (AgentStatus.PENDING_INIT, AgentStatus.RUNNING):
                    continue
                content = event.content or ''
                if status == AgentStatus.COMPLETED and content:
                    content = f'completed · {content[:300]}'
                self._emit('agent_event', agent=name, status=status,
                           content=content)
                dirty = True
            elif event.kind == 'tool_start' and show_activity:
                brief = _tool_brief(event.tool_args)
                self._emit('agent_event', agent=name, status='tool',
                           content=f'{event.tool_name}{brief}')
                dirty = True
            elif event.kind == 'tool_result' and show_output:
                self._emit('agent_event', agent=name, status='tool_result',
                           content=event.content[:500])
                dirty = True
            elif event.kind == 'tool_output_line' and show_output:
                self._emit('agent_event', agent=name, status='tool_output',
                           content=event.content[-500:])
                dirty = True
        if dirty:
            self._emit('agents_refresh')

    def _emit(self, kind: str, **payload: Any) -> None:
        self._sink.emit(kind, payload)


def _tool_brief(args: dict[str, Any]) -> str:
    """A short, human-readable summary of a tool call's arguments."""
    if not isinstance(args, dict) or not args:
        return ''
    parts = []
    for key, value in list(args.items())[:3]:
        text = str(value).replace('\n', ' ')
        if len(text) > 60:
            text = text[:57] + '...'
        parts.append(f'{key}={text}')
    return f"({', '.join(parts)})" if parts else ''
