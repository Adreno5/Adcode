"""The main streaming ReAct agent loop.

The loop runs on a worker thread so the UI can render and accept input while
tokens stream in.  It owns the conversation context, executes tools, performs
automatic compaction, drains sub-agent mailboxes, and persists sessions.  The
UI observes it through a thread-safe event queue and a locked snapshot.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from adcode.agents.manager import AgentManager
from adcode.config import AppConfig, ModelConfig, ProviderConfig
from adcode.diff import FileDiff
from adcode.goals import GoalStore
from adcode.images import (
    MARKER_RE as IMAGE_MARKER_RE, FileIdCache, has_image_marker, image_marker,
    strip_image_markers,
)
from adcode.llm.client import LLMClient
from adcode.llm.context import (
    ConversationContext, StablePrefix, TokenBudgetExceeded, messages_need_repair,
    repair_messages, stream_compact,
)
from adcode.llm.usage import (
    TokenBudget, effective_reserved, estimate_tokens, merge_usage,
    normalize_tool_result, normalize_usage,
)
from adcode.session import (
    LAST_SESSION, SessionPayload, SessionStore, TranscriptItem, new_session_name,
)
from adcode.skills import SkillRegistry
from adcode.system_prompt import build_system_prompt
from adcode.tools.registry import ToolRegistry
from adcode.tools.runtime import ToolContext, pop_context, push_context

MAX_CONTINUATIONS = 3
CHECKPOINT_INTERVAL = 30.0

ASIDE_SYSTEM = (
    'Answer this side question as a brief, self-contained aside. Use the current '
    'conversation as context but do not continue the main task, do not call tools, '
    'and keep the answer short. Do not use emoji.'
)


@dataclass
class AgentSnapshot:
    status: str = 'idle'
    turn: int = 0
    turn_in: int = 0
    turn_out: int = 0
    total_in: int = 0
    total_out: int = 0
    context_tokens: int = 0
    context_window: int = 0
    provider: str = ''
    model: str = ''
    effort: str = 'medium'
    session_name: str = ''
    session_title: str = ''
    queued: int = 0
    agents: list[dict[str, Any]] | None = None


class AgentLoop:
    def __init__(self, config: AppConfig, registry: ToolRegistry,
                 store: SessionStore, goals: GoalStore, skills: SkillRegistry,
                 emit: Callable[[str, dict[str, Any]], None],
                 mcp_manager: Any = None, cwd: str = '') -> None:
        self.config = config
        self.registry = registry
        self.store = store
        self.goals = goals
        self.skills = skills
        self.emit = emit
        self.mcp_manager = mcp_manager
        # Tools and project instructions resolve against this directory, which
        # may differ from the process cwd when the app is started with --cwd.
        self.cwd = cwd or os.getcwd()

        selected = config.selected()
        if selected is None:
            raise RuntimeError('no provider/model configured')
        self.provider, self.model = selected
        # Uploaded image ids are shared by the root loop and every sub-agent.
        self.image_files = FileIdCache(os.path.join(self.cwd, 'sessions',
                                                    'images', 'files.json'))
        self.client = self._make_client(self.provider)

        self.system = build_system_prompt(config, goals, skills, cwd=self.cwd,
                                          vision=self.model.vision)
        schemas = registry.schemas(self.provider.wire, vision=self.model.vision)
        self.context = ConversationContext(
            StablePrefix.create(
                provider=self.provider.name, endpoint=self.provider.base_url,
                model=self.model.id, system=self.system, tools=schemas,
                generation=1, wire=self.provider.wire,
            ),
            [],
        )
        self.items: list[TranscriptItem] = []
        self.turn = 0
        self.session_name = ''
        self.session_title = ''
        self.session_created = ''
        self.total_in = 0
        self.total_out = 0
        self.turn_in = 0
        self.turn_out = 0
        self.usage_records: list[dict[str, Any]] = []
        self.status = 'idle'
        self.agent_live: dict[str, dict[str, str]] = {}

        self._lock = threading.RLock()
        self._events: 'queue.Queue[tuple[str, dict[str, Any]]]' = queue.Queue()
        self._work: 'queue.Queue[tuple[str, str] | None]' = queue.Queue()
        self._stop = threading.Event()
        self._interrupt = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_checkpoint = 0.0
        self._turn_started = 0.0

        # Collaboration and skill tools live in the shared registry so root and
        # sub-agents see exactly the same surface.
        self.registry.unregister_source('agent')
        self.registry.unregister_source('skill')
        self.agents = AgentManager(
            config, self.provider, self.model, self.system, self.context,
            registry, goal_store=goals, cwd=self.cwd,
            file_cache=self.image_files,
        )
        if config.agents.enabled:
            self.registry.register_many(self.agents.tool_specs())
        self.registry.register_many(self.skills.tool_specs())

        self._refresh_prefix(force=True)
        self.agents.update_root(self.context, self.provider, self.model, self.system)

    def _make_client(self, provider: ProviderConfig) -> LLMClient:
        """Build a client wired to the workspace, image store and UI notices."""
        client = LLMClient(provider)
        client.base_dir = self.cwd
        client.file_cache = self.image_files
        client.warn = lambda text: self._append_notice(text, level='warning')
        return client

    # ── lifecycle ───────────────────────────────────────────
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name='adcode-agent-loop',
                                        daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        self._interrupt.set()
        self._work.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.agents.shutdown()

    @property
    def busy(self) -> bool:
        return self.status != 'idle'

    def submit(self, text: str, mode: str = 'chat') -> bool:
        text = text.strip()
        if not text:
            return False
        if not self.busy:
            # A new turn starts clean; while a turn is running the pending
            # interrupt must survive (the user pressed Esc for that turn).
            self._interrupt.clear()
        self._work.put((mode, text))
        self._emit('queued', text=text, mode=mode,
                   position=self._work.qsize())
        return True

    def interrupt(self) -> None:
        self._interrupt.set()

    def stop_turn(self) -> None:
        self._interrupt.set()

    def _quiesce(self, timeout: float = 2.0) -> bool:
        """Interrupt the active turn and wait until the worker is idle."""
        if not self.busy:
            return True
        self._interrupt.set()
        deadline = time.monotonic() + timeout
        while self.busy and time.monotonic() < deadline:
            time.sleep(0.01)
        return not self.busy

    def _require_idle(self, action: str) -> None:
        if not self._quiesce():
            raise RuntimeError(f'{action} while a turn is still running')

    def add_notice(self, text: str, level: str = 'info') -> None:
        self._append_notice(text, level)

    def add_diff_item(self, display: str, file_diff: FileDiff | None) -> None:
        if file_diff is None:
            self._append_notice('No working-tree changes.', 'info')
            return
        with self._lock:
            self.items.append(TranscriptItem(
                kind='tool', turn=self.turn, name='diff', display=display,
                status='done', diff=_diff_dict(file_diff)))
        self._emit('diff_added')

    def request_compact(self) -> None:
        if not self.busy:
            self._interrupt.clear()
        self._work.put(('compact', ''))

    # ── event delivery ──────────────────────────────────────
    def _emit(self, kind: str, **payload: Any) -> None:
        self._events.put((kind, payload))
        try:
            self.emit(kind, payload)
        except Exception:
            pass

    def drain_ui_events(self, limit: int = 2_000) -> list[tuple[str, dict[str, Any]]]:
        events: list[tuple[str, dict[str, Any]]] = []
        while len(events) < limit:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                break
        return events

    # ── snapshots for the UI ────────────────────────────────
    def snapshot(self) -> AgentSnapshot:
        with self._lock:
            context_tokens = self.context.estimated_tokens()
            return AgentSnapshot(
                status=self.status,
                turn=self.turn,
                turn_in=self.turn_in,
                turn_out=self.turn_out,
                total_in=self.total_in,
                total_out=self.total_out,
                context_tokens=context_tokens,
                context_window=self.model.context_window,
                provider=self.provider.name,
                model=self.model.id,
                effort=self.config.reasoning.effort,
                session_name=self.session_name,
                session_title=self.session_title,
                queued=self._work.qsize(),
                agents=self._agent_rows(),
            )

    def _agent_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in self.agents.list_agents():
            name = row['agent_name']
            if name == 'root':
                continue
            live = self.agent_live.get(name, {})
            rows.append({
                'name': name,
                'type': row.get('agent_type', ''),
                'status': row.get('agent_status'),
                'live': live.get('text', ''),
                'mode': live.get('mode', ''),
            })
        return rows

    def items_snapshot(self) -> list[TranscriptItem]:
        with self._lock:
            return list(self.items)

    # ── worker ──────────────────────────────────────────────
    def _run(self) -> None:
        while not self._stop.is_set():
            task = self._work.get()
            if task is None or self._stop.is_set():
                break
            mode, text = task
            try:
                if mode == 'aside':
                    self._run_aside(text)
                elif mode == 'compact':
                    self._run_compact()
                else:
                    self._run_chat(text)
            except Exception as error:
                self._append_notice(f'Request failed: {error}', level='error')
                self._emit('error', text=str(error))
            finally:
                with self._lock:
                    if self.status != 'idle':
                        self.status = 'idle'
                        self._emit('status', status='idle')
                self._interrupt.clear()
        with self._lock:
            self.status = 'idle'

    # ── main chat turn ──────────────────────────────────────
    def _run_chat(self, text: str) -> None:
        with self._lock:
            self.turn += 1
            turn = self.turn
            if not self.session_name:
                self.session_name = self.store.unique_name(new_session_name())
                self.session_created = time.strftime('%Y-%m-%d %H:%M')
            if not self.session_title:
                self.session_title = _title_from(text)
            self.turn_in = 0
            self.turn_out = 0
            self._turn_started = time.time()
            self.items.append(TranscriptItem(kind='user', turn=turn, text=text))
            self.status = 'connecting'
        self._emit('turn_start', turn=turn, text=text)
        self._emit('status', status='connecting')
        context = self.context
        context.stage({'role': 'user', 'content': text})
        tool_context = ToolContext(cwd=self.cwd, interrupt=self._interrupt,
                                   agent='root')
        token = push_context(tool_context)
        committed = False
        interrupted = False
        continuations = 0
        pending_mailbox: list[str] = []
        try:
            with self.agents.root_scope():
                # The turn runs until the model answers without tool calls; the
                # user's interrupt and the token budget are the only ways out,
                # so long investigations are not cut off at an arbitrary count.
                while True:
                    if self._stop.is_set():
                        break
                    if self._interrupt.is_set():
                        interrupted = True
                        break
                    pending_mailbox.extend(self._stage_mailbox(context))
                    self._refresh_system_and_prefix()
                    thinking, full, tool_calls, finish_reason = self._stream_request(
                        turn, context)
                    if self._interrupt.is_set():
                        interrupted = True
                        break
                    if tool_calls:
                        if not self._execute_tools(turn, context, tool_calls,
                                                   thinking, full):
                            interrupted = True
                            break
                        self._checkpoint()
                        continue
                    if finish_reason == 'length':
                        if continuations >= MAX_CONTINUATIONS:
                            self._append_notice(
                                'Output token limit reached; continuation stopped.',
                                level='warning')
                            if full:
                                context.stage({
                                    'role': 'assistant', 'content': full,
                                    'reasoning_content': thinking,
                                })
                            context.commit()
                            committed = True
                            break
                        continuations += 1
                        continuation = _continuation_prompt(thinking, full)
                        context.stage({'role': 'assistant', 'content': continuation})
                        continue
                    if finish_reason == 'incomplete':
                        self._append_notice(
                            'The response ended without a finish signal; the answer '
                            'may be truncated.', level='warning')
                    if full:
                        context.stage({
                            'role': 'assistant', 'content': full,
                            'reasoning_content': thinking,
                        })
                    context.commit()
                    committed = True
                    break
                if not committed and not interrupted and self._interrupt.is_set():
                    interrupted = True
        except TokenBudgetExceeded as error:
            # Nothing in this turn can be sent: the history does not fit even
            # after compaction, so keeping the draft would break every later
            # request.  The already committed rounds stay.
            context.rollback()
            self._append_notice(f'Token budget exceeded: {error}', level='error')
            self._emit('error', text=str(error))
        except Exception as error:
            # A failed request is not a reason to forget the work that already
            # happened; keep the user message and every completed tool round.
            self._append_notice(f'Request failed: {error}', level='error')
            self._emit('error', text=str(error))
        finally:
            pop_context(token)
            with self._lock:
                self.status = 'idle'
            if not committed:
                # Money in the transcript -- tool calls that ran, an answer that
                # was already streaming -- must survive the turn.  Dropping the
                # whole draft on Esc used to erase minutes of investigation.
                if not self._commit_partial(context):
                    # Nothing survived (for example the budget check refused the
                    # request before it was sent): restore the consumed mailbox
                    # so the next turn still sees sub-agent results.
                    context.rollback()
                    self.agents.requeue_mailbox(pending_mailbox)
            if interrupted:
                self._append_notice('[interrupted]', level='warning')
                self._emit('interrupted')
            self._emit('turn_end', turn=turn, status='interrupted' if interrupted else 'done',
                       stats={
                           'input_tokens': self.turn_in,
                           'output_tokens': self.turn_out,
                           'duration': round(time.time() - self._turn_started, 2),
                       })
            self.save_active()

    def _stream_request(self, turn: int, context: ConversationContext
                        ) -> tuple[str, str, list[dict[str, Any]], str]:
        """Run one streamed model request and return its accumulated pieces."""
        budget = TokenBudget(
            context_window=self.model.context_window,
            reserved_output=effective_reserved(
                self.model.context_window, self.config.context.reserved_output,
                self.model.max_output_tokens),
            stable_tokens=estimate_tokens({
                'system': context.stable_prefix.system,
                'tools': context.stable_prefix.tools,
            }),
        )
        if (self.config.context.auto_compact
                and context.needs_compaction(
                    budget, self.config.context.compact_threshold)):
            self._compact(context, budget)
        context.enforce_budget(budget, self.config.context.keep_recent_turns)
        self._refresh_prefix(force=False)
        prefix = context.stable_prefix
        max_tokens = max(256, min(self.config.max_output_tokens,
                                  self.model.max_output_tokens,
                                  budget.reserved_output))
        thinking = ''
        full = ''
        tool_calls: list[dict[str, Any]] = []
        finish_reason = ''
        request_usage: dict[str, Any] | None = None
        thinking_started = time.time()
        thinking_closed = False
        text_started = 0.0

        def close_thinking() -> None:
            nonlocal thinking_closed
            if thinking_closed:
                return
            thinking_closed = True
            if thinking:
                self._append_thinking(turn, thinking,
                                      time.time() - thinking_started)
                self._emit('thinking_end', text=thinking,
                           duration=round(time.time() - thinking_started, 2))

        # Nothing is on the wire yet: the request is only sent when the
        # generator below is first advanced, and the provider can take a while
        # to produce its first byte (DNS, TLS, proxy, a cold or queued model).
        # That wait is a phase of its own -- ``connecting`` -- and it ends when
        # real model output arrives, not when the turn is scheduled.
        connecting = True

        def enter_connecting(retrying: bool = False) -> None:
            nonlocal connecting
            connecting = True
            with self._lock:
                self.status = 'connecting'
            self._emit('status', status='connecting', retrying=retrying)

        def start_streaming() -> None:
            nonlocal connecting
            if not connecting:
                return
            connecting = False
            with self._lock:
                self.status = 'thinking'
            self._emit('status', status='thinking')

        enter_connecting()
        stream_started = time.perf_counter()
        for event in self.client.chat(
            context.request_messages(), self.model,
            system=prefix.system,
            max_tokens=max_tokens,
            tools=list(prefix.tools),
            prompt_cache_key=prefix.fingerprint,
            effort=self.config.reasoning.effort,
            force_reasoning=self.config.reasoning.force,
        ):
            if self._interrupt.is_set() or self._stop.is_set():
                break
            kind = event.get('kind', '')
            if kind == 'retry':
                # The attempt failed before anything streamed, so the next one
                # is still a connection attempt, not model output.
                enter_connecting(retrying=True)
                self._append_notice(
                    f'Reconnecting ({event.get("attempt", 1)}): '
                    f'{str(event.get("content", ""))[:160]}', level='warning')
                continue
            start_streaming()
            if kind == 'thinking':
                delta = str(event.get('content', ''))
                if delta:
                    thinking += delta
                    with self._lock:
                        self.status = 'thinking'
                    self._emit('thinking_delta', text=delta)
            elif kind == 'text':
                delta = str(event.get('content', ''))
                if delta:
                    close_thinking()
                    if not full:
                        text_started = time.time()
                    full += delta
                    with self._lock:
                        self.status = 'answering'
                    self._emit('text_delta', text=delta)
            elif kind == 'tool_call':
                tool_calls.append(event)
            elif kind == 'usage':
                normalized = normalize_usage(event.get('usage'))
                if (normalized.get('input_tokens') is not None
                        or normalized.get('output_tokens') is not None):
                    request_usage = merge_usage(request_usage, normalized)
            elif kind == 'finish':
                finish_reason = str(event.get('reason', ''))
        close_thinking()
        if full:
            self._append_assistant(turn, full)
            self._emit('text_end', text=full,
                       duration=round(time.time() - text_started, 2))
        if request_usage is not None:
            request_usage.update({
                'prefix_generation': prefix.generation,
                'prefix_hash': prefix.fingerprint,
                'provider': prefix.provider,
                'model': prefix.model,
                'latency_ms': round((time.perf_counter() - stream_started) * 1000),
            })
            self._record_usage(request_usage)
        return thinking, full, tool_calls, finish_reason

    def _execute_tools(self, turn: int, context: ConversationContext,
                       tool_calls: list[dict[str, Any]], thinking: str = '',
                       full: str = '') -> bool:
        """Execute a tool batch; returns False when the turn was interrupted."""
        draft: list[dict[str, Any]] = []
        image_markers: list[str] = []
        # Providers occasionally omit call ids; the assistant message and every
        # matching tool result must agree on the same fallback id.
        stamp = int(time.time() * 1000)
        call_ids = [
            str(call.get('id') or f'call_{stamp}_{index}')
            for index, call in enumerate(tool_calls)
        ]
        with self._lock:
            self.status = 'tool'
        self._emit('status', status='tool')
        for index, call in enumerate(tool_calls):
            if self._interrupt.is_set() or self._stop.is_set():
                return False
            name = str(call.get('name', ''))
            args = call.get('args', {}) if isinstance(call.get('args'), dict) else {}
            call_id = call_ids[index]
            display = self._display_tool(name)
            item = TranscriptItem(kind='tool', turn=turn, call_id=call_id,
                                  name=name, display=display, args=args,
                                  status='running')
            with self._lock:
                self.items.append(item)
            self._emit('tool_start', turn=turn, call_id=call_id, name=name,
                       display=display, args=args)
            started = time.time()

            def sink(line: str, _call_id: str = call_id) -> None:
                self._emit('tool_output', call_id=_call_id, text=line)

            tool_context = ToolContext(cwd=self.cwd, interrupt=self._interrupt,
                                       agent='root', live_sink=sink)
            token = push_context(tool_context)
            try:
                result = self.registry.execute(name, args)
            finally:
                pop_context(token)
            duration = time.time() - started
            item.status = 'error' if result.error else 'done'
            item.output = result.text
            item.duration = duration
            item.diff = _diff_dict(result.file_diff)
            self._emit('tool_end', turn=turn, call_id=call_id, name=name,
                       display=display, output=result.text,
                       diff=item.diff, duration=round(duration, 2),
                       error=result.error)
            normalized = normalize_tool_result(name, result.text)['text']
            draft.append({
                'role': 'tool', 'tool_call_id': call_id, 'content': normalized,
            })
            for reference in IMAGE_MARKER_RE.findall(normalized):
                marker = image_marker(reference)
                if marker not in image_markers:
                    image_markers.append(marker)
            if name == 'update_goal':
                self._refresh_system_and_prefix(force=True)
            if self._interrupt.is_set() or self._stop.is_set():
                # Persist the calls that did run before the interrupt, so the
                # model does not have to repeat them next turn.
                self._stage_tool_round(context, tool_calls[:index + 1],
                                       call_ids[:index + 1], draft, thinking,
                                       full, image_markers)
                return False

        # Persist the assistant tool-call message together with its results.
        self._stage_tool_round(context, tool_calls, call_ids, draft, thinking,
                               full, image_markers)
        with self._lock:
            self.status = 'connecting'
        self._emit('status', status='connecting')
        return True

    def _stage_tool_round(self, context: ConversationContext,
                          tool_calls: list[dict[str, Any]],
                          call_ids: list[str], results: list[dict[str, Any]],
                          thinking: str, full: str,
                          image_markers: list[str]) -> None:
        """Stage one assistant tool-call message and its results."""
        assistant_message: dict[str, Any] = {
            'role': 'assistant',
            'content': full or None,
            'reasoning_content': thinking,
            'tool_calls': [{
                'id': call_ids[index], 'type': 'function',
                'function': {
                    'name': str(call.get('name', '')),
                    'arguments': json.dumps(call.get('args', {}), ensure_ascii=False),
                },
            } for index, call in enumerate(tool_calls)],
        }
        context.stage(assistant_message)
        context.stage(*results)
        if image_markers:
            # Tool results are plain text on every wire, so the picture travels
            # in its own user block right after them.  The leading tag keeps
            # turn_starts() from treating it as a user turn.
            context.stage({
                'role': 'user',
                'content': '<tool_images>\n' + '\n'.join(image_markers)
                           + '\n</tool_images>',
            })

    # ── partial turns ───────────────────────────────────────
    def _commit_partial(self, context: ConversationContext) -> bool:
        """Commit the completed part of a turn that never reached its answer.

        A turn that ran tools and then failed or was interrupted used to be
        rolled back whole: the model lost every file it had read, every command
        it had run and every result it had seen, so it started the next turn
        from scratch (and often repeated the same work).  Only *complete* tool
        rounds survive -- a half-streamed reply or a call that never ran is
        dropped, because neither can be sent to a provider as-is.

        Returns False when nothing worth keeping was staged.
        """
        draft = context.draft
        if not draft:
            return False
        # Keep the longest prefix that ends with a tool result: that is exactly
        # the part where every tool call has an answer.
        cut = -1
        for index, message in enumerate(draft):
            if message.get('role') == 'tool':
                cut = index
        if cut < 0:
            return False
        kept = draft[:cut + 1]
        repaired, changed = repair_messages(kept)
        if changed:
            # A batch that was cut in half leaves calls without results; the
            # synthetic result keeps the history sendable.
            self._append_notice(
                f'Kept {len(kept)} message(s) of the interrupted turn; '
                f'{changed} unfinished step(s) were closed.', level='warning')
        draft[:] = repaired
        context.commit()
        return True

    # ── compaction ──────────────────────────────────────────
    def _compact(self, context: ConversationContext, budget: TokenBudget) -> None:
        self._emit('compaction_start')
        with self._lock:
            self.status = 'compacting'
        self._emit('status', status='compacting')
        parts: list[str] = []
        usage: dict[str, Any] | None = None

        def emit_delta(delta: str) -> None:
            parts.append(delta)
            self._emit('compaction_delta', text=delta)

        def compact_stream(messages: list[dict[str, Any]], **kwargs: Any):
            nonlocal usage
            for event in self.client.chat(
                messages, self.model,
                effort=self.config.context.compaction_reasoning_effort,
                **kwargs,
            ):
                if event.get('kind') == 'usage':
                    normalized = normalize_usage(event.get('usage'))
                    if (normalized.get('input_tokens') is not None
                            or normalized.get('output_tokens') is not None):
                        usage = merge_usage(usage, normalized)
                if self._interrupt.is_set() or self._stop.is_set():
                    # stream_compact catches this and falls back to the
                    # deterministic summary instead of keeping partial text.
                    raise RuntimeError('compaction interrupted')
                yield event

        removed = stream_compact(
            context, max(1024, budget.usable() - estimate_tokens(context.draft)),
            compact_stream, emit_delta, self.config.context.keep_recent_turns,
        )
        if usage is not None:
            usage.update({'provider': self.provider.name, 'model': self.model.id,
                          'prefix_generation': context.stable_prefix.generation})
            self._record_usage(usage)
        text = ''.join(parts).strip()
        summary = text or '(deterministic summary)'
        self._append_notice(
            f'Context compacted: {removed} messages summarized.', level='info')
        self._emit('compaction_end', text=summary, removed=removed)
        self._refresh_prefix(force=True)
        # The follow-up request has not been sent yet, so the turn returns to
        # the connection phase rather than to "thinking".
        with self._lock:
            self.status = 'connecting'
        self._emit('status', status='connecting')

    def _run_compact(self) -> None:
        if self._interrupt.is_set() or self._stop.is_set():
            self._append_notice('Compaction cancelled.', level='info')
            return
        budget = TokenBudget(
            context_window=self.model.context_window,
            reserved_output=effective_reserved(
                self.model.context_window, self.config.context.reserved_output,
                self.model.max_output_tokens),
            stable_tokens=estimate_tokens({
                'system': self.context.stable_prefix.system,
                'tools': self.context.stable_prefix.tools,
            }),
        )
        self._compact(self.context, budget)
        self._append_notice('Manual compaction finished.', level='info')

    # ── aside questions ─────────────────────────────────────
    def _run_aside(self, question: str) -> None:
        if self._interrupt.is_set() or self._stop.is_set():
            return
        with self._lock:
            self.status = 'aside'
        self._emit('aside_start', question=question)
        self._emit('status', status='aside')
        messages = list(self.context.committed) + [
            {'role': 'user', 'content': question},
        ]
        parts: list[str] = []
        usage: dict[str, Any] | None = None
        started = time.time()
        try:
            for event in self.client.chat(
                messages, self.model,
                system=f'{self.system}\n\n{ASIDE_SYSTEM}',
                max_tokens=max(256, min(4096, self.config.max_output_tokens,
                                        self.model.max_output_tokens)),
                tools=None,
                effort='low',
                force_reasoning=self.config.reasoning.force,
            ):
                if self._interrupt.is_set() or self._stop.is_set():
                    break
                kind = event.get('kind')
                if kind == 'text':
                    delta = str(event.get('content', ''))
                    parts.append(delta)
                    self._emit('aside_delta', text=delta)
                elif kind == 'usage':
                    normalized = normalize_usage(event.get('usage'))
                    if (normalized.get('input_tokens') is not None
                            or normalized.get('output_tokens') is not None):
                        usage = merge_usage(usage, normalized)
        except Exception as error:
            self._emit('error', text=str(error))
        if usage is not None:
            usage.update({
                'provider': self.provider.name, 'model': self.model.id,
                'prefix_generation': self.context.stable_prefix.generation,
                'latency_ms': round((time.time() - started) * 1000),
            })
            self._record_usage(usage)
        answer = ''.join(parts).strip()
        with self._lock:
            self.items.append(TranscriptItem(kind='aside', turn=self.turn,
                                             text=answer, display=question))
            self.status = 'idle'
        self._emit('aside_end', question=question, answer=answer)
        self._emit('status', status='idle')
        self.save_active()

    # ── transcript helpers ──────────────────────────────────
    def _append_thinking(self, turn: int, text: str, duration: float) -> None:
        mode = self.config.thinking.mode
        if not self.config.thinking.show_completed or mode == 'hidden':
            return
        with self._lock:
            self.items.append(TranscriptItem(
                kind='thinking', turn=turn, text=text, duration=round(duration, 2)))

    def _append_assistant(self, turn: int, text: str) -> None:
        if not text:
            return
        with self._lock:
            self.items.append(TranscriptItem(kind='assistant', turn=turn, text=text))

    def _append_notice(self, text: str, level: str = 'info') -> None:
        with self._lock:
            self.items.append(TranscriptItem(kind='notice', turn=self.turn,
                                             text=text, level=level))
        self._emit('notice', text=text, level=level)

    def _stage_mailbox(self, context: ConversationContext) -> list[str]:
        mailbox = self.agents.consume_mailbox()
        if mailbox:
            context.stage({
                'role': 'user',
                'content': '<subagent_updates>\n' + '\n'.join(mailbox)
                           + '\n</subagent_updates>',
            })
            self._emit('notice', text=f'Received {len(mailbox)} sub-agent update(s).',
                       level='info')
        return mailbox

    def _display_tool(self, name: str) -> str:
        if self.mcp_manager is not None and name.startswith('mcp__'):
            return self.mcp_manager.tool_display(name)
        return name

    # ── sub-agent events ────────────────────────────────────
    def pump_agent_events(self) -> None:
        """Drain sub-agent events into the transcript and live status."""
        settings = self.config.agents
        drained = self.agents.drain_events()
        for event in drained:
            try:
                if event.kind in ('thinking_delta', 'text_delta', 'compaction_delta'):
                    live = self.agent_live.setdefault(event.agent_name, {})
                    mode = {'thinking_delta': 'thinking',
                            'text_delta': 'answer',
                            'compaction_delta': 'compacting'}[event.kind]
                    if live.get('mode') != mode:
                        live['mode'] = mode
                        live['text'] = ''
                    live['text'] = (live.get('text', '') + event.content)[-4000:]
                    continue
                if event.kind == 'usage':
                    # Sub-agent tokens are real spending; fold them into the
                    # session totals instead of dropping the event.
                    try:
                        usage = json.loads(event.content or '{}')
                    except ValueError:
                        continue
                    if not isinstance(usage, dict):
                        continue
                    with self._lock:
                        self.total_in += int(usage.get('input_tokens') or 0)
                        self.total_out += int(usage.get('output_tokens') or 0)
                    self._emit('usage', **usage)
                    continue
                if event.kind == 'status':
                    self.agent_live.pop(event.agent_name, None)
                    if event.status in ('pending_init', 'running'):
                        if event.status == 'pending_init' and settings.show_activity:
                            self._append_agent_item(event.agent_name, 'started')
                        continue
                    if settings.show_activity:
                        detail = event.content.strip() if event.content else event.status
                        self._append_agent_item(event.agent_name,
                                                f'{event.status} · {detail[:300]}'
                                                if event.status not in ('completed',)
                                                else f'completed · {detail[:300]}')
                    self._emit('agent_event', agent=event.agent_name,
                               status=event.status, content=event.content)
                elif event.kind == 'tool_start' and settings.show_activity:
                    self._append_agent_item(
                        event.agent_name,
                        f'{event.tool_name} {_tool_brief(event.tool_args)}'.strip())
                elif event.kind == 'tool_result' and settings.show_output:
                    self._append_agent_item(
                        event.agent_name, event.content[:500])
                elif event.kind == 'tool_output_line' and settings.show_output:
                    self.agent_live.setdefault(event.agent_name, {})
                    self.agent_live[event.agent_name]['text'] = event.content[-500:]
            except Exception:
                continue
        if drained:
            self._emit('agents_refresh')

    def _append_agent_item(self, agent: str, text: str) -> None:
        with self._lock:
            self.items.append(TranscriptItem(kind='agent', turn=self.turn,
                                             agent=agent, text=text))

    # ── system/prefix management ────────────────────────────
    def _refresh_system_and_prefix(self, force: bool = False) -> None:
        system = build_system_prompt(self.config, self.goals, self.skills,
                                     cwd=self.cwd,
                                     vision=self.model.vision)
        if force or system != self.context.stable_prefix.system:
            if system != self.context.stable_prefix.system:
                with self._lock:
                    self.system = system
            self._refresh_prefix(force=True)
            self.agents.update_root(self.context, self.provider, self.model, self.system)

    def _refresh_prefix(self, force: bool = False) -> None:
        schemas = self.registry.schemas(self.provider.wire,
                                        vision=self.model.vision)
        current = self.context.stable_prefix
        # ``self.system`` is authoritative: ``_refresh_system_and_prefix`` has
        # already folded new goals, instructions or skills into it.  Using the
        # old prefix text here silently discarded those updates.
        system = self.system
        candidate = StablePrefix.create(
            provider=self.provider.name, endpoint=self.provider.base_url,
            model=self.model.id, system=system, tools=schemas,
            generation=current.generation, wire=self.provider.wire,
        )
        if force or candidate.fingerprint != current.fingerprint:
            self.context.stable_prefix = StablePrefix.create(
                provider=self.provider.name, endpoint=self.provider.base_url,
                model=self.model.id, system=system, tools=schemas,
                generation=current.generation + 1, wire=self.provider.wire,
            )
        self.agents.update_root(self.context, self.provider, self.model, self.system)

    # ── usage ───────────────────────────────────────────────
    def _record_usage(self, usage: dict[str, Any]) -> None:
        with self._lock:
            input_tokens = usage.get('input_tokens') or 0
            output_tokens = usage.get('output_tokens') or 0
            self.turn_in += input_tokens
            self.turn_out += output_tokens
            self.total_in += input_tokens
            self.total_out += output_tokens
            self.usage_records.append(dict(usage))
            del self.usage_records[:-200]
        self._emit('usage', **usage)

    # ── persistence ─────────────────────────────────────────
    def payload(self) -> SessionPayload:
        with self._lock:
            context = self.context
            # Include the in-flight turn so a crash during a long autonomous
            # run loses at most one checkpoint interval, not the whole turn.
            messages = [dict(message) for message in context.committed]
            messages.extend(dict(message) for message in context.draft)
            return SessionPayload(
                name=self.session_name,
                title=self.session_title,
                created=self.session_created,
                provider=self.provider.name,
                model=self.model.id,
                turn=self.turn,
                prefix_generation=context.stable_prefix.generation,
                messages=messages,
                compaction_archive=list(context.archived),
                items=list(self.items),
            )

    def save_active(self) -> None:
        payload = self.payload()
        if not payload.items and not payload.messages:
            return
        try:
            if payload.name:
                self.store.save(payload)
            self.store.save(payload, LAST_SESSION)
        except Exception:
            pass

    def _checkpoint(self) -> None:
        now = time.time()
        if now - self._last_checkpoint < CHECKPOINT_INTERVAL:
            return
        self._last_checkpoint = now
        self.save_active()

    def save_as(self, name: str) -> None:
        name = name.strip()
        if not name:
            raise ValueError('session name must not be empty')
        with self._lock:
            previous = self.session_name
            self.session_name = name
        payload = self.payload()
        self.store.save(payload)
        self.store.save(payload, LAST_SESSION)
        if previous and previous != name:
            self._append_notice(f'Session saved as {name}.', level='info')

    def load_session(self, name: str) -> bool:
        payload = self.store.load(name)
        if payload is None:
            return False
        self.apply_payload(payload)
        return True

    def apply_payload(self, payload: SessionPayload) -> None:
        self._require_idle('cannot load a session')
        provider = self.config.provider_by_name(payload.provider) or self.provider
        model = provider.model(payload.model) or (
            provider.models[0] if provider.models else self.model)
        with self._lock:
            self.provider = provider
            self.model = model
            self.client = self._make_client(provider)
            self.config.select(provider.name, model.id)
            self.session_name = payload.name if payload.name != LAST_SESSION else ''
            self.session_title = payload.title
            self.session_created = payload.created
            self.turn = payload.turn
            self.items = list(payload.items)
            self.context.committed[:] = payload.messages
            self.context.draft.clear()
            self.context.archived[:] = payload.compaction_archive
            self.context.invalidate()
            self.system = build_system_prompt(self.config, self.goals, self.skills,
                                              cwd=self.cwd,
                                              vision=model.vision)
            prefix = StablePrefix.create(
                provider=provider.name, endpoint=provider.base_url,
                model=model.id, system=self.system,
                tools=self.registry.schemas(provider.wire, vision=model.vision),
                generation=max(1, payload.prefix_generation), wire=provider.wire,
            )
            self.context.stable_prefix = prefix
            self.turn_in = 0
            self.turn_out = 0
            self.status = 'idle'
        self.agents.update_root(self.context, self.provider, self.model, self.system)
        self._emit('session_loaded', name=self.session_name or payload.name,
                   title=self.session_title)

    def clear(self) -> None:
        self._require_idle('cannot clear the conversation')
        with self._lock:
            self.context.committed.clear()
            self.context.draft.clear()
            self.context.archived.clear()
            self.context.invalidate()
            self.items.clear()
            self.turn = 0
            self.session_name = ''
            self.session_title = ''
            self.session_created = ''
            self.turn_in = 0
            self.turn_out = 0
            self.total_in = 0
            self.total_out = 0
            self.usage_records.clear()
            self.context.stable_prefix = StablePrefix.create(
                provider=self.provider.name, endpoint=self.provider.base_url,
                model=self.model.id, system=self.system,
                tools=self.registry.schemas(self.provider.wire,
                                            vision=self.model.vision),
                generation=1, wire=self.provider.wire,
            )
        try:
            self.store.save(SessionPayload(name=''), LAST_SESSION)
        except Exception:
            pass
        self._emit('session_cleared')

    def rewind(self, turn: int) -> str:
        """Drop the given turn and everything after it; returns the user text."""
        self._require_idle('cannot rewind the conversation')
        with self._lock:
            from adcode.session import SessionStore as _Store
            starts = _Store.turn_starts(self.context.committed)
            # Transcript turn ids survive compaction, but the retained history
            # starts later.  Align the oldest retained user message with the
            # suffix of transcript turns instead of assuming ``turn - 1``.
            turn_ids = sorted({item.turn for item in self.items
                               if item.kind == 'user' and item.turn > 0})
            index = len(starts) - sum(1 for value in turn_ids if value >= turn)
            if index < 0:
                # The requested turn was archived by compaction; roll back to
                # the oldest retained turn instead of wiping the whole context.
                index = 0
            text = ''
            if 0 <= index < len(starts):
                cut = starts[index]
                message = self.context.committed[cut]
                text = str(message.get('content') or '')
                marker = '</conversation_summary>'
                if text.startswith('<conversation_summary>') and marker in text:
                    text = text.split(marker, 1)[1].lstrip('\n')
                del self.context.committed[cut:]
            else:
                self.context.committed.clear()
            self.context.draft.clear()
            self.context.invalidate()
            self.items = [item for item in self.items if item.turn < turn]
            self.turn = max(0, turn - 1)
            self.status = 'idle'
        self._emit('session_rewound', turn=turn)
        self.save_active()
        return text

    def fork(self, name: str = '') -> str:
        self._require_idle('cannot fork the conversation')
        with self._lock:
            source = self.session_name
            if name:
                new_name = name.strip()
            else:
                new_name = self.store.unique_name(f'{new_session_name()}_fork')
            if not new_name:
                raise ValueError('session name must not be empty')
            self.session_name = new_name
            self.session_title = f'Fork of {self.session_title or source or "session"}'
        payload = self.payload()
        self.store.save(payload)
        self.store.save(payload, LAST_SESSION)
        self._append_notice(f'Forked into session {new_name}.', level='info')
        self.save_active()
        return new_name

    # ── model switching ─────────────────────────────────────
    def set_model(self, provider: ProviderConfig, model: ModelConfig) -> None:
        self._require_idle('cannot switch models')
        with self._lock:
            self.provider = provider
            self.model = model
            self.client = self._make_client(provider)
            self.config.select(provider.name, model.id)
            self.context.stable_prefix = StablePrefix.create(
                provider=provider.name, endpoint=provider.base_url,
                model=model.id, system=self.system,
                tools=self.registry.schemas(provider.wire, vision=model.vision),
                generation=self.context.stable_prefix.generation + 1,
                wire=provider.wire,
            )
            self.turn_in = 0
            self.turn_out = 0
        self.agents.update_root(self.context, self.provider, self.model, self.system)
        self._emit('model_changed', provider=provider.name, model=model.id)

    def refresh_tools(self) -> None:
        """Re-register dynamic tools after MCP or configuration changes."""
        self.registry.unregister_source('agent')
        self.registry.unregister_source('skill')
        if self.config.agents.enabled:
            self.registry.register_many(self.agents.tool_specs())
        self.registry.register_many(self.skills.tool_specs())
        # Skills and delegation limits also appear in the system prompt, so
        # rebuild it instead of only refreshing the tool schemas.
        self._refresh_system_and_prefix(force=True)
        self._emit('tools_refreshed')


def _title_from(text: str) -> str:
    visible = strip_image_markers(text).strip()
    if not visible and has_image_marker(text):
        return 'Image'
    line = visible.splitlines()[0] if visible else ''
    line = ' '.join(line.split())
    return line[:60]


def _continuation_prompt(thinking: str, full: str) -> str:
    parts = []
    if thinking:
        parts.append(f'<reasoning>\n{thinking}\n</reasoning>\n')
    if full:
        parts.append(f'<output>\n{full}\n</output>\n')
    parts.append(
        '\n[The previous response was cut off by the output token limit. '
        'Continue exactly where it stopped. Do not repeat anything above. '
        'Keep producing the rest of your reasoning, then finish the task.]')
    return ''.join(parts)


def _diff_dict(file_diff: FileDiff | None) -> dict[str, Any] | None:
    if file_diff is None:
        return None
    return {
        'path': file_diff.path,
        'unified': file_diff.unified,
        'additions': file_diff.additions,
        'deletions': file_diff.deletions,
        'omitted': file_diff.omitted,
    }


def _tool_brief(args: dict[str, Any]) -> str:
    for key in ('command', 'path', 'pattern', 'url', 'task_name', 'target', 'name'):
        value = args.get(key)
        if value:
            text = str(value).replace('\n', ' ')
            return text[:80] + ('...' if len(text) > 80 else '')
    return ''
