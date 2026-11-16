"""The AdCode full-screen application.

The render loop composites every widget onto a canvas each frame and writes the
whole screen in one terminal write.  The transcript is a virtual scrollback
owned by the app, and the agent loop streams from a worker thread into the
event queue consumed here.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

from adcode.agent_loop import AgentLoop, AgentSnapshot
from adcode.config import AppConfig, ConfigStore, ProviderConfig, ModelConfig
from adcode.goals import GoalStore
from adcode.images import (
    ImageStore, count_image_markers, file_uri_to_path, image_marker, is_image_url,
    read_clipboard_image, resolve_image,
)
from adcode.mcp.manager import get_mcp_manager
from adcode.paths import goals_file, sessions_db
from adcode.session import DbSessionStore, TranscriptItem
from adcode.skills import SkillRegistry
from adcode.theme import Palette, build_palette
from adcode.tools.builtin import shutdown_shell_sessions
from adcode.tools.registry import ToolRegistry
from adcode.tui.canvas import BOLD, DIM, Canvas, text_width, truncate_text
from adcode.tui.dialogs import (
    CommandPalette, ConfirmDialog, DialogItem, MessageDialog, Overlay,
    SelectDialog, TextPrompt,
)
from adcode.tui.input import (
    InputThread, KeyEvent, get_clipboard, set_clipboard,
)
from adcode.tui.keymap import KeyMap
from adcode.tui.markdown import MarkdownRenderer
from adcode.tui.screen import TerminalScreen
from adcode.tui.settings import SettingsOverlay
from adcode.tui.text import (
    Line, Span, add_span, line_text_between, line_width, plain_text,
    split_line_columns,
)
from adcode.tui.transcript import LiveRender, TranscriptRenderer

SPINNER = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
# A column past the end of a line, so a selection can run to its end.
_ALL_COLUMNS = 1 << 30
# Fallback frame rate when the configuration carries no value.
_DEFAULT_FPS = 60
# The status light crosses the status line once per second.
SWEEP_PERIOD = 1.0
# The UI error log is capped; a bug that fires every frame would fill it.
_LOG_MAX_BYTES = 2 * 1024 * 1024
# How wide the light is, in cells.
SWEEP_SPAN = 12
# How long the event loop sleeps when there is nothing to draw. A keypress
# wakes it up right away, so this only bounds the poll.
_IDLE_POLL = 0.15
# Upper bound on the events handled per frame, so a flooded queue cannot starve
# rendering (a paste arrives as one event, so this only caps bursts).
_MAX_INPUT_BATCH = 128


class PromptState:
    def __init__(self) -> None:
        self.text = ''
        self.cursor = 0
        self.anchor: int | None = None
        self.history: list[str] = []
        self.history_index = 0
        self.history_draft = ''
        self.undo: list[tuple[str, int]] = []
        self.redo: list[tuple[str, int]] = []
        self.completion_index = 0
        self.completion_closed_for = ''

    def snapshot(self) -> None:
        if not self.undo or self.undo[-1] != (self.text, self.cursor):
            self.undo.append((self.text, self.cursor))
            del self.undo[:-200]
            self.redo.clear()

    def set_text(self, text: str, cursor: int | None = None) -> None:
        self.snapshot()
        self.text = text
        self.cursor = len(text) if cursor is None else max(0, min(len(text), cursor))
        self.anchor = None

    def insert(self, text: str) -> None:
        clean = text.replace('\r\n', '\n').replace('\r', '\n')
        self.snapshot()
        start, end = self.selection()
        self.text = self.text[:start] + clean + self.text[end:]
        self.cursor = start + len(clean)
        self.anchor = None

    def selection(self) -> tuple[int, int]:
        if self.anchor is None:
            return self.cursor, self.cursor
        return min(self.anchor, self.cursor), max(self.anchor, self.cursor)

    def delete_backward(self) -> None:
        start, end = self.selection()
        if start != end:
            self.snapshot()
            self.text = self.text[:start] + self.text[end:]
            self.cursor = start
            self.anchor = None
        elif self.cursor > 0:
            self.snapshot()
            self.text = self.text[:self.cursor - 1] + self.text[self.cursor:]
            self.cursor -= 1

    def delete_forward(self) -> None:
        start, end = self.selection()
        if start != end:
            self.snapshot()
            self.text = self.text[:start] + self.text[end:]
            self.cursor = start
            self.anchor = None
        elif self.cursor < len(self.text):
            self.snapshot()
            self.text = self.text[:self.cursor] + self.text[self.cursor + 1:]

    def cut(self) -> str:
        start, end = self.selection()
        if start == end:
            return ''
        text = self.text[start:end]
        self.snapshot()
        self.text = self.text[:start] + self.text[end:]
        self.cursor = start
        self.anchor = None
        return text

    def copy(self) -> str:
        start, end = self.selection()
        return self.text[start:end] if start != end else ''


def _format_tokens(value: int) -> str:
    if value >= 1_000_000:
        return f'{value / 1_000_000:.1f}M'
    if value >= 1_000:
        return f'{value / 1_000:.1f}k'
    return str(value)


def _row_offset(line: Line) -> int:
    """Left column of a transcript row.

    Rows normally sit one cell in from the screen edge; a row that declares a
    bleed (a tool call row and its accent bar) starts at column 0 instead,
    keeping its own gutter so the text stays aligned.
    """
    return max(0, 1 - min(getattr(line, 'bleed', 0), 1))


def _prompt_layout(text: str, cursor: int, anchor: int | None, width: int
                   ) -> tuple[list[Line], int, int]:
    """Wrap the input into visual rows and locate the caret."""
    width = max(4, width)
    content_width = max(1, width - 2)
    selection_start, selection_end = (
        (min(anchor, cursor), max(anchor, cursor)) if anchor is not None
        else (cursor, cursor)
    )
    rows: list[Line] = []
    caret_row = 0
    caret_col = 2
    global_offset = 0
    logical_lines = text.split('\n')
    for logical_index, logical in enumerate(logical_lines):
        chunks: list[tuple[str, int]] = []
        current = ''
        current_start = global_offset
        current_width = 0
        for char in logical:
            size = max(0, text_width(char))
            if current_width and current_width + size > content_width:
                chunks.append((current, current_start))
                current = ''
                current_start += len(chunks[-1][0])
                current_width = 0
            current += char
            current_width += size
        chunks.append((current, current_start))
        for chunk_index, (chunk, start_offset) in enumerate(chunks):
            line: Line = []
            prefix = '> ' if (logical_index == 0 and chunk_index == 0) else '  '
            add_span(line, Span(prefix, 'accent' if prefix == '> ' else 'muted',
                                attrs=BOLD if prefix == '> ' else 0))
            for char_index, char in enumerate(chunk):
                offset = start_offset + char_index
                selected = selection_start <= offset < selection_end
                add_span(line, Span(
                    char,
                    'selection_fg' if selected else 'text',
                    'selection_bg' if selected else None,
                ))
            rows.append(line)
            row_start = start_offset
            row_end = start_offset + len(chunk)
            if row_start <= cursor <= row_end:
                if cursor == row_end and chunk_index < len(chunks) - 1:
                    # Caret at the wrap boundary belongs to the next row.
                    continue
                caret_row = len(rows) - 1
                caret_col = 2 + sum(text_width(char) for char in chunk[:cursor - row_start])
        # The newline itself: caret immediately before it sits at the row end.
        if logical_index < len(logical_lines) - 1:
            if global_offset + len(logical) == cursor:
                # Find the row whose end is this offset.
                if rows:
                    caret_row = len(rows) - 1
                    caret_col = line_width(rows[-1])
            global_offset += len(logical) + 1
        else:
            global_offset += len(logical)
    if not rows:
        rows = [[Span('> ', 'accent', attrs=BOLD)]]
    return rows, caret_row, caret_col


class AdCodeApp:
    def __init__(self, cwd: str | None = None, resume: bool = True,
                 config_path: str | os.PathLike[str] | None = None,
                 screen: TerminalScreen | None = None,
                 input_queue: 'queue.Queue[KeyEvent] | None' = None,
                 input_thread: InputThread | None = None) -> None:
        self.cwd = cwd or os.getcwd()
        self.store = ConfigStore(config_path)
        self.config: AppConfig = self.store.load()
        self.keymap = KeyMap(self.config.keys)
        # The terminal pieces are injectable so a harness can drive the real
        # application loop without owning a console (see the test harness).
        self.screen = screen if screen is not None else TerminalScreen()
        self.palette: Palette = build_palette(self.config.ui.theme, self.config.ui.accent)
        self.canvas = Canvas(*self._screen_size(), self.palette)
        self.input_queue: 'queue.Queue[KeyEvent]' = (
            input_queue if input_queue is not None else queue.Queue())
        self.input_thread = (
            input_thread if input_thread is not None
            else InputThread(self.input_queue))
        self.posted: 'queue.Queue[Callable[[], None]]' = queue.Queue()
        self.session_store = DbSessionStore(db_path=str(sessions_db()))
        self.images = ImageStore(os.path.join(self.cwd, 'sessions', 'images'),
                                 self.cwd)
        self.goals = GoalStore(str(goals_file()))
        self.skills = SkillRegistry(self.config.skills, self.cwd).scan()
        self.registry = ToolRegistry()
        self.mcp = get_mcp_manager(self.registry)
        self.loop: AgentLoop | None = None
        self.standalone_items: list[TranscriptItem] = []
        self.transcript = TranscriptRenderer()
        self.prompt = PromptState()
        self.overlay: Overlay | None = None
        self.running = True
        self.dirty = True
        self.follow = True
        self.scroll = 0
        self.phase = 'idle'
        self.config_version = 0
        self.btw_mode = False
        self.exit_armed = False
        self.last_render = 0.0
        self._prompt_cache: tuple[tuple[Any, ...], tuple[list[Line], int, int]] | None = None
        self._snapshot_cache: AgentSnapshot | None = None
        # A text selection in the transcript: the anchor and the active end
        # are transcript coordinates, so the selection survives a scroll.
        self._select_anchor: tuple[int, int] | None = None
        self._select_end: tuple[int, int] | None = None
        self._selecting = False
        self._view_lines: list[Line] = []
        self._view_start = 0
        self._view_top = 0
        self._last_cursor: tuple[int, int] | None = None
        self._cursor_visible: bool | None = None
        self.last_interrupt = 0.0
        self.last_width = 0
        self.last_title = ''
        self._last_total = 0
        self._last_viewport = 0
        self._last_max_scroll = 0
        self.live_thinking_text = ''
        self.live_thinking = MarkdownRenderer(self._content_width())
        self.live_thinking_started = 0.0
        self.live_answer_text = ''
        self.live_answer = MarkdownRenderer(self._content_width())
        self.live_compaction = MarkdownRenderer(self._content_width())
        self.live_aside_question = ''
        self.live_aside = MarkdownRenderer(self._content_width())
        self.live_tool_output: list[str] = []
        self.active_tool = ''
        self.status_hint = ''
        # The arguments of the tool that is running right now; the concise
        # level shows them next to the tool name as the current action.
        self.status_args: dict[str, Any] = {}
        self.status_tool_name = ''
        # Wall clock start of the current connection attempt, so the footer can
        # show how long the provider has been silent.
        self.connect_started = 0.0
        self.connect_retrying = False
        self._resume = resume
        self._started_at = time.time()
        self._tool_signature = self._tool_config_signature()
        # The tip shown under the welcome logo; re-rolled whenever the
        # conversation is cleared so the empty pane suggests something new.
        from adcode.commands import random_tip
        self.welcome_tip = random_tip()

    # ── boot ────────────────────────────────────────────────
    def _screen_size(self) -> tuple[int, int]:
        size = self.screen.size()
        return size.columns, size.rows

    def _content_width(self) -> int:
        columns, _ = self._screen_size()
        return max(16, columns - 2)

    def run(self) -> None:
        self.screen.enter()
        self.input_thread.start()
        try:
            self._build_loop()
            if self._resume and self.loop is not None:
                self._auto_resume()
            if self.loop is None:
                from adcode.tui.provider_ui import ProviderWizard
                self.post(lambda: ProviderWizard(self).run_add(first_run=True))
            self._loop()
        finally:
            self._shutdown()

    def _build_loop(self) -> None:
        selected = self.config.selected()
        if selected is None:
            self.loop = None
            return
        provider, model = selected
        self.loop = AgentLoop(
            self.config, self.registry, self.session_store, self.goals,
            self.skills, emit=self._wake, mcp_manager=self.mcp, cwd=self.cwd,
        )
        self.loop.start()
        self._start_mcp_load()
        self._update_title()

    def _start_mcp_load(self) -> None:
        servers = list(self.config.mcp_servers)
        if not servers or self.loop is None:
            return

        def worker() -> None:
            self.mcp.load(servers, self._mcp_status)
            self.post(self._after_mcp_load)

        threading.Thread(target=worker, name='adcode-mcp-load', daemon=True).start()

    def _mcp_status(self, name: str, state: str, detail: str) -> None:
        if state == 'connecting':
            self.post(lambda: self.notify(f'MCP {name}: connecting…', 'info'))
        elif state == 'ready':
            self.post(lambda: self.notify(f'MCP {name}: ready ({detail})', 'success'))
        elif state == 'failed':
            self.post(lambda: self.notify(f'MCP {name}: failed — {detail}', 'error'))

    def _after_mcp_load(self) -> None:
        if self.loop is not None:
            self.loop.refresh_tools()
        self.dirty = True

    def _auto_resume(self) -> None:
        payload = self.session_store.latest()
        if payload is None or not payload.messages or self.loop is None:
            return
        self.loop.apply_payload(payload)
        self.loop.add_notice(
            f'Resumed session "{payload.title or payload.name}" '
            f'({len(payload.messages)} messages).', level='info')
        self.follow = True
        self.dirty = True

    # ── shutdown ────────────────────────────────────────────
    def _shutdown(self) -> None:
        try:
            self.input_thread.stop()
            if self.loop is not None:
                self.loop.shutdown()
                self.loop.save_active()
            self.mcp.shutdown()
            shutdown_shell_sessions()
            self.store.save()
        finally:
            self.screen.leave()

    def request_exit(self) -> None:
        self.running = False

    # ── thread plumbing ─────────────────────────────────────
    def _wake(self, kind: str, payload: dict[str, Any]) -> None:
        # The loop's own queue is authoritative; this only shortens the wait.
        pass

    def post(self, function: Callable[[], None]) -> None:
        self.posted.put(function)

    def open_overlay(self, overlay: Overlay) -> None:
        self.overlay = overlay
        self.dirty = True

    def dismiss_overlay(self, overlay: Overlay | None = None,
                        *, drop_input: bool = False) -> None:
        """Close the current modal.

        ``drop_input`` is for dismissals the user did not perform -- a spinner
        that closes itself, for example.  Keys typed at that spinner must not
        survive into the prompt underneath it.  A close the user asked for with
        Esc or Enter keeps the queue intact: keys pressed after that are meant
        for the prompt, and the input batch is handled in arrival order.
        """
        if overlay is not None and self.overlay is not overlay:
            return
        self.overlay = None
        if drop_input:
            self._drop_pending_input()
        self.dirty = True

    def _drop_pending_input(self) -> None:
        while True:
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                return

    # ── notifications ───────────────────────────────────────
    def notify(self, text: str, level: str = 'info') -> None:
        if self.loop is not None:
            self.loop.add_notice(text, level)
        else:
            self.standalone_items.append(TranscriptItem(kind='notice', text=text,
                                                        level=level))
        self.dirty = True

    # ── configuration ───────────────────────────────────────
    def save_config(self) -> None:
        self.store.save()

    def config_changed(self) -> None:
        self.save_config()
        self.config_version += 1
        self.keymap = KeyMap(self.config.keys)
        self.palette = build_palette(self.config.ui.theme, self.config.ui.accent)
        self.canvas.palette = self.palette
        self.transcript.invalidate()
        signature = self._tool_config_signature()
        if signature != self._tool_signature:
            self._tool_signature = signature
            self.skills.settings = self.config.skills
            self.skills.scan()
            if self.loop is not None:
                self.loop.refresh_tools()
        if self.loop is not None:
            # The hold settings steer the engine's retry loop: they have to
            # reach it while it runs.
            self.loop.apply_hold()
        self.dirty = True

    def _tool_config_signature(self) -> tuple[Any, ...]:
        return (
            self.config.agents.enabled,
            self.config.agents.team_enabled,
            self.config.skills.enabled,
            self.config.skills.include_project,
            tuple(self.config.skills.extra_paths),
        )

    def refresh_tools(self) -> None:
        if self.loop is not None:
            self.loop.refresh_tools()
        self.dirty = True

    def select_model(self, provider: ProviderConfig, model: ModelConfig) -> None:
        self.config.select(provider.name, model.id)
        self.save_config()
        if self.loop is not None:
            self.loop.set_model(provider, model)
        self.notify(f'Model: {provider.name} / {model.id}', 'info')
        self.dirty = True

    def after_provider_change(self) -> None:
        selected = self.config.selected()
        if selected is None:
            if self.loop is not None:
                self.loop.save_active()
                self.loop.shutdown()
                self.loop = None
            return
        provider, model = selected
        if self.loop is None:
            self._build_loop()
        else:
            self.loop.set_model(provider, model)
        self.dirty = True

    def clear_conversation(self) -> None:
        if self.loop is not None:
            self.loop.clear()
        self.standalone_items.clear()
        self.transcript.invalidate()
        self.follow = True
        self.scroll = 0
        # No notice: the empty pane with the welcome logo is the feedback.
        from adcode.commands import random_tip
        self.welcome_tip = random_tip()
        self._update_title()

    def load_session(self, name: str) -> None:
        if self.loop is None:
            self.notify('No active conversation to load into.', 'warning')
            return
        if self.loop.load_session(name):
            self.transcript.invalidate()
            self.follow = True
            self.scroll = 0
            self.dirty = True
        else:
            self.notify(f'Session not found: {name}', 'error')

    # ── provider selection helpers ──────────────────────────
    def ensure_selection(self) -> tuple[ProviderConfig, ModelConfig] | None:
        return self.config.selected()

    @property
    def items(self) -> list[TranscriptItem]:
        if self.loop is not None:
            return self.loop.items_snapshot()
        return list(self.standalone_items)

    def snapshot(self) -> AgentSnapshot:
        """The agent snapshot, taken once per frame.

        Header and footer ask for the same numbers; walking the conversation
        twice per frame was pure waste.
        """
        cached = self._snapshot_cache
        if cached is not None:
            return cached
        if self.loop is not None:
            snap = self.loop.snapshot()
        else:
            selected = self.config.selected()
            snap = AgentSnapshot(
                provider=selected[0].name if selected else '',
                model=selected[1].id if selected else '',
                effort=self.config.reasoning.effort,
            )
        snap.status = self.phase
        self._snapshot_cache = snap
        return snap

    def _drop_snapshot(self) -> None:
        """Forget the cached snapshot after the state changed."""
        self._snapshot_cache = None

    def _update_title(self) -> None:
        if self.loop is None:
            return
        snap = self.loop.snapshot()
        title = f'AdCode - {snap.session_title}' if snap.session_title else 'AdCode'
        if title != self.last_title:
            self.last_title = title
            self.screen.set_title(title)

    # ── event loop ──────────────────────────────────────────
    def _loop(self) -> None:
        while self.running:
            # A frame that is due has to be drawn on time: the loop never
            # sleeps longer than one frame interval while something is
            # pending or an animation is running. An idle app sleeps long and
            # wakes up on the next input event.
            interval = self._render_interval()
            timeout = (interval if (self.dirty or self._needs_animation())
                       else _IDLE_POLL)
            try:
                event = self.input_queue.get(timeout=timeout)
            except queue.Empty:
                event = None
            try:
                self._process_posted()
                if self.loop is not None:
                    self.loop.pump_agent_events()
                    self._apply_loop_events()
                self._dispatch_input(event)
                self._check_resize()
                self._tick_animation()
                self._render_if_due()
            except Exception as error:
                # A UI exception must never take down the session silently.
                # The traceback goes to the log: a one-line message is not
                # enough to find a rendering bug.
                self._log_ui_error(error)
                try:
                    self.notify(f'UI error: {type(error).__name__}: {error}',
                                'error')
                except Exception:
                    pass

    def _log_ui_error(self, error: BaseException) -> None:
        """Write the traceback of a UI error next to the session database.

        The file is capped: a bug that fires every frame would otherwise
        fill the disk with the same traceback.
        """
        import traceback
        try:
            from adcode import paths
            target = paths.app_dir() / 'adcode.log'
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size > _LOG_MAX_BYTES:
                # Keep the tail: the newest traceback is the interesting one.
                with target.open('rb') as stream:
                    stream.seek(-_LOG_MAX_BYTES // 2, os.SEEK_END)
                    tail = stream.read()
                target.write_bytes(tail)
            with target.open('a', encoding='utf-8') as stream:
                stream.write(f'--- {time.strftime("%Y-%m-%d %H:%M:%S")} '
                             f'UI error\n')
                traceback.print_exception(type(error), error,
                                          error.__traceback__, file=stream)
        except Exception:
            pass

    def _dispatch_input(self, first: KeyEvent | None) -> None:
        """Handle the queued input batch before painting the next frame.

        Reading one event per frame builds a backlog while a frame is painted,
        and a backlog outlives the surface it was typed for: keys typed at a
        dialog would still be queued when the dialog closed and then land in
        the prompt behind it.  Draining the batch in arrival order applies each
        key to the screen state the user was actually looking at.
        """
        if first is not None:
            self._handle_event(first)
        for _ in range(_MAX_INPUT_BATCH):
            try:
                queued = self.input_queue.get_nowait()
            except queue.Empty:
                return
            self._handle_event(queued)

    def _process_posted(self) -> None:
        while True:
            try:
                function = self.posted.get_nowait()
            except queue.Empty:
                return
            try:
                function()
            except Exception as error:
                self.notify(f'Background error: {error}', 'error')

    def _apply_loop_events(self) -> None:
        if self.loop is None:
            return
        events = self.loop.drain_ui_events()
        for kind, payload in events:
            self._apply_event(kind, payload)
        if events:
            self._update_title()
            self.dirty = True

    def _apply_event(self, kind: str, payload: dict[str, Any]) -> None:
        if kind == 'turn_start':
            self._reset_live()
            # A turn opens by sending a request; no model output exists yet, so
            # the honest phase is "connecting" until the first stream event.
            self._enter_connecting()
            self.follow = True
            self.scroll = 0
        elif kind == 'thinking_delta':
            if self.phase != 'thinking':
                self._reset_live()
                self.phase = 'thinking'
            if not self.live_thinking_text:
                self.live_thinking_started = time.time()
            self.live_thinking_text += payload.get('text', '')
            self.live_thinking.feed(payload.get('text', ''))
        elif kind == 'thinking_end':
            self.live_thinking_text = ''
            self.live_thinking = MarkdownRenderer(self._content_width())
            if self.phase == 'thinking':
                self.phase = 'idle'
        elif kind == 'text_delta':
            if self.phase != 'answering':
                self.live_answer = MarkdownRenderer(self._content_width())
                self.live_answer_text = ''
                self.phase = 'answering'
            self.live_answer_text += payload.get('text', '')
            self.live_answer.feed(payload.get('text', ''))
        elif kind == 'text_end':
            self.live_answer = MarkdownRenderer(self._content_width())
            self.live_answer_text = ''
            if self.phase == 'answering':
                self.phase = 'idle'
        elif kind == 'tool_start':
            self.live_answer = MarkdownRenderer(self._content_width())
            self.live_answer_text = ''
            self.active_tool = payload.get('call_id', '')
            self.live_tool_output = []
            self.phase = 'tool'
            self.status_hint = payload.get('display', '')
            self.status_args = payload.get('args', {}) or {}
            self.status_tool_name = payload.get('name', '') or ''
        elif kind == 'tool_output':
            if payload.get('call_id') == self.active_tool:
                self.live_tool_output.append(payload.get('text', ''))
                del self.live_tool_output[:-30]
        elif kind == 'tool_end':
            self.active_tool = ''
            self.live_tool_output = []
            self.status_args = {}
            self.status_tool_name = ''
            self.phase = 'thinking'
        elif kind == 'compaction_start':
            self.phase = 'compacting'
            self.live_compaction = MarkdownRenderer(self._content_width())
        elif kind == 'compaction_delta':
            self.live_compaction.feed(payload.get('text', ''))
        elif kind == 'compaction_end':
            self.live_compaction = MarkdownRenderer(self._content_width())
            self.phase = 'thinking'
        elif kind == 'usage':
            self.dirty = True
        elif kind == 'notice':
            level = payload.get('level', 'info')
            if level == 'error':
                self.phase = 'idle'
        elif kind == 'error':
            self.phase = 'idle'
        elif kind == 'interrupted':
            self._reset_live()
            self.phase = 'idle'
        elif kind == 'turn_end':
            self._reset_live()
            self.phase = 'idle'
            self.status_hint = ''
        elif kind == 'aside_start':
            self.phase = 'aside'
            self.live_aside_question = payload.get('question', '')
            self.live_aside = MarkdownRenderer(self._content_width())
        elif kind == 'aside_delta':
            self.live_aside.feed(payload.get('text', ''))
        elif kind == 'aside_end':
            self._reset_live()
            self.phase = 'idle'
        elif kind == 'status':
            status = payload.get('status', self.phase)
            if status == 'connecting':
                self._enter_connecting(bool(payload.get('retrying')))
            else:
                if self.phase == 'connecting':
                    self.connect_started = 0.0
                    self.connect_retrying = False
                self.phase = status
        elif kind == 'session_loaded' or kind == 'session_rewound':
            self.transcript.invalidate()
            self._reset_live()
            self.phase = 'idle'
            self.follow = True
            self.scroll = 0
            self._update_title()
        elif kind == 'session_cleared':
            self.transcript.invalidate()
            self._reset_live()
            self.phase = 'idle'
            self.follow = True
            self.scroll = 0
            self._update_title()
        elif kind == 'agents_refresh':
            self.dirty = True
        elif kind == 'model_changed':
            self.dirty = True
        self.dirty = True

    def _reset_live(self) -> None:
        width = self._content_width()
        self.live_thinking_text = ''
        self.live_thinking = MarkdownRenderer(width)
        self.live_answer_text = ''
        self.live_answer = MarkdownRenderer(width)
        self.live_compaction = MarkdownRenderer(width)
        self.live_aside_question = ''
        self.live_aside = MarkdownRenderer(width)
        self.live_tool_output = []
        self.active_tool = ''

    def _enter_connecting(self, retrying: bool = False) -> None:
        """Show the wait before the first streamed byte.

        A retry restarts the clock but keeps the same phase, so a flapping
        connection reads as one wait instead of a flicker between states.
        """
        if self.phase != 'connecting' or retrying or not self.connect_started:
            self.connect_started = time.time()
        self.connect_retrying = bool(retrying)
        self.phase = 'connecting'

    def _connect_elapsed(self) -> float:
        if not self.connect_started:
            return 0.0
        return max(0.0, time.time() - self.connect_started)

    def _needs_animation(self) -> bool:
        return self.phase in ('connecting', 'thinking', 'answering', 'tool',
                              'compacting', 'aside')

    def _tick_animation(self) -> None:
        if self._needs_animation():
            self.dirty = True

    def _check_resize(self) -> None:
        columns, rows = self.screen.size()
        if columns != self.canvas.width or rows != self.canvas.height:
            self.canvas.resize(columns, rows)
            self.live_thinking = MarkdownRenderer(max(16, columns - 2))
            self.live_answer = MarkdownRenderer(max(16, columns - 2))
            self.live_compaction = MarkdownRenderer(max(16, columns - 2))
            self.live_aside = MarkdownRenderer(max(16, columns - 2))
            if self.live_thinking_text:
                self.live_thinking.feed(self.live_thinking_text)
            if self.live_answer_text:
                self.live_answer.feed(self.live_answer_text)
            self.transcript.invalidate()
            self.dirty = True

    # ── input ───────────────────────────────────────────────
    def _handle_event(self, event: KeyEvent) -> None:
        # One key quits, whatever is on the screen -- unless an overlay is
        # capturing a new binding, where the quit chord is the value typed.
        capturing = self.overlay is not None and \
            getattr(self.overlay, 'capture_mode', False)
        if event.kind == 'key' and not capturing \
                and event.key == self.keymap.key_of('quit'):
            self.request_exit()
            return
        if self.overlay is not None:
            # A modal owns every event while it is open.  Keys, pastes and the
            # wheel used to fall through to the transcript and the prompt,
            # because the mouse branch ran before this check and stopped there.
            overlay = self.overlay
            overlay.handle(event)
            if overlay.closed and self.overlay is overlay:
                self.dismiss_overlay(overlay)
            self.dirty = True
            return
        if event.kind == 'mouse':
            if event.action == 'wheel_up':
                self.follow = False
                self.scroll = min(self._max_scroll(), self.scroll
                                  + max(1, self.config.ui.mouse_scroll_lines))
                self.dirty = True
            elif event.action == 'wheel_down':
                self.scroll = max(0, self.scroll
                                  - max(1, self.config.ui.mouse_scroll_lines))
                self.follow = self.scroll <= 0
                self.dirty = True
            elif event.action in ('press', 'move', 'release'):
                self._handle_selection(event)
            return
        if event.kind == 'paste':
            self.prompt.insert(event.text)
            self.dirty = True
            return
        if event.kind == 'text':
            # Route typing through PromptState so the selection and undo stacks
            # stay consistent.
            self.prompt.insert(event.text)
            self.dirty = True
            return
        if event.kind != 'key':
            return
        key = event.key
        if key != self.keymap.key_of('exit_chord'):
            self.exit_armed = False
        if key == 'esc':
            # Cancel is a fixed gesture: close the completion, interrupt the
            # turn, or drop the selection.
            if self._completion_visible():
                self.prompt.completion_closed_for = self.prompt.text
                self.prompt.anchor = None
            elif self.phase != 'idle' and self.loop is not None:
                self.loop.interrupt()
            else:
                self.prompt.anchor = None
            self.dirty = True
            return
        if key == 'enter':
            # Submit is a fixed gesture too.
            self._handle_enter(event)
            return
        action = self.keymap.action_of(key)
        if action is None:
            return
        if action == 'scroll_up':
            self.follow = False
            self.scroll = min(self._max_scroll(), self.scroll + self._page_size())
            self.dirty = True
            return
        if action == 'scroll_down':
            self.scroll = max(0, self.scroll - self._page_size())
            self.follow = self.scroll <= 0
            self.dirty = True
            return
        if action == 'redraw':
            self.dirty = True
            return
        if action == 'palette':
            self.open_command_palette()
            return
        if action == 'sessions':
            self.open_sessions_browser()
            return
        if action == 'exit_chord':
            if self._copy_selection():
                return
            if self.prompt.selection()[0] != self.prompt.selection()[1]:
                copied = self.prompt.copy()
                if copied and not set_clipboard(copied):
                    self.status_hint = 'the clipboard is not available'
                self.prompt.anchor = None
            elif self.prompt.text:
                self.prompt.set_text('')
            elif self.overlay is None and self.phase != 'idle' and self.loop is not None:
                self.loop.interrupt()
            elif self.exit_armed:
                self.request_exit()
            else:
                self.exit_armed = True
                chord = self.keymap.key_of('exit_chord') or 'ctrl+c'
                self.status_hint = f'press {chord} again to exit'
            self.dirty = True
            return
        if action == 'exit_eof':
            if not self.prompt.text:
                self.request_exit()
            return
        if action == 'paste':
            # The standard paste chords stay attached as fixed aliases.
            self._paste_from_clipboard()
            return
        if action == 'attach_image':
            self.attach_clipboard_image()
            return
        if action == 'cut':
            cut = self.prompt.cut()
            if cut and not set_clipboard(cut):
                self.status_hint = 'the clipboard is not available'
            self.dirty = True
            return
        if action == 'select_all':
            self.prompt.anchor = 0
            self.prompt.cursor = len(self.prompt.text)
            self.dirty = True
            return
        if action == 'undo':
            self._undo()
            return
        if action == 'redo':
            self._redo()
            return
        if action == 'newline':
            self.prompt.insert('\n')
            self.dirty = True
            return
        if action == 'complete':
            self._apply_completion()
            return
        if action == 'history_up':
            if self._completion_visible():
                self._move_completion(-1)
            else:
                self._history_move(-1)
            self.dirty = True
            return
        if action == 'history_down':
            if self._completion_visible():
                self._move_completion(1)
            else:
                self._history_move(1)
            self.dirty = True
            return
        if action == 'move_left':
            self._move_horizontal(-1, event.shift)
            return
        if action == 'move_right':
            self._move_horizontal(1, event.shift)
            return
        if action == 'line_start':
            self._move_line_edge(start=True, whole=False, extend=event.shift)
            return
        if action == 'line_end':
            self._move_line_edge(start=False, whole=False, extend=event.shift)
            return
        if action == 'delete_backward':
            self.prompt.delete_backward()
            self.dirty = True
            return
        if action == 'delete_forward':
            self.prompt.delete_forward()
            self.dirty = True
            return
        if action == 'jump_start':
            self.prompt.cursor = 0
            self.prompt.anchor = None
            self.dirty = True
            return
        if action == 'jump_end':
            self.prompt.cursor = len(self.prompt.text)
            self.prompt.anchor = None
            self.dirty = True
            return

    def _handle_enter(self, event: KeyEvent) -> None:
        # A trailing backslash is the portable newline gesture.
        if self.prompt.text.endswith('\\') and \
                self.prompt.cursor == len(self.prompt.text):
            self.prompt.snapshot()
            self.prompt.text = self.prompt.text[:-1]
            self.prompt.cursor = min(self.prompt.cursor, len(self.prompt.text))
            # Insert without taking a second undo snapshot for the same edit.
            self.prompt.text += '\n'
            self.prompt.cursor = len(self.prompt.text)
            self.dirty = True
            return
        if self._completion_visible():
            candidates = self._completion_candidates()
            if candidates:
                index = min(self.prompt.completion_index, len(candidates) - 1)
                completed = candidates[index][0]
                if (self.prompt.text.strip()
                        and completed.rstrip() != self.prompt.text.rstrip()):
                    self.prompt.set_text(completed)
                    self.prompt.cursor = len(completed)
                    self.dirty = True
                    return
        text = self.prompt.text.strip()
        if not text:
            return
        self.prompt.history.append(self.prompt.text)
        self.prompt.history_index = len(self.prompt.history)
        self.prompt.history_draft = ''
        self.prompt.set_text('')
        self.prompt.completion_index = 0
        self.follow = True
        self.scroll = 0
        self.submit(text)

    def submit(self, text: str) -> None:
        from adcode.commands import dispatch
        if text.startswith('/'):
            dispatch(self, text)
            return
        if self.loop is None:
            self.notify('No provider is configured. Use /provider add.', 'warning')
            return
        mode = 'aside' if self.btw_mode else 'chat'
        if self.btw_mode:
            self.btw_mode = False
        self.loop.submit(text, mode)
        self.dirty = True

    def _paste_from_clipboard(self) -> None:
        """Insert the clipboard text, or attach the clipboard image."""
        text = get_clipboard()
        if text:
            self.prompt.insert(text)
        else:
            # A copied screenshot has no text form; attach it instead.
            self.attach_clipboard_image()
        self.dirty = True

    # ── image attachments ───────────────────────────────────
    def attach_clipboard_image(self) -> bool:
        data = read_clipboard_image()
        if data:
            return self.attach_image_bytes(data)
        # A file copied in the file manager lands on the clipboard as its path.
        clipboard = get_clipboard().strip()
        if clipboard and '\n' not in clipboard and \
                self._looks_like_image_reference(clipboard):
            return self.attach_image_file(clipboard)
        self.notify('No image in the clipboard. Copy a screenshot, then press alt+v.',
                    'warning')
        return False

    def _looks_like_image_reference(self, text: str) -> bool:
        value = file_uri_to_path(text)
        if value.startswith(('http://', 'https://')):
            return is_image_url(value)
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(self.cwd) / path
        return resolve_image(str(path), self.cwd) is not None

    def attach_image_bytes(self, data: bytes) -> bool:
        try:
            reference = self.images.save(data)
        except ValueError as error:
            self.notify(f'Cannot attach image: {error}', 'warning')
            return False
        except OSError as error:
            self.notify(f'Cannot store image: {error}', 'error')
            return False
        self.prompt.insert(f'{image_marker(reference)} ')
        self.dirty = True
        name = reference.rsplit('/', 1)[-1]
        size = (f'{len(data) / 1_048_576:.1f} MB' if len(data) >= 1_048_576
                else f'{max(1, round(len(data) / 1024))} KB')
        warning = self._vision_warning()
        if warning:
            self.notify(f'Attached {name} ({size}). {warning}', 'warning')
        else:
            self.notify(f'Attached {name} ({size}).', 'info')
        return True

    def _vision_warning(self) -> str:
        """Warn when the selected model is not known to accept images."""
        selected = self.config.selected()
        if selected is None or selected[1].vision:
            return ''
        return (f'{selected[1].id} is not marked as vision-capable; '
                f'it may reject the image.')

    def attach_image_file(self, reference: str) -> bool:
        reference = file_uri_to_path(reference)
        if not reference:
            self.notify('Usage: /image <path or https url>', 'warning')
            return False
        if reference.startswith(('http://', 'https://')):
            self.prompt.insert(f'{image_marker(reference)} ')
            self.dirty = True
            self.notify(f'Attached {reference}', 'info')
            return True
        path = Path(reference).expanduser()
        if not path.is_absolute():
            path = Path(self.cwd) / path
        if resolve_image(str(path), self.cwd) is None:
            self.notify(
                f'Unsupported image: {reference} (png, jpeg, gif or webp, '
                f'up to 5 MB)', 'error')
            return False
        try:
            return self.attach_image_bytes(path.read_bytes())
        except OSError as error:
            self.notify(f'Cannot read image: {error}', 'error')
            return False

    def _move_horizontal(self, delta: int, extend: bool) -> None:
        if extend:
            if self.prompt.anchor is None:
                self.prompt.anchor = self.prompt.cursor
        else:
            self.prompt.anchor = None
        self.prompt.cursor = max(0, min(len(self.prompt.text),
                                        self.prompt.cursor + delta))
        self.dirty = True

    def _move_line_edge(self, start: bool, whole: bool,
                        extend: bool = False) -> None:
        text = self.prompt.text
        cursor = self.prompt.cursor
        if extend:
            if self.prompt.anchor is None:
                self.prompt.anchor = cursor
        else:
            self.prompt.anchor = None
        if whole:
            self.prompt.cursor = 0 if start else len(text)
        else:
            if start:
                position = text.rfind('\n', 0, cursor)
                self.prompt.cursor = 0 if position == -1 else position + 1
            else:
                position = text.find('\n', cursor)
                self.prompt.cursor = len(text) if position == -1 else position
        self.dirty = True

    def _history_move(self, delta: int) -> None:
        history = self.prompt.history
        if not history:
            return
        if self.prompt.text and self.prompt.history_index >= len(history):
            self.prompt.history_draft = self.prompt.text
        index = self.prompt.history_index + delta
        index = max(0, min(len(history), index))
        self.prompt.history_index = index
        value = (self.prompt.history_draft if index >= len(history)
                 else history[index])
        self.prompt.set_text(value)
        self.prompt.cursor = len(value)
        self.prompt.anchor = None

    def _undo(self) -> None:
        if not self.prompt.undo:
            return
        self.prompt.redo.append((self.prompt.text, self.prompt.cursor))
        self.prompt.text, self.prompt.cursor = self.prompt.undo.pop()
        self.prompt.anchor = None
        self.dirty = True

    def _redo(self) -> None:
        if not self.prompt.redo:
            return
        self.prompt.undo.append((self.prompt.text, self.prompt.cursor))
        self.prompt.text, self.prompt.cursor = self.prompt.redo.pop()
        self.prompt.anchor = None
        self.dirty = True

    # ── slash completion ────────────────────────────────────
    def _completion_candidates(self) -> list[tuple[str, str, str]]:
        from adcode.commands import completion_candidates
        text = self.prompt.text
        if not text.startswith('/') or '\n' in text:
            return []
        if self.prompt.completion_closed_for == text:
            return []
        return completion_candidates(text)

    def _completion_visible(self) -> bool:
        return bool(self._completion_candidates())

    def _move_completion(self, delta: int) -> None:
        candidates = self._completion_candidates()
        if not candidates:
            return
        self.prompt.completion_index = (
            self.prompt.completion_index + delta) % len(candidates)

    def _apply_completion(self) -> None:
        candidates = self._completion_candidates()
        if not candidates:
            return
        index = min(self.prompt.completion_index, len(candidates) - 1)
        completed = candidates[index][0]
        self.prompt.set_text(completed)
        self.prompt.cursor = len(completed)
        self.prompt.completion_index = 0
        self.dirty = True

    # ── scrolling ───────────────────────────────────────────
    def _page_size(self) -> int:
        columns, rows = self.screen.size()
        return max(3, self._viewport_height(rows) - 2)

    def _max_scroll(self) -> int:
        rows = self.screen.size().rows
        viewport = self._last_viewport or self._viewport_height(rows)
        total = self._last_total or len(self._transcript_lines())
        return max(0, total - viewport)

    # ── transcript selection ─────────────────────────────────
    def _content_position(self, x: int, y: int) -> tuple[int, int] | None:
        """Map a screen cell to a transcript line and a column in it.

        The mouse reports 1-based cells, so the row and the column are moved
        one to the left before they are used.
        """
        row = y - 1 - self._view_top
        if row < 0 or row >= len(self._view_lines):
            return None
        line = self._view_lines[row]
        column = x - 1 - _row_offset(line)
        return self._view_start + row, max(0, column)

    def _handle_selection(self, event: KeyEvent) -> None:
        """Start, extend or end a text selection in the transcript."""
        if event.action == 'press':
            if event.button not in (0, -1):
                return
            position = self._content_position(event.x, event.y)
            if position is None:
                self._clear_selection()
                self.dirty = True
                return
            self._select_anchor = position
            self._select_end = position
            self._selecting = True
            self.dirty = True
            return
        if event.action == 'move':
            if not self._selecting:
                return
            position = self._content_position(event.x, event.y)
            if position is not None:
                self._select_end = position
                self.dirty = True
            return
        if event.action == 'release':
            self._selecting = False
            self.dirty = True

    def _clear_selection(self) -> None:
        self._select_anchor = None
        self._select_end = None
        self._selecting = False

    def _selection_range(self) -> tuple[int, int, int, int] | None:
        """The selected range as (start line, start column, end line, end column)."""
        anchor = self._select_anchor
        end = self._select_end
        if anchor is None or end is None:
            return None
        if anchor == end:
            return None
        if anchor <= end:
            return anchor[0], anchor[1], end[0], end[1]
        return end[0], end[1], anchor[0], anchor[1]

    def _row_selection(self, index: int) -> tuple[int, int] | None:
        """The selected column range of one transcript line."""
        selection = self._selection_range()
        if selection is None:
            return None
        start_line, start_col, end_line, end_col = selection
        if index < start_line or index > end_line:
            return None
        if index == start_line:
            return start_col, end_col if index == end_line else _ALL_COLUMNS
        if index == end_line:
            return 0, end_col
        return 0, _ALL_COLUMNS

    def _selection_text(self) -> str:
        """The text the selection covers."""
        selection = self._selection_range()
        if selection is None:
            return ''
        start_line, start_col, end_line, end_col = selection
        lines = self._transcript_lines()
        parts: list[str] = []
        for index in range(max(0, start_line), min(end_line + 1, len(lines))):
            line = lines[index]
            if index == start_line and index == end_line:
                parts.append(line_text_between(line, start_col, end_col))
            elif index == start_line:
                parts.append(line_text_between(line, start_col, _ALL_COLUMNS))
            elif index == end_line:
                parts.append(line_text_between(line, 0, end_col))
            else:
                parts.append(plain_text(line))
        return '\n'.join(part.rstrip() for part in parts)

    def _copy_selection(self) -> bool:
        """Copy the selected transcript text; False when nothing is selected."""
        text = self._selection_text()
        if not text.strip():
            return False
        if not set_clipboard(text):
            self.status_hint = 'the clipboard is not available'
        else:
            self.status_hint = f'copied {len(text)} characters'
        self.dirty = True
        return True

    def _prompt_rows(self, columns: int) -> tuple[list[Line], int, int]:
        """Wrap the input, reusing the previous result when nothing changed.

        Wrapping rebuilds one span per character, so a long prompt used to
        cost more than the rest of the frame together and froze typing.
        """
        width = max(16, columns - 2)
        key = (self.prompt.text, self.prompt.cursor, self.prompt.anchor, width)
        cached = self._prompt_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        layout = _prompt_layout(self.prompt.text, self.prompt.cursor,
                                self.prompt.anchor, width)
        self._prompt_cache = (key, layout)
        return layout

    def _transcript_lines(self) -> list[Line]:
        width = max(16, self.screen.size().columns - 2)
        live = LiveRender(
            phase=self.phase,
            thinking_lines=self.live_thinking.lines() if self.phase == 'thinking' else [],
            thinking_started=self.live_thinking_started,
            answer_lines=self.live_answer.lines() if self.phase == 'answering' else [],
            compaction_lines=self.live_compaction.lines() if self.phase == 'compacting' else [],
            aside_question=self.live_aside_question,
            aside_lines=self.live_aside.lines() if self.phase == 'aside' else [],
            tool_name=self.status_hint,
            tool_running=self.phase == 'tool',
            tool_output=self.live_tool_output,
        )
        return self.transcript.render(self.items, live, width, self.config,
                                      self.config_version, time.time(),
                                      self.palette)

    # ── rendering ───────────────────────────────────────────
    def _viewport_height(self, rows: int) -> int:
        header = 2 if self.config.ui.show_header else 0
        return max(1, rows - header - 1 - 1 - 3)

    def _render_interval(self) -> float:
        """Seconds between two frames, derived from the configured rate."""
        fps = int(getattr(self.config.ui, 'fps', 0) or _DEFAULT_FPS)
        return 1.0 / max(1, min(240, fps))

    def _render_if_due(self) -> None:
        now = time.monotonic()
        if not self.dirty:
            return
        if now - self.last_render < self._render_interval():
            return
        self.last_render = now
        self._render()
        # The flag goes down after the frame is drawn, so a reader that waits
        # for a clean app also waits for the picture it describes.
        self.dirty = False

    def _render(self) -> None:
        # The snapshot belongs to one frame; the next frame reads it again.
        self._drop_snapshot()
        columns, rows = self.screen.size()
        if columns != self.canvas.width or rows != self.canvas.height:
            self.canvas.resize(columns, rows)
        self.canvas.palette = self.palette
        self.canvas.clear()
        now = time.time()
        header_h = 2 if self.config.ui.show_header else 0
        if self.config.ui.show_header:
            self._draw_header(columns)
        prompt_rows, caret_row, caret_col = self._prompt_rows(columns)
        max_prompt = max(3, min(10, rows // 3))
        window_start = 0
        if len(prompt_rows) > max_prompt:
            window_start = max(0, min(caret_row - max_prompt + 1,
                                      len(prompt_rows) - max_prompt))
        visible_prompt = prompt_rows[window_start:window_start + max_prompt]
        prompt_h = max(1, len(visible_prompt))
        footer_y = rows - 1
        prompt_top = footer_y - prompt_h
        separator_y = prompt_top - 1
        transcript_top = header_h
        viewport_h = max(1, separator_y - transcript_top)
        lines = self._transcript_lines()
        total = len(lines)
        max_scroll = max(0, total - viewport_h)
        if not self.follow and total > self._last_total:
            # Anchor a scrolled-up viewport to the same content as new lines
            # stream in, instead of letting it drift downward every frame.
            self.scroll += total - self._last_total
        self._last_total = total
        self._last_viewport = viewport_h
        self._last_max_scroll = max_scroll
        self.scroll = 0 if self.follow else min(self.scroll, max_scroll)
        start = max(0, min(total - viewport_h - self.scroll, max_scroll))
        visible = lines[start:start + viewport_h]
        self._view_lines = visible
        self._view_start = start
        self._view_top = transcript_top
        if not visible and self.phase == 'idle':
            # An empty pane: the welcome logo and a random command tip.
            from adcode.tui.welcome import paint_welcome
            paint_welcome(self.canvas, transcript_top, separator_y, columns,
                          self.palette, self.welcome_tip)
        else:
            for index, line in enumerate(visible):
                # Tool call rows declare a bleed so their accent bar sits flush
                # with the left edge; the gutter inside the row keeps the text
                # aligned.
                x = _row_offset(line)
                selected = self._row_selection(start + index)
                self._draw_line(x, transcript_top + index, line,
                                max(1, columns - x - 1), selected)
            self._draw_scrollbar(columns, transcript_top, viewport_h, total,
                                 start)
        self.canvas.hline(0, separator_y, columns, self.palette.border, alpha=110)
        for index, line in enumerate(visible_prompt):
            self._draw_line(0, prompt_top + index, line, columns)
        if self._completion_visible():
            self._draw_completions(columns, separator_y)
        self._draw_footer(columns, footer_y, max_scroll)
        cursor: tuple[int, int] | None = None
        if self.overlay is not None:
            self.overlay.render(self.canvas, columns, rows, self.palette, now)
            cursor = self.overlay.cursor()
        else:
            caret_screen_row = prompt_top + (caret_row - window_start)
            if 0 <= caret_row - window_start < len(visible_prompt):
                cursor = (min(columns - 1, caret_col), caret_screen_row)
        # Only the rows whose cells actually changed are written; the caret
        # escape is appended only when it moved.  A frame that changes nothing
        # writes nothing at all, which is what keeps an idle UI off the wire.
        payload = self.canvas.render_diff()
        if cursor is not None:
            parts = []
            if self._cursor_visible is not True:
                parts.append('\x1b[?25h')
                self._cursor_visible = True
            # Writing rows moves the terminal cursor to the end of the last
            # row it touched, so the caret has to be placed again whenever the
            # frame carried any bytes.  Only a frame that writes nothing can
            # trust the remembered position.
            if payload or cursor != self._last_cursor:
                parts.append(f'\x1b[{cursor[1] + 1};{cursor[0] + 1}H')
                self._last_cursor = cursor
            tail = ''.join(parts)
        else:
            tail = ''
            if self._cursor_visible is not False:
                tail = '\x1b[?25l'
                self._cursor_visible = False
            self._last_cursor = None
        if payload or tail:
            self.screen.flush_frame(payload + tail)

    def _draw_line(self, x: int, y: int, line: Line, width: int,
                   selection: tuple[int, int] | None = None) -> None:
        if not line:
            return
        cursor = x
        if selection is None:
            for span in line:
                if cursor - x >= width:
                    break
                fg = self._color(span.fg, 'text')
                bg = self._color(span.bg, None) if span.bg is not None else None
                available = width - (cursor - x)
                text = truncate_text(span.text, available)
                if not text:
                    continue
                cursor += self.canvas.draw_text(cursor, y, text, fg, bg,
                                                span.attrs, span.alpha)
            return
        start_col, end_col = selection
        for piece, span, selected in split_line_columns(line, start_col, end_col):
            if cursor - x >= width:
                break
            if selected:
                fg = self.palette.selection_fg
                bg = self.palette.selection_bg
            else:
                fg = self._color(span.fg, 'text')
                bg = self._color(span.bg, None) if span.bg is not None else None
            available = width - (cursor - x)
            text = truncate_text(piece, available)
            if not text:
                continue
            cursor += self.canvas.draw_text(cursor, y, text, fg, bg,
                                            span.attrs, span.alpha)

    def _color(self, role: Any, default: str | None) -> Any:
        from adcode.theme import resolve_role
        if role is None:
            return resolve_role(self.palette, default) if default else None
        return resolve_role(self.palette, role, default or 'text')

    def _draw_header(self, columns: int) -> None:
        self.canvas.fill_row(0, self.palette.status_bg)
        snap = self.snapshot()
        title = 'AdCode'
        self.canvas.draw_text(1, 0, title, self.palette.text, self.palette.status_bg,
                              BOLD)
        offset = 1 + len(title)
        session_title = snap.session_title or ''
        if session_title:
            self.canvas.draw_text(offset, 0, '  ·  ', self.palette.muted,
                                  self.palette.status_bg)
            offset += 5
            self.canvas.draw_text(offset, 0,
                                  truncate_text(session_title, max(1, columns // 3)),
                                  self.palette.subtext, self.palette.status_bg)
        right = f'{snap.provider} / {snap.model}'
        effort = f'  ·  {snap.effort}' if snap.effort else ''
        text = right + effort
        right_x = max(1, columns - 2 - text_width(text))
        if right_x > offset + 4:
            self.canvas.draw_text(right_x, 0, right, self.palette.subtext,
                                  self.palette.status_bg)
            self.canvas.draw_text(right_x + text_width(right), 0, effort,
                                  self.palette.muted, self.palette.status_bg)
        self.canvas.hline(0, 1, columns, self.palette.border, alpha=110)

    def _current_action(self) -> str:
        """What the agent is doing right now, for the status line.

        A merged chain shows no details, so the status line is where the
        running action is spelled out ("read  README.md").
        """
        name = self.status_hint or 'tool'
        if self.config.tools.chain != 'merged':
            return name
        from adcode.tui.transcript import tool_arg_brief
        brief = tool_arg_brief(self.status_tool_name or name, self.status_args)
        return f'{name}  {brief}' if brief else name

    def _draw_footer(self, columns: int, y: int, max_scroll: int) -> None:
        snap = self.snapshot()
        frame = SPINNER[int(time.time() * 10) % len(SPINNER)]
        phase = self.phase
        if phase == 'connecting':
            # Reconnecting is the same wait, but the user deserves to know the
            # first attempt already failed.
            verb = 'reconnecting' if self.connect_retrying else 'connecting'
            elapsed = self._connect_elapsed()
            suffix = f' {elapsed:.1f}s' if elapsed >= 1.0 else ''
            status = f'{frame} {verb}{suffix}'
        elif phase == 'thinking':
            status = f'{frame} thinking'
        elif phase == 'answering':
            status = f'{frame} writing'
        elif phase == 'tool':
            status = f'{frame} {self._current_action()}'
        elif phase == 'compacting':
            status = f'{frame} compacting'
        elif phase == 'aside':
            status = f'{frame} aside'
        else:
            status = 'ready'
        color = {
            'connecting': 'info', 'thinking': 'subtext', 'answering': 'text',
            'tool': 'accent', 'compacting': 'info', 'aside': 'accent',
            'idle': 'muted',
        }.get(phase, 'muted')
        # The token status sits at the right edge; the status line and the
        # extras get the room that is left of it.
        right = ''
        right_x = columns
        if self.config.ui.show_token_status:
            context_pct = 0
            if snap.context_window:
                context_pct = round(100 * snap.context_tokens / snap.context_window)
            right = (f'ctx {context_pct}%  ↑{_format_tokens(snap.total_in)} '
                     f'↓{_format_tokens(snap.total_out)}')
            right_x = max(1, columns - 1 - text_width(right))
        room = max(0, right_x - 1)
        x = 1
        x += self.canvas.draw_text(x, y, status, self._color(color, 'muted'),
                                   None, 0, 255, room)
        extras: list[str] = []
        queued = snap.queued
        if queued:
            extras.append(f'{queued} queued')
        attached = count_image_markers(self.prompt.text)
        if attached:
            extras.append(f'{attached} image{"s" if attached > 1 else ""}')
        if self.btw_mode:
            extras.append('aside question mode')
        if self.exit_armed:
            extras.append('ctrl+c again to exit')
        agents = [row for row in (snap.agents or []) if row.get('status') not in
                  ('completed', 'errored', 'interrupted', 'shutdown', None)
                  and not (isinstance(row.get('status'), dict))]
        if agents and self.config.agents.show_activity:
            extras.append(f'{len(agents)} agent{"s" if len(agents) > 1 else ""}')
        if self.config.tools.chain == 'merged':
            extras.append('merged steps')
        if self.config.thinking.mode == 'summary':
            extras.append('summary thinking')
        if self.config.hold.enabled:
            extras.append(f'hold ×{self.config.hold.max_retries}')
        if self.scroll > 0:
            extras.append(f'↑ {self.scroll} lines')
        selected = self._selection_range()
        if selected is not None:
            length = len(self._selection_text())
            extras.append(f'{length} selected · ctrl+c copies')
        # The extras are dropped from the right until they fit: half a word
        # on the line reads like a bug.  The separator in front of the first
        # one counts towards the room.
        separator = '  ·  '
        room = right_x - x - text_width(separator)
        fitting: list[str] = []
        for extra in extras:
            if text_width('  ·  '.join([*fitting, extra])) > room:
                break
            fitting.append(extra)
        if fitting:
            x += self.canvas.draw_text(x, y, separator, self.palette.border,
                                       None, 0, 255, right_x - x)
            self.canvas.draw_text(x, y, '  ·  '.join(fitting), self.palette.muted,
                                  None, 0, 255, right_x - x)
        self._draw_status_sweep(columns, y, phase)
        if right:
            self.canvas.draw_text(right_x, y, right, self.palette.muted)

    def _draw_status_sweep(self, columns: int, y: int, phase: str) -> None:
        """The light that runs along the status line while the agent works.

        It enters at the left edge and leaves at the right one a second
        later, so the line reads as alive without pulling the eye away from
        the text.  The wait for the provider (``connecting``) has no light:
        nothing is happening yet.
        """
        if phase not in ('thinking', 'answering', 'tool', 'compacting', 'aside'):
            return
        # The light enters from beyond the left edge and leaves beyond the
        # right one, so it is never half on screen.
        head = ((time.time() % SWEEP_PERIOD) / SWEEP_PERIOD
                * (columns + SWEEP_SPAN) - SWEEP_SPAN)
        for offset in range(SWEEP_SPAN):
            x = int(head) - offset
            if not (0 <= x < columns):
                continue
            fade = 1.0 - offset / SWEEP_SPAN
            self.canvas.fill_rect(x, y, 1, 1, self.palette.accent,
                                  alpha=int(20 + 110 * fade * fade))

    def _draw_scrollbar(self, columns: int, top: int, height: int, total: int,
                        start: int) -> None:
        if total <= height or height <= 0:
            return
        x = columns - 1
        self.canvas.vline(x, top, height, self.palette.border, alpha=80, char='│')
        thumb_height = max(1, height * height // total)
        thumb_start = top + (height - thumb_height) * start // max(1, total - height)
        for offset in range(thumb_height):
            self.canvas.put(x, thumb_start + offset, '┃', self.palette.muted)

    def _draw_completions(self, columns: int, separator_y: int) -> None:
        candidates = self._completion_candidates()
        if not candidates:
            return
        window = 8
        index = min(self.prompt.completion_index, len(candidates) - 1)
        top = max(0, min(index - window // 2, len(candidates) - window))
        visible = candidates[top:top + window]
        width = min(max(44, columns - 4), min(96, columns - 2))
        height = len(visible) + 2
        x = 1
        y = max(1, separator_y - height)
        self.canvas.box(x, y, width, height, self.palette.border,
                        bg=self.palette.panel, alpha=255)
        for row_index, (completed, label, description) in enumerate(visible):
            row_y = y + 1 + row_index
            selected = (top + row_index) == index
            bg = self.palette.selection_bg if selected else self.palette.panel
            if selected:
                self.canvas.fill_rect(x + 1, row_y, width - 2, 1, bg)
            self.canvas.draw_text(x + 2, row_y, '› ' if selected else '  ',
                                  self.palette.accent if selected else self.palette.muted,
                                  bg)
            self.canvas.draw_text(x + 4, row_y,
                                  truncate_text(label, max(8, min(32, width - 8))),
                                  self.palette.text if selected else self.palette.subtext,
                                  bg, BOLD if selected else 0)
            if description:
                desc_x = x + 4 + min(32, width // 2)
                self.canvas.draw_text(desc_x, row_y, truncate_text(
                    description, max(1, x + width - 2 - desc_x)),
                    self.palette.muted, bg, DIM)

    # ── prompt manipulation for commands ────────────────────
    def set_prompt(self, text: str, submit: bool = False) -> None:
        self.prompt.set_text(text)
        self.prompt.cursor = len(text)
        self.dismiss_overlay()
        if submit:
            self.submit(text.strip())

    def open_command_palette(self) -> None:
        """Open the command palette (ctrl+p)."""
        from adcode.commands import COMMANDS
        items = [DialogItem(command.name, command.name, command.description)
                 for command in COMMANDS]
        if not items:
            self.notify('No commands are registered.', 'warning')
            return
        self.open_overlay(CommandPalette('Command palette', items,
                                          self._run_command))

    def _run_command(self, name: str) -> None:
        from adcode.commands import dispatch
        dispatch(self, str(name))

    def open_settings(self) -> None:
        self.open_overlay(SettingsOverlay(self.config, self.config_changed))

    def open_keybindings(self) -> None:
        """Open the key bindings panel (/keys)."""
        from adcode.tui.keybindings_panel import KeybindingsPanel
        self.open_overlay(KeybindingsPanel(self.config.keys,
                                           self.config_changed))

    def open_sessions_browser(self) -> None:
        """Browse and load saved sessions (ctrl+h, /sessions)."""
        rows = self.session_store.list()
        if not rows:
            self.notify('No saved sessions.', 'info')
            return

        def choose(name: Any) -> None:
            self.load_session(str(name))
        self.open_select(
            'Sessions',
            [DialogItem(row['name'], row['title'] or row['name'],
                        f'{row["updated"]} · {row["messages"]} messages')
             for row in rows],
            choose, width=86)

    def open_message(self, title: str, message: str) -> None:
        self.open_overlay(MessageDialog(title, message))

    def open_select(self, title: str, items: list[DialogItem],
                    on_select: Callable[[Any], None],
                    width: int = 72) -> None:
        self.open_overlay(SelectDialog(title, items, on_select, width=width))

    def open_confirm(self, title: str, message: str,
                     on_result: Callable[[bool], None]) -> None:
        self.open_overlay(ConfirmDialog(title, message, on_result))

    def open_text_prompt(self, title: str, label: str, initial: str,
                         on_submit: Callable[[str], None],
                         hint: str = '', password: bool = False) -> None:
        self.open_overlay(TextPrompt(title, label, initial, on_submit,
                                     hint=hint, password=password))
