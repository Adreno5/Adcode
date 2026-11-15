"""Headless end-to-end harness for the real AdCode application.

``AppHarness`` starts the actual :class:`adcode.tui.app.AdCodeApp` main loop on
a worker thread, substituting only the terminal edges:

* a :class:`~harness.screen.FakeScreen` for the console, which captures every
  composited frame and never writes to a TTY;
* a :class:`~harness.input.ScriptedInput` for the reader thread, which feeds
  real ANSI key/mouse/paste sequences into the app's own parser;
* a temporary config and working directory (see
  :class:`~harness.workspace.Workspace`);
* a :class:`~harness.gateway.FakeLLMGateway` speaking the real HTTP protocol.

Everything in between -- the agent loop, LLM SDK clients, tool execution,
session persistence, rendering and command handling -- is the production code.
"""

from __future__ import annotations

import queue
import shutil
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable
from unittest import mock

import adcode.config as config_module
import adcode.models_dev as models_dev_module
import adcode.system_prompt as system_prompt_module
import adcode.tui.app as app_module
import adcode.tui.dialogs as dialogs_module
from adcode.tui.app import AdCodeApp
from harness.gateway import FakeLLMGateway
from harness.input import ScriptedInput
from harness.screen import FakeScreen, FrameSnapshot
from harness.script import EchoResponder, ModelRequest, Responder
from harness.workspace import DEFAULT_MODEL, Workspace, build_config


class HarnessError(RuntimeError):
    """The simulated app failed or a wait timed out."""


class AppHarness:
    def __init__(self, *, responder: Responder | None = None,
                 wire: str = 'openai chat completions',
                 width: int = 100, height: int = 30,
                 reasoning: bool = True,
                 config_mutator: Callable[[Any], None] | None = None,
                 project_files: dict[str, str] | None = None,
                 global_instructions: str | None = None,
                 keep: bool = False,
                 capture_vt: bool = True,
                 root: str | Path | None = None) -> None:
        self.workspace = Workspace(root=root, keep=keep)
        self.gateway = FakeLLMGateway(responder or EchoResponder())
        self.gateway.start()
        config = build_config(self.gateway.base_url, wire, reasoning=reasoning)
        if config_mutator is not None:
            config_mutator(config)
        self.workspace.write_config(config)
        if project_files:
            for relative, content in project_files.items():
                self.workspace.write_project_file(relative, content)
        if global_instructions is not None:
            self.workspace.write_global_instructions(global_instructions)

        self.wire = wire
        self.model_id = DEFAULT_MODEL
        self.screen = FakeScreen(width, height, capture_vt=capture_vt)
        self.input_queue: 'queue.Queue[Any]' = queue.Queue()
        self.input = ScriptedInput(self.input_queue)
        self.clipboard = ''
        self.clipboard_image: bytes | None = None
        self.artifacts: Any = None
        self.app: AdCodeApp | None = None
        self.error: BaseException | None = None
        self.finished = threading.Event()
        self._thread: threading.Thread | None = None
        self._patch_stack: ExitStack | None = None
        self._started = False
        self._closed = False

    # ── lifecycle ──────────────────────────────────────────
    def start(self) -> 'AppHarness':
        if self._started:
            return self
        self._started = True
        self._patch_stack = ExitStack()
        self._install_isolated_globals(self._patch_stack)
        try:
            self.app = AdCodeApp(
                cwd=str(self.workspace.project),
                resume=False,
                config_path=str(self.workspace.config_path),
                screen=self.screen,
                input_queue=self.input_queue,
                input_thread=self.input,
            )
            self.screen.bind(self.app.canvas)
            self._thread = threading.Thread(
                target=self._run, name='adcode-harness-app', daemon=True)
            self._thread.start()
            self.wait_until(
                lambda: self.app is not None and self.app.loop is not None
                and self.screen.latest is not None,
                timeout=20.0, description='the app to boot and paint a frame')
        except BaseException:
            self.close()
            raise
        return self

    def _run(self) -> None:
        try:
            assert self.app is not None
            self.app.run()
        except BaseException as error:  # surface thread crashes in the test
            self.error = error
        finally:
            self.finished.set()

    def __enter__(self) -> 'AppHarness':
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.gateway.release_all()
        thread = self._thread
        app = self.app
        if app is not None and thread is not None and thread.is_alive():
            try:
                app.request_exit()
            except Exception:
                pass
            thread.join(timeout=10.0)
            if thread.is_alive():
                try:
                    if app.loop is not None:
                        app.loop.interrupt()
                except Exception:
                    pass
                self.gateway.release_all()
                thread.join(timeout=5.0)
        self.gateway.stop()
        if self._patch_stack is not None:
            self._patch_stack.close()
            self._patch_stack = None
        self.workspace.cleanup()
        error = self.error
        if error is not None:
            raise HarnessError(f'app thread raised {error!r}') from error

    # ── isolation ──────────────────────────────────────────
    def _install_isolated_globals(self, stack: ExitStack) -> None:
        workspace = self.workspace
        stack.enter_context(mock.patch.object(
            system_prompt_module, 'GLOBAL_INSTRUCTIONS_PATH',
            workspace.global_instructions_path))
        stack.enter_context(mock.patch.object(
            models_dev_module, 'load_cached_catalog', lambda: None))
        stack.enter_context(mock.patch.object(
            config_module, 'encrypt_secret', lambda value: value))
        stack.enter_context(mock.patch.object(
            config_module, 'decrypt_secret', lambda value: value))
        stack.enter_context(mock.patch.object(
            app_module, 'get_clipboard', self._get_clipboard))
        stack.enter_context(mock.patch.object(
            app_module, 'set_clipboard', self._set_clipboard))
        stack.enter_context(mock.patch.object(
            app_module, 'read_clipboard_image', lambda: self.clipboard_image))
        stack.enter_context(mock.patch.object(
            dialogs_module, 'get_clipboard', self._get_clipboard))

    def _get_clipboard(self) -> str:
        return self.clipboard

    def _set_clipboard(self, text: str) -> bool:
        self.clipboard = text
        return True

    # ── waiting ────────────────────────────────────────────
    def wait_until(self, predicate: Callable[[], bool], timeout: float = 10.0,
                   description: str = 'condition') -> 'AppHarness':
        deadline = time.monotonic() + timeout
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            error = self.error
            if error is not None:
                raise HarnessError(
                    f'app thread crashed while waiting for {description}: '
                    f'{error!r}') from error
            try:
                if predicate():
                    return self
            except BaseException as error:  # keep the last predicate failure
                last_error = error
            time.sleep(0.01)
        screen = self.screen_text()
        detail = f'\n--- last frame ---\n{screen}' if screen else ''
        suffix = f'\nlast predicate error: {last_error!r}' if last_error else ''
        raise HarnessError(
            f'timed out after {timeout:.1f}s waiting for {description}{suffix}'
            f'{detail}')

    def wait_idle(self, timeout: float = 15.0) -> 'AppHarness':
        def idle() -> bool:
            app = self.app
            if app is None or app.phase != 'idle':
                return False
            if app.loop is None:
                return not app.dirty
            return (not app.loop.busy and app.loop._work.qsize() == 0
                    and not app.dirty)

        return self.wait_until(idle, timeout, 'the agent loop to become idle')

    def wait_for_text(self, needle: str, timeout: float = 10.0) -> FrameSnapshot:
        self.wait_until(
            lambda: (self.screen.latest is not None
                     and self.screen.latest.contains(needle)),
            timeout, f'the screen to contain {needle!r}')
        assert self.screen.latest is not None
        return self.screen.latest

    def wait_for_requests(self, count: int, timeout: float = 15.0) -> None:
        self.wait_until(lambda: len(self.gateway.requests) >= count,
                        timeout, f'{count} gateway request(s)')
        self.gateway.wait_for_requests(count, timeout)

    def wait_prompt(self, text: str | None = None, cursor: int | None = None,
                    timeout: float = 10.0) -> 'AppHarness':
        def matches() -> bool:
            app = self.app
            if app is None:
                return False
            if text is not None and app.prompt.text != text:
                return False
            if cursor is not None and app.prompt.cursor != cursor:
                return False
            return True

        return self.wait_until(
            matches, timeout,
            f'the prompt to be {text!r} with cursor {cursor!r}'
        ).wait_rendered(timeout)

    def wait_rendered(self, timeout: float = 10.0) -> 'AppHarness':
        return self.wait_until(
            lambda: (self.app is not None and not self.app.dirty
                     and self.screen.latest is not None),
            timeout, 'the latest frame to render')

    # ── driving ────────────────────────────────────────────
    def type_text(self, text: str) -> 'AppHarness':
        self._require_live()
        self.input.type_text(text)
        return self

    def press(self, *keys: str) -> 'AppHarness':
        self._require_live()
        for key in keys:
            self.input.press(key)
        return self

    def paste(self, text: str) -> 'AppHarness':
        self._require_live()
        self.input.paste(text)
        return self

    def wheel(self, action: str = 'wheel_up', x: int = 10,
              y: int = 10) -> 'AppHarness':
        self._require_live()
        self.input.wheel(action, x, y)
        return self

    def click(self, x: int = 10, y: int = 10) -> 'AppHarness':
        self._require_live()
        self.input.click(x, y)
        return self

    def submit(self, text: str, *, expect_requests: int = 1,
               timeout: float = 20.0) -> 'AppHarness':
        """Type a message, press Enter and wait for the turn to settle."""
        before = len(self.gateway.requests)
        self.type_text(text)
        self.press('enter')
        if expect_requests > 0:
            self.wait_until(
                lambda: len(self.gateway.requests) >= before + expect_requests,
                timeout, f'{expect_requests} request(s) for {text!r}')
        self.wait_idle(timeout)
        return self

    def run_command(self, text: str) -> 'AppHarness':
        """Submit a slash command through the prompt (no model request)."""
        self.type_text(text)
        self.press('enter')
        return self

    def resize(self, columns: int, rows: int, timeout: float = 10.0) -> 'AppHarness':
        self.screen.resize(columns, rows)
        self.wait_until(
            lambda: (self.app is not None
                     and self.app.canvas.width == columns
                     and self.app.canvas.height == rows
                     and self.screen.latest is not None
                     and self.screen.latest.width == columns),
            timeout, f'a {columns}x{rows} frame')
        return self

    def release(self) -> 'AppHarness':
        self.gateway.release_all()
        return self

    def _require_live(self) -> None:
        if self.app is None:
            raise HarnessError('harness has not been started')
        if self.error is not None:
            raise HarnessError(f'app thread crashed: {self.error!r}') from self.error
        if self.finished.is_set():
            raise HarnessError('the app loop has already exited')

    # ── inspection ─────────────────────────────────────────
    def screen_text(self) -> str:
        latest = self.screen.latest
        return latest.text() if latest is not None else ''

    def snapshot(self, label: str) -> FrameSnapshot:
        latest = self.screen.latest
        if latest is None:
            raise HarnessError('no frame has been rendered yet')
        if self.artifacts is not None:
            self.artifacts.frame(label, latest)
        return latest

    @property
    def phase(self) -> str:
        return self.app.phase if self.app is not None else ''

    @property
    def prompt_text(self) -> str:
        return self.app.prompt.text if self.app is not None else ''

    @property
    def prompt_cursor(self) -> int:
        return self.app.prompt.cursor if self.app is not None else -1

    @property
    def overlay(self) -> Any:
        return self.app.overlay if self.app is not None else None

    def transcript_items(self) -> list[dict[str, Any]]:
        app = self.app
        if app is None:
            return []
        return [item.to_dict() for item in app.items]

    def transcript_texts(self) -> list[tuple[str, str]]:
        app = self.app
        if app is None:
            return []
        return [(item.kind, item.text) for item in app.items]

    def state(self) -> dict[str, Any]:
        app = self.app
        if app is None:
            return {'started': False}
        loop = app.loop
        snapshot = loop.snapshot() if loop is not None else None
        return {
            'started': True,
            'phase': app.phase,
            'running': app.running,
            'prompt': app.prompt.text,
            'cursor': app.prompt.cursor,
            'overlay': type(app.overlay).__name__ if app.overlay else '',
            'scroll': app.scroll,
            'follow': app.follow,
            'session': app.loop.session_name if loop is not None else '',
            'loop_status': loop.status if loop is not None else '',
            'requests': len(self.gateway.requests),
            'model': snapshot.model if snapshot is not None else '',
            'context_tokens': snapshot.context_tokens if snapshot is not None else 0,
            'transcript_items': len(app.items),
        }

    def artifact_payload(self) -> dict[str, Any]:
        return {
            'scenario_state': self.state(),
            'transcript': self.transcript_items(),
            'config': self.workspace.config_json(),
            'session': self.workspace.session_payload(),
            'session_files': self.workspace.session_files(),
            'gateway_requests': [
                {
                    'index': request.index,
                    'wire': request.wire,
                    'path': request.path,
                    'model': request.model,
                    'system': request.system_text,
                    'tools': request.tool_names,
                    'reasoning_effort': request.reasoning_effort,
                    'body': request.body,
                }
                for request in self.gateway.requests
            ],
        }

    # ── passthroughs used in tests ─────────────────────────
    @property
    def requests(self) -> list[ModelRequest]:
        return self.gateway.requests

    def last_request(self) -> ModelRequest | None:
        return self.gateway.requests[-1] if self.gateway.requests else None

    def mark_cleanup_keep(self, keep: bool = True) -> None:
        """Keep the temporary tree after close (useful while debugging)."""
        self.workspace.keep = keep

    def copy_workspace(self, destination: str | Path) -> Path:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.workspace.root, destination, dirs_exist_ok=True)
        return destination
