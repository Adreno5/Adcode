"""One AdCode instance shared by every browser that connects.

The app is the production AdCodeApp.  Only its two terminal edges are
replaced: WebScreen publishes the composited cell grid instead of writing ANSI
to a console, and WebInput turns browser keystrokes into the same KeyEvent
objects the console reader produces.

Frames are diffed per row before they leave the process, so an idle UI sends
nothing and streaming text rewrites one row -- the same economy the terminal
path gets from Canvas.render_diff, but carrying cells instead of escape
sequences.

Wire format (JSON text frames):

  browser -> server
    {"t": "size",  "cols": 100, "rows": 30}
    {"t": "key",   "k": "ctrl+shift+left"}
    {"t": "text",  "v": "hello"}
    {"t": "paste", "v": "..."}
    {"t": "mouse", "a": "wheel_up", "x": 3, "y": 7, "b": 0}
    {"t": "modifier", "m": "alt", "a": "press"}     # a held modifier

  server -> browser
    {"t": "full",  "w": 100, "h": 30, "lines": [[y, runs], ...],
     "caret": [x, y], "show": true, "title": "..."}
    {"t": "rows",  "lines": [[y, runs], ...], "caret": [x, y], "show": true}
    {"t": "size",  "w": 80, "h": 24}
    {"t": "title", "v": "..."}
    {"t": "bell"}
    {"t": "error", "v": "..."}      # the app thread raised
    {"t": "closed", "v": "..."}     # the session ended

  A run is [text, fg, bg, attrs] with colours packed as 0xRRGGBB and attrs
  carrying the canvas attribute bits (BOLD 1, DIM 2, ITALIC 4, UNDERLINE 8,
  STRIKE 16; REVERSE is already folded into fg/bg).
"""

from __future__ import annotations

import json
import queue
import re
import threading
from typing import Any

from adcode.tui.app import AdCodeApp
from adcode.tui.input import KeyEvent
from adcode.web.input import WebInput
from adcode.web.screen import WebScreen

# The app appends the caret escape at the very end of a frame, so anchoring
# the match keeps row positioning sequences from being mistaken for it.
_CARET_RE = re.compile(r'\x1b\[(\d+);(\d+)H\Z')
_HIDE_CARET = '\x1b[?25l'
_SHOW_CARET = '\x1b[?25h'

# Bounds for the grid a browser may ask for.
MIN_COLUMNS, MAX_COLUMNS = 20, 400
MIN_ROWS, MAX_ROWS = 6, 200
DEFAULT_COLUMNS, DEFAULT_ROWS = 100, 30

# A client that falls this far behind is sent a whole frame instead of the
# backlog: dropping rows would leave its screen permanently out of date.
_MAX_BACKLOG = 48


def _clamp(value: Any, low: int, high: int, fallback: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(low, min(high, number))


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(',', ':'))


class WebClient:
    """One browser connection: an outbox, a writer thread, its grid."""

    def __init__(self, connection, session: 'WebSession') -> None:
        self.connection = connection
        self.session = session
        self._condition = threading.Condition()
        self._outbox: list[str] = []
        self._needs_full = False
        self._closed = False
        self._thread = threading.Thread(
            target=self._write_loop, name='adcode-web-writer', daemon=True)

    # -- lifecycle ------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._outbox.clear()
            self._condition.notify_all()
        try:
            self.connection.close()
        except Exception:
            pass

    @property
    def closed(self) -> bool:
        return self._closed

    # -- outbox ---------------------------------------------------
    def push(self, text: str) -> None:
        """Queue one message; never blocks the app thread."""
        with self._condition:
            if self._closed:
                return
            if len(self._outbox) >= _MAX_BACKLOG:
                self._outbox.clear()
                self._needs_full = True
            if self._needs_full:
                # The writer will rebuild a full frame; older diffs would be
                # superseded by it anyway.
                self._condition.notify()
                return
            self._outbox.append(text)
            self._condition.notify()

    def _write_loop(self) -> None:
        while True:
            with self._condition:
                while (not self._outbox and not self._needs_full
                       and not self._closed):
                    self._condition.wait(0.25)
                if self._closed:
                    return
                pending = self._outbox
                self._outbox = []
                needs_full = self._needs_full
                self._needs_full = False
            try:
                if needs_full:
                    self.connection.send_text(self.session.snapshot())
                    pending = []
                for text in pending:
                    self.connection.send_text(text)
            except Exception:
                self.close()
                return


class WebSession:
    """Owns the app, its thread, the connected browsers and the grid size."""

    def __init__(self, cwd: str | None = None, resume: bool = True,
                 config_path: str | None = None,
                 size: tuple[int, int] | None = None) -> None:
        self._lock = threading.RLock()
        self._grid = size or (DEFAULT_COLUMNS, DEFAULT_ROWS)
        self._clients: dict[WebClient, tuple[int, int] | None] = {}
        self._rows: list[list[Any] | None] = []
        self._width = 0
        self._height = 0
        self._caret: tuple[int, int] | None = None
        self._caret_visible = True
        self._title = ''
        self._closed = False
        self.error: BaseException | None = None
        self.canvas = None
        self.app: AdCodeApp | None = None
        self.finished = threading.Event()
        self.input_queue: 'queue.Queue[KeyEvent]' = queue.Queue()
        self.screen = WebScreen(self)
        self.input = WebInput(self.input_queue)
        self._cwd = cwd
        self._resume = resume
        self._config_path = config_path
        self._thread: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------
    def start(self) -> 'WebSession':
        """Build the app and run its main loop on a worker thread."""
        self.app = AdCodeApp(
            cwd=self._cwd, resume=self._resume,
            config_path=self._config_path,
            screen=self.screen, input_queue=self.input_queue,
            input_thread=self.input,
        )
        self.canvas = self.app.canvas
        self._thread = threading.Thread(
            target=self._run, name='adcode-web-app', daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            assert self.app is not None
            self.app.run()
        except BaseException as error:  # surfaced to the CLI and the browsers
            self.error = error
            self.broadcast(_dumps({'t': 'error', 'v': f'{error!r}'}))
        finally:
            self.finished.set()
            self.broadcast(_dumps({'t': 'closed', 'v': 'the session ended'}))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        app = self.app
        thread = self._thread
        if app is not None and thread is not None and thread.is_alive():
            try:
                app.request_exit()
            except Exception:
                pass
            thread.join(timeout=10.0)
        with self._lock:
            clients = list(self._clients)
            self._clients.clear()
        for client in clients:
            client.close()

    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    # -- geometry -------------------------------------------------
    def grid(self) -> tuple[int, int]:
        with self._lock:
            return self._grid

    def set_client_size(self, client: 'WebClient', columns: Any,
                        rows: Any) -> None:
        """Record what a browser can show; the grid follows the smallest."""
        with self._lock:
            if client not in self._clients:
                return
            self._clients[client] = (
                _clamp(columns, MIN_COLUMNS, MAX_COLUMNS, DEFAULT_COLUMNS),
                _clamp(rows, MIN_ROWS, MAX_ROWS, DEFAULT_ROWS),
            )
        self._recompute_grid()

    # -- clients ---------------------------------------------------
    def attach(self, connection) -> None:
        """Serve one browser connection until it goes away."""
        client = WebClient(connection, self)
        with self._lock:
            self._clients[client] = None
            snapshot = self.snapshot()
        client.start()
        client.push(snapshot)
        try:
            for raw in connection.messages():
                self.handle(client, raw)
        except Exception:
            pass
        finally:
            client.close()
            with self._lock:
                self._clients.pop(client, None)
            self._recompute_grid()

    def _recompute_grid(self) -> None:
        """Keep every browser on the smallest grid any of them asked for.

        Each window scales the grid up to fill its own screen, so the smallest
        request is the one nobody has to scroll sideways -- and when that
        window closes the grid grows back for the others.
        """
        with self._lock:
            known = [size for size in self._clients.values() if size]
            if not known:
                return  # keep the last grid: the next client may match it
            wanted = (min(size[0] for size in known),
                      min(size[1] for size in known))
            if wanted == self._grid:
                return
            self._grid = wanted
            self._rows = []
            self._width = self._height = 0
            clients = list(self._clients)
        app = self.app
        if app is not None:
            # The loop picks the new size up on its next pass.
            app.dirty = True
        message = _dumps({'t': 'size', 'w': wanted[0], 'h': wanted[1]})
        for target in clients:
            target.push(message)

    def broadcast(self, text: str) -> None:
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            client.push(text)

    # -- browser messages -----------------------------------------
    def handle(self, client: 'WebClient', raw: str) -> None:
        try:
            message = json.loads(raw)
        except ValueError:
            return
        if not isinstance(message, dict):
            return
        kind = message.get('t')
        if kind == 'size':
            self.set_client_size(client, message.get('cols'),
                                 message.get('rows'))
            return
        # A malformed message must cost the browser nothing more than itself.
        try:
            if kind == 'key':
                self.input.press(str(message.get('k') or ''))
            elif kind == 'text':
                self.input.type_text(str(message.get('v') or ''))
            elif kind == 'paste':
                self.input.paste(str(message.get('v') or ''))
            elif kind == 'mouse':
                self.input.mouse(str(message.get('a') or ''),
                                 int(message.get('x', 1)),
                                 int(message.get('y', 1)),
                                 int(message.get('b', 0) or 0))
            elif kind == 'modifier':
                self.input.modifier(str(message.get('m') or ''),
                                    str(message.get('a') or ''))
            elif kind == 'mod':
                self.input.modifier(str(message.get('k') or ''),
                                    str(message.get('a') or ''))
        except (TypeError, ValueError):
            return

    # -- frames ----------------------------------------------------
    def publish_frame(self, frame: str) -> None:
        """Turn one composited frame into the messages browsers receive."""
        canvas = self.canvas
        if canvas is None:
            return
        width, height = canvas.width, canvas.height
        caret = None
        match = _CARET_RE.search(frame)
        if match is not None:
            caret = (int(match.group(2)) - 1, int(match.group(1)) - 1)
        with self._lock:
            visible = self._caret_visible
            if frame.endswith(_HIDE_CARET):
                visible = False
            elif _SHOW_CARET in frame:
                visible = True
            resized = (width, height) != (self._width, self._height)
            if resized:
                self._width, self._height = width, height
                self._rows = [None] * height
            lines: list[list[Any]] = []
            for y in range(height):
                runs = [list(run) for run in canvas.row_runs(y)]
                if self._rows[y] != runs:
                    self._rows[y] = runs
                    lines.append([y, runs])
            caret_moved = (caret != self._caret
                           or visible != self._caret_visible)
            if caret is not None:
                self._caret = caret
            self._caret_visible = visible
            clients = list(self._clients)
            if not clients:
                return
            if resized:
                message = _dumps({'t': 'full', 'w': width, 'h': height,
                                  'lines': lines, 'caret': self._caret_json(),
                                  'show': visible, 'title': self._title})
            elif lines or caret_moved:
                message = _dumps({'t': 'rows', 'lines': lines,
                                  'caret': self._caret_json(),
                                  'show': visible})
            else:
                return
        for client in clients:
            client.push(message)

    def _caret_json(self) -> list[int] | None:
        caret = self._caret
        return list(caret) if caret is not None else None

    def snapshot(self) -> str:
        """A complete frame, for a browser that just connected."""
        with self._lock:
            lines = [[y, runs] for y, runs in enumerate(self._rows)
                     if runs is not None]
            return _dumps({'t': 'full', 'w': self._width or self._grid[0],
                           'h': self._height or self._grid[1],
                           'lines': lines, 'caret': self._caret_json(),
                           'show': self._caret_visible,
                           'title': self._title})

    # -- small screen services -------------------------------------
    def set_title(self, title: str) -> None:
        with self._lock:
            if title == self._title:
                return
            self._title = title
        self.broadcast(_dumps({'t': 'title', 'v': title}))

    def bell(self) -> None:
        self.broadcast(_dumps({'t': 'bell'}))

    def set_caret(self, x: int, y: int, visible: bool = True) -> None:
        with self._lock:
            self._caret = (int(x), int(y))
            self._caret_visible = bool(visible)

    def set_caret_visible(self, visible: bool) -> None:
        with self._lock:
            self._caret_visible = bool(visible)
