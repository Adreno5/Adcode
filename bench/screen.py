"""A terminal screen stand-in that records every frame the app paints.

``FakeScreen`` implements the same small interface as
:class:`adcode.tui.screen.TerminalScreen`, but instead of writing to a console
it snapshots the app's canvas *and* replays the serialized ANSI frame through a
:class:`~harness.vt.VirtualScreen`.  Tests get an exact cell grid, the text a
terminal would display, the cursor position and the raw escape stream.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from adcode.tui.canvas import Canvas
from adcode.tui.screen import TerminalSize
from bench.vt import VTCell, VirtualScreen

_CURSOR_RE = re.compile(r'\x1b\[(\d+);(\d+)H')


@dataclass(frozen=True)
class CellView:
    ch: str
    fg: tuple[int, int, int]
    bg: tuple[int, int, int]
    attrs: int = 0
    cont: bool = False


@dataclass(frozen=True)
class FrameSnapshot:
    """One rendered frame, captured at the moment it reached the screen."""

    index: int
    width: int
    height: int
    rows: tuple[tuple[CellView, ...], ...]
    raw: str
    cursor: tuple[int, int] | None
    cursor_visible: bool
    at: float
    vt_text: str = ''
    vt_mismatch: bool = False

    # ── views ───────────────────────────────────────────────
    def line(self, y: int, *, rstrip: bool = True) -> str:
        if not (0 <= y < self.height):
            return ''
        text = ''.join(
            cell.ch for cell in self.rows[y] if not cell.cont and cell.ch)
        return text.rstrip() if rstrip else text

    @property
    def lines(self) -> list[str]:
        return [self.line(y) for y in range(self.height)]

    def text(self, *, rstrip: bool = True) -> str:
        return '\n'.join(self.line(y, rstrip=rstrip)
                         for y in range(self.height))

    def nonempty_lines(self) -> list[str]:
        return [line for line in self.lines if line.strip()]

    def contains(self, needle: str) -> bool:
        return needle in self.text()

    def row_index(self, needle: str) -> int:
        for y, line in enumerate(self.lines):
            if needle in line:
                return y
        return -1

    def cells(self, y: int) -> tuple[CellView, ...]:
        return self.rows[y] if 0 <= y < self.height else ()

    # ── screenshot payload (see bench.screenshots) ────────
    def to_payload(self, label: str = '') -> dict[str, Any]:
        return {
            'label': label,
            'width': self.width,
            'height': self.height,
            'cursor': list(self.cursor) if self.cursor else None,
            'cursor_visible': self.cursor_visible,
            'text': self.text(),
            'rows': [
                [[cell.ch, list(cell.fg), list(cell.bg), cell.attrs, cell.cont]
                 for cell in row]
                for row in self.rows
            ],
        }


def _cells_from_canvas(canvas: Canvas) -> tuple[tuple[CellView, ...], ...]:
    rows: list[tuple[CellView, ...]] = []
    for y in range(canvas.height):
        row: list[CellView] = []
        for x in range(canvas.width):
            cell = canvas.cell(x, y)
            if cell is None:
                row.append(CellView(' ', (0, 0, 0), (0, 0, 0)))
                continue
            row.append(CellView(cell.ch, tuple(cell.fg), tuple(cell.bg),
                                cell.attrs, cell.cont))
        rows.append(tuple(row))
    return tuple(rows)


def _cells_from_vt(rows: tuple[tuple[VTCell, ...], ...]
                   ) -> tuple[tuple[CellView, ...], ...]:
    return tuple(
        tuple(CellView(cell.ch, tuple(cell.fg), tuple(cell.bg),
                       cell.attrs, cell.cont) for cell in row)
        for row in rows)


def _text_from_cells(rows: tuple[tuple[CellView, ...], ...]) -> str:
    lines: list[str] = []
    for row in rows:
        lines.append(''.join(
            cell.ch for cell in row if not cell.cont and cell.ch).rstrip())
    return '\n'.join(lines)


class FakeScreen:
    """Records frames, titles and control writes instead of touching a TTY."""

    def __init__(self, columns: int = 100, rows: int = 30, *,
                 capture_vt: bool = True, max_frames: int = 240) -> None:
        self._size = TerminalSize(max(1, columns), max(1, rows))
        self.capture_vt = capture_vt
        self.max_frames = max_frames
        self.frames: list[FrameSnapshot] = []
        self.latest: FrameSnapshot | None = None
        self.titles: list[str] = []
        self.writes: list[str] = []
        self.bells = 0
        self.entered = False
        self.left = False
        self.cursor: tuple[int, int] | None = None
        self.cursor_visible = True
        self._canvas: Canvas | None = None
        self._vt: VirtualScreen | None = None
        self._index = 0
        self._lock = threading.RLock()
        self.on_frame: Callable[[FrameSnapshot], None] | None = None

    # ── TerminalScreen interface ────────────────────────────
    def size(self) -> TerminalSize:
        return self._size

    def enter(self) -> None:
        self.entered = True

    def leave(self) -> None:
        self.left = True

    def write(self, text: str) -> None:
        with self._lock:
            self.writes.append(text)
            if text.startswith('\x1b]0;'):
                self.titles.append(text[4:].split('\x07', 1)[0])

    def flush_frame(self, frame: str) -> None:
        self._capture(frame)

    def set_title(self, title: str) -> None:
        with self._lock:
            self.titles.append(title)

    def bell(self) -> None:
        self.bells += 1

    def set_cursor(self, x: int, y: int, visible: bool = True) -> None:
        self.cursor = (x, y)
        self.cursor_visible = visible

    def hide_cursor(self) -> None:
        self.cursor_visible = False

    # ── harness extras ──────────────────────────────────────
    def bind(self, canvas: Canvas) -> None:
        """Attach the app canvas so frames are snapshotted cell-for-cell."""
        with self._lock:
            self._canvas = canvas
            self._vt = (VirtualScreen(canvas.width, canvas.height)
                        if self.capture_vt else None)

    @property
    def columns(self) -> int:
        return self._size.columns

    @property
    def rows_count(self) -> int:
        return self._size.rows

    def resize(self, columns: int, rows: int) -> None:
        with self._lock:
            self._size = TerminalSize(max(1, columns), max(1, rows))
            if self._vt is not None:
                self._vt.resize(self._size.columns, self._size.rows)

    def wait_for_frame(self, predicate: Callable[[FrameSnapshot], bool],
                       timeout: float = 10.0) -> FrameSnapshot:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            latest = self.latest
            if latest is not None and predicate(latest):
                return latest
            time.sleep(0.01)
        raise TimeoutError('no frame matched the predicate')

    # ── capture ────────────────────────────────────────────
    def _capture(self, raw: str) -> None:
        with self._lock:
            canvas = self._canvas
            if canvas is None:
                return
            canvas_rows = _cells_from_canvas(canvas)
            vt_text = ''
            vt_mismatch = False
            cursor = None
            if self._vt is not None:
                self._vt.feed(raw)
                vt_rows = _cells_from_vt(self._vt.rows())
                vt_text = _text_from_cells(vt_rows)
                # Compare cells, not just text: colours, attributes and wide
                # glyph continuation cells all matter for a rendered frame.
                vt_mismatch = vt_rows != canvas_rows
                cursor = self._vt.cursor()
            else:
                matches = _CURSOR_RE.findall(raw)
                if matches:
                    row, column = (int(value) for value in matches[-1])
                    cursor = (column - 1, row - 1)
            cursor_visible = not raw.rstrip().endswith('\x1b[?25l')
            self._index += 1
            snapshot = FrameSnapshot(
                index=self._index,
                width=canvas.width,
                height=canvas.height,
                rows=canvas_rows,
                raw=raw,
                cursor=cursor,
                cursor_visible=cursor_visible,
                at=time.time(),
                vt_text=vt_text,
                vt_mismatch=vt_mismatch,
            )
            self.latest = snapshot
            self.frames.append(snapshot)
            if len(self.frames) > self.max_frames:
                del self.frames[:len(self.frames) - self.max_frames]
            callback = self.on_frame
        if callback is not None:
            callback(snapshot)

    def frame_text(self, index: int = -1) -> str:
        with self._lock:
            if not self.frames:
                return ''
            return self.frames[index].text()

    def wait_for_text(self, needle: str, timeout: float = 10.0) -> FrameSnapshot:
        return self.wait_for_frame(lambda frame: frame.contains(needle), timeout)
