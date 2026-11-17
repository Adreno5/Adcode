"""A terminal screen that publishes frames to browsers.

The app knows nothing about the web: it composites a cell grid into its canvas
and hands the serialized frame to the screen.  This screen ignores those bytes
and lets the session publish the canvas itself, which is why a browser gets
exact colours and attributes instead of having to parse escape sequences.
"""

from __future__ import annotations

from typing import Any

from adcode.tui.screen import TerminalSize


class WebScreen:
    """The TerminalScreen interface, backed by a web session."""

    def __init__(self, session: Any) -> None:
        self._session = session

    # -- TerminalScreen interface ---------------------------------
    def size(self) -> TerminalSize:
        columns, rows = self._session.grid()
        return TerminalSize(columns, rows)

    def enter(self) -> None:
        """No terminal modes to set: the browser owns its own screen."""

    def leave(self) -> None:
        """Nothing to restore."""

    def write(self, text: str) -> None:
        """The app paints through flush_frame; stray writes are dropped."""

    def flush_frame(self, frame: str) -> None:
        self._session.publish_frame(frame)

    def set_title(self, title: str) -> None:
        self._session.set_title(title)

    def bell(self) -> None:
        self._session.bell()

    def set_cursor(self, x: int, y: int, visible: bool = True) -> None:
        self._session.set_caret(x, y, visible)

    def hide_cursor(self) -> None:
        self._session.set_caret_visible(False)
