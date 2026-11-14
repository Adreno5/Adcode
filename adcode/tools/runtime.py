"""Per-thread execution context for tools."""

from __future__ import annotations

import contextvars
import threading
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ToolContext:
    cwd: str = '.'
    interrupt: threading.Event = field(default_factory=threading.Event)
    live_sink: Callable[[str], None] | None = None
    agent: str = 'root'


_CURRENT: contextvars.ContextVar[ToolContext] = contextvars.ContextVar(
    'adcode_tool_context', default=ToolContext())


def current_context() -> ToolContext:
    return _CURRENT.get()


def push_context(context: ToolContext):
    return _CURRENT.set(context)


def pop_context(token) -> None:
    _CURRENT.reset(token)
