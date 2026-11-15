"""Scripted model turns used by the fake local LLM gateway.

A :class:`Turn` is one *model response*: optional reasoning, optional answer
text, optional tool calls and usage.  The gateway serializes it into whichever
wire protocol the client asked for, so the same script can drive the OpenAI
Chat Completions, OpenAI Responses and Anthropic Messages paths.

``hold`` is a deterministic pause: the gateway streams the reasoning, then
blocks on the event until the test releases it.  That makes "UI while the model
is still thinking" assertions reliable instead of timing-dependent.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class ToolCall:
    """A model-requested tool call with decoded arguments."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    id: str = ''

    def arguments_json(self) -> str:
        return json.dumps(self.args, ensure_ascii=False)


@dataclass
class TurnUsage:
    input_tokens: int = 128
    output_tokens: int = 32
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Turn:
    """One scripted model response."""

    thinking: str = ''
    text: str = ''
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish: str = 'stop'
    usage: TurnUsage = field(default_factory=TurnUsage)
    # Streaming realism / test gates.
    chunk_size: int = 14
    chunk_delay: float = 0.0
    hold: threading.Event | None = None
    hold_timeout: float = 30.0
    # Drop the HTTP connection after this many SSE events (0 = never).
    disconnect_after: int | None = None


@dataclass
class ModelRequest:
    """A request the fake gateway received, already JSON-decoded."""

    index: int
    wire: str
    path: str
    body: dict[str, Any]
    headers: dict[str, str]

    # ── common views ────────────────────────────────────────
    @property
    def model(self) -> str:
        return str(self.body.get('model') or '')

    @property
    def messages(self) -> list[dict[str, Any]]:
        if self.wire == 'openai responses':
            raw = self.body.get('input')
            if isinstance(raw, list):
                return [item for item in raw if isinstance(item, dict)]
            return []
        raw = self.body.get('messages')
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        return []

    @property
    def system_text(self) -> str:
        if self.wire == 'anthropic messages':
            raw = self.body.get('system')
        elif self.wire == 'openai responses':
            raw = self.body.get('instructions')
        else:
            raw = '\n'.join(
                str(message.get('content') or '')
                for message in self.messages if message.get('role') == 'system')
        if isinstance(raw, str):
            return raw
        if isinstance(raw, list):
            return '\n'.join(
                str(block.get('text') or '') for block in raw
                if isinstance(block, dict))
        return ''

    @property
    def tools(self) -> list[dict[str, Any]]:
        raw = self.body.get('tools')
        return [item for item in raw if isinstance(item, dict)] \
            if isinstance(raw, list) else []

    @property
    def tool_names(self) -> list[str]:
        names: list[str] = []
        for tool in self.tools:
            function = tool.get('function') if isinstance(tool.get('function'), dict) else tool
            name = function.get('name')
            if name:
                names.append(str(name))
        return names

    @property
    def reasoning_effort(self) -> str:
        if self.wire == 'openai responses':
            reasoning = self.body.get('reasoning')
            if isinstance(reasoning, dict):
                return str(reasoning.get('effort') or '')
            return ''
        if self.wire == 'anthropic messages':
            thinking = self.body.get('thinking')
            if isinstance(thinking, dict):
                return str(thinking.get('type') or '')
            return ''
        return str(self.body.get('reasoning_effort') or '')

    @property
    def all_text(self) -> str:
        """The raw request as text; handy for loose containment assertions."""
        return json.dumps(self.body, ensure_ascii=False)

    def last_user_text(self) -> str:
        for message in reversed(self.messages):
            role = message.get('role')
            if role not in ('user', None):
                continue
            content = message.get('content')
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    str(block.get('text') or block.get('content') or '')
                    for block in content if isinstance(block, dict)
                ]
                text = '\n'.join(part for part in parts if part)
                if text:
                    return text
        return ''


class Responder(Protocol):
    def respond(self, request: ModelRequest) -> Turn:
        ...


class EchoResponder:
    """A deterministic responder that reflects the last user message."""

    def __init__(self, thinking: str = 'The fake gateway received the request.',
                 prefix: str = 'Echo') -> None:
        self.thinking = thinking
        self.prefix = prefix

    def respond(self, request: ModelRequest) -> Turn:
        text = request.last_user_text().strip() or 'ping'
        first = text.splitlines()[0][:120]
        return Turn(
            thinking=self.thinking,
            text=(f'{self.prefix}: {first}\n\n'
                  'Streamed by the local fake gateway.'),
            usage=TurnUsage(input_tokens=max(16, len(request.all_text) // 4),
                            output_tokens=24),
        )


class ScriptedResponder:
    """Returns the scripted turns in order, one HTTP response at a time."""

    def __init__(self, turns: list[Turn] | None = None,
                 fallback: Turn | None = None) -> None:
        self.turns = list(turns or [])
        self.served = 0
        self._lock = threading.Lock()
        self.fallback = fallback or Turn(text='(scripted response exhausted)')

    def push(self, turn: Turn) -> None:
        with self._lock:
            self.turns.append(turn)

    def respond(self, request: ModelRequest) -> Turn:
        with self._lock:
            if self.served < len(self.turns):
                turn = self.turns[self.served]
                self.served += 1
                return turn
            return self.fallback


class CallbackResponder:
    """Builds a turn from a Python callable (useful in ad-hoc probes)."""

    def __init__(self, callback: Callable[[ModelRequest], Turn]) -> None:
        self.callback = callback

    def respond(self, request: ModelRequest) -> Turn:
        return self.callback(request)
