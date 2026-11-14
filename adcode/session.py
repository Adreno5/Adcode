"""Session persistence, transcript items, rewind and fork support."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

SESSION_VERSION = 2
LAST_SESSION = '__last__'


@dataclass
class TranscriptItem:
    """One visual block in the conversation transcript.

    The transcript is the UI's model of the conversation.  ``turn`` links every
    item back to the user message that started it, which is what rewind, fork
    and edit operate on.  The LLM history itself is stored separately in the
    conversation context so tool-call payloads stay provider-shaped.
    """

    kind: str                       # user | assistant | thinking | tool | notice |
                                    # aside | compaction | agent
    turn: int = 0
    text: str = ''
    call_id: str = ''
    name: str = ''
    display: str = ''
    args: dict[str, Any] = field(default_factory=dict)
    status: str = ''
    output: str = ''
    diff: dict[str, Any] | None = None
    duration: float = 0.0
    level: str = 'info'
    agent: str = ''
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> 'TranscriptItem | None':
        if not isinstance(raw, dict) or not raw.get('kind'):
            return None
        known = {field_name for field_name in cls.__dataclass_fields__}
        payload = {key: value for key, value in raw.items() if key in known}
        try:
            return cls(**payload)
        except TypeError:
            return None


@dataclass
class SessionPayload:
    name: str = ''
    title: str = ''
    created: str = ''
    updated: str = ''
    provider: str = ''
    model: str = ''
    turn: int = 0
    prefix_generation: int = 1
    messages: list[dict[str, Any]] = field(default_factory=list)
    compaction_archive: list[dict[str, Any]] = field(default_factory=list)
    items: list[TranscriptItem] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            'version': SESSION_VERSION,
            'name': self.name,
            'title': self.title,
            'created': self.created,
            'updated': self.updated,
            'provider': self.provider,
            'model': self.model,
            'turn': self.turn,
            'prefix_generation': self.prefix_generation,
            'messages': self.messages,
            'compaction_archive': self.compaction_archive,
            'items': [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> 'SessionPayload':
        items = [item for raw_item in raw.get('items') or []
                 if (item := TranscriptItem.from_dict(raw_item)) is not None]
        messages = list(raw.get('messages') or raw.get('history') or [])
        if not items and messages:
            items = _items_from_legacy_messages(messages)
        return cls(
            name=str(raw.get('name') or raw.get('session') or ''),
            title=str(raw.get('title') or ''),
            created=str(raw.get('created') or ''),
            updated=str(raw.get('updated') or ''),
            provider=str(raw.get('provider') or ''),
            model=str(raw.get('model') or ''),
            turn=int(raw.get('turn') or 0),
            prefix_generation=max(1, int(raw.get('prefix_generation') or 1)),
            messages=messages,
            compaction_archive=list(raw.get('compaction_archive') or []),
            items=items,
        )


def _items_from_legacy_messages(messages: list[dict[str, Any]]) -> list[TranscriptItem]:
    """Build a transcript for a session written by AdCode 1.x."""
    items: list[TranscriptItem] = []
    turn = 0
    for message in messages:
        role = message.get('role')
        content = message.get('content')
        text = content if isinstance(content, str) else ''
        if role == 'user':
            if text.startswith(('<conversation_summary>', '<subagent_updates>',
                                '<agent_mailbox>')):
                continue
            turn += 1
            items.append(TranscriptItem(kind='user', turn=turn, text=text))
        elif role == 'assistant':
            reasoning = message.get('reasoning_content') or ''
            if reasoning:
                items.append(TranscriptItem(kind='thinking', turn=turn,
                                            text=str(reasoning)))
            for call in message.get('tool_calls') or []:
                function = call.get('function', {}) if isinstance(call, dict) else {}
                items.append(TranscriptItem(
                    kind='tool', turn=turn, name=str(function.get('name', '')),
                    display=str(function.get('name', '')), status='done'))
            if text:
                items.append(TranscriptItem(kind='assistant', turn=turn, text=text))
    return items


class SessionStore:
    """Atomic JSON persistence under ``sessions/``."""

    def __init__(self, directory: str | os.PathLike[str] = 'sessions') -> None:
        self.directory = Path(directory)
        self._lock = threading.Lock()

    def path_for(self, name: str) -> Path:
        safe = name.strip().replace('/', '_').replace('\\', '_')
        return self.directory / f'{safe}.json'

    def save(self, payload: SessionPayload, as_name: str | None = None) -> None:
        payload.updated = time.strftime('%Y-%m-%d %H:%M')
        if not payload.created:
            payload.created = payload.updated
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.path_for(as_name or payload.name)
            fd, temp_name = tempfile.mkstemp(
                prefix=f'.{path.name}.', suffix='.tmp', dir=str(self.directory))
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                    json.dump(payload.to_json(), stream, ensure_ascii=False, indent=2)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp_name, path)
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)

    def load(self, name: str) -> SessionPayload | None:
        path = self.path_for(name)
        try:
            raw = json.loads(path.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict):
            return None
        payload = SessionPayload.from_json(raw)
        if not payload.name:
            payload.name = name
        return payload

    def delete(self, name: str) -> bool:
        try:
            self.path_for(name).unlink()
            return True
        except OSError:
            return False

    def list(self) -> list[dict[str, Any]]:
        if not self.directory.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for path in self.directory.glob('*.json'):
            name = path.stem
            if name == LAST_SESSION:
                continue
            try:
                raw = json.loads(path.read_text(encoding='utf-8-sig'))
            except (OSError, ValueError):
                continue
            if not isinstance(raw, dict):
                continue
            rows.append({
                'name': name,
                'title': str(raw.get('title') or name),
                'created': str(raw.get('created') or ''),
                'updated': str(raw.get('updated') or ''),
                'messages': len(raw.get('messages') or []),
            })
        rows.sort(key=lambda row: (row['updated'], row['created']), reverse=True)
        return rows

    def latest(self) -> SessionPayload | None:
        payload = self.load(LAST_SESSION)
        if payload is not None:
            # An explicitly cleared checkpoint must not fall through to an old
            # named session and resurrect the cleared conversation.
            return payload if payload.messages else None
        rows = self.list()
        if not rows:
            return None
        return self.load(rows[0]['name'])

    # ── rewind / fork helpers ───────────────────────────────
    @staticmethod
    def turn_starts(messages: Iterable[dict[str, Any]]) -> list[int]:
        """Indices of user messages that begin a turn (ignoring injected blocks)."""
        starts: list[int] = []
        for index, message in enumerate(messages):
            if message.get('role') != 'user':
                continue
            content = message.get('content')
            if not isinstance(content, str):
                content = str(content)
            if content.startswith(('<conversation_summary>', '<subagent_updates>',
                                   '<agent_mailbox>', '<tool_')):
                continue
            starts.append(index)
        return starts

    @classmethod
    def rewind(cls, payload: SessionPayload, turn: int,
               edit_text: str | None = None) -> SessionPayload:
        """Drop every item and message from ``turn`` onwards.

        ``turn`` is the 1-based transcript turn id.  When ``edit_text`` is given
        the caller re-submits the edited text as a fresh turn.
        """
        starts = cls.turn_starts(payload.messages)
        if 1 <= turn <= len(starts):
            cut = starts[turn - 1]
        else:
            cut = len(payload.messages)
        payload.messages = payload.messages[:cut]
        payload.items = [item for item in payload.items if item.turn < turn]
        payload.turn = max(0, turn - 1)
        return payload


def group_turns(items: Iterable[TranscriptItem]) -> list[tuple[int, TranscriptItem]]:
    """Return (turn, item) rows in order; notices with turn 0 stay at the top."""
    return [(item.turn, item) for item in items]


def new_session_name() -> str:
    return time.strftime('%Y%m%d_%H%M%S')
