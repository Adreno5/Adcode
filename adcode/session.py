"""SDK session persistence with AdCode's OpenCode ID on the SQLite route."""

from dataclasses import dataclass
from os import PathLike
from typing import Any

from harness.transcript import (
    DbSessionStore as _DbSessionStore,
    LAST_SESSION,
    RESERVED_SESSION_NAMES,
    SESSION_VERSION,
    SessionPayload,
    SessionStore,
    TranscriptItem,
    group_turns,
    new_session_name,
)


@dataclass
class OpenCodeSessionPayload(SessionPayload):
    opencode_session_id: str = ''


class DbSessionStore(_DbSessionStore):
    def __init__(self, store: Any = None, *,
                 db_path: str | PathLike[str] = 'sessions/agent.db') -> None:
        super().__init__(store=store, db_path=db_path)
        self.store.execute(
            'CREATE TABLE IF NOT EXISTS adcode_opencode_sessions ('
            'name TEXT PRIMARY KEY, session_id TEXT NOT NULL)')

    def save(self, payload: SessionPayload, as_name: str | None = None) -> None:
        super().save(payload, as_name=as_name)
        name = (payload.name if as_name is None else as_name).strip()
        if isinstance(payload, OpenCodeSessionPayload) and payload.opencode_session_id:
            self.store.execute(
                'INSERT INTO adcode_opencode_sessions (name, session_id) VALUES (?, ?)'
                ' ON CONFLICT(name) DO UPDATE SET session_id=excluded.session_id',
                (name, payload.opencode_session_id))
        else:
            self._delete_opencode_id(name)

    def load(self, name: str) -> SessionPayload | None:
        payload = super().load(name)
        if payload is None:
            return None
        rows = self.store.execute(
            'SELECT session_id FROM adcode_opencode_sessions WHERE name = ?', (name,))
        if not rows:
            return payload
        return OpenCodeSessionPayload(**vars(payload), opencode_session_id=rows[0]['session_id'])

    def delete(self, name: str) -> bool:
        deleted = super().delete(name)
        if deleted:
            self._delete_opencode_id(name)
        return deleted

    def _delete_opencode_id(self, name: str) -> None:
        self.store.execute('DELETE FROM adcode_opencode_sessions WHERE name = ?', (name,))


__all__ = [
    'DbSessionStore', 'LAST_SESSION', 'RESERVED_SESSION_NAMES',
    'SESSION_VERSION', 'SessionPayload', 'SessionStore', 'TranscriptItem',
    'group_turns', 'new_session_name',
]
