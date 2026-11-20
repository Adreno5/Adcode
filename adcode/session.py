"""SDK session persistence with AdCode's OpenCode ID on the SQLite route."""

from dataclasses import dataclass
from os import PathLike
from typing import Any

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
    'ArchivedSessionStore',
    'DbSessionStore', 'LAST_SESSION', 'RESERVED_SESSION_NAMES',
    'SESSION_VERSION', 'SessionPayload', 'SessionStore', 'TranscriptItem',
    'group_turns', 'new_session_name',
]


class ArchivedSessionStore(DbSessionStore):
    """The session store plus the set of sessions the user archived.

    Archiving hides a session and stops it taking new messages, but leaves the
    row, its messages and its transcript items exactly where the engine wrote
    them, so an archived session can still be read, restored (/respawn) or
    deleted (/perish).  The state lives in its own table: nothing about how a
    session is written or loaded changes, and a session the engine saves again
    stays archived.
    """

    _ARCHIVE_TABLE = 'adcode_archived_sessions'

    def __init__(self, store: Any = None, *,
                 db_path: str = 'sessions/agent.db') -> None:
        super().__init__(store, db_path=db_path)
        self.store.execute(
            f'CREATE TABLE IF NOT EXISTS {self._ARCHIVE_TABLE} ('
            ' name TEXT PRIMARY KEY, archived_at TEXT NOT NULL)')

    # ── the archive ───────────────────────────────────────
    def archived_names(self) -> set[str]:
        """Every archived session name, in one query."""
        rows = self.store.execute(f'SELECT name FROM {self._ARCHIVE_TABLE}')
        return {str(row['name']) for row in rows}

    def is_archived(self, name: str) -> bool:
        if not name:
            return False
        return bool(self.store.execute(
            f'SELECT name FROM {self._ARCHIVE_TABLE} WHERE name = ?', (name,)))

    def exists(self, name: str) -> bool:
        """Whether a session with this name is on disk."""
        return bool(name) and self._row_exists(name)

    def archive(self, name: str) -> bool:
        """Archive a saved session; False when there is nothing to archive."""
        if not self.exists(name):
            return False
        with self._lock:
            self.store.execute(
                f'INSERT INTO {self._ARCHIVE_TABLE} (name, archived_at)'
                ' VALUES (?, ?) ON CONFLICT(name) DO NOTHING',
                (name, self._now()))
        return True

    def unarchive(self, name: str) -> bool:
        """Restore an archived session; False when it was not archived."""
        if not self.is_archived(name):
            return False
        with self._lock:
            self.store.execute(
                f'DELETE FROM {self._ARCHIVE_TABLE} WHERE name = ?', (name,))
        return True

    # ── the list ──────────────────────────────────────────
    def delete(self, name: str) -> bool:
        """Delete a session and whatever the archive remembers about it."""
        removed = super().delete(name)
        with self._lock:
            self.store.execute(
                f'DELETE FROM {self._ARCHIVE_TABLE} WHERE name = ?', (name,))
        return removed

    def list(self, include_archived: bool = False) -> list[dict[str, Any]]:
        """Saved sessions, live ones only unless asked.

        Every row carries an "archived" flag.  Leaving archived sessions out by
        default is also what keeps them out of latest(): an archived session is
        never the one a new run resumes.
        """
        rows = super().list()
        archived = self.archived_names()
        out: list[dict[str, Any]] = []
        for row in rows:
            flagged = str(row['name']) in archived
            if flagged and not include_archived:
                continue
            out.append({**row, 'archived': flagged})
        return out
