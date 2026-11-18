"""Console's ``x-opencode-session`` header: one stable id per conversation.

OpenCode Console asks every harness to send a stable session id with each
request so it can route a conversation consistently and reuse its prompt
cache (see https://opencode.ai/docs/console/harnesses/#sessions and the Go
plan's requirement). AdCode tracks the ID independently of the engine's
display name so saving the same conversation under a new name keeps its ID.

The session database stores the name-to-ID bindings. An unnamed conversation
carries a fresh ID that can be bound once the engine assigns a name.

Whether the header is sent at all is the provider's ``opencode_session``
setting: ``auto`` (the default) sends it to Console endpoints only, so no
other gateway sees a header it does not know; ``on`` and ``off`` force it.
The request side is installed by :class:`adcode.llm.client.LLMClient` as an
HTTP request hook, so every wire and every request the provider SDKs make
carries it without threading anything through the request bodies.
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from infra.store.sqlite import SqliteStore

#: The header Console reads.
SESSION_HEADER = 'x-opencode-session'

#: The values ``ProviderConfig.opencode_session`` accepts.
POLICIES = ('auto', 'on', 'off')

#: The host Console is served from; ``auto`` sends the header there and to
#: its subdomains.
CONSOLE_HOST = 'opencode.ai'
_IMPORT_MARKER = 'legacy-json-imported'


def new_session_id() -> str:
    """A fresh Console session id (``ses_`` and 32 hex characters)."""
    return f'ses_{uuid.uuid4().hex}'


def is_console_endpoint(base_url: str) -> bool:
    try:
        url = urlsplit(base_url or '')
        host = url.hostname or ''
    except ValueError:
        return False
    return (url.scheme == 'https'
            and (host == CONSOLE_HOST or host.endswith('.' + CONSOLE_HOST))
            and url.path.removeprefix('/').partition('/')[0] in ('inference', 'zen'))


def sends_session_header(provider: Any) -> bool:
    """Whether requests to this provider carry the Console session header.

    ``auto`` -- the default -- is what keeps every other gateway untouched:
    the header only leaves the process for an OpenCode Console endpoint.
    """
    policy = str(getattr(provider, 'opencode_session', '') or 'auto')
    if policy == 'on':
        return True
    if policy == 'off':
        return False
    return is_console_endpoint(getattr(provider, 'base_url', ''))


class OpencodeSessions:
    """Console session ids, one per AdCode session name.

    The store is thread-safe because the id is read on the engine thread while
    a request is built and rotated by session commands.
    """

    def __init__(self, store: SqliteStore, *, legacy_path: Path | None = None) -> None:
        self._store = store
        self._lock = threading.RLock()
        self._id = new_session_id()
        self._bound = ''
        store.execute('CREATE TABLE IF NOT EXISTS adcode_opencode_sessions ('
                      'name TEXT PRIMARY KEY, session_id TEXT NOT NULL)')
        if legacy_path is not None:
            self._import_legacy(legacy_path)

    # ── the live id ─────────────────────────────────────────
    def current(self, name: str = '') -> str:
        """The id for the conversation saved under ``name``.

        An unnamed conversation keeps the id it already has; a name the store
        remembers adopts the remembered id, and a name it has not seen yet
        binds the live id and remembers it, so the next run reuses it.
        """
        name = str(name or '')
        with self._lock:
            if not name or name == self._bound:
                return self._id
            stored = self.of(name)
            if not stored:
                candidate = new_session_id() if self._bound else self._id
                self._store.execute(
                    'INSERT OR IGNORE INTO adcode_opencode_sessions VALUES (?, ?)',
                    (name, candidate))
                # Another instance may already have registered this name.
                stored = self.of(name) or candidate
            self._id = stored
            self._bound = name
            return self._id

    def bind(self, name: str) -> None:
        """Keep the current ID when this conversation is saved under another name."""
        with self._lock:
            self._store.execute(
                'INSERT INTO adcode_opencode_sessions VALUES (?, ?) '
                'ON CONFLICT(name) DO UPDATE SET session_id=excluded.session_id',
                (name, self._id))
            self._bound = name

    def forget(self) -> str:
        """Drop the binding: the next conversation starts under a new id."""
        with self._lock:
            self._id = new_session_id()
            self._bound = ''
            return self._id

    def remove(self, name: str) -> None:
        """Delete the binding when the saved conversation is permanently deleted."""
        with self._lock:
            if name:
                self._store.execute('DELETE FROM adcode_opencode_sessions WHERE name = ?',
                                    (name,))
            if name == self._bound:
                self.forget()

    def of(self, name: str) -> str:
        """The remembered id for a session name ('' when there is none)."""
        if not name:
            return ''
        rows = self._store.execute(
            'SELECT session_id FROM adcode_opencode_sessions WHERE name = ?', (name,))
        return rows[0]['session_id'] if rows else ''

    def _import_legacy(self, path: Path) -> None:
        # The empty name is reserved; an import must not revive deleted IDs.
        if self._store.execute('SELECT 1 FROM adcode_opencode_sessions WHERE name = ?', ('',)):
            return
        try:
            raw = json.loads(path.read_text('utf-8'))
        except (OSError, ValueError):
            return
        if isinstance(raw, dict):
            ids = {name: value for name, value in raw.items()
                   if name and isinstance(value, str) and value}
            ids[''] = _IMPORT_MARKER
            self._store.execute(
                'INSERT OR IGNORE INTO adcode_opencode_sessions '
                'SELECT key, value FROM json_each(?) '
                "WHERE NOT EXISTS (SELECT 1 FROM adcode_opencode_sessions WHERE name = '')",
                (json.dumps(ids),))
