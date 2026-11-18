"""Console's ``x-opencode-session`` header: one stable id per conversation.

OpenCode Console asks every harness to send a stable session id with each
request so it can route a conversation consistently and reuse its prompt
cache (see https://opencode.ai/docs/console/harnesses/#sessions and the Go
plan's requirement).  AdCode has no wire-level session of its own -- the
engine names a session only after its first save -- so the ids live here:

- one id per saved session name, remembered in
  ``~/.adcode/opencode_sessions.json`` so a resumed conversation keeps the id
  its prompt cache was written under;
- a conversation the engine has not named yet (its first turn) carries an
  unnamed id, bound to the name as soon as the engine assigns one;
- :meth:`OpencodeSessions.forget` drops the binding, for the sessions a delete
  or an archive takes away.

Whether the header is sent at all is the provider's ``opencode_session``
setting: ``auto`` (the default) sends it to Console endpoints only, so no
other gateway sees a header it does not know; ``on`` and ``off`` force it.
The request side is installed by :class:`adcode.llm.client.LLMClient` as an
HTTP request hook, so every wire and every request the provider SDKs make
carries it without threading anything through the request bodies.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from adcode import paths

#: The header Console reads.
SESSION_HEADER = 'x-opencode-session'

#: The values ``ProviderConfig.opencode_session`` accepts.
POLICIES = ('auto', 'on', 'off')

#: The host Console is served from; ``auto`` sends the header there and to
#: its subdomains.
CONSOLE_HOST = 'opencode.ai'


def new_session_id() -> str:
    """A fresh Console session id (``ses_`` and 32 hex characters)."""
    return f'ses_{uuid.uuid4().hex}'


def _host(base_url: str) -> str:
    try:
        return (urlsplit(str(base_url or '')).hostname or '').lower()
    except ValueError:
        return ''


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
    host = _host(getattr(provider, 'base_url', ''))
    return host == CONSOLE_HOST or host.endswith('.' + CONSOLE_HOST)


class OpencodeSessions:
    """Console session ids, one per AdCode session name.

    The store is thread-safe because the id is read on the engine thread while
    a request is built and rotated on the UI thread by session commands.

    Args:
        path: the json file the ids are remembered in; defaults to
            ``~/.adcode/opencode_sessions.json``.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self._path = (Path(path) if path is not None
                      else paths.opencode_sessions_file())
        self._lock = threading.RLock()
        self._ids: dict[str, str] | None = None
        # The live conversation's id: bound to a session name once the engine
        # has saved one, unbound ('' name) before that.
        self._id = new_session_id()
        self._bound = ''

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
            stored = self._load().get(name, '')
            self._id = stored or new_session_id()
            self._bound = name
            if not stored:
                self._remember(name, self._id)
            return self._id

    def forget(self) -> str:
        """Drop the binding: the next conversation starts under a new id."""
        with self._lock:
            self._id = new_session_id()
            self._bound = ''
            return self._id

    def of(self, name: str) -> str:
        """The remembered id for a session name ('' when there is none)."""
        if not name:
            return ''
        with self._lock:
            return self._load().get(name, '')

    # ── persistence ─────────────────────────────────────────
    def _load(self) -> dict[str, str]:
        with self._lock:
            if self._ids is None:
                ids: dict[str, str] = {}
                try:
                    raw = json.loads(self._path.read_text('utf-8'))
                except (OSError, ValueError):
                    raw = {}
                if isinstance(raw, dict):
                    ids = {str(name): str(value) for name, value in raw.items()
                           if isinstance(value, str) and value}
                self._ids = ids
            return self._ids

    def _remember(self, name: str, session_id: str) -> None:
        with self._lock:
            ids = self._load()
            ids[name] = session_id
            target = self._path
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix('.tmp')
                temporary.write_text(json.dumps(ids, indent=2, sort_keys=True),
                                     'utf-8')
                os.replace(temporary, target)
            except OSError:
                pass    # remembered for this process; the file is a shortcut
