"""The state database: everything AdCode persists lives in one file.

Earlier versions scattered state across the project being worked on --
``.harness_todo.json``, ``audit.jsonl``, ``wires.json``,
``predictor.json``, ``goals.json`` and a ``sessions/`` directory.  A
repository is for code; those files were never code, and ``git status``
had to learn to ignore them.

So the state moved:

* **documents** (the todo plan, the goals, the wire cache, the prediction
  state, the configuration) are rows in the ``documents`` table, read and
  written as whole documents;
* **the audit trail** is an append-only table;
* **the sessions** stay in the table the engine sdk owns.

The file itself lives in the state directory (never inside the project),
and the connection is shared with the session store so there is exactly
one writer per database.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

from infra.store import SqliteStore
from infra.store.documents import SqliteDocumentStore

from adcode import paths

#: The document keys the app uses.
KEY_CONFIG = 'config'
KEY_GOALS = 'goals'
KEY_WIRES = 'llm.wires'
KEY_PREDICTOR = 'policy.predictor'
KEY_TODO = 'todo.plan'
KEY_OPENCODE = 'llm.opencode_sessions'

#: The audit table.  Append-only: rows are added, never rewritten.
AUDIT_TABLE = 'adcode_audit'


class StateStore:
    """One database: documents, the audit trail, and the sessions."""

    def __init__(self, store: Optional[SqliteStore] = None, *,
                 path: Optional[str] = None) -> None:
        self._owns = store is None
        if store is None:
            target = Path(path or paths.db_path())
            # sqlite does not create the directory, only the file.
            target.parent.mkdir(parents=True, exist_ok=True)
            store = SqliteStore(str(target))
        self.store = store
        self.documents = SqliteDocumentStore(self.store)
        self._lock = threading.RLock()
        self._ensure_audit_table()

    # ── the tables ──────────────────────────────────────────
    def _ensure_audit_table(self) -> None:
        self.store.execute(
            f'CREATE TABLE IF NOT EXISTS {AUDIT_TABLE} ('
            ' id INTEGER PRIMARY KEY AUTOINCREMENT,'
            ' ts REAL NOT NULL,'
            ' kind TEXT NOT NULL,'
            ' session_id TEXT NOT NULL DEFAULT "",'
            ' tool TEXT NOT NULL DEFAULT "",'
            ' payload TEXT NOT NULL)')
        self.store.execute(
            f'CREATE INDEX IF NOT EXISTS {AUDIT_TABLE}_ts ON {AUDIT_TABLE} (ts)')

    # ── documents ───────────────────────────────────────────
    def load_document(self, key: str) -> Optional[dict[str, Any]]:
        return self.documents.load(key)

    def save_document(self, key: str, data: Any) -> None:
        self.documents.save(key, data)

    def delete_document(self, key: str) -> bool:
        return self.documents.delete(key)

    def document_keys(self, prefix: str = '') -> list[str]:
        return self.documents.keys(prefix)

    # ── the audit trail ─────────────────────────────────────
    def append_audit(self, event: dict[str, Any]) -> None:
        """Append one audit event.  Never raises: an audit write must not
        break the call it is auditing."""
        try:
            payload = json.dumps(event, ensure_ascii=False, default=str)
            with self._lock:
                self.store.execute(
                    f'INSERT INTO {AUDIT_TABLE} '
                    '(ts, kind, session_id, tool, payload) VALUES (?, ?, ?, ?, ?)',
                    (float(event.get('ts') or time.time()),
                     str(event.get('kind') or ''),
                     str(event.get('session_id') or ''),
                     str(event.get('tool') or ''),
                     payload))
        except Exception:      # noqa: BLE001 - the trail is best effort
            pass

    def audit_tail(self, count: int = 200) -> list[dict[str, Any]]:
        """The newest events, oldest first."""
        try:
            rows = self.store.execute(
                f'SELECT payload FROM {AUDIT_TABLE} ORDER BY id DESC LIMIT ?',
                (max(1, int(count)),))
        except Exception:      # noqa: BLE001
            return []
        events: list[dict[str, Any]] = []
        for row in reversed(rows):
            try:
                data = json.loads(row[0])
            except (TypeError, ValueError):
                continue
            if isinstance(data, dict):
                events.append(data)
        return events

    def audit_count(self) -> int:
        try:
            rows = self.store.execute(f'SELECT COUNT(*) FROM {AUDIT_TABLE}')
            return int(rows[0][0]) if rows else 0
        except Exception:      # noqa: BLE001
            return 0

    def close(self) -> None:
        if self._owns:
            try:
                self.store.close()
            except Exception:  # noqa: BLE001
                pass


_STORE: Optional[StateStore] = None
_STORE_LOCK = threading.Lock()


def state_store(path: Optional[str] = None) -> StateStore:
    """The shared state store (created on first use)."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = StateStore(path=str(path) if path else None)
        return _STORE


def reset_state_store() -> None:
    """Drop the shared store (tests, and when the state dir moves)."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is not None:
            _STORE.close()
        _STORE = None


def documents() -> Any:
    """The document store of the state database."""
    return state_store().documents


def legacy_candidates(name: str) -> list[Path]:
    """Where an older version may have left a file of this name.

    Two places: the state directory (where the app used to keep its
    state) and the project root (where it used to drop it, which is the
    mistake this whole move is about).
    """
    out: list[Path] = [paths.state_dir() / name]
    root = paths.project_root()
    if root:
        out.append(Path(root) / name)
    return out


def import_legacy_document(key: str, name: str) -> None:
    """Take over a document an older version left in a json file.

    The file is read once, written into the database, and then removed.
    A repository is for code, and these files were never code -- the same
    reason the state directory itself moved out of the project.
    """
    for target in legacy_candidates(name):
        try:
            if not target.exists():
                continue
            raw = json.loads(target.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError, TypeError):
            continue
        try:
            store = documents()
            if store.load(key) is None:
                store.save(key, raw if isinstance(raw, dict) else {'value': raw})
            target.unlink()
        except (OSError, ValueError):
            continue


def migrate_legacy_database() -> bool:
    """Move the database to its place next to the other state.

    The database used to live in a ``sessions`` subdirectory; it is one
    file now, holding the sessions, the documents and the audit trail.
    """
    legacy = paths.legacy_db_path()
    target = paths.db_path()
    try:
        if not legacy.exists() or target.exists():
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        legacy.replace(target)
        return True
    except OSError:
        return False


def merge_legacy_databases() -> int:
    """Take the sessions of the older databases into the new one.

    There may be two: the one in the state directory (which used to be
    ``sessions/adcode.db``) and the one the app kept **inside** the
    project -- the mistake this move is about.  The project's copy holds
    the most recent sessions (the app was writing them there), so it is
    merged last and wins.
    """
    root = paths.project_root()
    sources: list[Path] = [paths.legacy_db_path()]
    if root:
        sources.append(Path(root) / 'sessions' / 'adcode.db')
    taken = 0
    for legacy in sources:
        taken += _merge_database(legacy)
    return taken


def _merge_database(legacy: Path) -> int:
    """Copy the rows of one database into the current one."""
    target = paths.db_path()
    try:
        if not legacy.exists() or not target.exists():
            return 0
    except OSError:
        return 0
    taken = 0
    try:
        import sqlite3
        connection = sqlite3.connect(str(target), timeout=10)
        try:
            connection.execute('ATTACH DATABASE ? AS old', (str(legacy),))
            for table in ('agent_sessions', 'agent_messages', 'agent_items',
                          'adcode_archived_sessions'):
                try:
                    cursor = connection.execute(
                        f'INSERT OR REPLACE INTO {table} SELECT * FROM old.{table}')
                    taken += cursor.rowcount if cursor.rowcount > 0 else 0
                except Exception:      # noqa: BLE001 - a missing table is fine
                    continue
            connection.commit()
            connection.execute('DETACH DATABASE old')
        finally:
            connection.close()
    except Exception:      # noqa: BLE001 - the merge is best effort
        return 0
    try:
        legacy.unlink()
    except OSError:
        pass
    return taken


def migrate_legacy_images() -> int:
    """Take the images an older version kept in the project over.

    The markers in the transcripts point at absolute paths, so the files
    move **and** the markers are rewritten -- otherwise a conversation
    from yesterday would point at a file that is no longer there.
    """
    root = paths.project_root()
    if not root:
        return 0
    target = paths.images_dir()
    moved = 0
    for old_dir in (Path(root) / 'images', Path(root) / 'sessions' / 'images'):
        try:
            if not old_dir.is_dir():
                continue
            target.mkdir(parents=True, exist_ok=True)
            for item in list(old_dir.iterdir()):
                if not item.is_file():
                    continue
                destination = target / item.name
                try:
                    if destination.exists():
                        item.unlink()
                    else:
                        item.replace(destination)
                    moved += 1
                except OSError:
                    continue
            _rewrite_markers(str(old_dir), str(target))
            try:
                old_dir.rmdir()
            except OSError:
                pass
        except OSError:
            continue
    return moved


def _rewrite_markers(old_dir: str, new_dir: str) -> None:
    """Point the stored markers at the new place."""
    try:
        import sqlite3
        connection = sqlite3.connect(str(paths.db_path()), timeout=10)
        try:
            connection.execute(
                'UPDATE agent_messages SET content = REPLACE(content, ?, ?) '
                'WHERE content LIKE ?', (old_dir, new_dir, f'%{old_dir}%'))
            connection.commit()
        finally:
            connection.close()
    except Exception:      # noqa: BLE001 - the markers are best effort
        pass


def import_legacy_audit() -> int:
    """Take over an audit trail an older version kept in a jsonl file.

    Returns how many events were taken over.  The file is removed
    afterwards: the trail is in the database now.
    """
    taken = 0
    for target in legacy_candidates('audit.jsonl'):
        try:
            if not target.exists():
                continue
            lines = target.read_text(encoding='utf-8').splitlines()
        except OSError:
            continue
        store = state_store()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                store.append_audit(event)
                taken += 1
        try:
            target.unlink()
        except OSError:
            pass
    return taken
