"""Where AdCode keeps its state -- and where it refuses to keep it.

Everything the app writes (session database, goals, the wire cache, the
audit trail, the todo plan) lives under one directory, so running AdCode in
a project never scatters state into the project.  The project keeps only
what it owns: read-only instruction files (AGENTS.md, skills) and the work
the agent does.

Two rules, in order of importance:

1. **State never lands in a repository.**  The directory can be pointed
   anywhere with ``ADCODE_HOME``, but a directory that sits inside the
   project being worked on -- or inside any checkout at all -- is refused
   and the user-level location is used instead.  ``export
   ADCODE_HOME=$HOME/myproject`` is an easy accident to write, and it puts
   the database and the log into the repository the agent is editing.  A
   repository is for code; a session log is not.
2. **State that is structured goes to the database.**  The JSON documents
   (todo plan, goals, wire cache, prediction state, the configuration) are
   rows in one sqlite file; only the things that are genuinely files
   (images, the log) stay files.

The directory can be redirected per environment with ``ADCODE_HOME``, and
tests redirect it with :func:`set_app_dir`.
"""

from __future__ import annotations

import os
from pathlib import Path

# Redirect target used by the test bench; None means "use the default".
APP_DIR: Path | None = None

#: The project the app is working on.  Set once at startup; the state
#: directory is resolved against it (rule 1 above).
PROJECT_ROOT: str = ''


def set_app_dir(path: str | os.PathLike[str] | None) -> None:
    """Pin the state directory (tests); ``None`` restores the default."""
    global APP_DIR
    APP_DIR = Path(path) if path is not None else None


def set_project_root(path: str | os.PathLike[str] | None) -> None:
    """Pin the project the app is working on (the cwd, usually)."""
    global PROJECT_ROOT
    PROJECT_ROOT = str(Path(path).resolve()) if path else ''


def project_root() -> str:
    """The project the app is working on ('' when unknown)."""
    return PROJECT_ROOT


def default_dir() -> Path:
    """The user-level state directory: ``~/.adcode``."""
    return Path.home() / '.adcode'


def app_dir() -> Path:
    """The configured state directory: ``$ADCODE_HOME`` or ``~/.adcode``."""
    if APP_DIR is not None:
        return Path(APP_DIR)
    override = os.environ.get('ADCODE_HOME')
    if override:
        return Path(override).expanduser()
    return default_dir()


def _inside(candidate: Path, parent: str) -> bool:
    """Is ``candidate`` inside ``parent``?

    ``commonpath`` rather than a prefix compare: ``/home/x`` is inside
    ``/home``, but ``/homey`` is not.
    """
    if not parent:
        return False
    try:
        root = Path(parent).resolve()
        return os.path.commonpath([str(Path(candidate).resolve()),
                                   str(root)]) == str(root)
    except (ValueError, OSError):
        return False


def _in_repository(path: Path) -> bool:
    """Is this directory (or one of its parents) a repository?

    A repository is for code.  Session state is not code, and a state
    directory that sits inside one is exactly the mess this app stopped
    making -- ``ADCODE_HOME=$HOME/myproject`` is a very easy accident to
    write, and it puts the database and the log into the repository the
    agent is working on.

    The walk stops at the home directory: a home that is itself a
    repository would otherwise reject every location under it.
    """
    home = Path.home()
    current = path
    while True:
        try:
            if (current / '.git').exists() or (current / '.hg').exists():
                return True
        except OSError:
            return False
        if current == home or current == current.parent:
            return False
        current = current.parent


def state_dir() -> Path:
    """Where the state goes -- never inside a repository.

    The configured directory (``$ADCODE_HOME``) wins, unless it is
    unusable: sitting inside the project the app is working on, or
    sitting inside a repository at all.  Then the user-level location is
    used instead.  That is the whole point -- an agent working in a
    repository must not leave its session state in that repository.
    """
    base = app_dir()
    root = PROJECT_ROOT
    inside_project = bool(root) and _inside(base, root)
    if not inside_project and not _in_repository(base):
        return base
    for candidate in (default_dir(), Path.home() / '.local' / 'state' / 'adcode'):
        if _in_repository(candidate):
            continue
        if root and _inside(candidate, root):
            continue
        return candidate
    return base


def state_moved() -> bool:
    """Was the configured state directory rejected?"""
    return state_dir() != app_dir()


def db_path() -> Path:
    """The one database: sessions, documents and the audit trail."""
    return state_dir() / 'adcode.db'


def legacy_db_path() -> Path:
    """Where the database used to live (before it moved next to the state)."""
    return state_dir() / 'sessions' / 'adcode.db'


def sessions_db() -> Path:
    """The sqlite session database (the same file as :func:`db_path`)."""
    return db_path()


def images_dir() -> Path:
    """Where attached/stored images are kept."""
    return state_dir() / 'images'


def image_file_ids() -> Path:
    """The Files API id cache next to the image store."""
    return images_dir() / 'files.json'


def log_file() -> Path:
    """The log file.  A log is a file; it just is not in your repository."""
    return state_dir() / 'adcode.log'


# ── the legacy file locations ─────────────────────────────────────────
#
# These are the places earlier versions wrote their state to.  The state
# itself now lives in the database; these names stay because the files are
# still imported on first run, and because tools and tests ask for them.

def predictor_file() -> Path:
    """Where the prediction state used to be kept."""
    return state_dir() / 'predictor.json'


def wire_cache_file() -> Path:
    """Where the learned wires used to be kept."""
    return state_dir() / 'wires.json'


def goals_file() -> Path:
    """Where the goals used to be kept."""
    return state_dir() / 'goals.json'


def audit_file() -> Path:
    """Where the audit trail used to be kept (it is in the database now)."""
    return state_dir() / 'audit.jsonl'


def opencode_sessions_file() -> Path:
    """Where the console session ids used to be kept."""
    return state_dir() / 'opencode_sessions.json'
