"""Where AdCode keeps its state: ``~/.adcode`` by default.

Everything the app writes (session database, goals, image store, files-api
id cache) lives under this one directory, so running AdCode in a project
never scatters state into the project.  The project keeps only what it owns:
read-only instruction files (AGENTS.md, skills) and the work the agent does.

The directory can be redirected per environment with ``ADCODE_HOME``, and
tests redirect it with :func:`set_app_dir`.
"""

from __future__ import annotations

import os
from pathlib import Path

# Redirect target used by the test bench; None means "use the default".
APP_DIR: Path | None = None


def set_app_dir(path: str | os.PathLike[str] | None) -> None:
    """Pin the state directory (tests); ``None`` restores the default."""
    global APP_DIR
    APP_DIR = Path(path) if path is not None else None


def app_dir() -> Path:
    """The state directory: ``$ADCODE_HOME`` or ``~/.adcode``."""
    if APP_DIR is not None:
        return Path(APP_DIR)
    override = os.environ.get('ADCODE_HOME')
    if override:
        return Path(override).expanduser()
    return Path.home() / '.adcode'


def sessions_db() -> Path:
    """The sqlite session database."""
    return app_dir() / 'sessions' / 'adcode.db'


def images_dir() -> Path:
    """Where attached/stored images are kept."""
    return app_dir() / 'images'


def image_file_ids() -> Path:
    """The Files API id cache next to the image store."""
    return images_dir() / 'files.json'


def predictor_file() -> Path:
    return app_dir() / 'predictor.json'


def wire_cache_file() -> Path:
    """Wires learned to be the only ones an endpoint can stream."""
    return app_dir() / 'wires.json'


def goals_file() -> Path:
    """The durable goal store."""
    return app_dir() / 'goals.json'
