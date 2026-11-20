"""What the agent may do to the project folder.

Three levels, one per project:

``readonly``
    the agent may read the project; it may not change anything -- not a
    file inside the project, not a temporary file outside it.
``workspace``
    the agent may change files inside the project; nothing outside it.
``full``
    no boundary: the agent runs with the user's own rights.

The level belongs to the **folder**, not to the app: the configuration
remembers it per project root (``security.access_by_project``), so a
project the user trusts can be ``full`` while the default stays
``readonly``.  The default itself (``security.access``) applies to every
project without an entry of its own.

The same three names are used in three places, on purpose:

* the **policy** refuses the writing tools outright in ``readonly``;
* the **sandbox** fences the shell to the project in ``workspace``;
* the **file tools** refuse a path outside the project themselves.
"""

from __future__ import annotations

import os
from typing import Any

from harness.tools.sandbox.paths import is_within, real_path

#: The three levels, from the strictest to the loosest.
ACCESS_LEVELS = ('readonly', 'workspace', 'full')

#: The tools that change something on disk.
WRITE_TOOLS = ('write', 'edit')

#: What each level declares to the sandbox.
_SANDBOX_FOR = {
    'readonly': 'readonly',
    'workspace': 'workspace',
    'full': 'none',
}


def project_root(cwd: str = '') -> str:
    """The project folder, resolved."""
    base = cwd or os.getcwd()
    try:
        return os.path.realpath(base)
    except OSError:
        return os.path.abspath(base)


def effective_access(config: Any, cwd: str = '') -> str:
    """The level that applies to this project right now."""
    settings = getattr(config, 'security', None)
    if settings is None:
        return 'full'
    root = project_root(cwd)
    per_project = getattr(settings, 'access_by_project', None) or {}
    level = per_project.get(root)
    if level in ACCESS_LEVELS:
        return level
    level = getattr(settings, 'access', '')
    return level if level in ACCESS_LEVELS else 'full'


def declared_sandbox(access: str) -> str:
    """The sandbox level a tool declares under this access level."""
    return _SANDBOX_FOR.get(access, 'none')


def access_for_project(config: Any, cwd: str = '') -> str:
    """The level stored for this project (the default when there is none)."""
    settings = getattr(config, 'security', None)
    if settings is None:
        return 'full'
    root = project_root(cwd)
    per_project = getattr(settings, 'access_by_project', None) or {}
    level = per_project.get(root)
    if level in ACCESS_LEVELS:
        return level
    default = getattr(settings, 'access', '')
    return default if default in ACCESS_LEVELS else 'full'


def is_per_project(config: Any, cwd: str = '') -> bool:
    """Does this project have a level of its own?"""
    settings = getattr(config, 'security', None)
    if settings is None:
        return False
    root = project_root(cwd)
    per_project = getattr(settings, 'access_by_project', None) or {}
    return per_project.get(root) in ACCESS_LEVELS


def write_allowed(path: str, cwd: str, access: str) -> tuple[bool, str]:
    """May this path be written under this access level?

    Returns ``(allowed, reason)``; the reason is the sentence the model
    gets when the answer is no.
    """
    if access == 'full':
        return True, ''
    root = project_root(cwd)
    target = real_path(path, cwd)
    if access == 'readonly':
        return False, ('the project is read-only: this session may read '
                       'files but not change them')
    if not is_within(root, target):
        return False, (f'outside the project: {access} access only allows '
                       f'changes inside {root}')
    return True, ''


def describe(access: str) -> str:
    """One line describing a level, for the interface."""
    return {
        'readonly': 'Read-only: the project can be read, nothing can be changed',
        'workspace': 'Workspace: files inside the project may be changed',
        'full': 'Full access: no boundary, the agent runs as you do',
    }.get(access, access)
