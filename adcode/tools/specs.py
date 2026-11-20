"""What each tool is allowed to do.

The sdk's guard reads a ``ToolSpec`` off every tool: the side-effect grade,
the isolation it needs, whether it reaches the network.  A tool that declares
nothing is treated as the strictest case there is -- effectful, isolated,
approval required -- so the declarations below are what make the guard usable
rather than a wall of prompts.

The grades follow the sdk's vocabulary:

* ``pure``       -- reads only; replaying it changes nothing.
* ``idempotent`` -- writes, but writing twice is the same as writing once.
* ``effectful``  -- has an effect outside this workspace (a shell, the
  network, another agent).  These are the ones that may need a human.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from harness.tools.spec import ToolSpec

from adcode.access import declared_sandbox


def _spec(name: str, summary: str, *, effect: str = 'pure',
          sandbox: str = 'none', network: str = 'none',
          destructive: bool = False, group: str = 'builtin') -> ToolSpec:
    return ToolSpec(
        name=name,
        summary=summary,
        effect=effect,          # type: ignore[arg-type]
        sandbox=sandbox,        # type: ignore[arg-type]
        network=network,        # type: ignore[arg-type]
        destructive=destructive,
        owner='adcode',
        group=group,
    )


#: The built-in suite.  Reads are pure, writes are idempotent, the shell and
#: the network are effectful -- that is exactly the set a strict mode asks
#: about.
BUILTIN_SPECS: dict[str, ToolSpec] = {
    'read': _spec('read', 'Read a file from the filesystem', sandbox='readonly'),
    'glob': _spec('glob', 'List files by pattern', sandbox='readonly'),
    'grep': _spec('grep', 'Search file contents', sandbox='readonly'),
    'ls': _spec('ls', 'List directory entries', sandbox='readonly'),
    'diff': _spec('diff', 'Show the working-tree diff', sandbox='readonly'),
    'view_image': _spec('view_image', 'Look at an image', sandbox='readonly'),
    'write': _spec('write', 'Write a file', effect='idempotent',
                   sandbox='workspace'),
    'edit': _spec('edit', 'Edit a file', effect='idempotent',
                  sandbox='workspace'),
    'todo': _spec('todo', 'Track multi-step work', effect='idempotent',
                  sandbox='workspace'),
    'bash': _spec('bash', 'Run a shell command', effect='effectful',
                  sandbox='workspace', network='full', group='shell'),
    'fetch': _spec('fetch', 'Make an HTTP request', effect='effectful',
                   sandbox='workspace', network='allowlist', group='network'),
}

#: The collaboration suite: spawning a worker and messaging it are effects on
#: the session itself, so a strict mode asks about them.
COLLAB_SPECS: dict[str, ToolSpec] = {
    name: _spec(name, f'Collaboration: {name}', effect='effectful',
                sandbox='workspace', group='collab')
    for name in ('spawn_agent', 'send_message', 'followup_task',
                 'list_agents', 'wait_agent', 'interrupt_agent')
}

#: Tools that arrive at runtime (skills, MCP servers) are unknown to us, so
#: they get the middle grade: they may change the workspace and they may
#: reach the network, but they are not silently effectful.
def dynamic_spec(name: str, tool: Any = None) -> ToolSpec:
    """A spec for a tool that was registered at runtime."""
    read_only = bool(getattr(tool, 'read_only', False))
    if read_only:
        return _spec(name, str(getattr(tool, 'description', ''))[:120],
                     sandbox='readonly', group='dynamic')
    return _spec(name, str(getattr(tool, 'description', ''))[:120],
                 effect='effectful', sandbox='workspace', network='full',
                 group='dynamic')


def attach_spec(tool: Any) -> Any:
    """Give a tool its declared spec; return the tool.

    A tool that already carries one keeps it: the stricter of the two wins
    elsewhere, but here the tool's own word is the honest one.
    """
    if getattr(tool, 'spec', None) is not None:
        return tool
    name = str(getattr(tool, 'name', ''))
    spec = (BUILTIN_SPECS.get(name) or COLLAB_SPECS.get(name)
            or dynamic_spec(name, tool))
    try:
        tool.spec = spec
    except AttributeError:      # pragma: no cover - a frozen tool
        pass
    return tool


def attach_registry_specs(registry: Any) -> None:
    """Give every tool in a registry its declared spec.

    Built-in, skill, MCP and collaboration tools all end up in the same
    registry; the ones that arrive at runtime would otherwise fall into the
    strictest case, which would mean a prompt for every call.
    """
    for tool in list(getattr(registry, 'tools', lambda: [])()):
        attach_spec(tool)


def spec_for(name: str, tool: Any = None) -> ToolSpec:
    """The spec of a tool by name (for tests and for the audit view)."""
    declared = getattr(tool, 'spec', None)
    if isinstance(declared, ToolSpec):
        return declared
    return (BUILTIN_SPECS.get(name) or COLLAB_SPECS.get(name)
            or dynamic_spec(name, tool))


def set_sandbox_level(registry: Any, access: str) -> None:
    """Make the declared sandbox level match what is actually enforced.

    The tools declare the boundary they *need* ('workspace'); when the
    access level is looser or stricter, the declaration has to follow --
    otherwise the audit trail claims a fence that is not there, and the
    shell asks for the wrong path policy.

    The read-only tools stay read-only in every level: reading is allowed
    everywhere, that is what all three levels have in common.
    """
    declared = declared_sandbox(access)
    for tool in list(getattr(registry, 'tools', lambda: [])()):
        spec = getattr(tool, 'spec', None)
        if spec is None or not isinstance(spec, ToolSpec):
            continue
        # The **declared** level comes from the table, not from the tool's
        # current spec: the spec was already replaced once, and the second
        # call has to know what the tool wants, not what it was told last.
        name = str(getattr(tool, 'name', ''))
        base = BUILTIN_SPECS.get(name) or COLLAB_SPECS.get(name)
        if base is None or base.sandbox != 'workspace':
            continue
        # The spec is frozen: a new one is made and put on the tool, rather
        # than the field being written in place.
        try:
            tool.spec = replace(spec, sandbox=declared)
        except AttributeError:          # pragma: no cover - a frozen tool
            pass
