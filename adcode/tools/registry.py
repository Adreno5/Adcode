# adcode/tools/registry.py
# Tool registry, re-exported from the engine sdk.
#
# ManagedToolRegistry adds ownership tracking (builtin / skill / agent /
# mcp:<server> sources) and snapshot-diff execution for tools that declare
# ``snapshot``: the file is captured around the call and the unified diff is
# appended to the model text.  Execution returns ``(text, file_diff, error)``
# and failures follow the "Error: ..." text convention.

from harness.tools.base import Tool, ToolResult
from harness.tools.registry import AnnotatedTool, ManagedToolRegistry

ToolRegistry = ManagedToolRegistry

__all__ = [
    'Tool', 'ToolResult', 'ToolRegistry', 'AnnotatedTool',
    'ManagedToolRegistry',
]
