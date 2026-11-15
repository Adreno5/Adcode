# adcode/tools/runtime.py
# Per-turn tool context, re-exported from the engine sdk.
#
# The engine (harness.agent.ConversationEngine) pushes a ToolContext around
# every tool call; tools read cwd / interrupt / live_sink from here.

from harness.tools.context import (
    ToolContext,
    current_context,
    pop_context,
    push_context,
    resolve_path,
)

__all__ = [
    'ToolContext', 'current_context', 'pop_context', 'push_context',
    'resolve_path',
]
