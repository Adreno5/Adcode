"""The todo strip above the prompt.

It shows the plan the agent is working through: a header with the progress and
one row per item, marked by state.  Rendering only -- the plan comes from
``harness.todo`` through ``adcode.tools.todo``, and the app redraws when the
store's revision moves.
"""

from __future__ import annotations

from typing import Any

from adcode.tui.text import Line, Span, truncate_line

BOLD = 1

# Header plus items; small enough that the transcript keeps most of the screen.
MAX_ROWS = 7

_MARKERS = {
    'completed': ('✓ ', 'success'),
    'failed': ('✗ ', 'error'),
    'skipped': ('· ', 'muted'),
    'in_progress': ('▸ ', 'accent'),
    'pending': ('○ ', 'muted'),
}


def _short_id(item_id: str) -> str:
    """``item-03`` reads as ``3``; anything else is left alone."""
    tail = item_id.rsplit('-', 1)[-1]
    if item_id.startswith('item-') and tail.isdigit():
        return str(int(tail) + 1)
    return item_id


def _header(plan: Any, columns: int) -> Line:
    total = len(plan.items)
    done = plan.done_count
    line: Line = [
        Span('│ ', 'border'),
        Span('todo', 'subtext', attrs=BOLD),
        Span(f'  {done}/{total}', 'muted'),
    ]
    current = plan.current
    if plan.status == 'planning':
        line.append(Span('  planning…', 'muted'))
    elif current is not None and current.status != 'completed':
        line.append(Span('  ·  ', 'border'))
        line.append(Span(current.title, 'subtext'))
    return truncate_line(line, max(8, columns - 1), '…')


def panel_lines(plan: Any, columns: int, *,
                max_rows: int = MAX_ROWS) -> list[Line]:
    """Rows for the panel; an empty list means "draw nothing".

    The visible window keeps the current item in view, so a long plan never
    hides what is actually running.
    """
    if plan is None or not getattr(plan, 'items', None):
        return []
    columns = max(12, columns)
    budget = max(1, max_rows - 1)
    items = list(plan.items)
    if len(items) > budget:
        current = plan.current
        index = items.index(current) if current in items else 0
        start = max(0, min(index - budget // 2, len(items) - budget))
        items = items[start:start + budget]

    lines: list[Line] = [_header(plan, columns)]
    for item in items:
        marker, role = _MARKERS.get(item.status, ('○ ', 'muted'))
        finished = item.status in ('completed', 'skipped')
        line: Line = [
            Span('│ ', 'border'),
            Span(marker, role),
            Span(f'{_short_id(item.id)} ', 'muted'),
            Span(item.title, 'muted' if finished else 'text'),
        ]
        if item.status == 'in_progress':
            line[-1] = Span(item.title, 'subtext', attrs=BOLD)
        lines.append(truncate_line(line, columns - 1, '…'))
    return lines
