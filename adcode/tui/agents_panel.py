"""The sub-agent strip: what the background workers are doing.

One row per child agent, above the todo panel and the prompt: its state, how
long it has been running, the step it is on, and a tail of what it is
producing.  Rendering only -- the app assembles the view from the agent
manager's rows and the live text the pump collects.
"""

from __future__ import annotations

from typing import Any

from adcode.tui.text import Line, Span, truncate_line

BOLD = 1

# Header plus rows; sub-agents are transient, so this never needs to be big.
MAX_ROWS = 6

_MARKERS = {
    'running': ('▸ ', 'accent'),
    'pending_init': ('○ ', 'muted'),
    'completed': ('✓ ', 'success'),
    'errored': ('✗ ', 'error'),
    'interrupted': ('· ', 'warning'),
    'shutdown': ('· ', 'muted'),
}

WORDS = {
    'running': 'running',
    'pending_init': 'starting',
    'completed': 'done',
    'errored': 'failed',
    'interrupted': 'stopped',
    'shutdown': 'closed',
}

ACTIVE = ('running', 'pending_init')
FINISHED = ('completed', 'errored', 'interrupted', 'shutdown')


def format_elapsed(seconds: float) -> str:
    """``12s``, ``1m02s``; empty for an unknown duration."""
    if seconds is None or seconds < 0:
        return ''
    total = int(seconds)
    if total < 60:
        return f'{total}s'
    minutes, rest = divmod(total, 60)
    if minutes < 60:
        return f'{minutes}m{rest:02d}s'
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h{minutes:02d}m'


def _ordering(row: dict[str, Any]) -> tuple[int, float]:
    """Live agents first, then the most recently started."""
    status = str(row.get('status') or '')
    return (0 if status in ACTIVE else 1, -float(row.get('started') or 0.0))


def _row_line(row: dict[str, Any], columns: int) -> Line:
    status = str(row.get('status') or '')
    marker, role = _MARKERS.get(status, ('○ ', 'muted'))
    line: Line = [
        Span('│ ', 'border'),
        Span(marker, role),
        Span(str(row.get('name') or '?'), 'subtext',
             attrs=BOLD if status in ACTIVE else 0),
    ]
    word = WORDS.get(status, status)
    if word:
        line.append(Span(' ', None))
        line.append(Span(word, role if status != 'completed' else 'success'))
    elapsed = format_elapsed(row.get('elapsed'))
    if elapsed:
        line.append(Span(f' {elapsed}', 'muted'))
    step = str(row.get('step') or '').strip()
    if step and status in ACTIVE:
        line.append(Span(' · ', 'border'))
        line.append(Span(step, 'muted'))
    tail = str(row.get('tail') or '').strip().replace('\n', ' ')
    if tail and status in ACTIVE:
        line.append(Span(' · ', 'border'))
        line.append(Span(tail, 'text'))
    return truncate_line(line, max(10, columns - 1), '…')


def panel_lines(rows: list[dict[str, Any]], columns: int, *,
                max_rows: int = MAX_ROWS) -> list[Line]:
    """The panel rows; ``[]`` when there is nothing to show.

    Each row carries ``name``, ``status``, ``elapsed`` (seconds), ``step`` (the
    tool or action it is on) and ``tail`` (the end of what it just produced).
    """
    if not rows:
        return []
    ordered = sorted((row for row in rows if row.get('name') != 'root'),
                     key=_ordering)
    if not ordered:
        return []
    columns = max(16, columns)
    running = sum(1 for row in ordered if row.get('status') in ACTIVE)
    done = sum(1 for row in ordered if row.get('status') in FINISHED)
    header: Line = [Span('│ ', 'border'),
                    Span('agents', 'subtext', attrs=BOLD),
                    Span(f'  {running} running', 'muted' if running else 'muted')]
    if done:
        header.append(Span(f' · {done} done', 'muted'))
    lines: list[Line] = [truncate_line(header, max(10, columns - 1), '…')]

    budget = max(0, max_rows - 1)
    shown = ordered[:budget]
    hidden = len(ordered) - len(shown)
    if hidden:
        # the count needs a row of its own, so one agent less is drawn
        shown = ordered[:max(0, budget - 1)]
        hidden = len(ordered) - len(shown)
    for row in shown:
        lines.append(_row_line(row, columns))
    if hidden:
        lines.append(truncate_line(
            [Span('│ ', 'border'), Span(f'+{hidden} more', 'muted')],
            max(10, columns - 1), '…'))
    return lines
