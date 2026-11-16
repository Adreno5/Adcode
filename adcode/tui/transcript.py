"""Transcript item rendering for the full-screen TUI.

Every visual block in the conversation is rendered to styled logical lines.
Display policies from ``/config`` decide how thinking text and tool output are
shown: hidden, a single label, a partial negative-index window with optional
fade, or the complete block.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from adcode.config import AppConfig, ThinkingDisplay
from adcode.llm.usage import estimate_text_tokens
from adcode.images import summarize_image_markers
from adcode.session import TranscriptItem
from adcode.theme import RGB, Palette, lead_in_gradient
from adcode.tui.canvas import BOLD, DIM, char_width
from adcode.tui.markdown import MarkdownRenderer
from adcode.tui.text import (
    ROW_BAR, ROW_BLEED, Line, Row, Span, add_span, line_width, truncate_line,
    wrap_line, wrap_lines,
)


@dataclass
class LiveRender:
    phase: str = 'idle'
    thinking_lines: list[Line] = field(default_factory=list)
    thinking_started: float = 0.0
    answer_lines: list[Line] = field(default_factory=list)
    compaction_lines: list[Line] = field(default_factory=list)
    aside_question: str = ''
    aside_lines: list[Line] = field(default_factory=list)
    tool_name: str = ''
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_running: bool = False
    tool_output: list[str] = field(default_factory=list)


def _chain_names(chain: list[TranscriptItem]) -> list[tuple[str, int]]:
    """The tools a chain used, in the order they first appear, with counts."""
    names: list[str] = []
    counts: list[int] = []
    for item in chain:
        name = item.display or item.name or 'tool'
        if name in names:
            counts[names.index(name)] += 1
            continue
        names.append(name)
        counts.append(1)
    return list(zip(names, counts))


def tool_arg_brief(name: str, args: dict[str, Any]) -> str:
    if name == 'read':
        path = str(args.get('path', ''))
        offset = args.get('offset')
        limit = args.get('limit')
        if offset or limit:
            low = offset or 1
            high = (offset or 1) + (limit or 0) - 1 if limit else '?'
            path += f' [{low}:{high}]'
        return path
    if name == 'bash':
        sub = args.get('sub', 'execute')
        if sub == 'execute':
            return str(args.get('command', ''))
        if sub == 'write':
            value = str(args.get('input', ''))
            return value[:60] + ('...' if len(value) > 60 else '')
        return str(sub)
    for key in ('path', 'pattern', 'url', 'task_name', 'target', 'name',
                'team_name', 'query', 'message'):
        if key in args and args[key]:
            value = str(args[key]).replace('\n', ' ')
            return value
    if args:
        key = next(iter(args))
        return str(args[key]).replace('\n', ' ')
    return ''


def _thinking_suffix(config: ThinkingDisplay, now: float) -> str:
    if config.animation == 'fixed':
        return config.fixed_suffix
    rotation = config.rotation or ['...']
    durations = config.rotation_ms or [400] * len(rotation)
    total = sum(max(50, value) for value in durations)
    if total <= 0:
        return rotation[0]
    position = (now * 1000.0) % total
    elapsed = 0.0
    for frame, duration in zip(rotation, durations):
        elapsed += max(50, duration)
        if position < elapsed:
            return frame
    return rotation[-1]


def _apply_base_role(lines: list[Line], role: str) -> list[Line]:
    result: list[Line] = []
    for line in lines:
        converted: Line = []
        for span in line:
            fg = span.fg
            if fg is None or fg == 'text':
                fg = role
            add_span(converted, Span(span.text, fg, span.bg, span.attrs, span.alpha))
        result.append(converted)
    return result


def _apply_gradient(lines: list[Line], top_alpha: int) -> list[Line]:
    """Fade a visual window from full opacity at the bottom to ``top_alpha``."""
    if not lines:
        return lines
    count = len(lines)
    if count == 1:
        alphas = [255]
    else:
        alphas = [
            round(top_alpha + (255 - top_alpha) * (index / (count - 1)))
            for index in range(count)
        ]
    result: list[Line] = []
    for line, alpha in zip(lines, alphas):
        converted: Line = []
        for span in line:
            converted.append(Span(span.text, span.fg, span.bg, span.attrs,
                                  min(span.alpha, alpha)))
        result.append(converted)
    return result


def _paint_lead_in(line: Line, colors: list[RGB]) -> Line:
    """Recolour the first cells of ``line`` with a column-by-column background.

    Spans are split wherever the background changes, so a tool call row can fade
    its opening cells from a light accent tint into the plain panel grey.
    """
    if not colors or not line:
        return line
    result: Line = []
    column = 0
    for span in line:
        if column >= len(colors):
            add_span(result, Span(span.text, span.fg, span.bg, span.attrs,
                                  span.alpha))
            continue
        text = ''
        bg: str | RGB | None = span.bg
        for char in span.text:
            size = char_width(char)
            if size == 0:
                text += char
                continue
            target = colors[column] if column < len(colors) else span.bg
            if target != bg:
                add_span(result, Span(text, span.fg, bg, span.attrs, span.alpha))
                text = ''
                bg = target
            text += char
            column += size
        if text:
            add_span(result, Span(text, span.fg, bg, span.attrs, span.alpha))
    return result


def _ellipsis_line(mode: str, omitted: int, label: str) -> list[Line]:
    if omitted <= 0 or mode == 'none':
        return []
    if mode == 'count':
        text = f'({omitted} characters omitted)'
    else:
        text = label or '...'
    if not text:
        return []
    return [[Span(text, 'muted', attrs=DIM)]]


def _partial_block(lines: list[Line], width: int, from_index: int, to_index: int,
                   style: str, gradient_top: int, ellipsis: str,
                   ellipsis_label: str, bar: bool = False) -> list[Line]:
    content_width = max(4, width - (2 if bar else 0))
    visual = wrap_lines(lines, content_width)
    start = max(0, len(visual) + from_index) if from_index < 0 else min(from_index, len(visual))
    end = max(0, len(visual) + to_index) if to_index < 0 else min(to_index, len(visual) - 1)
    if end < start:
        start, end = end, start
    window = visual[start:end + 1]
    if style == 'gradient':
        window = _apply_gradient(window, gradient_top)
    omitted = 0
    if start > 0:
        from adcode.tui.text import plain_text
        omitted = sum(len(plain_text(line)) for line in visual[:start])
    output: list[Line] = []
    if start > 0:
        output.extend(_ellipsis_line(ellipsis, omitted, ellipsis_label))
    for line in window:
        if bar:
            prefixed: Line = [Span('│ ', 'border')]
            for span in line:
                add_span(prefixed, Span(span.text, span.fg, span.bg,
                                        span.attrs, span.alpha))
            output.append(prefixed)
        else:
            output.append(line)
    return output


class TranscriptRenderer:
    def __init__(self) -> None:
        self._cache: dict[int, tuple[tuple[Any, ...], list[Line]]] = {}

    def invalidate(self) -> None:
        self._cache.clear()

    # ── public entry point ──────────────────────────────────
    def render(self, items: list[TranscriptItem], live: LiveRender, width: int,
               config: AppConfig, config_version: int = 0,
               now: float | None = None,
               palette: Palette | None = None) -> list[Line]:
        now = time.time() if now is None else now
        width = max(16, width)
        merged = config.tools.chain == 'merged'
        output: list[Line] = []
        previous = ''
        index = 0
        while index < len(items):
            item = items[index]
            if merged and item.kind == 'tool':
                # A chain of calls collapses into one row with the tool names
                # and their counts; the details stay in the history.
                chain = [item]
                index += 1
                while index < len(items) and items[index].kind == 'tool':
                    chain.append(items[index])
                    index += 1
                block = self._chain_lines(chain, width, config, now, palette)
            else:
                block = self._item_lines(item, width, config, config_version,
                                         now, palette)
                index += 1
            if not block:
                continue
            if output and output[-1] and self._needs_gap(previous, item.kind):
                output.append([])
            output.extend(block)
            if item.kind == 'assistant':
                output.append([])
            elif item.kind == 'thinking':
                output.append([])
            previous = item.kind
        if items:
            self._prune(items)
        live_lines = self._live_lines(live, width, config, now)
        if live_lines:
            if output and output[-1]:
                output.append([])
            output.extend(live_lines)
        return output

    @staticmethod
    def _needs_gap(previous: str, current: str) -> bool:
        if current == 'user':
            return True
        if current == 'aside':
            return True
        if current == 'tool':
            return previous in ('assistant', 'thinking', 'user', 'aside')
        if current == 'assistant':
            return previous in ('tool', 'thinking', 'notice', 'agent')
        if current == 'compaction':
            return True
        return False

    def _prune(self, items: list[TranscriptItem]) -> None:
        alive = {id(item) for item in items}
        if len(self._cache) > len(items) * 2 + 32:
            self._cache = {key: value for key, value in self._cache.items()
                           if key in alive}

    # ── items ───────────────────────────────────────────────
    def _item_lines(self, item: TranscriptItem, width: int, config: AppConfig,
                    config_version: int, now: float,
                    palette: Palette | None = None) -> list[Line]:
        signature = (width, config_version, item.kind, item.text, item.status,
                     len(item.output), item.duration,
                     bool(item.diff), item.display, item.agent, item.level,
                     item.name, _args_signature(item.args),
                     _palette_signature(palette))
        cached = self._cache.get(id(item))
        if cached is not None and cached[0] == signature:
            return cached[1]
        lines = self._render_item(item, width, config, now, palette)
        self._cache[id(item)] = (signature, lines)
        return lines

    def _render_item(self, item: TranscriptItem, width: int,
                     config: AppConfig, now: float,
                     palette: Palette | None = None) -> list[Line]:
        if item.kind == 'user':
            return self._render_user(item, width)
        if item.kind == 'assistant':
            return self._render_assistant(item, width)
        if item.kind == 'thinking':
            return self._render_thinking(item, width, config)
        if item.kind == 'tool':
            return self._render_tool(item, width, config, now, palette)
        if item.kind == 'notice':
            return self._render_notice(item, width)
        if item.kind == 'aside':
            return self._render_aside(item, width)
        if item.kind == 'compaction':
            return self._render_compaction(item, width)
        if item.kind == 'agent':
            return self._render_agent(item, width)
        return []

    def _render_user(self, item: TranscriptItem, width: int) -> list[Line]:
        prefix = [Span('> ', 'accent', attrs=BOLD)]
        result: list[Line] = []
        for index, raw in enumerate(summarize_image_markers(item.text).split('\n')):
            line: Line = list(prefix) if index == 0 else [Span('  ', None)]
            add_span(line, Span(raw, 'text'))
            result.extend(wrap_line(line, width))
        return result or [prefix]

    def _render_assistant(self, item: TranscriptItem, width: int) -> list[Line]:
        renderer = MarkdownRenderer(width)
        renderer.feed(item.text)
        renderer.finish()
        return wrap_lines(renderer.lines(), width)

    def _render_thinking(self, item: TranscriptItem, width: int,
                         config: AppConfig) -> list[Line]:
        mode = config.thinking.mode
        if mode == 'hidden':
            return []
        if not config.thinking.show_completed:
            return []
        if mode == 'summary':
            # The reasoning is kept, but not spelled out: the row says how
            # much there was.
            tokens = estimate_text_tokens(item.text or '')
            return [[Span('thought of ', 'subtext'),
                     Span(f'{tokens} token{"s" if tokens != 1 else ""}',
                          'muted')]]
        if mode == 'label':
            label = config.thinking.label or 'Thinking'
            suffix = '' if not item.duration else f' · {item.duration:.1f}s'
            return [[Span(label, 'subtext', attrs=BOLD),
                     Span(suffix, 'muted')]]
        renderer = MarkdownRenderer(width)
        renderer.feed(item.text)
        renderer.finish()
        lines = _apply_base_role(renderer.lines(), 'subtext')
        if mode == 'full':
            return wrap_lines(lines, width)
        return _partial_block(
            lines, width, config.thinking.from_index, config.thinking.to_index,
            config.thinking.style, config.thinking.gradient_top_alpha,
            config.thinking.ellipsis, config.thinking.ellipsis_label,
        )

    def _render_tool(self, item: TranscriptItem, width: int, config: AppConfig,
                     now: float, palette: Palette | None = None) -> list[Line]:
        mode = config.tools.mode
        if mode == 'hidden':
            return []
        output: list[Line] = []
        if mode in ('call', 'both'):
            output.extend(self._tool_call_line(item, width, config, now,
                                               palette))
        if mode in ('result', 'both'):
            output.extend(self._tool_result_lines(item, width, config))
            if config.tools.show_diff and item.diff:
                output.extend(self._render_diff(item.diff, width))
        return output

    def _chain_lines(self, chain: list[TranscriptItem], width: int,
                     config: AppConfig, now: float,
                     palette: Palette | None = None) -> list[Line]:
        """One line for a whole chain of steps.

        The line names the tools the chain used and how often: ``bash x2,
        grep, write``.  The details stay in the history -- the row is the
        summary, not the work.
        """
        if not chain:
            return []
        if config.tools.mode == 'hidden':
            return []
        row_width = width + ROW_BLEED
        line: Line = [Span(ROW_BAR + ' ', 'accent', 'panel')]
        names = _chain_names(chain)
        for index, (name, count) in enumerate(names):
            if index:
                add_span(line, Span(', ', 'muted', 'panel'))
            add_span(line, Span(name, 'text', 'panel', attrs=BOLD))
            if count > 1:
                add_span(line, Span(f' x{count}', 'muted', 'panel'))
        last = chain[-1]
        if last.status == 'running':
            dots = '...'[:1 + int(now * 2) % 3]
            add_span(line, Span(' ' + dots, 'muted', 'panel'))
        elif last.status == 'error':
            add_span(line, Span('  !', 'error', 'panel'))
        line = truncate_line(line, row_width, ellipsis='…')
        if palette is not None:
            line = _paint_lead_in(line, lead_in_gradient(palette))
        used = line_width(line)
        if used < row_width:
            add_span(line, Span(' ' * (row_width - used), None, 'panel'))
        return [Row(line, ROW_BLEED)]

    def _tool_call_line(self, item: TranscriptItem, width: int, config: AppConfig,
                        now: float, palette: Palette | None = None) -> list[Line]:
        # The row bleeds one cell left of the transcript margin, so it is one
        # cell wider than the wrapped text around it.  The accent bar is a
        # left-edge glyph: it keeps the one-cell gutter but puts its ink on the
        # screen edge instead of in the middle of that cell.
        row_width = width + ROW_BLEED
        line: Line = [Span(ROW_BAR + ' ', 'accent', 'panel')]
        add_span(line, Span(item.display or item.name, 'text', 'panel', attrs=BOLD))
        if config.tools.show_args:
            brief = tool_arg_brief(item.name, item.args)
            if brief:
                add_span(line, Span('  ', None, 'panel'))
                add_span(line, Span(brief, 'muted', 'panel'))
        status = item.status
        if status == 'running':
            dots = '...'[:1 + int(now * 2) % 3]
            add_span(line, Span(' ' + dots, 'muted', 'panel'))
        elif status == 'error':
            add_span(line, Span('  !', 'error', 'panel'))
        line = truncate_line(line, row_width, ellipsis='…')
        if palette is not None:
            line = _paint_lead_in(line, lead_in_gradient(palette))
        used = line_width(line)
        if used < row_width:
            add_span(line, Span(' ' * (row_width - used), None, 'panel'))
        return [Row(line, ROW_BLEED)]

    def _tool_result_lines(self, item: TranscriptItem, width: int,
                           config: AppConfig) -> list[Line]:
        if not item.output:
            return []
        renderer = MarkdownRenderer(max(8, width - 2))
        renderer.feed(item.output)
        renderer.finish()
        lines = _apply_base_role(renderer.lines(), 'subtext')
        if item.status == 'error':
            lines = _apply_base_role(lines, 'error')
        if config.tools.result_mode == 'full':
            visual = wrap_lines(lines, max(8, width - 2))
            limit = config.tools.max_result_lines
            if limit and len(visual) > limit:
                visual = visual[:limit]
            return [[Span('│ ', 'border', attrs=DIM)]
                    + [Span(span.text, span.fg, span.bg, span.attrs, span.alpha)
                       for span in line]
                    for line in visual]
        return _partial_block(
            lines, width, config.tools.from_index, config.tools.to_index,
            config.tools.style, config.tools.gradient_top_alpha,
            config.tools.ellipsis, config.tools.ellipsis_label, bar=True,
        )

    def _render_diff(self, diff: dict[str, Any], width: int) -> list[Line]:
        from adcode.diff import FileDiff, diff_lines
        file_diff = FileDiff(
            path=str(diff.get('path', '')),
            unified=str(diff.get('unified', '')),
            additions=int(diff.get('additions', 0) or 0),
            deletions=int(diff.get('deletions', 0) or 0),
            omitted=bool(diff.get('omitted')),
        )
        if file_diff.omitted and not file_diff.unified:
            return [[Span('│ ', 'border'),
                     Span(file_diff.path, 'text', attrs=BOLD),
                     Span('  (file too large for an inline diff)', 'muted')]]
        header: Line = [Span('│ ', 'border'),
                        Span(file_diff.path, 'text', attrs=BOLD),
                        Span(f'  +{file_diff.additions}', 'diff_add'),
                        Span(f' -{file_diff.deletions}', 'diff_del')]
        output = [header]
        for entry in diff_lines(file_diff):
            if entry.kind == 'hunk':
                output.append([Span('│ ', 'border'),
                               Span(entry.text, 'info', attrs=DIM)])
                continue
            if entry.kind == 'meta':
                output.append([Span('│ ', 'border'), Span(entry.text, 'muted')])
                continue
            marker = {'add': '+', 'del': '-', 'ctx': ' '}[entry.kind]
            color = {'add': 'diff_add', 'del': 'diff_del', 'ctx': 'subtext'}[entry.kind]
            number = entry.new_number if entry.kind != 'del' else entry.old_number
            prefix_text = f'{(str(number) if number is not None else ""):>5} '
            line: Line = [Span('│ ', 'border'), Span(prefix_text, 'muted', attrs=DIM),
                          Span(marker, color), Span(entry.text, color)]
            for row in wrap_line(line, width):
                output.append(row)
        if file_diff.omitted:
            output.append([Span('│ ', 'border'),
                           Span('diff output truncated', 'muted')])
        return output

    def _render_notice(self, item: TranscriptItem, width: int) -> list[Line]:
        glyph = {'info': '·', 'success': '✓', 'warning': '!', 'error': '!'}.get(
            item.level, '·')
        role = {'info': 'muted', 'success': 'success',
                'warning': 'warning', 'error': 'error'}.get(item.level, 'muted')
        prefix = f'{glyph} '
        result: list[Line] = []
        for index, raw in enumerate(item.text.split('\n')):
            line: Line = [Span(prefix if index == 0 else '  ', role)]
            add_span(line, Span(raw, 'subtext' if item.level == 'info' else role))
            result.extend(wrap_line(line, width))
        return result

    def _render_aside(self, item: TranscriptItem, width: int) -> list[Line]:
        output: list[Line] = []
        question = item.display or ''
        for index, raw in enumerate(question.split('\n')):
            line: Line = [Span('? ' if index == 0 else '  ', 'accent')]
            add_span(line, Span(raw, 'subtext', attrs=BOLD))
            output.extend(wrap_line(line, width))
        renderer = MarkdownRenderer(width)
        renderer.feed(item.text)
        renderer.finish()
        output.extend(wrap_lines(_apply_base_role(renderer.lines(), 'subtext'), width))
        return output

    def _render_compaction(self, item: TranscriptItem, width: int) -> list[Line]:
        return [[Span('» ', 'accent'), Span(item.text or 'context compacted', 'muted')]]

    def _render_agent(self, item: TranscriptItem, width: int) -> list[Line]:
        line: Line = [Span('│ ', 'border'),
                      Span(item.agent, 'subtext', attrs=BOLD),
                      Span(' · ', 'border'),
                      Span(item.text, 'muted')]
        return wrap_line(line, width)

    # ── live state ──────────────────────────────────────────
    def _live_lines(self, live: LiveRender, width: int, config: AppConfig,
                    now: float) -> list[Line]:
        output: list[Line] = []
        phase = live.phase
        if phase == 'thinking':
            mode = config.thinking.mode
            if mode in ('label', 'summary'):
                # The summary mode keeps the running thought out of the way as
                # well: the label is enough while it is in flight.
                label = config.thinking.label or 'Thinking'
                output.append([
                    Span(label, 'subtext', attrs=BOLD),
                    Span(' ' + _thinking_suffix(config.thinking, now), 'muted'),
                ])
            elif mode == 'partial':
                output.extend(_partial_block(
                    _apply_base_role(live.thinking_lines, 'subtext'), width,
                    config.thinking.from_index, config.thinking.to_index,
                    config.thinking.style, config.thinking.gradient_top_alpha,
                    config.thinking.ellipsis, config.thinking.ellipsis_label,
                ))
            elif mode == 'full':
                output.extend(wrap_lines(
                    _apply_base_role(live.thinking_lines, 'subtext'), width))
        elif phase == 'answering':
            output.extend(wrap_lines(live.answer_lines, width))
        elif phase == 'tool':
            mode = config.tools.mode
            # The tool call line itself is a transcript item; only the running
            # output tail is rendered live here.  A merged chain shows no
            # details at all.
            if (mode in ('result', 'both') and live.tool_output
                    and config.tools.chain != 'merged'):
                tail = live.tool_output[-3:]
                for raw in tail:
                    output.extend(wrap_line(
                        [Span('│ ', 'border'), Span(raw, 'muted', attrs=DIM)], width))
        elif phase == 'compacting':
            output.append([Span('» ', 'accent'), Span('compacting context', 'muted')])
            tail = wrap_lines(_apply_base_role(live.compaction_lines, 'subtext'), width)
            output.extend(tail[-5:])
        elif phase == 'aside':
            question = live.aside_question or ''
            output.append([Span('? ', 'accent'), Span(question, 'subtext', attrs=BOLD)])
            output.extend(wrap_lines(
                _apply_base_role(live.aside_lines, 'subtext'), width))
        return output


def _args_signature(args: dict[str, Any]) -> tuple[str, ...]:
    try:
        return tuple(f'{key}={str(value)[:40]}' for key, value in sorted(args.items()))
    except Exception:
        return ()


def _palette_signature(palette: Palette | None) -> tuple[Any, ...]:
    """Cache key for the palette parts the transcript actually paints with."""
    if palette is None:
        return ()
    return (palette.name, palette.panel, palette.accent, palette.background)
