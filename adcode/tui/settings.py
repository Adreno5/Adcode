"""The ``/config`` settings browser.

The tree mirrors every user-facing setting: appearance, thinking display, tool
display, reasoning, context compaction, sub-agents, skills and MCP.  Interaction
follows the requested model: Up/Down move, Enter expands an item and selects
options, Esc cancels (restoring the values captured when the item was opened),
and Enter on the expanded title confirms the item.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from adcode.config import (
    REASONING_EFFORTS, AppConfig, get_by_path, set_by_path,
)
from adcode.theme import ACCENT_PRESETS, Palette
from adcode.tui.canvas import BOLD, DIM, Canvas, text_width, truncate_text
from adcode.tui.dialogs import (
    Overlay, _draw_backdrop, _draw_frame, _draw_hint, _panel_geometry, wheel_step,
)
from adcode.tui.input import KeyEvent, get_clipboard, is_paste_key


@dataclass
class SettingChoice:
    value: Any
    label: str
    description: str = ''
    children: list['SettingNode'] = field(default_factory=list)


@dataclass
class SettingNode:
    key: str
    label: str
    kind: str                      # group | choice | bool | number | text | list | label | action
    path: str = ''
    choices: list[SettingChoice] = field(default_factory=list)
    minimum: int = 0
    maximum: int = 100
    hint: str = ''
    action: Callable[[], None] | None = None
    value_text: Callable[[AppConfig], str] | None = None


@dataclass
class Row:
    kind: str                      # group | setting | option | label
    depth: int
    key: str
    node: SettingNode | None = None
    option: SettingChoice | None = None


def _choice(key: str, label: str, path: str,
            choices: list[SettingChoice], hint: str = '') -> SettingNode:
    return SettingNode(key=key, label=label, kind='choice', path=path,
                       choices=choices, hint=hint)


def _bool(key: str, label: str, path: str, hint: str = '') -> SettingNode:
    return SettingNode(key=key, label=label, kind='bool', path=path, hint=hint)


def _number(key: str, label: str, path: str, minimum: int, maximum: int,
            hint: str = '') -> SettingNode:
    return SettingNode(key=key, label=label, kind='number', path=path,
                       minimum=minimum, maximum=maximum, hint=hint)


def _text(key: str, label: str, path: str, hint: str = '') -> SettingNode:
    return SettingNode(key=key, label=label, kind='text', path=path, hint=hint)


def _label(key: str, label: str,
           value: Callable[[AppConfig], str]) -> SettingNode:
    return SettingNode(key=key, label=label, kind='label', value_text=value)


def _partial_children(prefix: str, from_path: str, to_path: str,
                      style_path: str, gradient_path: str,
                      ellipsis_path: str, ellipsis_label_path: str) -> list[SettingNode]:
    style = _choice(f'{prefix}.style', 'Display style', style_path, [
        SettingChoice('normal', 'Normal', 'Markdown text at full opacity'),
        SettingChoice('gradient', 'Gradient fade', 'Fade the top rows',
                      [_number(f'{prefix}.gradient', 'Top row alpha',
                               gradient_path, 0, 255,
                               '0 = invisible top row, 255 = no fade')]),
    ])
    ellipsis = _choice(f'{prefix}.ellipsis', 'Omitted content', ellipsis_path, [
        SettingChoice('none', 'No marker', 'Truncate silently'),
        SettingChoice('label', 'Ellipsis label', 'Show a small label',
                      [_text(f'{prefix}.ellipsis_label', 'Label text',
                             ellipsis_label_path)]),
        SettingChoice('count', 'Character count', 'Show how many characters were omitted'),
    ])
    return [
        _number(f'{prefix}.from', 'Window start (negative)', from_path, -500, -1,
                'e.g. -5 starts five visual rows before the end'),
        _number(f'{prefix}.to', 'Window end (negative)', to_path, -500, -1,
                'e.g. -1 includes the last visual row'),
        style, ellipsis,
    ]


def build_settings_tree(config: AppConfig) -> list[SettingNode]:

    accent_choices = [
        SettingChoice(hex_value, name, hex_value)
        for name, hex_value in ACCENT_PRESETS
    ]
    theme = _choice('ui.theme', 'Theme', 'ui.theme', [
        SettingChoice('dark', 'Dark', 'Dark canvas, light grey content'),
        SettingChoice('light', 'Light', 'Light canvas, dark content'),
    ])
    accent = _choice('ui.accent', 'Accent colour', 'ui.accent', accent_choices)
    accent_text = _text('ui.accent_custom', 'Custom accent (hex)', 'ui.accent',
                        'Type a #rrggbb value')
    appearance = [
        theme, accent, accent_text,
        _bool('ui.header', 'Show header', 'ui.show_header'),
        _bool('ui.tokens', 'Show token status', 'ui.show_token_status'),
        _number('ui.scroll', 'Mouse wheel lines', 'ui.mouse_scroll_lines', 1, 30),
        _number('ui.fps', 'Frames per second', 'ui.fps', 10, 240),
    ]

    thinking_choices = [
        SettingChoice('hidden', 'Hidden', 'Accept reasoning silently', []),
        SettingChoice('label', 'Label only', 'One animated label line', [
            _text('thinking.label', 'Label text', 'thinking.label',
                  'e.g. Thinking, Reasoning, Polishing'),
            _choice('thinking.animation', 'Suffix animation', 'thinking.animation', [
                SettingChoice('fixed', 'Fixed suffix', '', [
                    _text('thinking.fixed_suffix', 'Fixed suffix', 'thinking.fixed_suffix'),
                ]),
                SettingChoice('rotating', 'Rotating frames', '', [
                    _text('thinking.rotation', 'Frames (comma separated)',
                          'thinking.rotation'),
                    _text('thinking.rotation_ms', 'Frame durations (ms)',
                          'thinking.rotation_ms'),
                ]),
            ]),
        ]),
        SettingChoice('partial', 'Partial window', 'Show a configurable window', [
            *_partial_children('thinking.partial', 'thinking.from_index',
                               'thinking.to_index', 'thinking.style',
                               'thinking.gradient_top_alpha', 'thinking.ellipsis',
                               'thinking.ellipsis_label'),
            _bool('thinking.show_completed', 'Keep after completion',
                  'thinking.show_completed'),
        ]),
        SettingChoice('full', 'Full', 'Show everything', [
            _bool('thinking.show_completed', 'Keep after completion',
                  'thinking.show_completed'),
        ]),
        SettingChoice('summary', 'Token count',
                      'Keep the reasoning, but only say how much there was', [
            _bool('thinking.show_completed', 'Keep after completion',
                  'thinking.show_completed'),
        ]),
    ]
    thinking_item = _choice('thinking.mode', 'Thinking display', 'thinking.mode',
                            thinking_choices)

    result_children = [
        _choice('tools.result_mode', 'Result display', 'tools.result_mode', [
            SettingChoice('partial', 'Partial window', 'Show a configurable window', [
                *_partial_children('tools.partial', 'tools.from_index',
                                   'tools.to_index', 'tools.style',
                                   'tools.gradient_top_alpha', 'tools.ellipsis',
                                   'tools.ellipsis_label'),
            ]),
            SettingChoice('full', 'Full', 'Show the complete result', [
                _number('tools.max_lines', 'Maximum lines (0 = unlimited)',
                        'tools.max_result_lines', 0, 10_000),
            ]),
        ]),
        _bool('tools.diff', 'Show file diffs', 'tools.show_diff'),
    ]
    chain_item = _choice('tools.chain', 'Chain of calls', 'tools.chain', [
        SettingChoice('separate', 'One per row', 'Every call gets its own row'),
        SettingChoice('merged', 'Merged',
                      'A chain of calls folds into one row with the tool '
                      'names and their counts'),
    ])
    tool_item = _choice('tools.mode', 'Tool display', 'tools.mode', [
        SettingChoice('hidden', 'Hidden', 'Execute silently', []),
        SettingChoice('call', 'Call line only', 'One compact call line', [
            _bool('tools.args', 'Show arguments', 'tools.show_args'),
        ]),
        SettingChoice('result', 'Result only', 'Only the result section', result_children),
        SettingChoice('both', 'Call line and result', 'Both sections', [
            _bool('tools.args', 'Show arguments', 'tools.show_args'),
            *result_children,
        ]),
    ])

    footer = [
        _bool('footer.sweep', 'Running light', 'footer.sweep_enabled'),
        _number('footer.sweep_period', 'Light period (s)', 'footer.sweep_period',
                1, 60),
        _number('footer.sweep_span', 'Light width (cells)', 'footer.sweep_span',
                1, 200),
        _bool('footer.progress', 'Progress bar', 'footer.progress_enabled'),
        _number('footer.progress_window', 'Sample freshness (s)',
                'footer.progress_window', 1, 600),
        _number('footer.progress_samples', 'Samples before the bar',
                'footer.progress_samples', 1, 100),
        _bool('footer.done', 'Green when finished', 'footer.done_enabled'),
        _number('footer.done_wipe', 'Wipe duration (s)', 'footer.done_wipe',
                0, 60),
        _bool('footer.done_hold', 'Stay green until input', 'footer.done_hold'),
        _text('footer.done_color', 'Green colour (#rrggbb)', 'footer.done_color',
              'Empty uses the palette green'),
        _text('footer.done_text_color', 'Font colour (#rrggbb)',
              'footer.done_text_color', 'Empty uses white'),
    ]

    hold = [
        _bool('hold.enabled', 'Keep retrying', 'hold.enabled'),
        _number('hold.retries', 'Maximum retries', 'hold.max_retries',
                0, 10_000),
        _number('hold.delay', 'Seconds between retries', 'hold.delay', 0, 600),
    ]

    reasoning = [
        _choice('reasoning.effort', 'Reasoning effort', 'reasoning.effort',
                [SettingChoice(value, value.capitalize()) for value in REASONING_EFFORTS]),
        _bool('reasoning.force', 'Force reasoning parameters', 'reasoning.force',
              'Send reasoning fields even when model metadata is missing'),
        _number('max_output', 'Max output tokens', 'max_output_tokens', 256, 2_000_000),
    ]

    context = [
        _bool('context.auto', 'Auto compact', 'context.auto_compact'),
        _number('context.threshold', 'Compact threshold %', 'context.compact_threshold',
                40, 99),
        _number('context.keep', 'Keep recent turns', 'context.keep_recent_turns', 0, 20),
        _number('context.reserved', 'Reserved output tokens',
                'context.reserved_output', 256, 2_000_000),
        _choice('context.effort', 'Compaction effort', 'context.compaction_reasoning_effort',
                [SettingChoice(value, value.capitalize()) for value in REASONING_EFFORTS]),
    ]

    type_names = sorted(config.agent_types)
    agents = [
        _bool('agents.enabled', 'Enable sub-agents', 'agents.enabled'),
        _bool('agents.teams', 'Enable agent teams', 'agents.team_enabled'),
        _number('agents.max', 'Max concurrent sub-agents', 'agents.max_concurrent', 1, 128),
        _number('agents.depth', 'Max delegation depth', 'agents.max_depth', 1, 32),
        _number('agents.wait', 'Wait timeout (ms)', 'agents.wait_timeout_ms',
                0, 3_600_000),
        _choice('agents.fork', 'Default fork turns', 'agents.default_fork_turns', [
            SettingChoice('all', 'All completed turns'),
            SettingChoice('none', 'No history'),
        ]),
        _bool('agents.activity', 'Show agent activity', 'agents.show_activity'),
        _bool('agents.output', 'Show agent output', 'agents.show_output'),
        _choice('agents.type', 'Default agent type', 'agents.default_type',
                [SettingChoice(name, name) for name in type_names]),
        _text('agents.suffix', 'Agent system suffix', 'agents.system_suffix'),
        _label('agents.types', 'Configured types',
               lambda cfg: ', '.join(sorted(cfg.agent_types))),
        _label('agents.teams_list', 'Configured teams',
               lambda cfg: ', '.join(sorted(cfg.agent_teams)) or '(none)'),
    ]

    skills = [
        _bool('skills.enabled', 'Enable skills', 'skills.enabled'),
        _bool('skills.project', 'Include project skills', 'skills.include_project'),
        _text('skills.paths', 'Extra skill paths', 'skills.extra_paths',
              'Separate paths with ;'),
        _label('skills.roots', 'Scanned roots',
               lambda cfg: '  '.join(
                   str(path) for path in _skill_roots(cfg))),
    ]

    mcp = [
        _label('mcp.servers', 'Configured servers',
               lambda cfg: ', '.join(str(server.get('name', '?'))
                                     for server in cfg.mcp_servers) or '(none)'),
    ]

    return [
        SettingNode(key='group:Appearance', label='Appearance', kind='group',
                    choices=[], value_text=None),
        *appearance,
        SettingNode(key='group:Work steps', label='Work steps', kind='group',
                    choices=[]),
        thinking_item,
        tool_item,
        chain_item,
        SettingNode(key='group:Footer', label='Footer status', kind='group',
                    choices=[]),
        *footer,
        SettingNode(key='group:Hold', label='Hold', kind='group', choices=[]),
        *hold,
        SettingNode(key='group:Reasoning', label='Reasoning', kind='group', choices=[]),
        *reasoning,
        SettingNode(key='group:Context', label='Context', kind='group', choices=[]),
        *context,
        SettingNode(key='group:Agents', label='Sub-agents and teams', kind='group',
                    choices=[]),
        *agents,
        SettingNode(key='group:Skills', label='Skills', kind='group', choices=[]),
        *skills,
        SettingNode(key='group:MCP', label='MCP', kind='group', choices=[]),
        *mcp,
    ]


def _skill_roots(config: AppConfig):
    from adcode.skills import default_skill_roots
    return default_skill_roots(config.skills)


class SettingsOverlay(Overlay):
    title = 'Settings'

    def __init__(self, config: AppConfig, on_change: Callable[[], None],
                 on_close: Callable[[], None] | None = None) -> None:
        super().__init__()
        self.config = config
        self.on_change = on_change
        self.on_close = on_close
        self.expanded: list[str] = []
        self.row_key = ''
        self.row = 0
        self.top = 0
        self.editing = False
        self.edit_key = ''
        self.edit_value = ''
        self.edit_error = ''
        self.edit_select_all = False
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.nodes = build_settings_tree(config)

    # ── tree flattening ─────────────────────────────────────
    def _flatten(self) -> list[Row]:
        rows: list[Row] = []

        def walk(nodes: list[SettingNode], depth: int) -> None:
            for node in nodes:
                if node.kind == 'group':
                    rows.append(Row('group', depth, node.key, node))
                    continue
                if node.kind == 'label':
                    rows.append(Row('label', depth, f'label:{node.key}', node))
                    continue
                rows.append(Row('setting', depth, node.key, node))
                if node.key not in self.expanded:
                    continue
                if node.kind in ('choice', 'bool'):
                    options = node.choices if node.kind == 'choice' else [
                        SettingChoice(True, 'Yes', 'Enabled'),
                        SettingChoice(False, 'No', 'Disabled'),
                    ]
                    current = self._get(node)
                    for option in options:
                        option_key = f'{node.key}::{option.value}'
                        rows.append(Row('option', depth + 1, option_key, node, option))
                        if _same(option.value, current) and option.children:
                            walk(option.children, depth + 2)

        walk(self.nodes, 0)
        return rows

    def _sync_cursor(self, rows: list[Row]) -> None:
        """Point ``row`` at the row remembered by ``row_key``.

        The key, not the index, is the source of truth: expanding and
        collapsing options inserts and removes rows, so an index captured
        before a change can silently point at a different setting.
        """
        if not self.row_key:
            return
        selectable = self._selectable(rows)
        if not selectable:
            self.row = 0
            return
        positions = {rows[index].key: index for index in selectable}
        key = self.row_key
        while key:
            index = positions.get(key)
            if index is not None:
                self.row = index
                self.row_key = key
                return
            key = key.rsplit('::', 1)[0] if '::' in key else ''
        # The remembered row is gone (an option collapsed into its parent):
        # fall back to the closest selectable row that still exists.
        position = sum(1 for index in selectable if index <= self.row)
        self.row = selectable[max(0, position - 1)]
        self.row_key = rows[self.row].key

    def _refocus(self, key: str) -> None:
        """Remember ``key`` and move the cursor onto it right away."""
        self.row_key = key
        self._sync_cursor(self._flatten())

    def _get(self, node: SettingNode) -> Any:
        if not node.path:
            return None
        try:
            return get_by_path(self.config, node.path)
        except Exception:
            return None

    def _set(self, node: SettingNode, value: Any) -> None:
        if not node.path:
            return
        try:
            set_by_path(self.config, node.path, value)
        except Exception:
            return

    def _snapshot_for(self, node: SettingNode) -> dict[str, Any]:
        paths: set[str] = set()

        def collect(current: SettingNode) -> None:
            if current.path:
                paths.add(current.path)
            for choice in current.choices:
                for child in choice.children:
                    collect(child)

        collect(node)
        values: dict[str, Any] = {}
        for path in paths:
            try:
                values[path] = copy.deepcopy(get_by_path(self.config, path))
            except Exception:
                pass
        return values

    def _restore(self, snapshot: dict[str, Any]) -> None:
        for path, value in snapshot.items():
            try:
                set_by_path(self.config, path, value)
            except Exception:
                pass
        self.on_change()

    # ── event handling ──────────────────────────────────────
    def _selectable(self, rows: list[Row]) -> list[int]:
        return [index for index, row in enumerate(rows)
                if row.kind in ('setting', 'option')]

    def _move(self, delta: int) -> None:
        rows = self._flatten()
        self._sync_cursor(rows)
        selectable = self._selectable(rows)
        if not selectable:
            return
        if self.row in selectable:
            position = (selectable.index(self.row) + delta) % len(selectable)
        else:
            position = 0 if delta > 0 else len(selectable) - 1
        self.row = selectable[position]
        self.row_key = rows[self.row].key

    def handle(self, event: KeyEvent) -> bool:
        rows = self._flatten()
        self._sync_cursor(rows)
        if self.editing:
            return self._handle_edit(event)
        if event.kind == 'paste':
            return True
        if event.kind == 'mouse':
            step = wheel_step(event)
            if step:
                self._move(step)
            return True
        if event.kind != 'key':
            return True
        key = event.key
        if key == 'esc':
            if self.expanded:
                node_key = self.expanded.pop()
                snapshot = self.snapshots.pop(node_key, None)
                if snapshot is not None:
                    self._restore(snapshot)
                self._refocus(node_key)
                self.on_change()
            else:
                self.closed = True
                if self.on_close is not None:
                    self.on_close()
            return True
        if key == 'up':
            self._move(-1)
            return True
        if key == 'down':
            self._move(1)
            return True
        if key == 'pgup':
            for _ in range(8):
                self._move(-1)
            return True
        if key == 'pgdn':
            for _ in range(8):
                self._move(1)
            return True
        if key == 'home':
            selectable = self._selectable(rows)
            if selectable:
                self.row = selectable[0]
                self.row_key = rows[self.row].key
            return True
        if key == 'end':
            selectable = self._selectable(rows)
            if selectable:
                self.row = selectable[-1]
                self.row_key = rows[self.row].key
            return True
        if key in ('enter', 'right'):
            if 0 <= self.row < len(rows):
                self._activate(rows[self.row])
            return True
        if key == 'left':
            if self.expanded:
                node_key = self.expanded.pop()
                snapshot = self.snapshots.pop(node_key, None)
                if snapshot is not None:
                    self._restore(snapshot)
                self._refocus(node_key)
            return True
        if key == 'ctrl+s':
            self.on_change()
            self.closed = True
            if self.on_close is not None:
                self.on_close()
            return True
        return True

    def _activate(self, row: Row) -> None:
        node = row.node
        if node is None:
            return
        if row.kind == 'option' and row.option is not None:
            self._set(node, row.option.value)
            self.on_change()
            if row.option.children:
                if node.key not in self.expanded:
                    self.expanded.append(node.key)
                    self.snapshots.setdefault(node.key, self._snapshot_for(node))
                self._refocus(f'{node.key}::{row.option.value}')
            else:
                self._confirm(node)
            return
        if node.kind in ('choice', 'bool'):
            if node.key in self.expanded:
                self._confirm(node)
            else:
                self.expanded.append(node.key)
                self.snapshots[node.key] = self._snapshot_for(node)
            return
        if node.kind == 'number':
            self._begin_edit(node, str(self._get(node)))
            return
        if node.kind == 'text':
            self._begin_edit(node, self._edit_text_for(node))
            return
        if node.kind == 'list':
            self._begin_edit(node, self._edit_text_for(node))
            return
        if node.kind == 'action' and node.action is not None:
            node.action()
            return

    def _confirm(self, node: SettingNode) -> None:
        if node.key in self.expanded:
            self.expanded.remove(node.key)
            self.snapshots.pop(node.key, None)
        # Keep the cursor on the item that was just confirmed instead of
        # letting the stale option key fall back to the top of the tree.
        self._refocus(node.key)
        self.on_change()

    def _edit_text_for(self, node: SettingNode) -> str:
        value = self._get(node)
        if isinstance(value, list):
            return ', '.join(str(item) for item in value)
        return '' if value is None else str(value)

    def _begin_edit(self, node: SettingNode, initial: str) -> None:
        self.editing = True
        self.edit_key = node.key
        self.edit_value = initial
        self.edit_error = ''
        self.edit_select_all = True

    def _handle_edit(self, event: KeyEvent) -> bool:
        node = self._find_node(self.edit_key)
        if event.kind == 'paste':
            text = event.text.replace('\n', ' ')
            if self.edit_select_all:
                self.edit_value = text
                self.edit_select_all = False
            else:
                self.edit_value += text
            return True
        if event.kind == 'key' and is_paste_key(event.key):
            pasted = get_clipboard().replace('\n', ' ')
            if self.edit_select_all:
                self.edit_value = pasted
                self.edit_select_all = False
            else:
                self.edit_value += pasted
            return True
        if event.kind == 'text':
            if self.edit_select_all:
                self.edit_value = ''
                self.edit_select_all = False
            self.edit_value += event.text
            return True
        if event.kind != 'key':
            return True
        if event.key == 'esc':
            self.editing = False
            self.edit_error = ''
            return True
        if event.key == 'backspace':
            if self.edit_select_all:
                self.edit_value = ''
                self.edit_select_all = False
            else:
                self.edit_value = self.edit_value[:-1]
            return True
        if event.key == 'ctrl+u':
            self.edit_value = ''
            self.edit_select_all = False
            return True
        if event.key == 'enter':
            if node is None:
                self.editing = False
                return True
            if self._apply_edit(node, self.edit_value):
                self.editing = False
                self.edit_error = ''
                self.on_change()
            else:
                self.edit_error = 'invalid value'
            return True
        return True

    def _find_node(self, key: str) -> SettingNode | None:
        found: list[SettingNode] = []

        def walk(nodes: list[SettingNode]) -> None:
            for node in nodes:
                if node.key == key:
                    found.append(node)
                for choice in node.choices:
                    walk(choice.children)

        walk(self.nodes)
        return found[0] if found else None

    def _apply_edit(self, node: SettingNode, text: str) -> bool:
        if node.kind == 'number':
            try:
                value = int(text.strip())
            except ValueError:
                return False
            value = max(node.minimum, min(node.maximum, value))
            self._set(node, value)
            return True
        if node.kind == 'text':
            if node.path in ('thinking.rotation',):
                frames = [item.strip() for item in text.split(',') if item.strip()]
                if not frames:
                    return False
                self._set(node, frames)
                return True
            if node.path in ('thinking.rotation_ms',):
                try:
                    values = [int(item.strip()) for item in text.split(',')
                              if item.strip()]
                except ValueError:
                    return False
                if not values:
                    return False
                rotation = list(self.config.thinking.rotation)
                values = values[:len(rotation)]
                while len(values) < len(rotation):
                    values.append(values[-1] if values else 400)
                self._set(node, [max(50, min(5_000, value)) for value in values])
                return True
            if node.path == 'skills.extra_paths':
                parts = [item.strip() for item in text.replace(';', os.pathsep).split(os.pathsep)
                         if item.strip()]
                self._set(node, parts)
                return True
            self._set(node, text)
            return True
        if node.kind == 'list':
            parts = [item.strip() for item in text.split(',') if item.strip()]
            self._set(node, parts)
            return True
        return False

    # ── rendering ───────────────────────────────────────────
    def render(self, canvas: Canvas, width: int, height: int,
               palette: Palette, now: float) -> None:
        x, y, w, h = _panel_geometry(width, height, min(96, width - 6),
                                     desired_height=min(max(12, height - 6),
                                                        height - 2))
        _draw_backdrop(canvas, width, height, palette)
        _draw_frame(canvas, x, y, w, h, palette, 'Settings')
        rows = self._flatten()
        self._sync_cursor(rows)
        body_top = y + 2
        body_height = max(1, h - 5)
        if rows:
            if self.row < self.top:
                self.top = self.row
            elif self.row >= self.top + body_height:
                self.top = self.row - body_height + 1
            self.top = max(0, min(self.top, len(rows) - 1))
        row_y = body_top
        for index in range(self.top, min(len(rows), self.top + body_height)):
            row = rows[index]
            selected = index == self.row and row.kind in ('setting', 'option')
            bg = palette.selection_bg if selected else None
            if bg is not None:
                canvas.fill_rect(x + 1, row_y, w - 2, 1, bg)
            indent = 2 + row.depth * 2
            if row.kind == 'group':
                canvas.draw_text(x + 2, row_y, truncate_text(row.node.label, w - 4),
                                 palette.accent, None, BOLD)
            elif row.kind == 'label':
                label = row.node.label + ':'
                canvas.draw_text(x + indent, row_y, truncate_text(label, w // 2),
                                 palette.subtext, bg)
                value = row.node.value_text(self.config) if row.node.value_text else ''
                label_width = text_width(label)
                canvas.draw_text(x + indent + label_width + 1, row_y,
                                 truncate_text(value, max(1, w - indent - label_width - 3)),
                                 palette.muted, bg, DIM)
            elif row.kind == 'setting':
                expanded = row.node.key in self.expanded
                marker = '▾ ' if expanded else '▸ '
                canvas.draw_text(x + indent, row_y, marker,
                                 palette.accent if expanded else palette.muted, bg)
                label = row.node.label
                canvas.draw_text(x + indent + 2, row_y,
                                 truncate_text(label, max(8, w // 2)),
                                 palette.text, bg, BOLD if selected else 0)
                value = self._display_value(row.node)
                if value:
                    value_width = text_width(value)
                    value_x = x + w - 3 - value_width
                    if value_x > x + indent + 4 + text_width(label):
                        canvas.draw_text(value_x, row_y, value,
                                         palette.subtext, bg)
            elif row.kind == 'option':
                current = self._get(row.node)
                chosen = _same(row.option.value, current)
                glyph = '● ' if chosen else '○ '
                canvas.draw_text(x + indent, row_y, glyph,
                                 palette.accent if chosen else palette.muted, bg)
                canvas.draw_text(x + indent + 2, row_y,
                                 truncate_text(row.option.label, w - indent - 6),
                                 palette.text, bg, BOLD if selected else 0)
                if row.option.description:
                    desc_x = x + indent + 3 + text_width(row.option.label)
                    if desc_x < x + w - 4:
                        canvas.draw_text(desc_x, row_y, truncate_text(
                            row.option.description, x + w - 4 - desc_x),
                            palette.muted, bg, DIM)
            if selected and self.editing and row.kind == 'setting' \
                    and row.node.key == self.edit_key:
                editor = f'  {self.edit_value}█'
                canvas.draw_text(x + w // 2, row_y, truncate_text(
                    editor, max(1, w // 2 - 2)), palette.accent, bg)
            row_y += 1
        if self.edit_error:
            canvas.draw_text(x + 2, y + h - 3, self.edit_error, palette.error)
        hint = ('type a value · enter apply · esc cancel' if self.editing
                else '↑↓ move · enter expand/select · esc back · ctrl+s save')
        _draw_hint(canvas, x, y + h - 2, w, hint, palette)

    def _display_value(self, node: SettingNode) -> str:
        if node.kind == 'bool':
            return 'on' if self._get(node) else 'off'
        if node.kind in ('number',):
            return str(self._get(node))
        if node.kind == 'text':
            value = self._get(node)
            if isinstance(value, list):
                return f'{len(value)} items'
            text = '' if value is None else str(value)
            return truncate_text(text.replace('\n', ' '), 30)
        if node.kind == 'list':
            value = self._get(node) or []
            return f'{len(value)} items'
        if node.kind == 'choice':
            current = self._get(node)
            for choice in node.choices:
                if _same(choice.value, current):
                    return choice.label
            return str(current) if current is not None else ''
        return ''


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    return left == right
