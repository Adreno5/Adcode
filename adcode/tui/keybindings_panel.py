"""The bindings panel (``/keys``): view and rebind every app shortcut.

Each row shows one action from :data:`adcode.tui.keymap.BINDINGS` with its
configured key.  Enter starts a capture; the next key event becomes the
binding.  Changes are applied to ``config.keys`` right away and persisted
through ``on_change`` (``AdCodeApp.config_changed``), so they survive a
restart and reach the dispatcher immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from adcode.theme import Palette
from adcode.tui.canvas import BOLD, DIM, Canvas, text_width, truncate_text
from adcode.tui.dialogs import (
    Overlay, _draw_backdrop, _draw_frame, _draw_hint, _draw_scroll_marker,
    _panel_geometry, wheel_step,
)
from adcode.tui.input import KeyEvent
from adcode.tui.keymap import (
    BINDINGS, UNBOUND, Binding, binding_of, canonical_key, validate_key_name,
)

_LABEL_WIDTH = 26
_KEY_WIDTH = 14


@dataclass
class _Row:
    kind: str                 # header | binding
    group: str = ''
    binding: Binding | None = None


class KeybindingsPanel(Overlay):
    def __init__(self, settings: object, on_change: Callable[[], None]) -> None:
        super().__init__()
        self.title = 'Key bindings'
        self.settings = settings
        self.on_change = on_change
        self.rows = self._build_rows()
        self.index = self._selectable()[0]
        self.top = 0
        self.mode = 'list'          # list | capture | conflict
        self.message = ''
        # (action, key, other action) while a take-over is being confirmed.
        self.pending: tuple[str, str, str] | None = None

    @property
    def capture_mode(self) -> bool:
        """True while the panel owns every keypress (app suppresses quit)."""
        return self.mode in ('capture', 'conflict')

    # ── row model ───────────────────────────────────────────
    @staticmethod
    def _build_rows() -> list[_Row]:
        rows: list[_Row] = []
        group = None
        for item in BINDINGS:
            if item.group != group:
                group = item.group
                rows.append(_Row(kind='header', group=group))
            rows.append(_Row(kind='binding', binding=item))
        return rows

    def _selectable(self) -> list[int]:
        return [index for index, row in enumerate(self.rows)
                if row.kind == 'binding']

    def _move(self, delta: int) -> None:
        selectable = self._selectable()
        if not selectable:
            return
        if self.index not in selectable:
            self._select(selectable[0])
            return
        position = (selectable.index(self.index) + delta) % len(selectable)
        self.index = selectable[position]

    def _select(self, position: int) -> None:
        selectable = self._selectable()
        if selectable:
            self.index = selectable[max(0, min(position, len(selectable) - 1))]

    def _selected_binding(self) -> Binding:
        row = self.rows[self.index]
        assert row.binding is not None
        return row.binding

    def _key_of(self, action: str) -> str:
        value = getattr(self.settings, action, '')
        return '' if value == UNBOUND else str(value)

    def _holder_of(self, key: str, exclude: str) -> Binding | None:
        """The action (other than ``exclude``) currently bound to ``key``."""
        for item in BINDINGS:
            if item.action != exclude and self._key_of(item.action) == key:
                return item
        return None

    # ── event handling ──────────────────────────────────────
    def handle(self, event: KeyEvent) -> bool:
        if self.mode == 'capture':
            return self._handle_capture(event)
        if self.mode == 'conflict':
            return self._handle_conflict(event)
        return self._handle_list(event)

    def _handle_list(self, event: KeyEvent) -> bool:
        if event.kind == 'mouse':
            step = wheel_step(event)
            if step:
                self._move(step)
            return True
        if event.kind == 'paste':
            return True
        if event.kind == 'text':
            if event.text.lower() == 'r':
                self._reset_selected()
            return True
        if event.kind != 'key':
            return True
        key = event.key
        if key == 'esc':
            self.closed = True
            return True
        if key == 'up':
            self._move(-1)
            return True
        if key == 'down':
            self._move(1)
            return True
        if key == 'pgup':
            self._move(-8)
            return True
        if key == 'pgdn':
            self._move(8)
            return True
        if key == 'home':
            self._select(0)
            return True
        if key == 'end':
            self._select(len(self._selectable()) - 1)
            return True
        if key == 'enter':
            self.mode = 'capture'
            self.message = ''
            return True
        return True

    def _handle_capture(self, event: KeyEvent) -> bool:
        binding = self._selected_binding()
        if event.kind == 'mouse':
            return True
        if event.kind == 'paste':
            self.message = 'pastes are text; press a key combination'
            return True
        if event.kind == 'text':
            self.message = ('bare characters are reserved for typing; '
                            'press a ctrl/alt/function key')
            return True
        if event.kind != 'key':
            return True
        key = canonical_key(event.key)
        if key == 'esc':
            self.mode = 'list'
            self.message = ''
            return True
        error = validate_key_name(key)
        if error is not None:
            self.message = error
            return True
        other = self._holder_of(key, binding.action)
        if other is not None:
            self.pending = (binding.action, key, other.action)
            self.mode = 'conflict'
            self.message = ''
            return True
        self._apply(binding.action, key)
        return True

    def _handle_conflict(self, event: KeyEvent) -> bool:
        if event.kind != 'key':
            return True
        if event.key == 'enter':
            action, key, other = self.pending or ('', '', '')
            if action:
                setattr(self.settings, other, UNBOUND)
                self._apply(action, key)
                self.message = f'{binding_of(other).label} is now unbound'
            return True
        if event.key == 'esc':
            self.mode = 'list'
            self.message = ''
        return True

    # ── changes ─────────────────────────────────────────────
    def _apply(self, action: str, key: str) -> None:
        setattr(self.settings, action, key)
        self.mode = 'list'
        self.pending = None
        self.on_change()
        self.message = f'{binding_of(action).label} bound to {key}'

    def _reset_selected(self) -> None:
        binding = self._selected_binding()
        note = ''
        other = self._holder_of(binding.default, binding.action)
        if other is not None:
            setattr(self.settings, other.action, UNBOUND)
            note = f' · {other.label} unbound'
        setattr(self.settings, binding.action, binding.default)
        self.on_change()
        self.message = f'{binding.label} reset to {binding.default}{note}'

    # ── rendering ───────────────────────────────────────────
    def render(self, canvas: Canvas, width: int, height: int,
               palette: Palette, now: float) -> None:
        x, y, w, h = _panel_geometry(
            width, height, 78,
            desired_height=min(max(9, len(self.rows) + 6), height - 2))
        _draw_backdrop(canvas, width, height, palette)
        selectable = self._selectable()
        position = selectable.index(self.index) if self.index in selectable else 0
        total = len(selectable)
        title = f'{self.title}  {position + 1}/{total}'
        _draw_frame(canvas, x, y, w, h, palette, title)
        body_height = max(1, h - 5)
        if self.index < self.top:
            self.top = self.index
        elif self.index >= self.top + body_height:
            self.top = self.index - body_height + 1
        row = y + 2
        for offset in range(body_height):
            absolute = self.top + offset
            if absolute >= len(self.rows):
                break
            self._draw_row(canvas, x, w, row, absolute, self.rows[absolute],
                           palette)
            row += 1
        _draw_scroll_marker(canvas, x + w - 2, y + 2, body_height,
                            len(self.rows), self.top, palette)
        status = self._status_text()
        if status:
            colour = palette.accent if self.capture_mode else palette.subtext
            canvas.draw_text(x + 2, y + h - 3, truncate_text(
                status, max(1, w - 4)), colour, None, DIM)
        _draw_hint(canvas, x, y + h - 2, w, self._hint_text(), palette)

    def _draw_row(self, canvas: Canvas, x: int, w: int, row: int,
                  absolute: int, item: _Row, palette: Palette) -> None:
        if item.kind == 'header':
            canvas.draw_text(x + 2, row, item.group.upper(),
                             palette.muted, None, DIM)
            return
        binding = item.binding
        assert binding is not None
        selected = absolute == self.index
        bg = palette.selection_bg if selected else None
        fg = palette.selection_fg if selected else palette.text
        if bg is not None:
            canvas.fill_rect(x + 1, row, w - 2, 1, bg)
        marker = '› ' if selected else '  '
        canvas.draw_text(x + 2, row, marker,
                         palette.accent if selected else palette.muted,
                         bg, BOLD if selected else 0)
        label = truncate_text(binding.label, _LABEL_WIDTH)
        canvas.draw_text(x + 4, row, label, fg, bg, BOLD if selected else 0)
        desc_x = x + 4 + _LABEL_WIDTH + 2
        key_text = self._key_of(binding.action) or '(unbound)'
        key_x = max(desc_x, x + w - 2 - text_width(key_text))
        if key_x > desc_x:
            canvas.draw_text(desc_x, row, truncate_text(
                binding.description, key_x - 1 - desc_x),
                palette.muted, bg, DIM)
        key_colour = palette.muted if not self._key_of(binding.action) \
            else palette.accent
        canvas.draw_text(key_x, row, key_text, key_colour, bg,
                         BOLD if selected else 0)

    def _status_text(self) -> str:
        if self.mode == 'capture':
            label = self._selected_binding().label
            return f'press a key for «{label}» · esc cancels'
        if self.mode == 'conflict' and self.pending is not None:
            _, key, other = self.pending
            return (f'«{key}» is bound to {binding_of(other).label} -- '
                    'take over? enter yes · esc no')
        return self.message

    def _hint_text(self) -> str:
        if self.mode == 'capture':
            return 'press a key · esc cancel'
        if self.mode == 'conflict':
            return 'enter take over · esc cancel'
        return 'enter rebind · r reset · esc close'
