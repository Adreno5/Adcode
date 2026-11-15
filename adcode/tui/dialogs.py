"""Modal overlays: selection lists, multi-select, prompts and confirmations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from adcode.theme import Palette
from adcode.tui.canvas import BOLD, DIM, Canvas, text_width, truncate_text
from adcode.tui.input import KeyEvent, get_clipboard, is_paste_key
from adcode.tui.markdown import render_markdown
from adcode.tui.text import Line, wrap_lines


@dataclass
class DialogItem:
    value: Any
    label: str
    description: str = ''
    selected: bool = False


class Overlay:
    title = ''

    def __init__(self) -> None:
        self.closed = False
        self._cursor: tuple[int, int] | None = None

    def handle(self, event: KeyEvent) -> bool:
        return False

    def render(self, canvas: Canvas, width: int, height: int,
               palette: Palette, now: float) -> None:
        raise NotImplementedError

    def cursor(self) -> tuple[int, int] | None:
        return self._cursor

    def close(self) -> None:
        self.closed = True


def _panel_geometry(width: int, height: int, desired: int,
                    desired_height: int | None = None) -> tuple[int, int, int, int]:
    panel_width = min(max(desired, 40), max(20, width - 4))
    if desired_height is None:
        panel_height = min(max(7, height - 4), max(6, height - 2))
    else:
        panel_height = min(max(7, desired_height), max(6, height - 2))
    x = max(0, (width - panel_width) // 2)
    y = max(0, (height - panel_height) // 2)
    return x, y, panel_width, panel_height


def _draw_backdrop(canvas: Canvas, width: int, height: int,
                   palette: Palette) -> None:
    canvas.fill_rect(0, 0, width, height, (0, 0, 0), alpha=170)


def _draw_frame(canvas: Canvas, x: int, y: int, w: int, h: int,
                palette: Palette, title: str) -> None:
    canvas.box(x, y, w, h, palette.border, bg=palette.panel, alpha=255,
               title=title, title_color=palette.accent)


def _draw_hint(canvas: Canvas, x: int, y: int, w: int, hint: str,
               palette: Palette) -> None:
    canvas.draw_text(x + 1, y, truncate_text(hint, max(1, w - 2)),
                     palette.muted, None, DIM)


def _draw_spans(canvas: Canvas, x: int, y: int, line: Line,
                palette: Palette) -> None:
    """Draw markdown spans, resolving semantic colour roles."""
    from adcode.theme import resolve_role
    cursor = x
    for span in line:
        fg = resolve_role(palette, span.fg, 'text')
        bg = resolve_role(palette, span.bg, 'text') if span.bg is not None else None
        cursor += canvas.draw_text(cursor, y, span.text, fg, bg,
                                   span.attrs, span.alpha)


class MessageDialog(Overlay):
    def __init__(self, title: str, message: str,
                 on_close: Callable[[], None] | None = None) -> None:
        super().__init__()
        self.title = title
        self.message = message
        self.on_close = on_close

    def handle(self, event: KeyEvent) -> bool:
        if event.kind == 'key' and event.key in ('enter', 'esc', 'q'):
            self.closed = True
            if self.on_close is not None:
                self.on_close()
            return True
        return True

    def render(self, canvas: Canvas, width: int, height: int,
               palette: Palette, now: float) -> None:
        lines = wrap_lines(render_markdown(self.message, 64), 64)
        x, y, w, h = _panel_geometry(width, height, 72,
                                     desired_height=min(6 + len(lines), height - 2))
        _draw_backdrop(canvas, width, height, palette)
        _draw_frame(canvas, x, y, w, h, palette, self.title)
        body = wrap_lines(render_markdown(self.message, max(10, w - 6)), max(10, w - 6))
        for index, line in enumerate(body[:h - 5]):
            _draw_spans(canvas, x + 3, y + 2 + index, line, palette)
        _draw_hint(canvas, x, y + h - 2, w, 'enter / esc close', palette)


class SelectDialog(Overlay):
    def __init__(self, title: str, items: list[DialogItem],
                 on_select: Callable[[Any], None],
                 on_cancel: Callable[[], None] | None = None,
                 allow_filter: bool = True, width: int = 72,
                 footer: str = '') -> None:
        super().__init__()
        self.title = title
        self.items = items
        self.on_select = on_select
        self.on_cancel = on_cancel
        self.allow_filter = allow_filter
        self.desired_width = width
        self.footer = footer
        self.filter = ''
        self.index = 0
        self.top = 0

    def _visible_items(self) -> list[DialogItem]:
        if not self.filter:
            return self.items
        lowered = self.filter.lower()
        return [
            item for item in self.items
            if lowered in item.label.lower() or lowered in item.description.lower()
        ]

    def handle(self, event: KeyEvent) -> bool:
        if event.kind == 'paste':
            if self.allow_filter:
                self.filter += event.text.replace('\r', '').replace('\n', ' ')
                self.index = 0
            return True
        if event.kind == 'text':
            if self.allow_filter:
                self.filter += event.text
                self.index = 0
            return True
        if event.kind != 'key':
            return True
        key = event.key
        visible = self._visible_items()
        if key == 'esc':
            self.closed = True
            if self.on_cancel is not None:
                self.on_cancel()
            return True
        if key == 'up':
            if visible:
                self.index = (self.index - 1) % len(visible)
            return True
        if key == 'down':
            if visible:
                self.index = (self.index + 1) % len(visible)
            return True
        if key == 'pgup':
            self.index = max(0, self.index - 8)
            return True
        if key == 'pgdn':
            if visible:
                self.index = min(len(visible) - 1, self.index + 8)
            return True
        if key == 'enter':
            if visible:
                self.closed = True
                self.on_select(visible[min(self.index, len(visible) - 1)].value)
            return True
        if key == 'backspace':
            if self.filter:
                self.filter = self.filter[:-1]
                self.index = 0
            return True
        if is_paste_key(key):
            self._paste_into_filter()
            return True
        return True

    def _paste_into_filter(self) -> None:
        """Paste the clipboard into the filter box."""
        if not self.allow_filter:
            return
        text = get_clipboard()
        if not text:
            return
        self.filter += text.replace('\r', '').replace('\n', ' ')
        self.index = 0

    def render(self, canvas: Canvas, width: int, height: int,
               palette: Palette, now: float) -> None:
        x, y, w, h = _panel_geometry(width, height, self.desired_width,
                                     desired_height=min(
                                         max(9, len(self.items) + 6), height - 2))
        _draw_backdrop(canvas, width, height, palette)
        visible = self._visible_items()
        total = len(visible)
        if total == 0:
            _draw_frame(canvas, x, y, w, h, palette, self.title)
            canvas.draw_text(x + 3, y + 2, '(nothing to select)', palette.muted)
            _draw_hint(canvas, x, y + h - 2, w, 'esc close', palette)
            return
        self.index = max(0, min(self.index, total - 1))
        body_height = h - 5
        if body_height < 1:
            body_height = 1
        if self.index < self.top:
            self.top = self.index
        elif self.index >= self.top + body_height:
            self.top = self.index - body_height + 1
        title = f'{self.title}  {self.index + 1}/{total}'
        _draw_frame(canvas, x, y, w, h, palette, title)
        row = y + 2
        for offset in range(body_height):
            position = self.top + offset
            if position >= total:
                break
            item = visible[position]
            selected = position == self.index
            bg = palette.selection_bg if selected else None
            fg = palette.selection_fg if selected else palette.text
            if bg is not None:
                canvas.fill_rect(x + 1, row, w - 2, 1, bg)
            marker = '› ' if selected else '  '
            canvas.draw_text(x + 2, row, marker,
                             palette.accent if selected else palette.muted,
                             bg, BOLD if selected else 0)
            label_width = max(8, w - 6 - text_width(item.description))
            label = truncate_text(item.label, label_width)
            canvas.draw_text(x + 4, row, label, fg, bg, BOLD if selected else 0)
            if item.description:
                desc_x = x + 5 + text_width(label)
                canvas.draw_text(desc_x, row, truncate_text(
                    item.description, max(1, x + w - 3 - desc_x)),
                    palette.muted, bg, DIM)
            row += 1
        if self.allow_filter and self.filter:
            canvas.draw_text(x + 2, y + h - 3, 'filter: ' + self.filter,
                             palette.subtext, None, DIM)
        hint = self.footer or '↑↓ move · enter select · esc cancel'
        if self.allow_filter:
            hint += ' · type to filter'
        _draw_hint(canvas, x, y + h - 2, w, hint, palette)


class MultiSelectDialog(Overlay):
    def __init__(self, title: str, items: list[DialogItem],
                 on_submit: Callable[[list[Any]], None],
                 on_cancel: Callable[[], None] | None = None,
                 allow_filter: bool = True, allow_manual: bool = False,
                 manual_label: str = 'Add manually', width: int = 78) -> None:
        super().__init__()
        self.title = title
        self.items = items
        self.on_submit = on_submit
        self.on_cancel = on_cancel
        self.allow_filter = allow_filter
        self.allow_manual = allow_manual
        self.manual_label = manual_label
        self.desired_width = width
        self.filter = ''
        self.index = 0
        self.top = 0
        self.manual_mode = False
        self.manual_text = ''

    def _visible(self) -> list[DialogItem]:
        items = list(self.items)
        if self.filter:
            lowered = self.filter.lower()
            items = [item for item in items
                     if lowered in item.label.lower()
                     or lowered in item.description.lower()]
        if self.allow_manual:
            items.append(DialogItem('__manual__', f'＋ {self.manual_label}',
                                    'type a model id', False))
        return items

    def handle(self, event: KeyEvent) -> bool:
        visible = self._visible()
        if self.manual_mode:
            if event.kind == 'paste':
                self.manual_text += event.text.replace('\r', '').replace('\n', '')
                return True
            if event.kind == 'text':
                self.manual_text += event.text
                return True
            if event.kind == 'key':
                if event.key == 'esc':
                    self.manual_mode = False
                    self.manual_text = ''
                elif event.key == 'backspace':
                    self.manual_text = self.manual_text[:-1]
                elif is_paste_key(event.key):
                    pasted = get_clipboard()
                    if pasted:
                        self.manual_text += pasted.replace('\r', '').replace('\n', '')
                elif event.key == 'enter':
                    value = self.manual_text.strip()
                    if value:
                        self.items.append(DialogItem(value, value, 'manual', True))
                        self.filter = ''
                        self.index = len(self.items) - 1
                    self.manual_mode = False
                    self.manual_text = ''
            return True
        if event.kind == 'paste':
            if self.allow_filter:
                self.filter += event.text.replace('\r', '').replace('\n', ' ')
                self.index = 0
            return True
        if event.kind == 'key':
            key = event.key
            if key == 'esc':
                self.closed = True
                if self.on_cancel is not None:
                    self.on_cancel()
                return True
            if key == 'up':
                if visible:
                    self.index = (self.index - 1) % len(visible)
                return True
            if key == 'down':
                if visible:
                    self.index = (self.index + 1) % len(visible)
                return True
            if key in (' ', 'space'):
                self._toggle(visible)
                return True
            if key == 'enter':
                item = visible[min(self.index, len(visible) - 1)] if visible else None
                if item is not None and item.value == '__manual__':
                    self.manual_mode = True
                    return True
                selected = [candidate.value for candidate in self.items
                            if candidate.selected and candidate.value != '__manual__']
                self.closed = True
                self.on_submit(selected)
                return True
            if key == 'backspace':
                if self.filter:
                    self.filter = self.filter[:-1]
                    self.index = 0
                return True
            if is_paste_key(key):
                if self.allow_filter:
                    pasted = get_clipboard()
                    if pasted:
                        self.filter += pasted.replace('\r', '').replace('\n', ' ')
                        self.index = 0
                return True
            if key == 'pgup':
                self.index = max(0, self.index - 10)
                return True
            if key == 'pgdn':
                if visible:
                    self.index = min(len(visible) - 1, self.index + 10)
                return True
        if event.kind == 'text':
            if event.text == 'a' and event.ctrl:
                return True
            if event.text == ' ':
                self._toggle(visible)
            elif self.allow_filter:
                self.filter += event.text
                self.index = 0
            return True
        return True

    def _toggle(self, visible: list[DialogItem]) -> None:
        if not visible:
            return
        item = visible[min(self.index, len(visible) - 1)]
        if item.value == '__manual__':
            self.manual_mode = True
            return
        item.selected = not item.selected
        for original in self.items:
            if original.value == item.value:
                original.selected = item.selected
                break

    def render(self, canvas: Canvas, width: int, height: int,
               palette: Palette, now: float) -> None:
        x, y, w, h = _panel_geometry(width, height, self.desired_width,
                                     desired_height=min(
                                         max(9, len(self._visible()) + 6), height - 2))
        _draw_backdrop(canvas, width, height, palette)
        visible = self._visible()
        total = len(visible)
        body_height = max(1, h - 6)
        if self.index < self.top:
            self.top = self.index
        elif self.index >= self.top + body_height:
            self.top = self.index - body_height + 1
        selected_count = sum(1 for item in self.items if item.selected)
        _draw_frame(canvas, x, y, w, h, palette,
                    f'{self.title}  {selected_count} selected')
        row = y + 2
        for offset in range(body_height):
            position = self.top + offset
            if position >= total:
                break
            item = visible[position]
            selected = position == self.index
            bg = palette.selection_bg if selected else None
            if bg is not None:
                canvas.fill_rect(x + 1, row, w - 2, 1, bg)
            check = '[x]' if item.selected else '[ ]'
            if item.value == '__manual__':
                check = ' + '
            canvas.draw_text(x + 2, row, check,
                             palette.accent if item.selected else palette.muted, bg)
            label = item.label
            if self.manual_mode and item.value == '__manual__':
                label = self.manual_text + '█'
            canvas.draw_text(x + 6, row, truncate_text(label, max(8, w - 10)),
                             palette.selection_fg if selected else palette.text,
                             bg, BOLD if selected else 0)
            if item.description:
                canvas.draw_text(
                    x + 7 + text_width(truncate_text(label, max(8, w - 10))), row,
                    truncate_text(item.description,
                                  max(1, x + w - 3 - (x + 7 + text_width(label)))),
                    palette.muted, bg, DIM)
            row += 1
        if self.allow_filter and self.filter:
            canvas.draw_text(x + 2, y + h - 3, 'filter: ' + self.filter,
                             palette.subtext, None, DIM)
        hint = '↑↓ move · space toggle · enter confirm · esc cancel'
        if self.allow_manual:
            hint += ' · + adds a custom entry'
        _draw_hint(canvas, x, y + h - 2, w, hint, palette)


class CommandPalette(Overlay):
    """A searchable list of commands, opened with ctrl+p.

    The palette is a filter box plus a list: typing narrows the commands,
    enter runs the selected one and esc closes the panel.
    """

    def __init__(self, title: str, items: list[DialogItem],
                 on_select: Callable[[Any], None],
                 on_cancel: Callable[[], None] | None = None,
                 width: int = 72) -> None:
        super().__init__()
        self.title = title
        self.items = items
        self.on_select = on_select
        self.on_cancel = on_cancel
        self.desired_width = width
        self.filter = ''
        self.index = 0
        self.top = 0

    def _visible(self) -> list[DialogItem]:
        if not self.filter:
            return self.items
        needle = self.filter.lower()
        return [item for item in self.items
                if needle in item.label.lower()
                or needle in item.description.lower()]

    def handle(self, event: KeyEvent) -> bool:
        visible = self._visible()
        if event.kind == 'mouse':
            step = _wheel_step(event)
            if step and visible:
                self.index = max(0, min(len(visible) - 1, self.index + step * 3))
            return True
        if event.kind == 'paste':
            self.filter += event.text.replace('\r', '').replace('\n', ' ')
            self.index = 0
            return True
        if event.kind == 'text':
            self.filter += event.text
            self.index = 0
            return True
        if event.kind != 'key':
            return True
        key = event.key
        if key == 'esc':
            self.closed = True
            if self.on_cancel is not None:
                self.on_cancel()
            return True
        if key == 'up':
            if visible:
                self.index = (self.index - 1) % len(visible)
            return True
        if key == 'down':
            if visible:
                self.index = (self.index + 1) % len(visible)
            return True
        if key == 'pgup':
            self.index = max(0, self.index - 8)
            return True
        if key == 'pgdn':
            if visible:
                self.index = min(len(visible) - 1, self.index + 8)
            return True
        if key == 'backspace':
            self.filter = self.filter[:-1]
            self.index = 0
            return True
        if key == 'enter':
            if visible:
                self.closed = True
                self.on_select(visible[min(self.index, len(visible) - 1)].value)
            return True
        return True

    def render(self, canvas: Canvas, width: int, height: int,
               palette: Palette, now: float) -> None:
        visible = self._visible()
        panel_width = min(self.desired_width, max(30, width - 4))
        rows = min(len(visible), 12)
        # border + search + spacer + list + hint + border
        panel_height = min(max(6, rows + 5), max(6, height - 4))
        x = max(0, (width - panel_width) // 2)
        y = max(0, min(4, (height - panel_height) // 2))
        _draw_backdrop(canvas, width, height, palette)
        _draw_frame(canvas, x, y, panel_width, panel_height, palette, self.title)

        inner = max(10, panel_width - 6)
        search = self.filter or ''
        canvas.draw_text(x + 3, y + 1, '> ', palette.accent, None, BOLD)
        canvas.draw_text(x + 5, y + 1, truncate_text(search, inner - 1),
                         palette.text)
        caret = x + 5 + text_width(truncate_text(search, inner - 1))
        self._cursor = (min(x + panel_width - 2, caret), y + 1)

        body_height = max(1, panel_height - 5)
        if visible:
            self.index = max(0, min(self.index, len(visible) - 1))
        if self.index < self.top:
            self.top = self.index
        elif self.index >= self.top + body_height:
            self.top = self.index - body_height + 1
        row = y + 3
        for offset in range(body_height):
            position = self.top + offset
            if position >= len(visible):
                break
            item = visible[position]
            selected = position == self.index
            bg = palette.selection_bg if selected else None
            if bg is not None:
                canvas.fill_rect(x + 1, row, panel_width - 2, 1, bg)
            marker = '› ' if selected else '  '
            canvas.draw_text(x + 2, row, marker,
                             palette.accent if selected else palette.muted, bg)
            canvas.draw_text(x + 4, row,
                             truncate_text(item.label, max(8, inner - 4)),
                             palette.selection_fg if selected else palette.text,
                             bg, BOLD if selected else 0)
            if item.description:
                label_width = text_width(truncate_text(item.label, max(8, inner - 4)))
                desc_x = x + 5 + label_width
                canvas.draw_text(desc_x, row, truncate_text(
                    item.description, max(1, x + panel_width - 2 - desc_x)),
                    palette.muted, bg, DIM)
            row += 1
        if not visible:
            canvas.draw_text(x + 3, y + 3, 'no matching command', palette.muted)
        _draw_hint(canvas, x, y + panel_height - 2, panel_width,
                   '↑↓ move · enter run · esc close', palette)


class TextPrompt(Overlay):
    def __init__(self, title: str, label: str, initial: str,
                 on_submit: Callable[[str], None],
                 on_cancel: Callable[[], None] | None = None,
                 hint: str = '', password: bool = False,
                 width: int = 72) -> None:
        super().__init__()
        self.title = title
        self.label = label
        self.text = initial
        self.cursor_pos = len(initial)
        self.on_submit = on_submit
        self.on_cancel = on_cancel
        self.hint = hint
        self.password = password
        self.desired_width = width

    def _insert(self, text: str) -> None:
        clean = text.replace('\r', '').replace('\n', '')
        self.text = self.text[:self.cursor_pos] + clean + self.text[self.cursor_pos:]
        self.cursor_pos += len(clean)

    def handle(self, event: KeyEvent) -> bool:
        if event.kind == 'paste':
            self._insert(event.text)
            return True
        if event.kind == 'text':
            self._insert(event.text)
            return True
        if event.kind != 'key':
            return True
        key = event.key
        if key == 'esc':
            self.closed = True
            if self.on_cancel is not None:
                self.on_cancel()
            return True
        if key == 'enter':
            self.closed = True
            self.on_submit(self.text)
            return True
        if key == 'left':
            self.cursor_pos = max(0, self.cursor_pos - 1)
        elif key == 'right':
            self.cursor_pos = min(len(self.text), self.cursor_pos + 1)
        elif key == 'home':
            self.cursor_pos = 0
        elif key == 'end':
            self.cursor_pos = len(self.text)
        elif key == 'backspace':
            if self.cursor_pos > 0:
                self.text = (self.text[:self.cursor_pos - 1]
                             + self.text[self.cursor_pos:])
                self.cursor_pos -= 1
        elif key == 'delete':
            self.text = self.text[:self.cursor_pos] + self.text[self.cursor_pos + 1:]
        elif key == 'ctrl+a':
            self.cursor_pos = len(self.text)
        elif key == 'ctrl+u':
            self.text = ''
            self.cursor_pos = 0
        elif is_paste_key(key):
            self._insert(get_clipboard())
        return True

    def render(self, canvas: Canvas, width: int, height: int,
               palette: Palette, now: float) -> None:
        x, y, w, h = _panel_geometry(width, height, max(self.desired_width, 50),
                                     desired_height=10)
        _draw_backdrop(canvas, width, height, palette)
        _draw_frame(canvas, x, y, w, h, palette, self.title)
        canvas.draw_text(x + 3, y + 2, self.label, palette.subtext, None, BOLD)
        inner_width = max(8, w - 6)
        shown = '•' * len(self.text) if self.password else self.text
        # Scroll the input horizontally to keep the caret visible.
        start = 0
        while text_width(shown[start:self.cursor_pos]) > inner_width - 1 and start < self.cursor_pos:
            start += 1
        visible = truncate_text(shown[start:], inner_width, ellipsis='')
        canvas.draw_text(x + 3, y + 4, visible, palette.text)
        caret_col = x + 3 + text_width(shown[start:self.cursor_pos])
        self._cursor = (min(x + w - 2, caret_col), y + 4)
        if self.hint:
            canvas.draw_text(x + 3, y + 6, truncate_text(self.hint, inner_width),
                             palette.muted, None, DIM)
        _draw_hint(canvas, x, y + h - 2, w, 'enter confirm · esc cancel', palette)


class ConfirmDialog(Overlay):
    def __init__(self, title: str, message: str,
                 on_result: Callable[[bool], None]) -> None:
        super().__init__()
        self.title = title
        self.message = message
        self.on_result = on_result

    def handle(self, event: KeyEvent) -> bool:
        # Printable letters arrive as text events; accept y/n there too.
        if event.kind == 'text':
            if event.text.lower() == 'y':
                self.closed = True
                self.on_result(True)
            elif event.text.lower() == 'n':
                self.closed = True
                self.on_result(False)
            return True
        if event.kind == 'key':
            if event.key in ('enter', 'y'):
                self.closed = True
                self.on_result(True)
                return True
            if event.key in ('esc', 'n'):
                self.closed = True
                self.on_result(False)
                return True
            if event.key == 'ctrl+c':
                self.closed = True
                self.on_result(False)
                return True
        return True

    def render(self, canvas: Canvas, width: int, height: int,
               palette: Palette, now: float) -> None:
        lines = wrap_lines(render_markdown(self.message, 56), 56)
        x, y, w, h = _panel_geometry(width, height, 64,
                                     desired_height=min(8 + len(lines), height - 2))
        _draw_backdrop(canvas, width, height, palette)
        _draw_frame(canvas, x, y, w, h, palette, self.title)
        body = wrap_lines(render_markdown(self.message, max(10, w - 6)), max(10, w - 6))
        for index, line in enumerate(body[:h - 5]):
            _draw_spans(canvas, x + 3, y + 2 + index, line, palette)
        _draw_hint(canvas, x, y + h - 2, w, 'enter / y confirm · esc / n cancel',
                   palette)
