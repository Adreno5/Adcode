"""The approval prompt.

When the security mode is ``ask``, an effectful tool call waits for a yes
before it runs.  The question is asked once per tool: answering "always
allow" remembers it for the session, so the agent is not interrupted on
every call.

Keys:

* ``y`` / ``enter`` -- allow this call
* ``a``            -- allow this tool for the rest of the session
* ``n`` / ``esc``  -- refuse (the model is told the call was refused)
"""

from __future__ import annotations

from typing import Any, Callable

from adcode.tui.settings import ACCESS_LABELS
from adcode.tui.dialogs import (Overlay, ScrollingBody, _draw_backdrop,
                                _draw_frame, _draw_hint, _draw_spans,
                                _panel_geometry)
from adcode.tui.input import KeyEvent
from adcode.theme import Palette


class ApprovalDialog(Overlay):
    """One tool call, waiting for a decision.

    The dialog says what the call would be allowed to change: the access
    level of the project is part of the question, because "may this run"
    and "what may it touch" are the same question to the user.
    """

    def __init__(self, request: Any,
                 on_decision: Callable[[str], None], *,
                 access: str = '', project: str = '') -> None:
        super().__init__()
        self.request = request
        self.on_decision = on_decision
        self.access = str(access or '')
        self.project = str(project or '')
        self.title = 'Approve tool call'
        self.body = ScrollingBody()

    # ── what to show ────────────────────────────────────────
    def _lines(self) -> list[str]:
        request = self.request
        tool = str(getattr(request, 'tool', '') or '?')
        effect = str(getattr(request, 'effect', '') or '')
        risk = str(getattr(request, 'risk', '') or '')
        destructive = bool(getattr(request, 'destructive', False))
        preview = str(getattr(request, 'args_preview', '') or '')
        rows = [f'tool     {tool}']
        if effect:
            rows.append(f'effect   {effect}')
        if risk or destructive:
            rows.append(f'risk     {risk}{"  (irreversible)" if destructive else ""}')
        detail = str(getattr(request, 'detail', '') or '')
        if detail:
            rows.append(f'why      {detail}')
        if self.access:
            rows.append(f'access   {ACCESS_LABELS.get(self.access, self.access)}')
        if self.project:
            rows.append(f'project  {self.project}')
        if preview:
            rows.append('')
            rows.append('arguments')
            rows.append(preview)
        return rows

    def handle(self, event: KeyEvent) -> bool:
        self.body.handle(event)
        if event.kind == 'text':
            choice = event.text.lower()
            if choice in ('y', 'a', 'n'):
                self._decide(choice)
            return True
        if event.kind == 'key':
            if event.key in ('enter', 'y'):
                self._decide('y')
                return True
            if event.key == 'a':
                self._decide('a')
                return True
            if event.key in ('esc', 'n', 'ctrl+c'):
                self._decide('n')
                return True
        return True

    def _decide(self, choice: str) -> None:
        self.closed = True
        try:
            self.on_decision(choice)
        except Exception:      # noqa: BLE001 - the waiter must not hang
            pass

    def render(self, canvas, width: int, height: int,
               palette: Palette, now: float) -> None:
        text = '\n'.join(self._lines())
        preview = self.body.lines(text, 56)
        x, y, w, h = _panel_geometry(width, height, 68,
                                     desired_height=min(11 + len(preview),
                                                        height - 2))
        _draw_backdrop(canvas, width, height, palette)
        _draw_frame(canvas, x, y, w, h, palette, self.title)
        body_rows = max(1, h - 5)
        body = self.body.layout(text, max(10, w - 6), body_rows)
        top = self.body.top
        for index, line in enumerate(body[top:top + body_rows]):
            _draw_spans(canvas, x + 3, y + 2 + index, line, palette)
        _draw_hint(canvas, x, y + h - 2, w,
                   'y allow once · a always allow this tool · n refuse',
                   palette)
