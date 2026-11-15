"""Manual, screenshot-producing scenarios for the AdCode harness.

Run one with::

    python -m harness run chat --out harness-out

Each scenario drives the real application (fake keyboard/mouse, fake local LLM
gateway, temporary config) and captures labelled frames along the way.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from harness.app import AppHarness
from harness.script import ScriptedResponder, ToolCall, Turn

LONG_ANSWER = '\n'.join(
    f'{index + 1:02d}. streamed transcript row for scroll testing -- '
    f'the quick brown fox jumps over the lazy dog.'
    for index in range(80))


@dataclass(frozen=True)
class Scenario:
    name: str
    summary: str
    run: Callable[[AppHarness], None]


def _wait_reasoning(h: AppHarness, needle: str) -> None:
    h.wait_until(
        lambda: h.phase == 'thinking' and needle in h.screen_text(),
        timeout=15.0, description='the reasoning stream to reach the screen')


def _chat(h: AppHarness) -> None:
    hold = threading.Event()
    h.gateway.responder = ScriptedResponder([Turn(
        thinking=('The gateway is a real HTTP server on localhost. '
                  'It streams reasoning first, then the answer.'),
        text=('## Fake gateway\n\n'
              'This answer travelled through a real SSE stream.\n\n'
              '- the app used the production OpenAI client\n'
              '- the request carried the real system prompt and tool schemas\n'
              '- the frame below is the composited terminal canvas'),
        hold=hold,
    )])
    h.snapshot('idle')
    h.type_text('Explain what the fake gateway does')
    h.press('enter')
    _wait_reasoning(h, 'real HTTP server on localhost')
    h.snapshot('thinking-stream')
    h.release()
    h.wait_idle(20.0)
    h.wait_for_text('Fake gateway')
    h.snapshot('answer')


def _tool_call(h: AppHarness) -> None:
    h.gateway.responder = ScriptedResponder([
        Turn(
            thinking='I should write the requested file with the write tool.',
            tool_calls=[ToolCall(
                name='write',
                args={'path': 'notes.txt',
                      'content': 'hello from a scripted tool call\n'},
                id='call_write_1',
            )],
        ),
        Turn(text=('Wrote `notes.txt` through the real tool registry. '
                   'The file is in the temporary workspace.')),
    ])
    h.snapshot('idle')
    h.type_text('Create notes.txt with a greeting')
    h.press('enter')
    h.wait_until(lambda: len(h.requests) >= 2 and h.phase == 'idle',
                 timeout=20.0,
                 description='the tool round-trip to finish')
    h.wait_for_text('notes.txt')
    h.snapshot('tool-result')


def _keyboard(h: AppHarness) -> None:
    h.type_text('helo')
    h.press('left')
    h.type_text('l')                 # hello
    h.press('home', 'delete')
    h.type_text('H')                 # Hello
    h.press('ctrl+end')
    h.type_text(' world')
    h.snapshot('edited')
    h.press('shift+home')
    h.snapshot('selection')
    h.press('ctrl+x')
    h.snapshot('cut')
    h.press('ctrl+v')
    h.press('ctrl+a')
    h.type_text('final draft')
    h.paste('\nsecond line from a bracketed paste')
    h.snapshot('pasted-multiline')


def _commands(h: AppHarness) -> None:
    h.type_text('/th')
    h.wait_for_text('Set the thinking display mode')
    h.snapshot('command-palette')
    h.press('tab')
    h.type_text('full')
    h.press('enter')
    h.wait_for_text('Thinking display: full')
    h.snapshot('command-applied')

    h.run_command('/config')
    h.wait_until(lambda: h.overlay is not None, timeout=10.0,
                 description='the settings overlay')
    h.wait_for_text('Appearance')
    h.snapshot('settings-panel')
    h.press('down')                  # select Theme
    h.press('enter')                 # expand
    h.press('down', 'down')          # Dark -> Light
    h.press('enter')                 # choose Light
    h.wait_until(
        lambda: h.app is not None and h.app.config.ui.theme == 'light',
        timeout=10.0, description='the theme to change')
    h.wait_until(
        lambda: h.app is not None and h.app.palette.name == 'light'
        and not h.app.dirty,
        timeout=10.0, description='the light frame to render')
    h.snapshot('theme-light')
    h.press('ctrl+s')
    h.wait_until(lambda: h.overlay is None and not h.app.dirty, timeout=10.0,
                 description='the settings overlay to close')
    h.snapshot('settings-saved')


def _scroll(h: AppHarness) -> None:
    h.gateway.responder = ScriptedResponder([Turn(
        thinking='A long answer is the fastest way to overflow the viewport.',
        text=LONG_ANSWER,
    )])
    h.snapshot('idle')
    h.type_text('Render a long answer')
    h.press('enter')
    h.wait_until(lambda: len(h.requests) >= 1 and h.phase == 'idle',
                 timeout=20.0, description='the long answer')
    h.wait_for_text('80. streamed transcript row')
    h.snapshot('bottom')
    h.wheel('wheel_up')
    h.wheel('wheel_up')
    h.wait_until(lambda: h.app is not None and h.app.scroll > 0,
                 timeout=10.0, description='the transcript to scroll up')
    h.snapshot('scrolled-up')
    h.wheel('wheel_down')
    h.wheel('wheel_down')
    h.wait_until(
        lambda: h.app is not None and h.app.follow,
        timeout=10.0, description='the transcript to follow the tail again')


def _resize(h: AppHarness) -> None:
    h.gateway.responder = ScriptedResponder([Turn(
        thinking='A short answer so the layout is easy to compare.',
        text=('Resize the simulated terminal; the canvas, transcript viewport, '
              'prompt and footer should all reflow.'),
    )])
    h.type_text('Show the layout')
    h.press('enter')
    h.wait_idle(20.0)
    h.wait_until(lambda: h.app is not None and not h.app.dirty, timeout=10.0,
                 description='the initial answer frame')
    h.snapshot('resize-100x30')
    h.resize(72, 20)
    h.wait_until(lambda: h.app is not None and not h.app.dirty, timeout=10.0,
                 description='the 72x20 frame')
    h.snapshot('resize-72x20')
    h.resize(120, 40)
    h.wait_until(lambda: h.app is not None and not h.app.dirty, timeout=10.0,
                 description='the 120x40 frame')
    h.snapshot('resize-120x40')


def _interrupt(h: AppHarness) -> None:
    hold = threading.Event()
    h.gateway.responder = ScriptedResponder([Turn(
        thinking='This reasoning stream is deliberately held open.',
        text='This answer should be interrupted before it is committed.',
        hold=hold,
    )])
    h.type_text('Start a slow turn')
    h.press('enter')
    _wait_reasoning(h, 'deliberately held open')
    h.snapshot('in-flight')
    h.press('esc')
    h.release()
    h.wait_idle(20.0)
    h.wait_for_text('[interrupted]')
    h.snapshot('interrupted')


SCENARIOS: dict[str, Scenario] = {
    scenario.name: scenario for scenario in [
        Scenario('chat', 'Stream reasoning and an answer from the fake gateway.',
                 _chat),
        Scenario('tool-call', 'Exercise a scripted write tool call end to end.',
                 _tool_call),
        Scenario('keyboard', 'Edit, select, cut, paste and undo with real key '
                              'sequences.', _keyboard),
        Scenario('commands', 'Drive the slash palette, settings overlay and '
                              'theme change.', _commands),
        Scenario('scroll', 'Render a long answer and scroll it with the SGR '
                            'mouse wheel.', _scroll),
        Scenario('resize', 'Re-render the frame at several terminal sizes.',
                 _resize),
        Scenario('interrupt', 'Interrupt a held-open stream with Esc.',
                 _interrupt),
    ]
}


def list_scenarios() -> list[Scenario]:
    return list(SCENARIOS.values())
