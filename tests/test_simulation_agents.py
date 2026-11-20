"""Sub-agent visualisation: the live strip, the browser and the output view.

The root agent spawns a real child through the ``spawn_agent`` tool; the child
streams from the fake gateway and is held mid-stream so the panel can be read
while it runs.
"""

from __future__ import annotations

import re
import threading

import pytest

from bench.script import ToolCall, Turn


def _enable_agents(config) -> None:
    """The harness turns sub-agents off by default; these tests need them."""
    config.agents.enabled = True


@pytest.fixture()
def agent_harness(harness_factory):
    def make(**kwargs):
        kwargs.setdefault('config_mutator', _enable_agents)
        return harness_factory(**kwargs)
    return make


class ScoutResponder:
    """Root delegates once, then wraps up; the child reports with a hold."""

    def __init__(self, hold: threading.Event | None = None) -> None:
        self.hold = hold
        self.child_turns = 0

    def respond(self, request):
        if 'role: sub-agent' in request.system_text:
            self.child_turns += 1
            return Turn(thinking='reading the repository layout',
                        text='scout: all clear, nothing broken',
                        hold=self.hold)
        if any(message.get('role') == 'tool' for message in request.messages):
            return Turn(text='scout reported back')
        return Turn(tool_calls=[ToolCall(
            name='spawn_agent',
            args={'task_name': 'scout', 'message': 'look around the repo'},
            id='call_spawn_1')])


def _prompt_row(frame) -> int:
    for row in range(len(frame.lines) - 1, -1, -1):
        if frame.lines[row].startswith('>'):
            return row
    return -1


def _strips(frame) -> list[str]:
    """The panel rows directly above the prompt (agents, then todo)."""
    row = _prompt_row(frame) - 1
    rows: list[str] = []
    while row >= 0 and frame.lines[row].startswith('│'):
        rows.append(frame.lines[row])
        row -= 1
    return list(reversed(rows))



def _wait_no_agents(harness, timeout: float = 20.0) -> None:
    """The strip disappears once no sub-agent is running any more."""
    harness.wait_until(
        lambda: harness.screen.latest is not None
        and not any('agents' in line for line in _strips(harness.screen.latest)),
        timeout=timeout, description='the sub-agent panel to clear')


def _run_turn(harness, text: str, expect_requests: int = 1) -> None:
    harness.type_text(text)
    harness.wait_prompt(text)
    harness.press('enter')
    harness.wait_until(lambda: len(harness.requests) >= expect_requests,
                       timeout=20.0, description='a model request')
    harness.wait_idle(30.0)


def test_a_running_sub_agent_shows_above_the_prompt(agent_harness):
    harness = agent_harness()
    hold = threading.Event()
    harness.gateway.responder = ScoutResponder(hold)

    _run_turn(harness, 'delegate the survey', expect_requests=2)

    frame = harness.wait_for_text('scout')
    strips = _strips(frame)
    agents = [line for line in strips if 'agents' in line or 'scout' in line]
    assert agents, frame.text()
    assert 'agents' in agents[0] and '1 running' in agents[0], agents
    assert any('▸ scout' in line and 'running' in line for line in agents), agents
    assert agents[-1] != _prompt_row(frame), 'the strip sits above the input bar'
    # the footer counts sub-agents only: the root is always "running" while a
    # turn is in flight and must not be counted
    assert '1 agent' in frame.lines[-1], frame.lines[-1]
    assert not frame.vt_mismatch, 'serialized ANSI must match the canvas'

    hold.set()
    harness.wait_until(
        lambda: harness.screen.latest is not None
        and not any('scout' in line for line in _strips(harness.screen.latest)),
        timeout=20.0, description='the sub-agent panel to go away')
    frame = harness.screen.latest
    assert frame is not None
    assert ' agent' not in frame.lines[-1], frame.lines[-1]
    assert not frame.vt_mismatch


def test_the_elapsed_clock_keeps_ticking_while_the_root_is_idle(agent_harness):
    """No events arrive while a child holds, so the app must repaint itself."""
    harness = agent_harness()
    hold = threading.Event()
    harness.gateway.responder = ScoutResponder(hold)
    _run_turn(harness, 'delegate the survey', expect_requests=2)

    def elapsed() -> int:
        frame = harness.screen.latest
        match = re.search(r'scout running (\d+)s', frame.text() if frame else '')
        return int(match.group(1)) if match else -1

    harness.wait_until(lambda: elapsed() >= 1, timeout=10.0,
                       description='the strip elapsed clock to tick')
    hold.set()
    _wait_no_agents(harness)


def test_the_output_of_a_finished_agent_is_readable(agent_harness):
    harness = agent_harness()
    harness.gateway.responder = ScoutResponder()
    _run_turn(harness, 'delegate the survey', expect_requests=2)
    _wait_no_agents(harness)

    # /agents show <name> opens the agent's own words
    _run_turn(harness, '/agents show scout')
    frame = harness.wait_for_text('all clear')
    assert 'Agent · scout' in frame.text(), frame.text()


def test_the_agents_browser_lists_the_child(agent_harness):
    harness = agent_harness()
    harness.gateway.responder = ScoutResponder()
    _run_turn(harness, 'delegate the survey', expect_requests=2)
    _wait_no_agents(harness)

    # ctrl+g opens the browser; enter opens the highlighted agent
    harness.press('ctrl+g')
    harness.wait_for_text('Agents')
    frame = harness.screen.latest
    assert frame is not None and 'scout' in frame.text(), frame.text() if frame else ''

    harness.press('enter')
    harness.wait_for_text('all clear')
