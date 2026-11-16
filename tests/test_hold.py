"""The ``/hold`` mode and the interrupted-round context.

Three things decide whether a long session survives an interruption:

- the reasoning of a round that never finished stays in the history, so the
  next turn continues instead of re-planning the same work;
- an interrupt reaches the provider, so the stream stops at the next chunk;
- ``/hold`` re-sends a request whose stream broke until it works, the user
  interrupts, or the retries run out.
"""

import asyncio
import time

import pytest

import adcode.agent_loop as agent_loop
from adcode.agent_loop import AgentLoop
from adcode.config import HoldSettings
from adcode.engine import UNFINISHED_NOTE, AdCodeEngine
from adcode.goals import GoalStore
from adcode.session import DbSessionStore
from adcode.skills import SkillRegistry
from adcode.tools.registry import ToolRegistry


class ScriptedClient:
    """A client whose rounds are scripted by the last user message."""

    def __init__(self, provider=None, model=None, *, base_dir='', file_cache=None,
                 warn=None):
        self.provider = provider
        self.calls = 0
        self.script: list = []
        self.prompts: list[list[str]] = []

    async def stream(self, messages, tools=None, **kwargs):
        msgs = [m.to_dict() if hasattr(m, 'to_dict') else dict(m)
                for m in messages]
        self.prompts.append([str(m.get('content') or '')[:60] for m in msgs])
        self.calls += 1
        for event in self.script:
            yield event

    def invoke(self, messages, tools=None, **kwargs):
        raise NotImplementedError

    def list_models(self):
        return ['test-model']


def _loop(config, tmp_path, monkeypatch, client=None):
    monkeypatch.setattr(agent_loop, 'LLMClient',
                        lambda *args, **kwargs: client or ScriptedClient())
    instance = AgentLoop(
        config, ToolRegistry(),
        DbSessionStore(db_path=tmp_path / 'sessions' / 'adcode.db'),
        GoalStore(tmp_path / 'goals.json'),
        SkillRegistry(config.skills, str(tmp_path)),
        emit=lambda kind, payload: None,
    )
    instance.start()
    return instance


def wait_idle(loop, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not loop.busy and loop._work.qsize() == 0:
            time.sleep(0.1)
            if not loop.busy:
                return
        time.sleep(0.02)
    raise AssertionError('the loop did not become idle')


# ── the interrupted round keeps its reasoning ───────────────
def test_an_interrupted_round_keeps_its_reasoning(config, tmp_path,
                                                   monkeypatch):
    class Slow(ScriptedClient):
        async def stream(self, messages, tools=None, **kwargs):
            yield {'reasoning': 'I read the file and I know what to change: '
                                'the cursor is parked at the end of the prompt.'}
            await asyncio.sleep(1.0)
            yield {'finish': 'stop'}

    client = Slow()
    client.script = []
    loop = _loop(config, tmp_path, monkeypatch, client)
    try:
        loop.submit('fix the cursor')
        time.sleep(0.3)
        # The user interrupts while the model is still talking.
        loop.interrupt()
        wait_idle(loop)

        committed = loop.context.committed
        assert committed, 'the interrupted turn must not vanish'
        contents = [str(message.get('content') or '') for message in committed]
        assert any(UNFINISHED_NOTE in text for text in contents), contents
        assert any('the cursor is parked' in text for text in contents), contents
        # The user request stays in the history as well.
        assert any('fix the cursor' in text for text in contents), contents
    finally:
        loop.shutdown()


def test_a_cut_response_keeps_its_reasoning(config, tmp_path, monkeypatch):
    client = ScriptedClient()
    client.script = [
        {'reasoning': 'The plan is to move the caret back after the frame.'},
        {'finish': 'incomplete'},
    ]
    loop = _loop(config, tmp_path, monkeypatch, client)
    try:
        loop.submit('fix the cursor')
        wait_idle(loop)

        committed = loop.context.committed
        contents = [str(message.get('content') or '') for message in committed]
        assert any(UNFINISHED_NOTE in text for text in contents), contents
        assert any('move the caret back' in text for text in contents), contents
    finally:
        loop.shutdown()


# ── hold ───────────────────────────────────────────────────
def test_hold_retries_a_cut_response_until_it_works(config, tmp_path,
                                                    monkeypatch):
    client = ScriptedClient()

    class Flaky(ScriptedClient):
        async def stream(self, messages, tools=None, **kwargs):
            self.calls += 1
            if self.calls < 3:
                yield {'reasoning': 'partial'}
                yield {'finish': 'incomplete'}
                return
            yield {'reasoning': 'finally'}
            yield {'finish': 'stop'}

    flaky = Flaky()
    flaky.script = []
    loop = _loop(config, tmp_path, monkeypatch, flaky)
    try:
        config.hold = HoldSettings(enabled=True, max_retries=5, delay=0)
        loop.apply_hold()
        loop.submit('go')
        wait_idle(loop)
        assert flaky.calls == 3, flaky.calls
        # The successful attempt is the one that counts: the partial answers
        # of the failed attempts are not in the history.
        last = loop.context.committed[-1]
        assert 'finally' in str(last.get('reasoning_content') or ''), last
        assert 'partial' not in str(last.get('reasoning_content') or ''), last
    finally:
        loop.shutdown()


def test_hold_gives_up_after_the_configured_retries(config, tmp_path,
                                                    monkeypatch):
    class Broken(ScriptedClient):
        async def stream(self, messages, tools=None, **kwargs):
            self.calls += 1
            yield {'reasoning': 'partial'}
            yield {'finish': 'incomplete'}

    broken = Broken()
    broken.script = []
    loop = _loop(config, tmp_path, monkeypatch, broken)
    try:
        config.hold = HoldSettings(enabled=True, max_retries=3, delay=0)
        loop.apply_hold()
        loop.submit('go')
        wait_idle(loop)
        # One attempt plus three retries.
        assert broken.calls == 4, broken.calls
    finally:
        loop.shutdown()


def test_without_hold_a_cut_response_ends_the_turn(config, tmp_path,
                                                    monkeypatch):
    client = ScriptedClient()
    client.script = [{'reasoning': 'partial'}, {'finish': 'incomplete'}]
    loop = _loop(config, tmp_path, monkeypatch, client)
    try:
        config.hold = HoldSettings(enabled=False)
        loop.apply_hold()
        loop.submit('go')
        wait_idle(loop)
        assert client.calls == 1, client.calls
    finally:
        loop.shutdown()


def test_hold_stops_when_the_user_interrupts(config, tmp_path, monkeypatch):
    class Slow(ScriptedClient):
        async def stream(self, messages, tools=None, **kwargs):
            self.calls += 1
            yield {'reasoning': 'partial'}
            yield {'finish': 'incomplete'}
            await asyncio.sleep(0.4)

    slow = Slow()
    slow.script = []
    loop = _loop(config, tmp_path, monkeypatch, slow)
    try:
        config.hold = HoldSettings(enabled=True, max_retries=50, delay=0.1)
        loop.apply_hold()
        loop.submit('go')
        time.sleep(0.2)
        loop.interrupt()
        wait_idle(loop)
        # The retry loop stops at the interrupt instead of hammering on.
        assert slow.calls < 50, slow.calls
    finally:
        loop.shutdown()


# ── the interrupt reaches the provider ─────────────────────
def test_the_stream_stops_when_the_turn_is_interrupted(config, tmp_path,
                                                        monkeypatch):
    """The provider must not keep streaming after an interrupt."""
    chunks = []

    class Slow(ScriptedClient):
        async def stream(self, messages, tools=None, **kwargs):
            for index in range(200):
                chunks.append(index)
                yield {'reasoning': f'chunk {index}'}
                await asyncio.sleep(0.01)

    slow = Slow()
    slow.script = []
    loop = _loop(config, tmp_path, monkeypatch, slow)
    try:
        loop.submit('go')
        time.sleep(0.2)
        loop.interrupt()
        time.sleep(0.4)
        # The stream did not run to its end.
        assert len(chunks) < 200, len(chunks)
    finally:
        loop.shutdown()


def test_the_hold_settings_reach_the_engine(config, tmp_path, monkeypatch):
    loop = _loop(config, tmp_path, monkeypatch)
    try:
        config.hold = HoldSettings(enabled=True, max_retries=7, delay=0.5)
        loop.apply_hold()
        engine = loop._engine
        assert isinstance(engine, AdCodeEngine)
        assert engine.hold_enabled is True
        assert engine.hold_max_retries == 7
        assert engine.hold_delay == 0.5
    finally:
        loop.shutdown()


# ── the reasoning reaches the model on every wire ───────────
def test_the_reasoning_travels_as_text_on_the_wires_without_the_field():
    """The Responses and the Anthropic wire have no ``reasoning_content``."""
    from adcode.llm.client import _reasoning_as_text, _responses_items

    message = {'role': 'assistant', 'content': None,
               'reasoning_content': 'the plan is to move the caret'}
    # The chat wire keeps the field it knows.
    assert _reasoning_as_text(dict(message), 'chat')['reasoning_content'] \
        == 'the plan is to move the caret'
    # The others carry it as the text of the turn.
    carried = _reasoning_as_text(dict(message), 'responses')
    assert carried['content'] == 'the plan is to move the caret'
    assert 'reasoning_content' not in carried
    items = _responses_items(carried)
    assert items[0]['content'][0]['text'] == 'the plan is to move the caret'


def test_an_assistant_turn_with_text_keeps_its_reasoning_field():
    from adcode.llm.client import _reasoning_as_text

    message = {'role': 'assistant', 'content': 'the answer',
               'reasoning_content': 'the plan'}
    for wire in ('chat', 'responses', 'anthropic'):
        assert _reasoning_as_text(dict(message), wire) == message


# ── the command ─────────────────────────────────────────────
def test_the_hold_command_sets_the_settings(config):
    from adcode.commands import dispatch

    class FakeApp:
        def __init__(self) -> None:
            self.config = config
            self.notices: list[str] = []

        def notify(self, text, level='info') -> None:
            self.notices.append(text)

        def config_changed(self) -> None:
            pass

    app = FakeApp()
    dispatch(app, '/hold on')
    assert config.hold.enabled is True
    dispatch(app, '/hold 42')
    assert config.hold.max_retries == 42
    dispatch(app, '/hold off')
    assert config.hold.enabled is False
    # The bare command toggles.
    dispatch(app, '/hold')
    assert config.hold.enabled is True
