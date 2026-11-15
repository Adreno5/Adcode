import asyncio
import time

import pytest

import adcode.agent_loop as agent_loop
from adcode.agent_loop import AgentLoop
from adcode.goals import GoalStore
from adcode.session import SessionStore
from adcode.skills import SkillRegistry
from adcode.tools.registry import AnnotatedTool, ToolRegistry


class ScriptedClient:
    """Deterministic IChatClient: replays sdk-shaped stream events.

    Rounds are chosen by the last user text, like the old chat() scripts.
    """

    instances: list['ScriptedClient'] = []

    def __init__(self, provider, model=None, *, base_dir='', file_cache=None,
                 warn=None):
        self.provider = provider
        self.model = model.id if model is not None else 'test-model'
        self.base_dir = base_dir
        self.requests: list[list[dict]] = []
        self.kwargs: list[dict] = []
        ScriptedClient.instances.append(self)

    async def stream(self, messages, tools=None, **kwargs):
        # the engine passes ChatMessage objects; normalize once, like the
        # real clients do
        msgs = [m.to_dict() if hasattr(m, 'to_dict') else dict(m)
                for m in messages]
        self.requests.append(msgs)
        self.kwargs.append(kwargs)
        system = next((str(m.get('content') or '') for m in msgs
                       if m.get('role') == 'system'), '')
        last_user = next((str(m.get('content') or '') for m in reversed(msgs)
                          if m.get('role') == 'user'), '')
        if 'compactor' in system.lower():
            yield 'SUMMARY'
            yield {'finish': 'stop'}
            return
        if 'side question' in system.lower():
            yield 'aside answer'
            yield {'finish': 'stop'}
            return
        if 'use tool' in last_user and not any(
                m.get('role') == 'tool' for m in msgs):
            yield {'reasoning': 'Need a tool.'}
            yield {'tool_calls': [{'index': 0, 'id': 'c1', 'name': 'probe',
                                   'arguments': '{"value": "x"}'}]}
            yield {'finish': 'tool_calls'}
            return
        yield {'reasoning': 'Thinking...'}
        yield 'Answer '
        yield '**done**.'
        yield {'usage': {'input_tokens': 12, 'output_tokens': 4,
                         'input_tokens_details': {'cached_tokens': 5}}}
        yield {'finish': 'stop'}

    def invoke(self, messages, tools=None, **kwargs):
        raise NotImplementedError

    def list_models(self):
        return ['test-model']


@pytest.fixture()
def loop(config, tmp_path, monkeypatch):
    monkeypatch.setattr(agent_loop, 'LLMClient', ScriptedClient)
    registry = ToolRegistry()
    registry.register(AnnotatedTool(
        name='probe', description='probe',
        parameters={'type': 'object',
                    'properties': {'value': {'type': 'string'}},
                    'required': ['value']},
        handler=lambda value: 'probe:' + str(value)))
    ScriptedClient.instances = []
    instance = AgentLoop(
        config, registry, SessionStore(tmp_path / 'sessions'),
        GoalStore(tmp_path / 'goals.json'),
        SkillRegistry(config.skills, str(tmp_path)),
        emit=lambda kind, payload: None,
    )
    instance.start()
    yield instance
    instance.shutdown()


def wait_idle(loop, timeout=5.0):
    deadline = time.time() + timeout
    quiet_since = None
    while time.time() < deadline:
        idle = not loop.busy and loop._work.qsize() == 0
        if idle:
            quiet_since = quiet_since or time.time()
            if time.time() - quiet_since >= 0.1:
                return
        else:
            quiet_since = None
        time.sleep(0.01)
    raise TimeoutError('the engine never went idle')


def _kinds(loop):
    return [kind for kind, _ in loop.drain_ui_events()]


def test_simple_turn(loop):
    loop.submit('hello there')
    wait_idle(loop)
    kinds = _kinds(loop)
    for kind in ('turn_start', 'status', 'thinking_delta', 'text_delta',
                 'usage', 'turn_end'):
        assert kind in kinds, kinds
    assert _kinds(loop) == []
    items = loop.items_snapshot()
    assert [item.kind for item in items] == ['user', 'thinking', 'assistant']
    assert items[-1].text == 'Answer **done**.'
    # the committed history stays provider-shaped
    roles = [m['role'] for m in loop.context.committed]
    assert roles == ['user', 'assistant']
    snap = loop.snapshot()
    assert snap.turn == 1 and snap.total_in == 12 and snap.total_out == 4
    assert snap.provider == 'Test' and snap.model == 'test-model'


def test_tool_turn(loop):
    loop.submit('please use tool')
    wait_idle(loop)
    kinds = _kinds(loop)
    assert 'tool_start' in kinds and 'tool_end' in kinds
    items = loop.items_snapshot()
    # the tool round's reasoning lands as its own thinking item, before the
    # tool executes
    assert [item.kind for item in items] == ['user', 'thinking', 'tool',
                                             'thinking', 'assistant']
    assert items[2].output == 'probe:x'
    roles = [m['role'] for m in loop.context.committed]
    assert roles == ['user', 'assistant', 'tool', 'assistant']
    assert loop.context.committed[2]['content'] == 'probe:x'
    # the tool result reaches the next model call
    second = ScriptedClient.instances[-1].requests[-1]
    assert any(m.get('role') == 'tool' and m.get('content') == 'probe:x'
               for m in second)


def test_usage_normalization_reaches_the_event(loop):
    loop.submit('hello there')
    wait_idle(loop)
    usage = [payload for kind, payload in loop.drain_ui_events()
             if kind == 'usage']
    assert usage, 'usage events must reach the ui'
    event = usage[-1]
    assert event['input_tokens'] == 12
    assert event['cached_input_tokens'] == 5
    assert event['uncached_input_tokens'] == 7
    assert event['prefix_generation'] >= 1
    assert event['prefix_hash'].startswith('sha256:')


def test_compaction_trigger(config, tmp_path, monkeypatch):
    monkeypatch.setattr(agent_loop, 'LLMClient', ScriptedClient)
    config.context.auto_compact = True
    config.context.compact_threshold = 5
    config.selected()[1].context_window = 2_000
    loop = AgentLoop(config, ToolRegistry(), SessionStore(tmp_path / 's'),
                     GoalStore(tmp_path / 'g.json'),
                     SkillRegistry(config.skills, str(tmp_path)),
                     emit=lambda kind, payload: None)
    loop.start()
    try:
        loop.submit('hello there')
        wait_idle(loop)
        kinds = _kinds(loop)
        assert 'compaction_start' in kinds and 'compaction_end' in kinds
    finally:
        loop.shutdown()


def test_interrupt(config, tmp_path, monkeypatch):
    class SlowClient(ScriptedClient):
        async def stream(self, messages, tools=None, **kwargs):
            self.requests.append(list(messages))
            yield 'partial '
            await asyncio.sleep(1.5)
            yield 'answer'
            yield {'finish': 'stop'}

    monkeypatch.setattr(agent_loop, 'LLMClient', SlowClient)
    loop = AgentLoop(config, ToolRegistry(), SessionStore(tmp_path / 's'),
                     GoalStore(tmp_path / 'g.json'),
                     SkillRegistry(config.skills, str(tmp_path)),
                     emit=lambda kind, payload: None)
    loop.start()
    try:
        loop.submit('hello there')
        time.sleep(0.3)
        loop.interrupt()
        wait_idle(loop, timeout=10)
        events = loop.drain_ui_events()
        kinds = [kind for kind, _ in events]
        assert 'interrupted' in kinds
        turn_end = [payload for kind, payload in events if kind == 'turn_end'][-1]
        assert turn_end['status'] == 'interrupted'
        # a text-only draft is rolled back: nothing half-finished is committed
        assert loop.context.committed == []
    finally:
        loop.shutdown()


def test_aside_does_not_touch_history(loop):
    loop.submit('hello there')
    wait_idle(loop)
    loop.drain_ui_events()
    committed_before = list(loop.context.committed)
    loop.submit('quick side question?', mode='aside')
    wait_idle(loop)
    kinds = _kinds(loop)
    assert 'aside_start' in kinds and 'aside_delta' in kinds and 'aside_end' in kinds
    assert list(loop.context.committed) == committed_before
    aside = loop.items_snapshot()[-1]
    assert aside.kind == 'aside' and aside.text == 'aside answer'


def test_persistence_and_rewind(loop):
    loop.submit('hello there')
    wait_idle(loop)
    loop.save_as('persisted')
    payload = loop.payload()
    assert payload.name == 'persisted'
    assert len(payload.messages) == 2
    # rewind drops the turn and reports the dropped user text
    dropped = loop.rewind(1)
    assert dropped == 'hello there'
    assert loop.context.committed == []
    assert loop.snapshot().turn == 0
    loop.drain_ui_events()


def test_fork_and_save(loop):
    loop.submit('hello there')
    wait_idle(loop)
    name = loop.fork()
    assert name.endswith('_fork')
    loop.drain_ui_events()


def test_clear(loop):
    loop.submit('hello there')
    wait_idle(loop)
    loop.clear()
    assert loop.context.committed == []
    assert loop.items_snapshot() == []
    assert loop.snapshot().turn == 0
    assert 'session_cleared' in _kinds(loop)


def test_apply_payload_resume(loop):
    loop.submit('hello there')
    wait_idle(loop)
    payload = loop.payload()
    loop.clear()
    loop.apply_payload(payload)
    assert len(loop.context.committed) == 2
    assert 'session_loaded' in _kinds(loop)


def test_load_session_from_store(loop):
    loop.submit('hello there')
    wait_idle(loop)
    loop.save_active()
    loop.clear()
    name = loop.snapshot().session_name
    assert loop.load_session(name) is True
    assert len(loop.context.committed) == 2


def test_tools_and_agent_tools_registered(loop):
    names = set(loop._registry.names())
    assert {'read', 'write', 'edit', 'bash', 'grep'} <= names
    if loop.config.agents.enabled:
        assert {'spawn_agent', 'send_message', 'update_goal'} <= names
    loop.refresh_tools()
    assert set(loop._registry.names()) >= names
    loop.drain_ui_events()
