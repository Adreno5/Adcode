import time

import pytest

import adcode.agent_loop as agent_loop
from adcode.agent_loop import AgentLoop
from adcode.goals import GoalStore
from adcode.session import SessionStore
from adcode.skills import SkillRegistry
from adcode.tools.registry import ToolRegistry, ToolSpec


class ScriptedClient:
    """Deterministic streaming client used by the loop tests."""

    def __init__(self, provider):
        self.provider = provider
        self.requests = []

    def chat(self, messages, model, *, system='', max_tokens=8192, tools=None,
             prompt_cache_key=None, effort='medium', force_reasoning=False):
        self.requests.append(messages)
        if 'compactor' in system.lower():
            yield {'kind': 'text', 'content': 'SUMMARY'}
            yield {'kind': 'finish', 'reason': 'stop'}
            return
        if 'side question' in system.lower():
            yield {'kind': 'text', 'content': 'aside answer'}
            yield {'kind': 'finish', 'reason': 'stop'}
            return
        last_user = next((str(message.get('content') or '')
                          for message in reversed(messages)
                          if message.get('role') == 'user'), '')
        if 'use tool' in last_user and not any(
                message.get('role') == 'tool' for message in messages):
            yield {'kind': 'thinking', 'content': 'Need a tool.'}
            yield {'kind': 'tool_call', 'id': 'c1', 'name': 'probe',
                   'args': {'value': 'x'}}
            yield {'kind': 'finish', 'reason': 'tool_calls'}
            return
        yield {'kind': 'thinking', 'content': 'Thinking...'}
        yield {'kind': 'text', 'content': 'Answer '}
        yield {'kind': 'text', 'content': '**done**.'}
        yield {'kind': 'usage', 'usage': {
            'input_tokens': 12, 'output_tokens': 4,
            'input_tokens_details': {'cached_tokens': 5}}}
        yield {'kind': 'finish', 'reason': 'stop'}


@pytest.fixture()
def loop(config, tmp_path, monkeypatch):
    monkeypatch.setattr(agent_loop, 'LLMClient', ScriptedClient)
    registry = ToolRegistry()
    registry.register(ToolSpec(
        'probe', 'probe', {'type': 'object'}, lambda args: 'probe:' + args['value']))
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
        time.sleep(0.02)
    raise AssertionError('loop did not become idle')


def test_simple_turn(loop):
    loop.submit('hello')
    wait_idle(loop)
    assert loop.snapshot().status == 'idle'
    assert [message['role'] for message in loop.context.committed] == [
        'user', 'assistant']
    assert loop.context.committed[1]['content'] == 'Answer **done**.'
    assert loop.context.committed[1]['reasoning_content'] == 'Thinking...'
    kinds = [item.kind for item in loop.items]
    assert kinds == ['user', 'thinking', 'assistant']
    assert loop.total_in == 12 and loop.total_out == 4
    assert loop.usage_records[-1]['cached_input_tokens'] == 5


def test_tool_turn(loop):
    loop.submit('please use tool')
    wait_idle(loop)
    roles = [message['role'] for message in loop.context.committed]
    assert roles == ['user', 'assistant', 'tool', 'assistant']
    tool_item = next(item for item in loop.items if item.kind == 'tool')
    assert tool_item.name == 'probe'
    assert tool_item.status == 'done'
    assert 'probe:x' in tool_item.output
    assert tool_item.duration >= 0


def test_view_image_tool_attaches_the_picture(loop, tmp_path):
    """A tool result that names an image becomes a user image block."""
    from adcode import images

    (tmp_path / 'shot.png').write_bytes(
        images.encode_png(2, 2, bytes([1, 2, 3, 255] * 4)))
    loop.cwd = str(tmp_path)
    loop.client.base_dir = str(tmp_path)
    rounds = {'n': 0}

    def chat(messages, model, **kwargs):
        rounds['n'] += 1
        if rounds['n'] == 1:
            yield {'kind': 'tool_call', 'id': 'c1', 'name': 'view_image',
                   'args': {'path': 'shot.png'}}
            yield {'kind': 'finish', 'reason': 'tool_calls'}
            return
        yield {'kind': 'text', 'content': 'I can see it.'}
        yield {'kind': 'finish', 'reason': 'stop'}

    loop.client.chat = chat
    loop.submit('look at the screenshot')
    wait_idle(loop)
    roles = [message['role'] for message in loop.context.committed]
    assert roles == ['user', 'assistant', 'tool', 'user', 'assistant']
    tool_message = loop.context.committed[2]
    assert 'Loaded shot.png (2x2 image/png' in tool_message['content']
    injected = loop.context.committed[3]['content']
    assert injected.startswith('<tool_images>')
    assert '[image: sessions/images/paste-' in injected
    # The picture is only expanded for the request; history keeps the marker.
    assert 'base64' not in injected
    tool_item = next(item for item in loop.items if item.kind == 'tool')
    assert tool_item.name == 'view_image'
    assert tool_item.status == 'done'


def test_turn_is_not_capped_by_tool_rounds(loop):
    """A long investigation is only ended by the model, not a fixed count."""
    rounds = 130
    sessions = {'n': 0}

    def long_chat(messages, model, **kwargs):
        sessions['n'] += 1
        if sessions['n'] <= rounds:
            yield {'kind': 'tool_call', 'id': f'c{sessions["n"]}',
                   'name': 'probe', 'args': {'value': str(sessions['n'])}}
            yield {'kind': 'finish', 'reason': 'tool_calls'}
            return
        yield {'kind': 'text', 'content': 'finished.'}
        yield {'kind': 'finish', 'reason': 'stop'}

    loop.client.chat = long_chat
    loop.submit('keep going')
    wait_idle(loop, timeout=30)
    assert not any('tool iteration limit' in item.text for item in loop.items
                   if item.kind == 'notice')
    assert [item.kind for item in loop.items].count('tool') == rounds
    assert loop.context.committed[-1]['content'] == 'finished.'


def test_compaction_trigger(loop):
    loop.model.context_window = 8_000
    loop.config.max_output_tokens = 4_000
    loop.config.context.reserved_output = 4_000
    for index in range(12):
        loop.context.committed.append({'role': 'user', 'content': f'u{index} ' + 'x' * 300})
        loop.context.committed.append({'role': 'assistant',
                                       'content': f'a{index} ' + 'y' * 300})
    loop.config.context.compact_threshold = 50
    loop.submit('continue')
    wait_idle(loop)
    assert loop.context.archived
    assert any('SUMMARY' in str(message.get('content'))
               for message in loop.context.committed)
    assert any('Context compacted' in item.text for item in loop.items
               if item.kind == 'notice')


def test_interrupt(loop):
    def slow_chat(messages, model, **kwargs):
        for index in range(30):
            yield {'kind': 'text', 'content': f'chunk{index} '}
            time.sleep(0.02)
        yield {'kind': 'finish', 'reason': 'stop'}

    loop.client.chat = slow_chat
    loop.submit('slow')
    time.sleep(0.1)
    assert loop.busy
    loop.interrupt()
    wait_idle(loop)
    assert any(item.kind == 'notice' and '[interrupted]' in item.text
               for item in loop.items)
    # Esc keeps the work that already happened: the user message and the tokens
    # that were already on screen stay in the history.  Dropping them made the
    # next request disagree with the transcript and lost the whole turn.
    assert [message['role'] for message in loop.context.committed] == [
        'user', 'assistant']
    assert loop.context.committed[0]['content'] == 'slow'
    assert loop.context.committed[1]['content'].startswith('chunk0')
    assert loop.context.draft == []


def test_aside_does_not_touch_history(loop):
    loop.submit('main question')
    wait_idle(loop)
    before = len(loop.context.committed)
    loop.submit('what is side?', 'aside')
    wait_idle(loop)
    assert len(loop.context.committed) == before
    aside = next(item for item in loop.items if item.kind == 'aside')
    assert aside.text == 'aside answer'
    assert aside.display == 'what is side?'


def test_persistence_and_rewind(loop, tmp_path):
    loop.submit('first')
    wait_idle(loop)
    loop.submit('second')
    wait_idle(loop)
    payload = loop.payload()
    assert payload.name
    assert len(payload.items) >= 4
    text = loop.rewind(2)
    assert text == 'second'
    assert [message['content'] for message in loop.context.committed] == [
        'first', 'Answer **done**.']
    assert all(item.turn < 2 for item in loop.items)


def test_fork_and_save(loop):
    loop.submit('hello')
    wait_idle(loop)
    name = loop.fork('target')
    assert name == 'target'
    assert loop.session_name == 'target'
    assert loop.store.load('target') is not None
    loop.save_as('copy')
    assert loop.store.load('copy') is not None


def test_clear(loop):
    loop.submit('hello')
    wait_idle(loop)
    loop.clear()
    assert loop.items == []
    assert loop.context.committed == []
    assert loop.snapshot().session_name == ''


def test_apply_payload_resume(loop):
    loop.submit('hello')
    wait_idle(loop)
    payload = loop.payload()
    loop.clear()
    loop.apply_payload(payload)
    assert loop.context.committed == payload.messages
    assert loop.session_name == payload.name
    assert [item.kind for item in loop.items] == ['user', 'thinking', 'assistant']
    assert loop.snapshot().status == 'idle'


def test_load_session_from_store(loop):
    loop.submit('hello')
    wait_idle(loop)
    loop.save_as('stored')
    loop.clear()
    assert loop.load_session('stored')
    assert loop.session_name == 'stored'
    assert loop.context.committed
    assert not loop.load_session('missing')
