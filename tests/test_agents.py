import time

import pytest

import adcode.agents.manager as manager_module
from adcode.agents.manager import AgentManager, AgentStatus
from adcode.llm.context import ConversationContext, StablePrefix
from adcode.tools.registry import ToolRegistry


class AgentClient:
    def __init__(self, provider):
        self.provider = provider

    def chat(self, messages, model, *, system='', max_tokens=8192, tools=None,
             prompt_cache_key=None, effort='medium', force_reasoning=False):
        if 'slow' in str(messages[-1].get('content', '')):
            for index in range(50):
                yield {'kind': 'text', 'content': 'tick '}
                time.sleep(0.01)
            yield {'kind': 'finish', 'reason': 'stop'}
            return
        yield {'kind': 'text', 'content': 'child result'}
        yield {'kind': 'finish', 'reason': 'stop'}


@pytest.fixture()
def manager(config, tmp_path, monkeypatch):
    monkeypatch.setattr(manager_module, 'LLMClient', AgentClient)
    prefix = StablePrefix.create(
        provider='Test', endpoint='http://x', model='test-model',
        system='root system', tools=[])
    context = ConversationContext(prefix, [
        {'role': 'user', 'content': 'root question'},
        {'role': 'assistant', 'content': 'root answer'},
    ])
    instance = AgentManager(config, config.providers[0], config.providers[0].models[0],
                            'root system', context, ToolRegistry())
    yield instance
    instance.shutdown()


def wait_for(manager, name, statuses=('completed',), timeout=4.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = manager._records.get(name)
        if record is not None and record.status in statuses:
            return record
        time.sleep(0.02)
    raise AssertionError(f'{name} did not reach {statuses}; '
                         f'last={manager._records.get(name).status if manager._records.get(name) else None}')


def test_spawn_and_mailbox(manager):
    with manager.root_scope():
        record = manager.spawn_agent('child', 'do the work', 'general', 'none')
    assert record.scope.name == 'root/child'
    wait_for(manager, 'root/child')
    rows = manager.list_agents()
    assert any(row['agent_name'] == 'root/child' for row in rows)
    mailbox = manager.consume_mailbox('root')
    assert any('subagent_update' in message for message in mailbox)
    events = manager.drain_events()
    assert any(event.kind == 'status' for event in events)


def test_safe_history_fork_turns(manager):
    with manager.root_scope():
        record = manager.spawn_agent('child', 'work', 'general', 'all')
    assert record.scope.context.committed[0]['content'] == 'root question'
    with manager.root_scope():
        none = manager.spawn_agent('empty', 'work', 'general', 'none')
    assert none.scope.context.committed == []


def test_limits_and_duplicates(manager):
    manager.config.agents.max_depth = 1
    with manager.root_scope():
        manager.spawn_agent('child', 'work', 'general', 'none')
        with pytest.raises(ValueError):
            manager.spawn_agent('child', 'work', 'general', 'none')
    child_scope = manager._records['root/child'].scope
    with manager.scope(child_scope):
        with pytest.raises(RuntimeError):
            manager.spawn_agent('grandchild', 'work', 'general', 'none')


def test_disabled_agents(config, monkeypatch):
    monkeypatch.setattr(manager_module, 'LLMClient', AgentClient)
    config.agents.enabled = False
    prefix = StablePrefix.create(provider='P', endpoint='http://x', model='m',
                                 system='s', tools=[])
    manager = AgentManager(config, config.providers[0],
                           config.providers[0].models[0], 's',
                           ConversationContext(prefix, []), ToolRegistry())
    with manager.root_scope():
        with pytest.raises(RuntimeError):
            manager.spawn_agent('child', 'work', 'general', 'none')
    manager.shutdown()


def test_team_spawn(manager):
    with manager.root_scope():
        records = manager.spawn_team('build', 'build it', 'none')
    assert len(records) == len(manager.config.agent_teams['build'])
    for record in records:
        wait_for(manager, record.scope.name)
    rows = manager.list_agents('root/build')
    assert len(rows) == len(records)


def test_interrupt_agent(manager):
    with manager.root_scope():
        record = manager.spawn_agent('slow', 'slow work', 'general', 'none')
    time.sleep(0.1)
    with manager.root_scope():
        previous = manager.interrupt_agent('slow')
    assert previous in ('pending_init', 'running')
    wait_for(manager, 'root/slow', statuses=(AgentStatus.INTERRUPTED,))
    assert record.interrupt.is_set()


def test_send_message_and_followup(manager):
    with manager.root_scope():
        manager.spawn_agent('child', 'work', 'general', 'none')
    wait_for(manager, 'root/child')
    with manager.root_scope():
        manager.send_message('child', 'a note')
        manager.send_message('child', 'more work', trigger=True)
    time.sleep(0.1)
    wait_for(manager, 'root/child', statuses=(AgentStatus.COMPLETED, AgentStatus.RUNNING))
    record = manager._records['root/child']
    assert record.inbox or record.pending_turns or record.status == AgentStatus.COMPLETED


def test_agent_tool_specs(manager):
    names = {spec.name for spec in manager.tool_specs()}
    assert {'spawn_agent', 'spawn_team', 'send_message', 'followup_task',
            'list_agents', 'wait_agent', 'interrupt_agent', 'update_goal'} == names
