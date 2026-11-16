"""Sub-agent collaboration tests for the sdk-backed AgentManager wrapper."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from adcode.agents.manager import AgentEvent, AgentManager, AgentStatus
from adcode.config import AgentsSettings, ModelConfig, ProviderConfig
from adcode.goals import GoalStore
from harness.agent import ConversationEngine, EngineConfig, ModelInfo
from harness.tools.registry import AnnotatedTool, ManagedToolRegistry


class ScriptedClient:
    """Shared IChatClient stub; replies scripted per call."""

    instances: list['ScriptedClient'] = []

    def __init__(self, provider=None, model=None, *, base_dir='', file_cache=None,
                 warn=None):
        self.provider = provider
        self.rounds: list[list[str]] = [['child result']]
        self.calls = 0
        self.prompts: list[list[str]] = []
        ScriptedClient.instances.append(self)

    async def stream(self, messages, tools=None, **kwargs):
        # the engine passes ChatMessage objects; normalize once
        msgs = [m.to_dict() if hasattr(m, 'to_dict') else dict(m)
                for m in messages]
        self.prompts.append([str(m.get('content', ''))[:40] for m in msgs])
        index = min(self.calls, len(self.rounds) - 1)
        self.calls += 1
        last_user = next((str(m.get('content') or '') for m in reversed(msgs)
                          if m.get('role') == 'user'), '')
        if 'slowly' in last_user:
            await asyncio.sleep(1.0)
        for piece in self.rounds[index]:
            yield piece
        yield {'finish': 'stop'}

    def invoke(self, messages, tools=None, **kwargs):
        raise NotImplementedError


def _config() -> Any:
    from adcode.config import AppConfig

    cfg = AppConfig()
    provider = ProviderConfig(name='Test', base_url='http://localhost:9/v1',
                              api_key='k', wire='openai chat completions',
                              models=[ModelConfig(id='test-model')])
    other = ProviderConfig(name='Other', base_url='http://localhost:10/v1',
                           api_key='k', wire='openai chat completions',
                           models=[ModelConfig(id='other-model')])
    cfg.providers = [provider, other]
    cfg.select('Test', 'test-model')
    agent_type = type(cfg.agent_types['general'])
    cfg.agent_types['scout'] = agent_type(
        name='scout', description='scouts ahead', system='be brief',
        model='Test::test-model', tools=['read'], max_turns=6)
    cfg.agent_types['remote'] = agent_type(
        name='remote', description='runs elsewhere', system='be remote',
        model='Other::other-model', tools=[], max_turns=6)
    cfg.agents.enabled = True
    return cfg


def _registry() -> ManagedToolRegistry:
    registry = ManagedToolRegistry()
    registry.register(AnnotatedTool(
        name='read', description='read', read_only=True,
        parameters={'type': 'object',
                    'properties': {'path': {'type': 'string'}},
                    'required': ['path']},
        handler=lambda path: 'file body'), source='builtin')
    return registry


def _model_info() -> ModelInfo:
    return ModelInfo(id='test-model', provider='Test',
                     endpoint='http://localhost:9/v1', wire='chat',
                     context_window=32_000, max_output_tokens=2_048)


def _setup(tmp_path, rounds: list[str] | None = None):
    """Build (root engine, manager, client) under a running loop."""
    cfg = _config()
    client = ScriptedClient()
    if rounds:
        client.rounds = rounds
    registry = _registry()
    root = ConversationEngine(
        client=client, registry=registry, store=None, cwd=str(tmp_path),
        model=_model_info(),
        config=EngineConfig(agents=AgentsSettings(max_concurrent=4)),
        system_provider=lambda: 'root system')
    manager = AgentManager(
        cfg, cfg.agents, registry, client, _model_info(),
        goal_store=GoalStore(tmp_path / 'goals.json'),
        client_factory=lambda provider, model: ScriptedClient(provider, model),
        cwd=str(tmp_path))
    manager._provider = cfg.providers[0]
    manager._model_config = cfg.providers[0].models[0]
    manager.attach(root)
    return root, manager, client


async def _wait_child(manager, name: str, timeout: float = 5.0) -> None:
    async def poll():
        while manager._agents[name].status not in (
                'completed', 'errored', 'interrupted'):
            await asyncio.sleep(0.02)
    await asyncio.wait_for(poll(), timeout)


def test_spawn_delivers_update_to_parent_mailbox(tmp_path):
    async def main():
        root, manager, client = _setup(tmp_path)
        record = await manager.spawn('scout', 'scout the area')
        assert record.name == 'scout'
        await _wait_child(manager, 'scout')
        updates = manager.consume_mailbox(root)
        assert len(updates) == 1
        assert '<subagent_update agent="scout" status="completed">' in updates[0]
        assert 'child result' in updates[0]
        rows = manager.agents_snapshot()
        assert any(row['name'] == 'scout' and row['status'] == 'completed'
                   for row in rows)

    asyncio.run(main())


def test_agent_type_resolves_model_and_tools(tmp_path):
    created: list[tuple[str, str]] = []

    async def main():
        cfg = _config()
        client = ScriptedClient()
        registry = _registry()
        root = ConversationEngine(
            client=client, registry=registry, cwd=str(tmp_path),
            model=_model_info(), system_provider=lambda: 'root system')
        manager = AgentManager(
            cfg, cfg.agents, registry, client, _model_info(),
            client_factory=lambda provider, model:
                created.append((provider.name, model.id))
                or ScriptedClient(provider, model),
            cwd=str(tmp_path))
        manager._provider = cfg.providers[0]
        manager._model_config = cfg.providers[0].models[0]
        manager.attach(root)
        # 'scout' pins the caller's model: the shared client is reused
        record = await manager.spawn('scout', 'go', agent_type='scout')
        assert record.engine._model.id == 'test-model'
        assert created == []
        assert record.engine._allowed_tools == {'read'}
        # 'remote' pins a cross-provider model: a dedicated client is built;
        # an empty tools list means no allowlist, like the reference engine
        record = await manager.spawn('remote', 'go', agent_type='remote')
        assert record.engine._model.id == 'other-model'
        assert created == [('Other', 'other-model')]
        assert record.engine._allowed_tools is None
        await _wait_child(manager, 'remote')

    asyncio.run(main())


def test_fork_turns_shapes_child_history(tmp_path):
    async def main():
        root, manager, client = _setup(tmp_path)
        root.conversation.committed.extend([
            {'role': 'user', 'content': 'first question'},
            {'role': 'assistant', 'content': 'first answer'},
            {'role': 'user', 'content': 'second question'},
            {'role': 'assistant', 'content': 'second answer'},
        ])
        record = await manager.spawn('all_child', 'go', fork_turns='all')
        assert [m['content'] for m in record.engine.conversation.committed] == [
            'first question', 'first answer', 'second question', 'second answer']
        record = await manager.spawn('none_child', 'go', fork_turns='none')
        assert record.engine.conversation.committed == []
        record = await manager.spawn('one_child', 'go', fork_turns=1)
        assert [m['content'] for m in record.engine.conversation.committed] == [
            'second question', 'second answer']
        await _wait_child(manager, 'all_child')

    asyncio.run(main())


def test_child_system_swaps_the_identity_for_its_own_model(tmp_path):
    """A child running another model must not claim the parent's model."""
    from adcode.system_prompt import identity_block

    async def main():
        root, manager, client = _setup(tmp_path)
        root._system_provider = lambda: identity_block('test-model', 'Test')
        record = await manager.spawn('remote', 'go', agent_type='remote')
        system = record.engine.system_prompt()
        assert 'model id: other-model' in system
        assert 'model id: test-model' not in system
        assert 'provider: Other' in system
        assert 'role: sub-agent' in system
        await _wait_child(manager, 'remote')

    asyncio.run(main())


def test_spawn_validates_input(tmp_path):
    async def main():
        root, manager, client = _setup(tmp_path)
        with pytest.raises(ValueError):
            await manager.spawn('Bad Name!', 'task')
        with pytest.raises(ValueError):
            await manager.spawn('worker', '   ')
        await manager.spawn('worker', 'task')
        with pytest.raises(ValueError):
            await manager.spawn('worker', 'again')

    asyncio.run(main())


def test_send_message_and_followup(tmp_path):
    async def main():
        root, manager, client = _setup(tmp_path)
        record = await manager.spawn('helper', 'stand by')
        result = manager.send_message('helper', 'hello there')
        assert result == {'delivered': True, 'turn_triggered': False}
        assert manager.consume_mailbox(record.engine) == ['From root: hello there']
        assert manager.send_message('ghost', 'x')['delivered'] is False
        result = manager.send_message('helper', 'go again', trigger=True)
        assert result['turn_triggered'] is True
        await _wait_child(manager, 'helper')

    asyncio.run(main())


def test_agent_and_goal_tools_registered(tmp_path):
    async def main():
        root, manager, client = _setup(tmp_path)
        names = set(manager._registry.names())
        assert {'spawn_agent', 'send_message', 'followup_task', 'list_agents',
                'wait_agent', 'interrupt_agent', 'update_goal'} <= names
        sources = {name: manager._registry.source_of(name) for name in
                   ('spawn_agent', 'update_goal')}
        assert set(sources.values()) == {'agent'}
        # the goal tool talks to the GoalStore
        goal_tool = next(tool for tool in manager._registry.tools()
                         if tool.name == 'update_goal')
        assert 'no active goal' in goal_tool.handler(status='completed')
        goal = manager.goal_store.create('ship it')
        payload = json.loads(goal_tool.handler(status='blocked',
                                               result='waiting'))
        assert payload['goal_id'] == goal.id and payload['status'] == 'blocked'
        assert manager.goal_store.get(goal.id).status == 'blocked'

    asyncio.run(main())


def test_drain_agent_events_maps_child_stream(tmp_path):
    async def main():
        root, manager, client = _setup(tmp_path)
        await manager.spawn('scout', 'go')
        await _wait_child(manager, 'scout')
        events = manager.drain_agent_events()
        kinds = {(event.agent_name, event.kind) for event in events}
        assert ('scout', 'status') in kinds
        lifecycle = [event for event in events
                     if event.agent_name == 'scout'
                     and event.status == 'completed']
        assert lifecycle and 'child result' in lifecycle[0].content
        # drain empties the queue
        assert manager.drain_agent_events() == []

    asyncio.run(main())


def test_drain_events_is_inert_for_the_engine(tmp_path):
    async def main():
        root, manager, client = _setup(tmp_path)
        # the engine's per-round pump must never consume child activity; the
        # facade owns it (see pump_agent_events)
        assert manager.drain_events() == []
        manager._agent_events.append(('agent_event', {'agent': 'x'}))
        assert manager.drain_events() == []
        assert len(manager.drain_agent_events()) == 1

    asyncio.run(main())


def test_wait_agent_reports_activity(tmp_path):
    async def main():
        root, manager, client = _setup(tmp_path)
        record = await manager.spawn('busy', 'work')
        waiter = asyncio.create_task(manager.wait_for_activity(1_000))
        await asyncio.sleep(0.05)
        manager._touch(record)
        result = await waiter
        assert result['timed_out'] is False
        stale = await manager.wait_for_activity(50)
        assert stale['timed_out'] is True

    asyncio.run(main())


def test_interrupt_agent(tmp_path):
    async def main():
        root, manager, client = _setup(tmp_path)
        record = await manager.spawn('slowpoke', 'work slowly')
        await asyncio.sleep(0.1)
        result = manager.interrupt_agent('slowpoke')
        assert 'previous_status' in result
        await _wait_child(manager, 'slowpoke')
        assert manager._agents['slowpoke'].status == AgentStatus.INTERRUPTED

    asyncio.run(main())


def test_a_failed_turn_gives_the_sub_agent_update_back(tmp_path):
    """A turn that never reaches the history must not eat the update.

    The sdk clears the inbox when it hands the updates to a turn, and the
    engine never puts them back: an interrupted turn used to throw away what
    the sub-agents found.  The manager keeps a copy and returns the updates
    that never made it into the history.
    """
    async def main():
        root, manager, client = _setup(tmp_path)
        record = await manager.spawn('scout', 'scout the area')
        await _wait_child(manager, 'scout')

        # The turn takes the updates out of the inbox...
        updates = manager.consume_mailbox(root)
        assert len(updates) == 1
        assert not manager.records()['root'].inbox

        # ...and the turn dies before the update reaches the history.
        lost = manager.settle_mailbox(root)
        assert lost == 1
        # The update is back in the inbox, so the next turn still sees it.
        assert len(manager.records()['root'].inbox) == 1
        again = manager.consume_mailbox(root)
        assert again == updates

        # A turn that did commit keeps the update and requeues nothing.
        root.conversation.stage({'role': 'user', 'content': updates[0]})
        root.conversation.commit()
        pending = manager.consume_mailbox(root)
        assert pending == []
        manager.settle_mailbox(root)
        assert not manager.records()['root'].inbox

    asyncio.run(main())
