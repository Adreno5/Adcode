"""Regression tests for defects found during the full-code review."""

from __future__ import annotations

import json
import pathlib
import time

import pytest

import adcode.agent_loop as agent_loop
from adcode.agent_loop import AgentLoop
from adcode.config import (
    AppConfig, ConfigStore, ModelConfig, ProviderConfig, config_from_dict,
)
from adcode.goals import GoalStore
from adcode.llm.client import LLMClient
from adcode.llm.context import ConversationContext, StablePrefix
from adcode.llm.usage import merge_usage, normalize_usage
from adcode.session import DbSessionStore, SessionPayload, SessionStore, TranscriptItem
from adcode.skills import SkillRegistry
from adcode.theme import build_palette
from adcode.tools.registry import ToolRegistry
from adcode.tools.runtime import ToolContext, pop_context, push_context
from adcode.tui.canvas import Canvas
from adcode.tui.dialogs import ConfirmDialog
from adcode.tui.input import KeyEvent


# ── configuration robustness ───────────────────────────────
def test_config_tolerates_malformed_payloads():
    assert config_from_dict({'version': 'abc'}).version == 2
    assert config_from_dict({'version': 2, 'providers': 123}).providers == []
    cfg = config_from_dict({
        'version': 2,
        'providers': [{'name': 'P', 'base_url': 'u', 'extra_headers': [1]}],
    })
    assert cfg.providers[0].extra_headers == {}
    cfg = config_from_dict({'version': 2, 'mcp_servers': [1, {'name': 'ok'}]})
    assert cfg.mcp_servers == [{'name': 'ok'}]


def test_provider_name_with_separator_is_selectable():
    cfg = AppConfig()
    cfg.providers = [ProviderConfig(name='A::B', base_url='u',
                                    models=[ModelConfig(id='m')])]
    cfg.selected_model = 'A::B::m'
    selected = cfg.selected()
    assert selected is not None and selected[0].name == 'A::B'
    assert selected[1].id == 'm'


# ── usage merging ──────────────────────────────────────────
def test_merge_usage_keeps_split_reports():
    merged = merge_usage(
        normalize_usage({'input_tokens': 100, 'cache_read_input_tokens': 40}),
        normalize_usage({'output_tokens': 7}),
    )
    assert merged['input_tokens'] == 100
    assert merged['cached_input_tokens'] == 40
    assert merged['output_tokens'] == 7
    assert merged['cache_hit_rate'] == pytest.approx(0.4)


# ── anthropic message shaping ──────────────────────────────
def test_anthropic_messages_stay_alternating():
    from infra.orchestrator.anthropic import AnthropicCompatClient

    messages = [
        {'role': 'user', 'content': 'first'},
        {'role': 'assistant', 'content': None,
         'tool_calls': [{'id': 'c1', 'type': 'function',
                         'function': {'name': 'read',
                                      'arguments': '{"path": "a"}'}}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'body'},
        {'role': 'user', 'content': '<subagent_updates>note</subagent_updates>'},
        {'role': 'assistant', 'content': ''},
        {'role': 'user', 'content': 'second'},
    ]
    _, converted = AnthropicCompatClient._split_system(messages)
    # consecutive same-role entries merge: the tool result and the injected
    # subagent block land in ONE user message, tool_result blocks first
    assert [message['role'] for message in converted] == [
        'user', 'assistant', 'user']
    last = converted[2]['content']
    assert last[0] == {'type': 'tool_result', 'tool_use_id': 'c1',
                       'content': 'body'}
    assert [block['type'] for block in last[1:]] == ['text', 'text']


# ── session store hardening ────────────────────────────────
def test_session_store_rejects_empty_name(tmp_path):
    store = SessionStore(tmp_path)
    with pytest.raises(ValueError):
        store.save(SessionPayload(name=''))


def test_session_paths_stay_inside_the_store(tmp_path):
    store = SessionStore(tmp_path / 'sessions')
    for name in ('..', 'C:evil', 'a/b', 'con', 'x' * 300):
        path = store.path_for(name)
        assert path.parent == tmp_path / 'sessions'
        assert path.name.endswith('.json')


def test_session_list_skips_goal_files(tmp_path):
    directory = tmp_path / 'sessions'
    directory.mkdir()
    (directory / 'goals.json').write_text(
        json.dumps({'version': 1, 'goals': []}), encoding='utf-8')
    store = SessionStore(directory)
    store.save(SessionPayload(name='real', messages=[
        {'role': 'user', 'content': 'hi'}]))
    assert [row['name'] for row in store.list()] == ['real']


def test_session_load_survives_malformed_payload(tmp_path):
    directory = tmp_path / 'sessions'
    directory.mkdir()
    (directory / 'broken.json').write_text(
        json.dumps({'turn': 'x', 'messages': {'a': 1},
                    'compaction_archive': 5}), encoding='utf-8')
    payload = SessionStore(directory).load('broken')
    assert payload is not None
    assert payload.messages == []
    assert payload.compaction_archive == []


def test_unique_session_name(tmp_path):
    store = SessionStore(tmp_path)
    store.save(SessionPayload(name='dup', messages=[
        {'role': 'user', 'content': 'x'}]))
    assert store.unique_name('dup') == 'dup_2'


def test_latest_does_not_resurrect_after_a_corrupt_checkpoint(tmp_path):
    store = SessionStore(tmp_path)
    store.save(SessionPayload(name='named', messages=[
        {'role': 'user', 'content': 'old'}]))
    store.path_for('__last__').write_text('{not json', encoding='utf-8')
    assert store.latest() is None


# ── compaction and rewind ──────────────────────────────────
def _prefixed_context(messages):
    prefix = StablePrefix.create(provider='P', endpoint='http://x', model='m',
                                 system='s', tools=[])
    return ConversationContext(prefix, list(messages))


def test_compaction_keeps_retained_user_turn_addressable():
    messages = []
    for index in range(6):
        messages.append({'role': 'user', 'content': f'u{index} ' + 'x' * 300})
        messages.append({'role': 'assistant', 'content': f'a{index} ' + 'y' * 300})
    context = _prefixed_context(messages)
    plan = context.compaction_plan(target_tokens=500, keep_recent_turns=1)
    summary = {'role': 'user',
               'content': '<conversation_summary>sum</conversation_summary>'}
    context.apply_compaction(plan, summary)
    starts = SessionStore.turn_starts(context.committed)
    assert context.committed[0]['content'].startswith('<conversation_summary>')
    assert len(starts) == 1
    assert context.committed[starts[0]]['content'].startswith('u5')


class _QuietClient:
    def __init__(self, provider, model=None, *, base_dir='', file_cache=None,
                 warn=None):
        self.provider = provider

    async def stream(self, messages, tools=None, **kwargs):
        yield 'ok'
        yield {'finish': 'stop'}

    def invoke(self, messages, tools=None, **kwargs):
        raise NotImplementedError


def _make_loop(config, tmp_path, monkeypatch):
    monkeypatch.setattr(agent_loop, 'LLMClient', _QuietClient)
    return AgentLoop(
        config, ToolRegistry(),
        DbSessionStore(db_path=tmp_path / 'sessions' / 'adcode.db'),
        GoalStore(tmp_path / 'goals.json'),
        SkillRegistry(config.skills, str(tmp_path)),
        emit=lambda kind, payload: None, cwd=str(tmp_path),
    )


def test_rewind_aligns_after_compaction(config, tmp_path, monkeypatch):
    loop = _make_loop(config, tmp_path, monkeypatch)
    loop.start()
    try:
        _rewind_after_compaction(loop)
    finally:
        loop.shutdown()


def _rewind_after_compaction(loop):
    loop.context.committed[:] = [
        {'role': 'user',
         'content': '<conversation_summary>old</conversation_summary>'},
        {'role': 'user', 'content': 'question five'},
        {'role': 'assistant', 'content': 'answer five'},
        {'role': 'user', 'content': 'question six'},
        {'role': 'assistant', 'content': 'answer six'},
    ]
    loop.items[:] = [TranscriptItem(kind='user', turn=turn, text=f'q{turn}')
                     for turn in range(1, 7)]
    loop._engine.turn = 6
    assert loop.rewind(5) == 'question five'
    assert [message['content'] for message in loop.context.committed] == [
        '<conversation_summary>old</conversation_summary>']
    assert [item.turn for item in loop.items] == [1, 2, 3, 4]


def test_system_prompt_tracks_goal_changes(config, tmp_path, monkeypatch):
    loop = _make_loop(config, tmp_path, monkeypatch)
    loop.start()
    try:
        assert '<active_goal>' not in loop._system_provider()
        loop.goals.create('ship it')
        assert '<active_goal>' in loop._system_provider()
    finally:
        loop.shutdown()


# ── tools ──────────────────────────────────────────────────
def test_tool_paths_resolve_against_the_tool_cwd(tmp_path):
    from adcode.tools.builtin import install_coding_tools

    registry = ToolRegistry()
    install_coding_tools(registry)
    context = ToolContext(cwd=str(tmp_path))
    token = push_context(context)
    try:
        result = registry.execute('write', {'path': 'nested/file.txt',
                                            'content': 'data\n'})
        assert not result[2]
        assert (tmp_path / 'nested' / 'file.txt').read_text() == 'data\n'
        listing = registry.execute('ls', {'path': '.'})[0]
        assert 'nested/' in listing
        read_back = registry.execute('read', {'path': 'nested/file.txt'})[0]
        assert 'data' in read_back
        edited = registry.execute('edit', {'path': 'nested/file.txt',
                                           'old': 'data', 'new': 'more'})
        assert 'Replaced' in edited[0]
        # The diff must describe the file in the tool cwd, not a same-named
        # relative path in the process cwd.
        assert edited[1] is not None
        assert 'more' in edited[1].unified
    finally:
        pop_context(token)


# ── rendering safety ───────────────────────────────────────
def test_canvas_never_emits_control_characters():
    canvas = Canvas(8, 1, build_palette('dark'))
    canvas.draw_text(0, 0, 'a\x1b[31mred', canvas.palette.text)
    frame = canvas.render_row(0)
    assert '\x1b[31m' not in frame
    assert all(not (ord(char) < 32) for char in frame.replace('\x1b', ''))


def test_canvas_drops_wide_glyph_that_cannot_fit():
    canvas = Canvas(3, 1, build_palette('dark'))
    canvas.draw_text(2, 0, '中', canvas.palette.text)
    assert canvas.cell(2, 0).ch == ' '


def test_markdown_html_tags_apply_attributes():
    from adcode.tui.canvas import BOLD, ITALIC, UNDERLINE
    from adcode.tui.markdown import render_markdown
    from adcode.tui.text import plain_text

    lines = render_markdown('<b>bold</b> plain <i>it</i> <u>u</u>', 60)
    spans = lines[0]
    attrs = {plain_text([span]).strip(): span.attrs for span in spans
             if plain_text([span]).strip()}
    assert attrs['bold'] & BOLD
    assert attrs['plain'] == 0
    assert attrs['it'] & ITALIC
    assert attrs['u'] & UNDERLINE


def test_confirm_dialog_accepts_printed_letters():
    results = []
    dialog = ConfirmDialog('Sure?', 'Remove it.', results.append)
    dialog.handle(KeyEvent(kind='text', text='y'))
    assert dialog.closed and results == [True]
    dialog = ConfirmDialog('Sure?', 'Remove it.', results.append)
    dialog.handle(KeyEvent(kind='text', text='n'))
    assert results == [True, False]


def test_session_store_roundtrip_preserves_prefix_generation(tmp_path):
    store = SessionStore(tmp_path)
    payload = SessionPayload(name='s', messages=[{'role': 'user', 'content': 'x'}],
                             prefix_generation=4)
    store.save(payload)
    loaded = store.load('s')
    assert loaded is not None and loaded.prefix_generation == 4


def test_config_store_save_is_atomic(tmp_path):
    path = tmp_path / 'config.json'
    store = ConfigStore(path)
    store.config.providers = [ProviderConfig(
        name='P', base_url='u', models=[ModelConfig(id='m')])]
    assert store.save()
    assert json.loads(path.read_text(encoding='utf-8'))['providers'][0]['name'] == 'P'
    assert not list(pathlib.Path(tmp_path).glob('*.tmp'))
