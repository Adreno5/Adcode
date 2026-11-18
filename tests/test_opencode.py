import pytest

from adcode.agent_loop import AgentLoop
from adcode.config import ModelConfig, ProviderConfig
from adcode.llm import client as llm
from adcode.net import httpx_module
from adcode.session import DbSessionStore, SessionPayload
from adcode.tools.registry import ToolRegistry


HEADER = 'x-opencode-session'


@pytest.fixture()
def requests(monkeypatch):
    httpx = httpx_module()
    seen = []

    def respond(request):
        seen.append(request)
        if request.url.host == 'opencode.ai' and 'redirect' in request.url.path:
            return httpx.Response(307, headers={'Location': 'https://example.test/v1/models'})
        if request.method == 'POST':
            return httpx.Response(200, headers={'Content-Type': 'text/event-stream'},
                                  content=b'')
        return httpx.Response(200, json={'object': 'list', 'data': []})

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(llm, 'build_http_client', lambda **kw: httpx.Client(
        transport=transport, follow_redirects=True, **kw))
    monkeypatch.setattr(llm, 'build_async_http_client', lambda **kw: httpx.AsyncClient(
        transport=transport, follow_redirects=True, **kw))
    return seen


@pytest.mark.parametrize(('base_url', 'wire'), [
    ('https://opencode.ai/inference/openai/v1', 'openai chat completions'),
    ('https://OPENCODE.AI/inference/openai/v1/', 'openai responses'),
    ('https://opencode.ai/inference/go/anthropic', 'anthropic messages'),
    ('https://opencode.ai/zen/v1', 'openai chat completions'),
    ('https://opencode.ai/zen/go/v1', 'openai chat completions'),
])
def test_opencode_session_header_is_stable_across_sync_and_async_sdk_calls(
        requests, base_url, wire):
    provider = ProviderConfig(name='Custom name', base_url=base_url, wire=wire,
                              models=[ModelConfig(id='test-model')])
    client = llm.LLMClient(provider)
    client.list_models()
    list(client.chat([{'role': 'user', 'content': 'hello'}]))
    llm.LLMClient(provider).list_models()
    assert requests[1].method == 'POST'
    suffix = {'openai chat completions': '/chat/completions',
              'openai responses': '/responses', 'anthropic messages': '/messages'}[wire]
    assert requests[1].url.path.endswith(suffix)
    first, second, other = [req.headers.get(HEADER) for req in requests]
    assert first and first == second
    assert other and other != first
    assert HEADER not in provider.extra_headers


@pytest.mark.parametrize('base_url', [
    'https://example.test/opencode.ai/inference/v1',
    'https://opencode.ai.example.test/inference/v1',
    'https://opencode.ai@example.test/inference/v1',
    'https://opencode.ai/docs/console',
    'https://opencode.ai/inference-other/v1',
])
def test_provider_name_or_lookalike_url_does_not_enable_header(requests, base_url):
    provider = ProviderConfig(name='OpenCode Go', base_url=base_url)
    llm.LLMClient(provider).list_models()
    assert HEADER not in requests[0].headers


def test_explicit_session_header_wins_case_insensitively(requests):
    provider = ProviderConfig(
        name='OpenCode', base_url='https://opencode.ai/inference/openai/v1',
        extra_headers={'X-OpenCode-Session': 'manual', 'X-Other': 'kept'})
    llm.fetch_model_ids(provider)
    assert requests[0].headers.get_list(HEADER) == ['manual']
    assert requests[0].headers['x-other'] == 'kept'


def test_automatic_header_is_not_forwarded_outside_opencode(requests):
    provider = ProviderConfig(name='OpenCode', base_url='https://opencode.ai/inference/redirect')
    llm.LLMClient(provider).list_models()
    assert requests[0].headers.get(HEADER)
    assert HEADER not in requests[1].headers


def test_session_header_follows_conversation_lifecycle(requests, tmp_path, config):
    provider, _ = config.selected()
    provider.base_url = 'https://opencode.ai/inference/openai/v1'

    def start():
        store = DbSessionStore(db_path=tmp_path / 'sessions.db')
        loop = AgentLoop(config, ToolRegistry(), store, None, None, cwd=str(tmp_path))
        loop.start()
        return loop, store

    def header(loop):
        list(loop.client.chat([{'role': 'user', 'content': 'hello'}]))
        return requests[-1].headers[HEADER]

    loop, store = start()
    try:
        loop.apply_payload(SessionPayload(name='existing', messages=[
            {'role': 'user', 'content': 'Hello'},
        ]))
        original = header(loop)
        loop.set_model(provider, ModelConfig(id='switched-model'))
        assert header(loop) == original
        loop.save_as('renamed')
        assert header(loop) == original
        forked = loop.fork()
        assert loop.load_session(forked)
        fork_header = header(loop)
        assert fork_header != original
        assert loop.load_session('renamed')
        assert header(loop) == original
        loop.save_active()
    finally:
        loop.shutdown()
        store.store.close()

    resumed, store = start()
    try:
        resumed.apply_payload(store.latest())
        assert header(resumed) == original
        assert resumed.load_session(forked)
        assert header(resumed) == fork_header
        resumed.clear()
        assert header(resumed) not in (original, fork_header)
    finally:
        resumed.shutdown()
        store.store.close()
