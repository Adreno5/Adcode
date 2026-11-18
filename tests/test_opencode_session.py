"""Console's ``x-opencode-session`` header.

Console asks every harness to send a stable session id with each request so it
can route a conversation consistently and reuse its prompt cache.  These tests
pin the three parts of the feature:

* which providers the header is sent to (``opencode_session`` policy);
* the ids themselves -- one per conversation, remembered across runs;
* the header actually reaching the wire, through the real SDKs, on all three
  wires, and following the conversation it belongs to.
"""

from __future__ import annotations

import time

import pytest

from adcode import paths
from adcode.agent_loop import AgentLoop
from adcode.config import (
    AppConfig, ModelConfig, ProviderConfig, config_from_dict, config_to_dict,
)
from adcode.goals import GoalStore
from adcode.llm.client import LLMClient
from adcode.llm.session import (
    OpencodeSessions, SESSION_HEADER, new_session_id, sends_session_header,
)
from adcode.session import DbSessionStore
from adcode.skills import SkillRegistry
from adcode.tools.registry import ToolRegistry
from bench.gateway import FakeLLMGateway
from bench.script import EchoResponder, ScriptedResponder, Turn
from bench.workspace import build_config

WIRES = [
    'openai chat completions',
    'openai responses',
    'anthropic messages',
]


def provider(base_url: str = 'http://localhost:1234/v1', *,
             wire: str = 'openai chat completions',
             policy: str = 'auto') -> ProviderConfig:
    return ProviderConfig(
        name='Harness', base_url=base_url, api_key='harness-key', wire=wire,
        opencode_session=policy,
        models=[ModelConfig(id='harness-model', context_window=100_000,
                            max_output_tokens=8_192)],
    )


def headers(request) -> dict[str, str]:
    return {name.lower(): value for name, value in request.headers.items()}


# ── the policy ────────────────────────────────────────────
def test_auto_sends_to_console_endpoints_only():
    for url in ('https://opencode.ai/inference',
                'https://opencode.ai/inference/go/openai/v1',
                'https://console.opencode.ai/inference',
                'HTTPS://OPENCODE.AI/inference'):
        assert sends_session_header(provider(url)), url
    for url in ('http://localhost:1234/v1',
                'https://api.deepseek.com/v1',
                'https://opencode.ai.evil.example/v1',
                'https://notopencode.ai/v1',
                ''):
        assert not sends_session_header(provider(url)), url


def test_on_and_off_override_the_host():
    assert sends_session_header(provider('http://localhost:1234/v1',
                                         policy='on'))
    assert not sends_session_header(provider('https://opencode.ai/inference',
                                             policy='off'))


def test_an_unknown_policy_behaves_like_auto():
    assert sends_session_header(provider('https://opencode.ai/inference',
                                         policy='sometimes'))
    assert not sends_session_header(provider('http://localhost:1234/v1',
                                             policy='sometimes'))
    # a provider definition that predates the setting
    legacy = provider('https://opencode.ai/inference')
    del legacy.opencode_session
    assert sends_session_header(legacy)


def test_the_policy_survives_the_config_file():
    config = AppConfig()
    config.providers = [provider('https://opencode.ai/inference', policy='off')]
    raw = config_to_dict(config)
    assert raw['providers'][0]['opencode_session'] == 'off'
    assert config_from_dict(raw).providers[0].opencode_session == 'off'
    # a config written before the setting existed comes back as auto
    raw['providers'][0].pop('opencode_session')
    assert config_from_dict(raw).providers[0].opencode_session == 'auto'


# ── the ids ───────────────────────────────────────────────
def test_a_session_id_looks_like_consoles():
    session_id = new_session_id()
    assert session_id.startswith('ses_')
    assert len(session_id) > 20
    assert new_session_id() != session_id


def test_one_id_per_session_name(tmp_path):
    store = OpencodeSessions(tmp_path / 'sessions.json')
    first = store.current('20250101_120000')
    assert store.current('20250101_120000') == first     # stable across turns
    # an unnamed conversation (the engine names a session after its first save)
    # keeps the id it already has
    assert store.current('') == first
    other = store.current('20250101_130000')
    assert other != first
    assert store.of('20250101_120000') == first


def test_a_new_run_reads_the_ids_back(tmp_path):
    path = tmp_path / 'sessions.json'
    bound = OpencodeSessions(path).current('20250101_120000')
    reopened = OpencodeSessions(path)
    assert reopened.of('20250101_120000') == bound
    assert reopened.current('20250101_120000') == bound


def test_forget_rotates_the_live_id(tmp_path):
    store = OpencodeSessions(tmp_path / 'sessions.json')
    bound = store.current('20250101_120000')
    after = store.forget()
    assert after != bound
    assert store.current('') == after
    # the forgotten conversation is still remembered by name
    assert store.of('20250101_120000') == bound


def test_an_unreadable_store_is_ignored(tmp_path):
    path = tmp_path / 'sessions.json'
    path.write_text('{ not json', 'utf-8')
    store = OpencodeSessions(path)
    session_id = store.current('20250101_120000')
    assert session_id.startswith('ses_')
    assert OpencodeSessions(path).of('20250101_120000') == session_id


# ── the wire ──────────────────────────────────────────────
@pytest.fixture()
def gateway():
    instance = FakeLLMGateway(ScriptedResponder([
        Turn(text='first answer'),
        Turn(text='second answer'),
    ])).start()
    yield instance
    instance.stop()


def chat(client: LLMClient, text: str = 'hello') -> None:
    list(client.chat([{'role': 'user', 'content': text}],
                     client.provider.models[0], system='SYSTEM',
                     max_tokens=256))


@pytest.mark.parametrize('wire', WIRES)
def test_the_header_reaches_the_wire(gateway, wire):
    gateway.responder = EchoResponder()
    client = LLMClient(provider(gateway.base_url, wire=wire, policy='on'),
                       session_id=lambda: 'ses_from_the_test')
    try:
        chat(client)
    finally:
        client.close()
    assert headers(gateway.requests[-1])[SESSION_HEADER] == 'ses_from_the_test'


@pytest.mark.parametrize('wire', WIRES)
def test_the_id_is_read_per_request(gateway, wire):
    """A session command moves the conversation; the next request follows it."""
    gateway.responder = EchoResponder()
    live = ['ses_first']
    client = LLMClient(provider(gateway.base_url, wire=wire, policy='on'),
                       session_id=lambda: live[0])
    try:
        chat(client)
        live[0] = 'ses_second'
        chat(client)
    finally:
        client.close()
    sent = [headers(request)[SESSION_HEADER] for request in gateway.requests]
    assert sent == ['ses_first', 'ses_second']


@pytest.mark.parametrize('wire', WIRES)
def test_other_gateways_never_see_the_header(gateway, wire):
    """``auto`` sends nothing to a gateway that is not Console."""
    gateway.responder = EchoResponder()
    client = LLMClient(provider(gateway.base_url, wire=wire),
                       session_id=lambda: 'ses_from_the_test')
    try:
        chat(client)
    finally:
        client.close()
    assert SESSION_HEADER not in headers(gateway.requests[-1])


def test_off_silences_the_header(gateway):
    gateway.responder = EchoResponder()
    client = LLMClient(provider(gateway.base_url, policy='off'),
                       session_id=lambda: 'ses_from_the_test')
    try:
        chat(client)
    finally:
        client.close()
    assert SESSION_HEADER not in headers(gateway.requests[-1])


def test_an_unwired_client_sends_nothing(gateway):
    """Discovery and other one-off clients have no conversation to report."""
    gateway.responder = EchoResponder()
    client = LLMClient(provider(gateway.base_url, policy='on'))
    try:
        chat(client)
    finally:
        client.close()
    assert SESSION_HEADER not in headers(gateway.requests[-1])


# ── the conversation ──────────────────────────────────────
@pytest.fixture()
def app_home(tmp_path):
    home = tmp_path / 'adcode-home'
    paths.set_app_dir(home)
    try:
        yield home
    finally:
        paths.set_app_dir(None)


def wait_idle(loop: AgentLoop, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    quiet_since = None
    while time.time() < deadline:
        if not loop.busy and loop._work.qsize() == 0:
            quiet_since = quiet_since or time.time()
            if time.time() - quiet_since >= 0.1:
                return
        else:
            quiet_since = None
        time.sleep(0.01)
    raise TimeoutError('the engine never went idle')


def test_the_id_follows_the_conversation(gateway, app_home, tmp_path):
    """One turn, a second turn, a delete and a load: the id tracks each one."""
    gateway.responder = EchoResponder()
    config = build_config(gateway.base_url)
    config.providers[0].opencode_session = 'on'
    loop = AgentLoop(
        config, ToolRegistry(),
        DbSessionStore(db_path=tmp_path / 'sessions' / 'adcode.db'),
        GoalStore(tmp_path / 'goals.json'),
        SkillRegistry(config.skills, str(tmp_path)),
        emit=lambda kind, payload: None, cwd=str(tmp_path))
    loop.start()
    try:
        loop.submit('first conversation')
        wait_idle(loop)
        loop.save_active()
        name = loop.session_name
        first = headers(gateway.requests[-1])[SESSION_HEADER]
        assert first == loop.opencode_session_id()
        assert first.startswith('ses_')
        # the binding is on disk, so a later run resumes the same id
        assert OpencodeSessions().of(name) == first

        loop.submit('still the same conversation')
        wait_idle(loop)
        assert headers(gateway.requests[-1])[SESSION_HEADER] == first

        # /perish: the row goes away, the id must not follow the next one
        loop.forget_session()
        loop.clear()
        loop.submit('a new conversation')
        wait_idle(loop)
        loop.save_active()
        second = headers(gateway.requests[-1])[SESSION_HEADER]
        assert second != first

        # /sessions: loading the old conversation brings its id back
        assert loop.load_session(name)
        loop.submit('back to the first conversation')
        wait_idle(loop)
        assert headers(gateway.requests[-1])[SESSION_HEADER] == first
    finally:
        loop.shutdown()
