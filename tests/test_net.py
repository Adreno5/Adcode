"""Proxy handling: SOCKS URLs, the HTTP clients and the urllib helpers."""

import urllib.error
import urllib.request

import pytest

from infra import net
from adcode.config import ModelConfig, ProviderConfig
from adcode.llm.client import LLMClient


httpx = net.httpx_module()

PROXY_VARS = ('http_proxy', 'HTTP_PROXY', 'https_proxy', 'HTTPS_PROXY',
              'all_proxy', 'ALL_PROXY', 'no_proxy', 'NO_PROXY')


@pytest.fixture(autouse=True)
def clean_proxy_env(monkeypatch):
    for name in PROXY_VARS:
        monkeypatch.delenv(name, raising=False)


# proxy URLs
def test_socks_urls_are_normalised():
    assert net.normalize_proxy_url('socks://127.0.0.1:1080') == \
        'socks5://127.0.0.1:1080'
    assert net.normalize_proxy_url('SOCKS5://127.0.0.1:1080') == \
        'socks5://127.0.0.1:1080'
    assert net.normalize_proxy_url('socks5h://127.0.0.1:1080') == \
        'socks5h://127.0.0.1:1080'
    assert net.normalize_proxy_url('http://127.0.0.1:7890') == \
        'http://127.0.0.1:7890'
    assert net.normalize_proxy_url('127.0.0.1:7890') == 'http://127.0.0.1:7890'


def test_socks4_is_rejected_with_a_clear_message():
    with pytest.raises(net.ProxyError) as info:
        net.normalize_proxy_url('socks4://127.0.0.1:1080')
    assert 'socks5' in str(info.value)


def test_environment_proxies_are_normalised(monkeypatch):
    monkeypatch.setenv('ALL_PROXY', 'socks://127.0.0.1:1080')
    proxies = net.environment_proxies()
    assert proxies
    for url in proxies.values():
        if url:
            assert url.startswith('socks5://')


def test_the_socks_proxy_is_detected(monkeypatch):
    assert net.socks_proxy_url() is None
    monkeypatch.setenv('ALL_PROXY', 'socks5://127.0.0.1:1080')
    assert net.socks_proxy_url() == 'socks5://127.0.0.1:1080'
    assert net.has_socks_proxy() is True
    monkeypatch.setenv('ALL_PROXY', 'http://127.0.0.1:7890')
    assert net.has_socks_proxy() is False


# HTTP clients
def test_the_http_client_mounts_the_socks_proxy(monkeypatch):
    monkeypatch.setenv('ALL_PROXY', 'socks5://127.0.0.1:1080')
    client = net.build_http_client(timeout=5)
    try:
        transports = [transport for transport in client._mounts.values()
                      if transport is not None]
        assert transports, 'the proxy was not mounted'
    finally:
        client.close()


def test_the_provider_client_accepts_a_socks_proxy(monkeypatch):
    """A SOCKS proxy used to crash the whole app on start-up."""
    monkeypatch.setenv('ALL_PROXY', 'socks5://127.0.0.1:1080')
    provider = ProviderConfig(name='P', base_url='http://127.0.0.1:1/v1',
                              api_key='k', models=[ModelConfig(id='m')])
    client = LLMClient(provider)
    try:
        assert client._http is not None
    finally:
        client.close()


def test_an_unsupported_proxy_scheme_is_reported(monkeypatch):
    monkeypatch.setenv('ALL_PROXY', 'socks4://127.0.0.1:1080')
    provider = ProviderConfig(name='P', base_url='http://127.0.0.1:1/v1',
                              api_key='k', models=[ModelConfig(id='m')])
    with pytest.raises(net.ProxyError):
        LLMClient(provider)


# urllib helpers
def test_urlopen_uses_httpx_behind_a_socks_proxy(monkeypatch):
    def handler(request):
        return httpx.Response(200, text='hello', headers={'X-Test': 'yes'})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(net, 'has_socks_proxy', lambda: True)
    monkeypatch.setattr(net, 'http_client', lambda: client)

    request = urllib.request.Request('https://example.com/data')
    with net.urlopen(request, timeout=5) as response:
        assert response.status == 200
        assert response.read() == b'hello'
        assert response.headers.get('X-Test') == 'yes'


def test_urlopen_reports_http_errors_with_the_body(monkeypatch):
    def handler(request):
        return httpx.Response(404, text='missing')

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(net, 'has_socks_proxy', lambda: True)
    monkeypatch.setattr(net, 'http_client', lambda: client)

    request = urllib.request.Request('https://example.com/data')
    with pytest.raises(urllib.error.HTTPError) as info:
        net.urlopen(request, timeout=5)
    assert info.value.code == 404
    assert info.value.read() == b'missing'


def test_urlopen_keeps_urllib_without_a_socks_proxy(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        raise urllib.error.HTTPError('https://example.com', 500, 'boom', {}, None)

    monkeypatch.setattr(net, 'has_socks_proxy', lambda: False)
    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)

    request = urllib.request.Request('https://example.com/data')
    with pytest.raises(urllib.error.HTTPError):
        net.urlopen(request, timeout=5)
    assert calls == [request]
