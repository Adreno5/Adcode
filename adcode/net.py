"""Network helpers with SOCKS proxy support.

The standard library only speaks HTTP proxies: with a ``socks5://`` proxy in
the environment every ``urllib`` request failed with ``unknown url type:
socks5``, and the OpenAI/Anthropic clients refused to start at all because
the HTTP library needs the ``socksio`` extra for SOCKS transports.

This module is the single place that decides how a request leaves the machine:

* proxy URLs are normalised (``socks://`` becomes ``socks5://``) so a value
  written by hand or read from the Windows registry never crashes the client,
* the HTTP library is used whenever a SOCKS proxy is configured, and ``urllib``
  otherwise, so the common case keeps its exact behaviour,
* the HTTP clients used by the provider SDKs are built here as well, which
  makes the SOCKS configuration effective for every wire protocol.
"""

from __future__ import annotations

import atexit
import io
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from email.message import Message
from typing import Any, Callable, Iterator

SOCKS_SCHEMES = frozenset({'socks', 'socks4', 'socks4a', 'socks5', 'socks5h'})

_PROXY_ENV_VARS = (
    'https_proxy', 'HTTPS_PROXY',
    'http_proxy', 'HTTP_PROXY',
    'all_proxy', 'ALL_PROXY',
)

_client_lock = threading.RLock()
_http_client: Any = None


class ProxyError(RuntimeError):
    """Raised when a configured proxy cannot be used."""


def _proxy_scheme(url: str) -> str:
    raw = (url or '').strip()
    if '://' not in raw:
        return 'http'
    return raw.partition('://')[0].lower()


def normalize_proxy_url(url: str) -> str:
    """Return a proxy URL the HTTP library understands.

    ``socks://`` is ambiguous in the wild (Windows writes it for SOCKS4), but
    every current proxy server speaks SOCKS5, so it is mapped onto
    ``socks5://``.  SOCKS4 is rejected with a message that says what to do
    instead of failing later with an opaque URL error.
    """
    raw = (url or '').strip()
    if not raw:
        return raw
    if '://' not in raw:
        raw = 'http://' + raw
    scheme, _, rest = raw.partition('://')
    scheme = scheme.lower()
    if scheme not in SOCKS_SCHEMES:
        return raw
    if scheme in ('socks4', 'socks4a'):
        raise ProxyError(
            f'SOCKS4 proxies are not supported ({raw}). '
            'Use a socks5:// proxy instead.')
    if scheme == 'socks':
        scheme = 'socks5'
    return f'{scheme}://{rest}'


def environment_proxies() -> dict[str, str | None]:
    """Proxy URLs per mount pattern, with SOCKS schemes normalised.

    the HTTP library resolves ``NO_PROXY`` itself, so its mapping is reused
    and only the URLs are rewritten.
    """
    httpx = httpx_module()
    try:
        get_environment_proxies = httpx._utils.get_environment_proxies
    except Exception:
        return {}
    try:
        raw = get_environment_proxies()
    except Exception:
        return {}
    proxies: dict[str, str | None] = {}
    for pattern, url in raw.items():
        if url is None:
            proxies[pattern] = None
            continue
        try:
            proxies[pattern] = normalize_proxy_url(url)
        except ProxyError:
            # An unsupported scheme must be reported, not silently ignored.
            raise
        except Exception:
            proxies[pattern] = url
    return proxies


def socks_proxy_url() -> str | None:
    """The SOCKS proxy configured in the environment, if there is one."""
    for name in _PROXY_ENV_VARS:
        value = os.environ.get(name)
        if not value:
            continue
        if _proxy_scheme(value) in SOCKS_SCHEMES:
            return normalize_proxy_url(value)
    return None


def has_socks_proxy() -> bool:
    """True when a SOCKS proxy is configured and has to be honoured."""
    try:
        return socks_proxy_url() is not None
    except ProxyError:
        return True


def httpx_module() -> Any:
    """The HTTP library the project depends on.

    AdCode pins ``httpx2``; the plain ``httpx`` is accepted as well, so the
    module keeps working in a checkout that still resolves the old package.
    """
    try:
        import httpx2 as module
    except ImportError:
        import httpx as module
    return module


def _socket_options() -> list[tuple[int, int, int | bool]]:
    import socket
    options: list[tuple[int, int, int | bool]] = [
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, True)]
    for name, level in (('TCP_KEEPINTVL', socket.IPPROTO_TCP),
                        ('TCP_KEEPCNT', socket.IPPROTO_TCP),
                        ('TCP_KEEPIDLE', socket.IPPROTO_TCP)):
        value = getattr(socket, name, None)
        if value is not None:
            options.append((level, value, 60 if name == 'TCP_KEEPINTVL' else 5))
    return options


def build_http_client(timeout: float = 600.0, follow_redirects: bool = True) -> Any:
    """Build an HTTP client that understands the configured proxy.

    The environment proxies are mounted by hand: the HTTP library builds
    its own mounts from the environment and refuses to start on a proxy
    scheme it does not know, which is the crash this module exists to prevent.
    """
    httpx = httpx_module()

    proxies = environment_proxies()
    mounts: dict[str, Any] = {}
    for pattern, url in proxies.items():
        if url is None:
            mounts[pattern] = None
            continue
        try:
            mounts[pattern] = httpx.HTTPTransport(
                proxy=url, socket_options=_socket_options())
        except ImportError as error:
            # The HTTP library needs the ``socksio`` extra for SOCKS
            # transports; without it the client cannot be built at all.
            raise ProxyError(
                'SOCKS proxy support is missing: install the project with its '
                'SOCKS extra (pip install \"httpx2[socks]\") or remove the '
                f'proxy from the environment. ({error})') from error
    return httpx.Client(
        transport=httpx.HTTPTransport(socket_options=_socket_options()),
        mounts=mounts,
        timeout=timeout,
        follow_redirects=follow_redirects,
    )


def http_client() -> Any:
    """A shared client for the urllib-style helpers."""
    global _http_client
    with _client_lock:
        if _http_client is None:
            _http_client = build_http_client(timeout=60.0)
            atexit.register(_close_http_client)
        return _http_client


def _close_http_client() -> None:
    global _http_client
    with _client_lock:
        client = _http_client
        _http_client = None
    if client is not None:
        try:
            client.close()
        except Exception:
            pass


def _headers(items: Any) -> 'Message':
    """Case-insensitive header mapping, like the one urllib returns."""
    headers = Message()
    for key, value in items:
        headers[str(key)] = str(value)
    return headers


@dataclass
class HttpResponse:
    """A minimal, urllib-shaped response."""

    status: int
    headers: 'Message' = field(default_factory=Message)
    body: bytes = b''
    url: str = ''
    _closed: bool = False

    def read(self, amount: int | None = None) -> bytes:
        if amount is None:
            return self.body
        chunk = self.body[:amount]
        self.body = self.body[amount:]
        return chunk

    def geturl(self) -> str:
        return self.url

    def getcode(self) -> int:
        return self.status

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> 'HttpResponse':
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def _request_parts(request: Any) -> tuple[str, str, dict[str, str], bytes | None]:
    url = getattr(request, 'full_url', None) or getattr(request, 'url', '')
    method = getattr(request, 'method', None) or 'GET'
    headers = dict(getattr(request, 'headers', {}) or {})
    data = getattr(request, 'data', None)
    if isinstance(data, str):
        data = data.encode('utf-8')
    return str(url), str(method), headers, data


def urlopen(request: Any, timeout: float = 30.0) -> HttpResponse:
    """``urllib.request.urlopen`` that also works behind a SOCKS proxy."""
    url, method, headers, data = _request_parts(request)
    if not has_socks_proxy():
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return HttpResponse(
                status=int(response.status),
                headers=_headers(response.headers.items()),
                body=body,
                url=str(response.geturl()),
            )
    httpx = httpx_module()
    try:
        response = http_client().request(
            method, url, headers=headers, content=data, timeout=timeout)
    except httpx.HTTPError as error:
        raise URLError(str(error)) from error
    body = response.content
    if response.status_code >= 400:
        # ``urllib.error.HTTPError`` carries the body so callers that read it
        # keep working; the standard library hands over a file object.
        raise urllib.error.HTTPError(
            url, response.status_code, response.reason_phrase,
            dict(response.headers), io.BytesIO(body))
    return HttpResponse(
        status=response.status_code,
        headers=_headers(response.headers.items()),
        body=body,
        url=str(response.url),
    )


class URLError(urllib.error.URLError):
    """Raised when a proxied request cannot be performed."""


def stream_lines(request: Any, timeout: float = 60.0,
                 on_open: Callable[[Any], None] | None = None) -> Iterator[str]:
    """Yield decoded lines of a streaming response (used by the SSE transport).

    The standard library has no streaming HTTP client, so a SOCKS proxy could
    not be used at all for server-sent events.  ``httpx`` streams the body and
    the caller keeps the usual line-by-line protocol.
    """
    url, method, headers, data = _request_parts(request)
    httpx = httpx_module()
    client = build_http_client(timeout=timeout, follow_redirects=True)
    try:
        with client.stream(method, url, headers=headers, content=data,
                           timeout=timeout) as response:
            if response.status_code >= 400:
                raise urllib.error.HTTPError(
                    url, response.status_code, response.reason_phrase,
                    dict(response.headers), io.BytesIO(b''))
            if on_open is not None:
                on_open(response)
            for line in response.iter_lines():
                yield line
    except httpx.HTTPError as error:
        raise URLError(str(error)) from error
    finally:
        try:
            client.close()
        except Exception:
            pass
