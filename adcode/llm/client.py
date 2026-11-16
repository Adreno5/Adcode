"""Provider streaming adapter on the engine sdk.

The engine (harness.agent.ConversationEngine) drives any ``IChatClient``.
This module is adcode's implementation: one provider connection, one wire
protocol, chosen by ``ProviderConfig.wire`` and built from the sdk's
OpenAI-compatible and Anthropic-compatible clients.  On top of the wire it
adds what stays application-shaped here:

- ``[image: path]`` markers expand into native image parts while the request
  is built, so base64 never enters the stored conversation; the sdk's
  FilesUploader decides whether a marker becomes a Files API id or an
  inline image, with the inline fallback when the provider refuses;
- ``provider.reasoning_extra_body`` merges into the request's extra body
  whenever a reasoning dial is in play;
- an interrupt stops the stream at the next chunk.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Callable, Generator

from infra.net import build_async_http_client, build_http_client
from infra.orchestrator.anthropic import AnthropicCompatClient
from infra.orchestrator.base import effort_stream_params, retrying_stream
from infra.orchestrator.files import (
    FilesUploader, file_reference_rejected, references_file,
)
from infra.orchestrator.openai import OpenAICompatClient

from harness.images import (
    FileIdCache, ImageRef, data_url, has_image_marker, image_marker,
    iter_segments, resolve_image,
)

from adcode.config import ModelConfig, ProviderConfig


def asyncio_run(coroutine: Any) -> Any:
    """asyncio.run that also works when a loop is already running here."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()


import anthropic
import openai


def _image_parts(text: Any, *, base_dir: str = '', wire: str = 'chat',
                 file_ids: Callable[[ImageRef], str] | None = None
                 ) -> list[dict[str, Any]] | None:
    """Replace ``[image: path]`` markers with chat-shaped image parts.

    Chat completions passes these parts through; the responses and anthropic
    converters in the sdk map them onto their native blocks.  A part carries
    either a data URL or - when ``file_ids`` resolves one and the wire
    supports it - a Files API id (``{"image_url": {"file_id": ...}}``).
    """
    if not isinstance(text, str) or not has_image_marker(text):
        return None
    parts: list[dict[str, Any]] = []
    replaced = False
    for segment, chunk in iter_segments(text):
        if segment == 'text':
            if chunk:
                parts.append({'type': 'text', 'text': chunk})
            continue
        ref = resolve_image(chunk, base_dir)
        if ref is None:
            # Keep the marker visible instead of silently dropping it.
            parts.append({'type': 'text', 'text': image_marker(chunk)})
            continue
        file_id = ''
        # Chat completions has no file id, so never spend an upload on it.
        if wire != 'chat' and file_ids is not None and ref.path is not None:
            file_id = file_ids(ref) or ''
        replaced = True
        if file_id:
            parts.append({'type': 'image_url', 'image_url': {'file_id': file_id}})
        else:
            parts.append({'type': 'image_url', 'image_url': {'url': data_url(ref)}})
    return parts if replaced else None


class LLMClient:
    """One provider connection, one wire, IChatClient-compatible.

    Args:
        provider: the provider definition (wire, endpoint, credentials).
        model: the selected model; its id names the wire's model field.
        base_dir: relative ``[image: ...]`` markers resolve against it.
        file_cache: optional Files API id cache (responses / anthropic wires).
        warn: optional sink for one-time notices about upload fallbacks.
    """

    def __init__(self, provider: ProviderConfig, model: ModelConfig | None = None,
                 *, base_dir: str = '', file_cache: FileIdCache | None = None,
                 warn: Callable[[str], None] | None = None,
                 sdk: Any | None = None) -> None:
        self.provider = provider
        self.name = provider.name
        self.wire = provider.wire
        # No explicit model: fall back to the provider's first one, so a
        # client built for discovery still names a model on the wire.
        self.model = (model or (provider.models[0] if provider.models
                                else None))
        self.model = self.model.id if self.model is not None else ''
        model = model or (provider.models[0] if provider.models else None)
        self.base_dir = base_dir
        self.file_cache = file_cache
        self.warn = warn
        self._model_config = model
        # Set by the engine: a threading.Event that is set when the user
        # interrupts.  The stream stops at the next chunk, so an interrupt is
        # not stuck behind a slow or endless provider response.
        self.interrupt: Any = None
        headers = dict(provider.extra_headers) if provider.extra_headers else None
        self._reasoning_extra_body = dict(provider.reasoning_extra_body or {})
        # The SDKs build their own HTTP client from the environment, and that
        # client refuses to start on a proxy scheme it does not know.  Both
        # transports are built here, so a SOCKS proxy works on every wire and
        # on both the sync and the async side.
        self._http = build_http_client(timeout=600.0)
        self._async_http = build_async_http_client(timeout=600.0)
        if 'openai' in self.wire:
            protocol = 'responses' if 'responses' in self.wire else 'chat'
            self._client: Any = OpenAICompatClient(
                api_key=provider.api_key or 'not-needed',
                base_url=provider.base_url,
                model=self.model,
                extra_headers=headers,
                protocol=protocol,
                max_retries=0,
                http_client=self._http,
                async_http_client=self._async_http,
            )
        else:
            self._client = AnthropicCompatClient(
                api_key=provider.api_key or 'not-needed',
                base_url=provider.base_url,
                model=self.model,
                extra_headers=headers,
                max_retries=0,
                timeout=600.0,
                http_client=self._http,
                async_http_client=self._async_http,
            )
        # The raw SDK object drives the Files API and model discovery; the
        # uploader owns the upload-once / inline-fallback trade-off.  ``sdk``
        # is a test seam: inject a fake raw client before the uploader exists.
        self._sdk = sdk if sdk is not None else self._client._client
        self._files = FilesUploader(
            self._sdk, file_cache, self._file_scope(),
            api='anthropic' if self.wire == 'anthropic messages' else 'openai',
            warn=self._note)

    def close(self) -> None:
        """Release the underlying HTTP connection pools."""
        for client in (self._http, self._async_http):
            try:
                client.close()
            except Exception:
                pass

    # ── synchronous adcode-shaped adapter ───────────────────
    async def _chat_events(self, messages: list[dict[str, Any]],
                           model: ModelConfig, *, system: str,
                           max_tokens: int, tools: list[dict[str, Any]] | None,
                           prompt_cache_key: str | None, effort: str,
                           force_reasoning: bool) -> AsyncGenerator[dict[str, Any], None]:
        """Async generator of adcode-shaped events over the sdk stream.

        Event kinds: retry / thinking / text / tool_call_start /
        tool_call_delta / tool_call / usage / finish.  The stream is wrapped
        with the sdk's stream-creation retry; the finish event is always
        emitted (a stream that ends without a signal reports 'incomplete').
        """
        wire = self._wire_kind()
        wire_kwargs = effort_stream_params(
            wire, effort=effort, reasoning_style=model.reasoning_style,
            reasoning_efforts=tuple(model.reasoning_efforts or ()),
            reasoning=bool(model.reasoning), force=force_reasoning,
            prompt_cache_key=prompt_cache_key)
        wire_kwargs = self._with_provider_extras(wire_kwargs)
        wire_kwargs['max_tokens'] = max_tokens

        msgs: list[Any] = []
        if system:
            msgs.append({'role': 'system', 'content': system})
        # stream() expands the image markers: it must see the raw stored text
        # so the file-id fallback can rebuild inline
        msgs.extend(messages)

        buffers: dict[int, dict[str, str]] = {}
        name_seen: set[int] = set()
        finish_reason: str | None = None

        def drain_tools() -> Generator[dict[str, Any], None, None]:
            for index in sorted(buffers):
                call = buffers[index]
                if not call['name']:
                    continue
                try:
                    args = json.loads(call['arguments'] or '{}')
                except ValueError:
                    args = {}
                if not isinstance(args, dict):
                    args = {'value': args}
                yield {'kind': 'tool_call', 'id': call['id'] or f'call_{index}',
                       'name': call['name'], 'args': args}

        async for event in retrying_stream(self, msgs, tools, attempts=3,
                                           **wire_kwargs):
            if isinstance(event, str):
                yield {'kind': 'text', 'content': event}
                continue
            if not isinstance(event, dict):
                continue
            retry = event.get('retry')
            if retry:
                yield {'kind': 'retry', 'content': retry.get('error', ''),
                       'attempt': retry.get('attempt', 0),
                       'delay': retry.get('delay', 0)}
                continue
            reasoning = event.get('reasoning')
            if reasoning:
                yield {'kind': 'thinking', 'content': reasoning}
            for item in event.get('tool_calls', []) or []:
                index = item.get('index', 0)
                buffer = buffers.setdefault(
                    index, {'id': '', 'name': '', 'arguments': ''})
                if item.get('id'):
                    buffer['id'] = item['id']
                name = item.get('name') or ''
                if name and name != buffer['name']:
                    buffer['name'] = name
                    if index not in name_seen:
                        name_seen.add(index)
                        yield {'kind': 'tool_call_start',
                               'id': buffer['id'] or str(index), 'name': name}
                fragment = item.get('arguments') or ''
                if fragment:
                    buffer['arguments'] += fragment
                    yield {'kind': 'tool_call_delta',
                           'id': buffer['id'] or str(index), 'name': name,
                           'delta': fragment}
            if event.get('usage'):
                yield {'kind': 'usage', 'usage': event['usage']}
            if event.get('finish'):
                finish_reason = str(event['finish'])
        for tool_event in drain_tools():
            yield tool_event
        yield {'kind': 'finish', 'reason': finish_reason or 'incomplete'}

    def chat(self, messages: list[dict[str, Any]], model: ModelConfig | None = None,
             *, system: str = '', max_tokens: int = 8_192,
             tools: list[dict[str, Any]] | None = None,
             prompt_cache_key: str | None = None,
             effort: str = 'medium', force_reasoning: bool = False,
             ) -> Generator[dict[str, Any], None, None]:
        """Synchronous generator of adcode-shaped events.

        Buffered convenience adapter for tools and tests: the whole stream is
        collected through a temporary event loop and then yielded.  Callers
        that need incremental events use ``stream()`` directly.  Must not be
        called from a thread with a running event loop.
        """
        model = model or self._model_config or ModelConfig()

        async def collect() -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            async for event in self._chat_events(
                    messages, model, system=system, max_tokens=max_tokens,
                    tools=tools, prompt_cache_key=prompt_cache_key,
                    effort=effort, force_reasoning=force_reasoning):
                out.append(event)
            return out

        yield from asyncio_run(collect())

    # ── IChatClient surface ─────────────────────────────────
    async def stream(self, messages: list[Any], tools: Any = None,
                     **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Stream one model call; yields str deltas and dict events.

        A provider that refuses a Files API reference (400/403/404/422)
        triggers exactly one inline rebuild, before anything reached the
        caller.

        The stream also stops at the next chunk when the turn was
        interrupted, so an interrupt is not stuck behind a slow or endless
        provider response.
        """
        prepared = self._prepared(messages)
        kwargs = self._with_provider_extras(kwargs)
        try:
            async for event in self._client.stream(prepared, tools, **kwargs):
                yield event
                if self._interrupted():
                    # The user stopped the turn: stop the stream too.
                    return
            return
        except (openai.APIError, anthropic.APIError) as error:
            if not references_file(prepared) or not file_reference_rejected(error):
                raise
            self._files.fallback_inline(f'file id rejected: {error}')
            prepared = self._prepared(messages)
            async for event in self._client.stream(prepared, tools, **kwargs):
                yield event
                if self._interrupted():
                    return

    def _interrupted(self) -> bool:
        """True when the user asked to stop."""
        event = self.interrupt
        if event is None:
            return False
        try:
            return bool(event.is_set())
        except Exception:
            return False

    def invoke(self, messages: list[Any], tools: Any = None,
               **kwargs: Any) -> Any:
        """One non-streaming call (protocol completeness; the engine streams)."""
        return self._client.invoke(self._prepared(messages), tools, **kwargs)

    # ── request building ────────────────────────────────────
    def _wire_kind(self) -> str:
        if self.wire == 'anthropic messages':
            return 'anthropic'
        if self.wire == 'openai responses':
            return 'responses'
        return 'chat'

    def _prepared(self, messages: list[Any]) -> list[Any]:
        """Expand image markers and normalize the message shapes.

        The engine hands over ChatMessage objects; normalize them first, or
        the markers inside their contents would never be expanded.  The chat
        shape itself is translated by the sdk's wire converters, which now
        take plain dicts too.
        """
        wire = self._wire_kind()
        prepared: list[Any] = []
        for message in messages:
            if not isinstance(message, dict):
                if hasattr(message, 'to_dict') and callable(message.to_dict):
                    message = message.to_dict()
                else:
                    prepared.append(message)
                    continue
            copy = dict(message)
            content = copy.get('content')
            # user content only: tool messages must stay plain text (their
            # protocol shape is a string on every wire), and the engine
            # mirrors tool-result images into a <tool_images> user message
            if (copy.get('role') == 'user' and isinstance(content, str)
                    and has_image_marker(content)):
                parts = _image_parts(content, base_dir=self.base_dir,
                                     wire=wire,
                                     file_ids=self._file_id)
                if parts is not None:
                    copy['content'] = parts
            prepared.append(copy)
        return prepared

    def _with_provider_extras(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Merge provider-private reasoning fields into the extra body."""
        if not self._reasoning_extra_body or 'openai' not in self.wire:
            return kwargs
        if not ({'reasoning_effort', 'reasoning', 'effort'} & kwargs.keys()):
            return kwargs
        merged = dict(self._reasoning_extra_body)
        merged.update(kwargs.get('extra_body') or {})
        kwargs['extra_body'] = merged
        return kwargs

    # ── Files API ───────────────────────────────────────────
    def _file_scope(self) -> str:
        from urllib.parse import urlsplit
        endpoint = urlsplit(self.provider.base_url or '').netloc.lower()
        return f'{self.provider.name}|{endpoint}'

    def _file_id(self, ref: ImageRef) -> str:
        """The Files API id for an image, uploaded once per endpoint."""
        if ref.path is None:
            return ''
        return self._files.file_id(ref.path, ref.media_type, ref.size)

    def _note(self, text: str) -> None:
        if self.warn is None:
            return
        try:
            self.warn(text)
        except Exception:
            pass

    # ── model discovery ─────────────────────────────────────
    def list_models(self) -> list[str]:
        """The model ids this endpoint exposes (the sdk walks the pages)."""
        return self._client.list_models()


def fetch_model_ids(provider: ProviderConfig, wire: str | None = None) -> list[str]:
    """Fetch model ids for a provider definition, optionally with another wire."""
    probe = ProviderConfig(
        name=provider.name, base_url=provider.base_url, api_key=provider.api_key,
        wire=wire or provider.wire, models=provider.models,
    )
    return LLMClient(probe).list_models()
