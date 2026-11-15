"""Streaming protocol adapters for the supported API wires.

All model operations in AdCode are streamed.  The adapters translate the
provider-specific event stream into a small shared vocabulary consumed by the
agent loop and the sub-agent runtime.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Generator

import anthropic
import openai

from adcode.config import ModelConfig, ProviderConfig
from adcode.net import build_http_client
from adcode.images import (
    FileIdCache, ImageRef, anthropic_source, data_url, has_image_marker,
    image_marker, iter_segments, resolve_image,
)

ChatEvent = dict[str, Any]

_EFFORT_RANK = {
    'low': 1, 'medium': 2, 'high': 3, 'xhigh': 4, 'max': 5, 'ultra': 6,
}
_EFFORT_ALIASES = {
    'xhigh': ('xhigh', 'high', 'max'),
    'ultra': ('ultra', 'max', 'xhigh'),
    'max': ('max', 'xhigh', 'high'),
}


def resolve_effort(configured: str, available: list[str] | tuple[str, ...]) -> str:
    """Clamp the user's effort to a value the model actually exposes."""
    if not available:
        return configured
    options = [str(item).lower() for item in available if item]
    if not options:
        return configured
    configured = configured.lower()
    if configured in options:
        return configured
    for alias in _EFFORT_ALIASES.get(configured, ()):
        if alias in options:
            return alias
    rank = _EFFORT_RANK.get(configured, 2)
    ranked = sorted(options, key=lambda item: abs(_EFFORT_RANK.get(item, 2) - rank))
    return ranked[0]


def _retry_generator(events_factory: Any, attempts: int = 3) -> Generator[ChatEvent, None, None]:
    """Retry only the stream *creation*; mid-stream failures surface to the caller."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        emitted = False
        try:
            stream = events_factory()
            for event in stream:
                emitted = True
                yield event
            return
        except (openai.APIError, anthropic.APIError) as error:
            status = getattr(error, 'status_code', 0) or getattr(error, 'status', 0)
            if emitted or status == 400 or attempt == attempts - 1:
                raise
            last_error = error
            wait = 2 ** attempt
            yield {'kind': 'retry', 'content': str(error)[:160], 'attempt': attempt + 1,
                   'delay': wait}
            time.sleep(wait)
        except Exception:
            raise
    if last_error is not None:
        raise last_error


def _tool_call_payload(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    try:
        args = json.loads(arguments) if arguments else {}
    except Exception:
        args = {}
    if not isinstance(args, dict):
        args = {'value': args}
    return {'kind': 'tool_call', 'id': call_id, 'name': name, 'args': args}


def _image_parts(text: Any, kind: str, base_dir: str = '',
                 file_ids: Callable[[ImageRef], str] | None = None
                 ) -> list[dict[str, Any]] | None:
    """Replace ``[image: path]`` markers with native content parts.

    ``kind`` selects the wire dialect: ``image_url`` for chat completions,
    ``input_image`` for the Responses API and ``anthropic`` for Messages.
    ``file_ids`` resolves an image to a Files API id; when it returns one the
    part only references the id.  Otherwise the image is inlined, and the bytes
    are read here, while the request is built, so base64 never enters the stored
    conversation.
    """
    if not isinstance(text, str) or not has_image_marker(text):
        return None
    text_type = 'input_text' if kind == 'input_image' else 'text'
    parts: list[dict[str, Any]] = []
    replaced = False
    for segment, chunk in iter_segments(text):
        if segment == 'text':
            if chunk:
                parts.append({'type': text_type, 'text': chunk})
            continue
        ref = resolve_image(chunk, base_dir)
        if ref is None:
            # Keep the marker visible instead of silently dropping it.
            parts.append({'type': text_type, 'text': image_marker(chunk)})
            continue
        file_id = ''
        # Chat completions has no file id, so never spend an upload on it.
        if kind != 'image_url' and file_ids is not None and ref.path is not None:
            file_id = file_ids(ref) or ''
        replaced = True
        if kind == 'image_url':
            parts.append({'type': 'image_url', 'image_url': {'url': data_url(ref)}})
        elif kind == 'input_image':
            if file_id:
                parts.append({'type': 'input_image', 'file_id': file_id,
                              'detail': 'auto'})
            else:
                parts.append({'type': 'input_image', 'image_url': data_url(ref),
                              'detail': 'auto'})
        else:
            parts.append({'type': 'image',
                          'source': anthropic_source(ref, file_id)})
    return parts if replaced else None


class LLMClient:
    """One provider connection, one wire protocol, streaming everywhere."""

    def __init__(self, provider: ProviderConfig) -> None:
        self.provider = provider
        self.name = provider.name
        self.wire = provider.wire
        # Set by the loop; relative ``[image: ...]`` markers resolve against it.
        self.base_dir = ''
        # Optional Files API cache and warning sink, also set by the loop.
        self.file_cache: FileIdCache | None = None
        self.warn: Callable[[str], None] | None = None
        self._upload_notes: set[str] = set()
        headers = dict(provider.extra_headers) if provider.extra_headers else None
        # The SDKs build their own HTTP client from the environment, and that
        # client refuses to start on a proxy scheme it does not know.  The
        # client is built here instead, so a SOCKS proxy in the environment
        # works for every wire protocol.
        self._http = build_http_client(timeout=600.0)
        if 'openai' in self.wire:
            self._client: Any = openai.OpenAI(
                base_url=provider.base_url,
                api_key=provider.api_key or 'not-needed',
                default_headers=headers,
                http_client=self._http,
                timeout=600.0,
                max_retries=0,
            )
        else:
            self._client = anthropic.Anthropic(
                base_url=provider.base_url,
                api_key=provider.api_key or 'not-needed',
                default_headers=headers,
                http_client=self._http,
                timeout=600.0,
                max_retries=0,
            )

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        try:
            self._client.close()
        except Exception:
            pass

    # ── protocol-neutral entry point ────────────────────────
    def chat(self, messages: list[dict[str, Any]], model: ModelConfig, *,
             system: str = '', max_tokens: int = 8_192,
             tools: list[dict[str, Any]] | None = None,
             prompt_cache_key: str | None = None,
             effort: str = 'medium', force_reasoning: bool = False,
             ) -> Generator[ChatEvent, None, None]:
        if self.wire == 'openai chat completions':
            yield from self._openai_chat(
                messages, model, system, max_tokens, tools, effort, force_reasoning)
        elif self.wire == 'openai responses':
            yield from self._openai_responses(
                messages, model, system, max_tokens, tools, prompt_cache_key,
                effort, force_reasoning)
        elif self.wire == 'anthropic messages':
            yield from self._anthropic(
                messages, model, system, max_tokens, tools, effort, force_reasoning)
        else:
            raise ValueError(f'unsupported wire protocol: {self.wire}')

    def reasoning_enabled(self, model: ModelConfig, force: bool) -> bool:
        return bool(model.reasoning or force)

    # ── image parts ─────────────────────────────────────────
    def _chat_content(self, text: str) -> list[dict[str, Any]] | None:
        """Chat-completions user content with native image parts."""
        return _image_parts(text, 'image_url', self.base_dir)

    def _responses_content(self, text: str) -> list[dict[str, Any]] | None:
        """Responses user content: ``input_text`` / ``input_image`` parts."""
        return _image_parts(text, 'input_image', self.base_dir, self._file_id)

    def _anthropic_content(self, text: str) -> list[dict[str, Any]] | None:
        """Messages user content: ``text`` / ``image`` blocks."""
        return _image_parts(text, 'anthropic', self.base_dir, self._file_id)

    # ── Files API ───────────────────────────────────────────
    def _file_scope(self) -> str:
        base = str(self.provider.base_url or '').rstrip('/')
        return f'{self.provider.name}|{base}'

    def _file_key(self, ref: ImageRef) -> str:
        path = ref.path
        if path is None:
            return ''
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        return f'{resolved}|{ref.size}'

    def _file_id(self, ref: ImageRef) -> str:
        """Return the Files API id for an image, uploading it once.

        An endpoint that rejects the upload is remembered in the cache, so the
        first failure costs one request rather than one per turn; the caller
        then falls back to an inline image.
        """
        cache = getattr(self, 'file_cache', None)
        if cache is None or ref.path is None:
            return ''
        scope = self._file_scope()
        key = self._file_key(ref)
        if not key:
            return ''
        # A rejected file reference disables the scope even while ids remain
        # cached: they are only valid until the provider says otherwise.
        if cache.unsupported(scope):
            return ''
        cached = cache.get_id(scope, key)
        if cached:
            return cached
        try:
            file_id = self._upload(ref)
        except Exception as error:
            self._upload_failed(scope, error)
            return ''
        if not file_id:
            self._upload_failed(scope, None)
            return ''
        cache.set_id(scope, key, file_id)
        return file_id

    def _upload(self, ref: ImageRef) -> str:
        """Upload one image with the provider's Files API and return its id."""
        upload = (ref.path.name, ref.path.read_bytes(), ref.media_type)
        if self.wire == 'anthropic messages':
            created = self._client.files.upload(file=upload)
        else:
            created = self._client.files.create(file=upload, purpose='vision')
        return str(getattr(created, 'id', '') or '')

    def _upload_failed(self, scope: str, error: Exception | None) -> None:
        status = 0
        if error is not None:
            status = int(getattr(error, 'status_code', 0)
                         or getattr(error, 'status', 0) or 0)
        # Only a definite refusal means the endpoint has no Files API; a
        # network blip or a 5xx should be retried on the next request.
        if error is None or status in (400, 401, 403, 404, 405, 415, 501):
            cache = getattr(self, 'file_cache', None)
            if cache is not None:
                cache.mark_unsupported(
                    scope, str(error) if error is not None else 'upload rejected')
            self._note(f'Files API unavailable at {scope}; images are sent inline.')
        else:
            self._note(f'Image upload failed ({error}); sending the image inline.')

    def _note(self, text: str) -> None:
        notes = self.__dict__.setdefault('_upload_notes', set())
        if text in notes:
            return
        notes.add(text)
        warn = getattr(self, 'warn', None)
        if warn is not None:
            try:
                warn(text)
            except Exception:
                pass

    @staticmethod
    def _has_file_id(payload: Any) -> bool:
        """True when a built request body references an uploaded file."""
        if isinstance(payload, dict):
            file_id = payload.get('file_id')
            if isinstance(file_id, str) and file_id:
                return True
            return any(LLMClient._has_file_id(value) for value in payload.values())
        if isinstance(payload, (list, tuple)):
            return any(LLMClient._has_file_id(value) for value in payload)
        return False

    def _fallback_inline(self, error: Exception) -> None:
        """Stop using uploaded file ids after the provider rejected one."""
        cache = getattr(self, 'file_cache', None)
        if cache is not None:
            cache.mark_unsupported(self._file_scope(), f'file id rejected: {error}')
        self._note('The provider rejected the uploaded image reference; '
                   'images are sent inline from now on.')

    @staticmethod
    def _file_id_rejected(error: Exception) -> bool:
        """True for the statuses a provider uses to refuse a file reference."""
        status = int(getattr(error, 'status_code', 0)
                     or getattr(error, 'status', 0) or 0)
        return status in (400, 403, 404, 422)

    def _openai_chat_messages(self, messages: list[dict[str, Any]]
                              ) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                prepared.append(message)
                continue
            parts = (self._chat_content(message.get('content'))
                     if message.get('role') == 'user' else None)
            if parts is None:
                prepared.append(message)
            else:
                prepared.append({**message, 'content': parts})
        return prepared

    # ── OpenAI chat completions ─────────────────────────────
    def _openai_chat(self, messages: list[dict[str, Any]], model: ModelConfig,
                     system: str, max_tokens: int,
                     tools: list[dict[str, Any]] | None,
                     effort: str, force: bool) -> Generator[ChatEvent, None, None]:
        msgs: list[dict[str, Any]] = self._openai_chat_messages(messages)
        if system:
            msgs = [{'role': 'system', 'content': system}, *msgs]
        kwargs: dict[str, Any] = {
            'model': model.id,
            'messages': msgs,
            'max_tokens': max_tokens,
            'stream': True,
            'stream_options': {'include_usage': True},
        }
        if tools:
            kwargs['tools'] = tools
        if self.reasoning_enabled(model, force):
            if model.reasoning_style in ('effort', 'none'):
                kwargs['reasoning_effort'] = resolve_effort(
                    effort, model.reasoning_efforts)
            extra_body = dict(self.provider.reasoning_extra_body)
            if model.reasoning_style == 'toggle' and not extra_body:
                extra_body = {'thinking': {'type': 'enabled'}}
            if extra_body:
                kwargs['extra_body'] = extra_body

        def factory() -> Generator[ChatEvent, None, None]:
            try:
                stream = self._client.chat.completions.create(**kwargs)
            except openai.APIStatusError as error:
                # Several OpenAI-compatible servers reject stream_options.
                if error.status_code != 400 or 'stream_options' not in kwargs:
                    raise
                kwargs.pop('stream_options', None)
                stream = self._client.chat.completions.create(**kwargs)
            yield from self._consume_openai_chat(stream)

        yield from _retry_generator(factory)

    def _consume_openai_chat(self, stream: Any) -> Generator[ChatEvent, None, None]:
        chat_buf: dict[int, dict[str, Any]] = {}
        name_seen: set[int] = set()
        finish_reason: str | None = None
        for chunk in stream:
            usage = getattr(chunk, 'usage', None)
            if usage is not None:
                yield {'kind': 'usage', 'usage': usage}
            choices = getattr(chunk, 'choices', None) or []
            if not choices:
                continue
            choice = choices[0]
            if getattr(choice, 'finish_reason', None):
                finish_reason = choice.finish_reason
            delta = getattr(choice, 'delta', None)
            if delta is None:
                continue
            reasoning = getattr(delta, 'reasoning_content', None) or ''
            if reasoning:
                yield {'kind': 'thinking', 'content': reasoning}
            for call in getattr(delta, 'tool_calls', None) or []:
                index = getattr(call, 'index', 0)
                buffer = chat_buf.setdefault(index, {'id': '', 'name': '', 'arguments': ''})
                if getattr(call, 'id', None):
                    buffer['id'] = call.id
                function = getattr(call, 'function', None)
                if function is not None and getattr(function, 'name', None):
                    buffer['name'] = function.name
                    if index not in name_seen:
                        name_seen.add(index)
                        yield {'kind': 'tool_call_start', 'id': buffer['id'] or str(index),
                               'name': buffer['name']}
                if function is not None and getattr(function, 'arguments', None):
                    buffer['arguments'] += function.arguments
                    yield {'kind': 'tool_call_delta', 'id': buffer['id'] or str(index),
                           'name': buffer['name'], 'delta': function.arguments}
            content = getattr(delta, 'content', None)
            if content:
                yield {'kind': 'text', 'content': content}
        for index in sorted(chat_buf):
            call = chat_buf[index]
            if call['name']:
                yield _tool_call_payload(call['id'] or f'call_{index}', call['name'],
                                         call['arguments'])
        if finish_reason:
            yield {'kind': 'finish', 'reason': finish_reason}
        else:
            # The stream ended without a finish signal: a dropped connection or
            # a proxy that omits finish_reason.  The caller warns instead of
            # treating a possibly truncated answer as complete.
            yield {'kind': 'finish', 'reason': 'incomplete'}

    # ── OpenAI responses ────────────────────────────────────
    def _openai_responses(self, messages: list[dict[str, Any]], model: ModelConfig,
                          system: str, max_tokens: int,
                          tools: list[dict[str, Any]] | None,
                          prompt_cache_key: str | None,
                          effort: str, force: bool) -> Generator[ChatEvent, None, None]:
        def build(file_ids: Callable[[ImageRef], str] | None) -> dict[str, Any]:
            kwargs: dict[str, Any] = {
                'model': model.id,
                'input': self._to_responses_input(messages, self.base_dir, file_ids),
                'instructions': system or None,
                'max_output_tokens': max_tokens,
                'stream': True,
            }
            if tools:
                kwargs['tools'] = self._to_responses_tools(tools)
            if prompt_cache_key:
                kwargs['prompt_cache_key'] = prompt_cache_key
            if self.reasoning_enabled(model, force):
                kwargs['reasoning'] = {
                    'effort': resolve_effort(effort, model.reasoning_efforts),
                    'summary': 'auto',
                }
            return kwargs

        def factory(payload: dict[str, Any]) -> Generator[ChatEvent, None, None]:
            stream = self._client.responses.create(**payload)
            yield from self._consume_openai_responses(stream)

        kwargs = build(self._file_id)
        emitted = False
        try:
            for event in _retry_generator(lambda: factory(kwargs)):
                # A retry notice is not model output; anything else means the
                # stream already reached the caller and must not be replayed.
                emitted = emitted or event.get('kind') != 'retry'
                yield event
        except (openai.APIError, anthropic.APIError) as error:
            if (emitted or not self._has_file_id(kwargs)
                    or not self._file_id_rejected(error)):
                raise
            self._fallback_inline(error)
            yield from _retry_generator(lambda: factory(build(None)))

    def _consume_openai_responses(self, stream: Any) -> Generator[ChatEvent, None, None]:
        call_buf: dict[str, dict[str, Any]] = {}
        summary_had_delta = False
        summary_stage_open = False
        summary_stage_closed = False
        finished = False
        for event in stream:
            etype = getattr(event, 'type', '')
            if etype == 'response.output_text.delta':
                delta = getattr(event, 'delta', None)
                if delta:
                    yield {'kind': 'text', 'content': delta}
            elif etype == 'response.reasoning_summary_part.added':
                summary_had_delta = False
                summary_stage_open = True
                summary_stage_closed = False
            elif etype == 'response.reasoning_summary_text.delta':
                delta = getattr(event, 'delta', None)
                if delta:
                    if summary_stage_closed:
                        summary_had_delta = False
                        summary_stage_closed = False
                    summary_had_delta = True
                    summary_stage_open = True
                    yield {'kind': 'thinking', 'content': delta}
            elif etype == 'response.reasoning_summary_text.done':
                if summary_stage_closed and not summary_stage_open:
                    summary_stage_closed = False
                    summary_had_delta = False
                summary_stage_open = True
                if not summary_had_delta:
                    text = getattr(event, 'text', None)
                    if text:
                        yield {'kind': 'thinking', 'content': text}
                if not summary_stage_closed:
                    yield {'kind': 'thinking', 'content': '\n'}
                summary_stage_open = False
                summary_stage_closed = True
                summary_had_delta = False
            elif etype == 'response.reasoning_summary_part.done':
                if summary_stage_open and not summary_stage_closed:
                    part_text = ''
                    if not summary_had_delta:
                        part = getattr(event, 'part', None)
                        part_text = getattr(part, 'text', None) or ''
                        if part_text:
                            yield {'kind': 'thinking', 'content': part_text}
                    if summary_had_delta or part_text:
                        yield {'kind': 'thinking', 'content': '\n'}
                    summary_stage_closed = True
                    summary_stage_open = False
                    summary_had_delta = False
            elif etype == 'response.reasoning_text.delta':
                delta = getattr(event, 'delta', None)
                if delta:
                    yield {'kind': 'thinking', 'content': delta}
            elif etype == 'response.output_item.added':
                item = getattr(event, 'item', None)
                if item is not None and getattr(item, 'type', '') == 'function_call':
                    item_id = getattr(item, 'id', '') or getattr(item, 'call_id', '')
                    call_id = getattr(item, 'call_id', '') or item_id
                    call_buf[item_id] = {
                        'id': call_id, 'name': getattr(item, 'name', ''),
                        'arguments': '',
                    }
                    yield {'kind': 'tool_call_start', 'id': call_id,
                           'name': call_buf[item_id]['name']}
            elif etype == 'response.function_call_arguments.delta':
                item_id = getattr(event, 'item_id', '')
                if item_id in call_buf:
                    delta = getattr(event, 'delta', '')
                    call_buf[item_id]['arguments'] += delta
                    yield {'kind': 'tool_call_delta', 'id': call_buf[item_id]['id'],
                           'name': call_buf[item_id]['name'], 'delta': delta}
            elif etype == 'response.function_call_arguments.done':
                item_id = getattr(event, 'item_id', '')
                if item_id in call_buf:
                    call = call_buf.pop(item_id)
                    final_arguments = getattr(event, 'arguments', '')
                    if final_arguments:
                        call['arguments'] = final_arguments
                    yield _tool_call_payload(call['id'], call['name'], call['arguments'])
            elif etype == 'response.completed':
                response = getattr(event, 'response', None)
                usage = getattr(response, 'usage', None) if response is not None else None
                if usage is not None:
                    yield {'kind': 'usage', 'usage': usage}
                if response is not None:
                    status = getattr(response, 'status', '')
                    if status == 'incomplete':
                        incomplete = getattr(response, 'incomplete_details', None)
                        reason = getattr(incomplete, 'reason', '') if incomplete else ''
                        yield {'kind': 'finish',
                               'reason': 'length' if reason == 'max_output_tokens' else 'incomplete'}
                    else:
                        yield {'kind': 'finish', 'reason': 'stop'}
                else:
                    yield {'kind': 'finish', 'reason': 'stop'}
                finished = True
        for call in call_buf.values():
            if call['name']:
                yield _tool_call_payload(call['id'], call['name'], call['arguments'])
        if not finished:
            yield {'kind': 'finish', 'reason': 'incomplete'}

    @staticmethod
    def _to_responses_input(messages: list[dict[str, Any]], base_dir: str = '',
                            file_ids: Callable[[ImageRef], str] | None = None,
                            ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for message in messages:
            role = message.get('role', '')
            if role == 'tool':
                result.append({
                    'type': 'function_call_output',
                    'call_id': message.get('tool_call_id', ''),
                    'output': str(message.get('content', '')),
                })
                continue
            content = message.get('content')
            parts = (_image_parts(content, 'input_image', base_dir, file_ids)
                     if role == 'user' else None)
            if parts is not None:
                result.append({'role': role, 'content': parts})
            elif content:
                result.append({'role': role, 'content': content})
            for call in message.get('tool_calls') or []:
                function = call.get('function', {})
                arguments = function.get('arguments', '{}')
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                result.append({
                    'type': 'function_call',
                    'call_id': call.get('id', ''),
                    'name': function.get('name', ''),
                    'arguments': arguments,
                })
        return result

    @staticmethod
    def _to_responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for tool in tools:
            function = tool.get('function')
            if tool.get('type') == 'function' and isinstance(function, dict):
                converted = {
                    'type': 'function',
                    'name': function.get('name', ''),
                    'description': function.get('description', ''),
                    'parameters': function.get('parameters', {'type': 'object'}),
                }
                if 'strict' in function:
                    converted['strict'] = function['strict']
                result.append(converted)
            else:
                result.append(tool)
        return result

    # ── Anthropic messages ──────────────────────────────────
    @staticmethod
    def _to_anthropic_msgs(messages: list[dict[str, Any]], base_dir: str = '',
                           file_ids: Callable[[ImageRef], str] | None = None,
                           ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []

        def merge(role: str, content: Any) -> None:
            """Append a message, merging consecutive same-role turns.

            The Messages API requires alternating roles; AdCode injects extra
            user blocks (sub-agent updates, summaries) that would otherwise
            produce two user messages in a row and a 400 response.
            """
            if isinstance(content, str) and not content.strip():
                return
            if isinstance(content, list) and not content:
                return
            if out and out[-1]['role'] == role:
                previous = out[-1]['content']
                if isinstance(previous, str) and isinstance(content, str):
                    out[-1]['content'] = f'{previous}\n\n{content}'
                else:
                    if isinstance(previous, str):
                        previous = ([{'type': 'text', 'text': previous}]
                                    if previous else [])
                    else:
                        previous = list(previous)
                    if isinstance(content, str):
                        content = [{'type': 'text', 'text': content}]
                    out[-1]['content'] = [*previous, *content]
                return
            out.append({'role': role, 'content': content})

        def flush_tool_results() -> None:
            if pending_tool_results:
                merge('user', list(pending_tool_results))
                pending_tool_results.clear()

        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get('role', '')
            if role == 'system':
                continue
            if role == 'tool':
                pending_tool_results.append({
                    'type': 'tool_result',
                    'tool_use_id': str(message.get('tool_call_id') or ''),
                    'content': str(message.get('content') or ''),
                })
                continue
            flush_tool_results()
            if role == 'assistant' and message.get('tool_calls'):
                blocks: list[dict[str, Any]] = []
                if message.get('content'):
                    blocks.append({'type': 'text', 'text': message['content']})
                for call in message['tool_calls'] or []:
                    if not isinstance(call, dict):
                        continue
                    function = call.get('function') or {}
                    try:
                        args = json.loads(function.get('arguments') or '{}')
                    except Exception:
                        args = {}
                    if not isinstance(args, dict):
                        args = {'value': args}
                    blocks.append({
                        'type': 'tool_use', 'id': str(call.get('id') or ''),
                        'name': str(function.get('name') or ''), 'input': args,
                    })
                merge('assistant', blocks)
            elif role == 'assistant':
                merge('assistant', message.get('content'))
            elif role == 'user':
                merge('user', _image_parts(message.get('content'), 'anthropic',
                                           base_dir, file_ids)
                      or message.get('content'))
        flush_tool_results()
        return out

    def _anthropic(self, messages: list[dict[str, Any]], model: ModelConfig,
                   system: str, max_tokens: int,
                   tools: list[dict[str, Any]] | None,
                   effort: str, force: bool) -> Generator[ChatEvent, None, None]:
        def build(file_ids: Callable[[ImageRef], str] | None) -> dict[str, Any]:
            kwargs: dict[str, Any] = {
                'model': model.id,
                'messages': self._to_anthropic_msgs(messages, self.base_dir, file_ids),
                'max_tokens': max_tokens,
            }
            if system:
                kwargs['system'] = system
            if tools:
                kwargs['tools'] = tools
            if self.reasoning_enabled(model, force):
                budget_map = {
                    'low': 2_048, 'medium': 4_096, 'high': 8_192, 'xhigh': 12_288,
                    'max': 16_384, 'ultra': 24_576,
                }
                budget = budget_map.get(
                    resolve_effort(effort, model.reasoning_efforts), 4_096)
                # The Messages API requires 1024 <= budget_tokens < max_tokens.
                budget = max(1_024, min(budget, max_tokens - 1))
                if budget < max_tokens:
                    kwargs['thinking'] = {'type': 'enabled', 'budget_tokens': budget}
            return kwargs

        def factory(payload: dict[str, Any]) -> Generator[ChatEvent, None, None]:
            buffers: dict[int, dict[str, Any]] = {}
            with self._client.messages.stream(**payload) as stream:
                for event in stream:
                    etype = getattr(event, 'type', '')
                    if etype == 'message_start':
                        usage = getattr(getattr(event, 'message', None), 'usage', None)
                        if usage is not None:
                            yield {'kind': 'usage', 'usage': usage}
                    elif etype == 'message_delta':
                        usage = getattr(event, 'usage', None)
                        if usage is not None:
                            yield {'kind': 'usage', 'usage': usage}
                    elif etype == 'content_block_start':
                        block = getattr(event, 'content_block', None)
                        if block is not None and getattr(block, 'type', '') == 'tool_use':
                            index = getattr(event, 'index', 0)
                            buffers[index] = {
                                'id': getattr(block, 'id', ''),
                                'name': getattr(block, 'name', ''),
                                'arguments': '',
                            }
                            yield {'kind': 'tool_call_start',
                                   'id': buffers[index]['id'],
                                   'name': buffers[index]['name']}
                    elif etype == 'content_block_delta':
                        delta = getattr(event, 'delta', None)
                        dtype = getattr(delta, 'type', '')
                        if dtype == 'thinking_delta':
                            text = getattr(delta, 'thinking', '')
                            if text:
                                yield {'kind': 'thinking', 'content': text}
                        elif dtype == 'text_delta':
                            text = getattr(delta, 'text', '')
                            if text:
                                yield {'kind': 'text', 'content': text}
                        elif dtype == 'input_json_delta':
                            index = getattr(event, 'index', 0)
                            if index in buffers:
                                partial = getattr(delta, 'partial_json', '')
                                buffers[index]['arguments'] += partial
                                yield {'kind': 'tool_call_delta',
                                       'id': buffers[index]['id'],
                                       'name': buffers[index]['name'],
                                       'delta': partial}
                final = stream.get_final_message()
            usage = getattr(final, 'usage', None)
            if usage is not None:
                yield {'kind': 'usage', 'usage': usage}
            for call in buffers.values():
                if call['name']:
                    yield _tool_call_payload(call['id'], call['name'], call['arguments'])
            stop_reason = getattr(final, 'stop_reason', None)
            yield {'kind': 'finish',
                   'reason': 'length' if stop_reason == 'max_tokens' else (stop_reason or 'stop')}

        kwargs = build(self._file_id)
        emitted = False
        try:
            for event in _retry_generator(lambda: factory(kwargs)):
                emitted = emitted or event.get('kind') != 'retry'
                yield event
        except (anthropic.APIError, openai.APIError) as error:
            if (emitted or not self._has_file_id(kwargs)
                    or not self._file_id_rejected(error)):
                raise
            self._fallback_inline(error)
            yield from _retry_generator(lambda: factory(build(None)))

    # ── model discovery ─────────────────────────────────────
    def list_models(self) -> list[str]:
        models: list[str] = []
        if 'openai' in self.wire:
            page = self._client.models.list()
            for item in getattr(page, 'data', page) or []:
                model_id = getattr(item, 'id', None) or (
                    item.get('id') if isinstance(item, dict) else None)
                if model_id:
                    models.append(str(model_id))
        else:
            # The Anthropic SDK paginates; a single page hides most models.
            after_id: str | None = None
            for _ in range(50):
                kwargs: dict[str, Any] = {'limit': 1000}
                if after_id:
                    kwargs['after_id'] = after_id
                try:
                    page = self._client.models.list(**kwargs)
                except TypeError:
                    page = self._client.models.list()
                data = getattr(page, 'data', None)
                if data is None and isinstance(page, list):
                    data = page
                data = list(data or [])
                for item in data:
                    model_id = getattr(item, 'id', None) or (
                        item.get('id') if isinstance(item, dict) else None)
                    if model_id:
                        models.append(str(model_id))
                if not getattr(page, 'has_more', False) or not data:
                    break
                after_id = getattr(page, 'last_id', None) or getattr(
                    data[-1], 'id', None)
                if not after_id:
                    break
        return sorted(dict.fromkeys(models))


def fetch_model_ids(provider: ProviderConfig, wire: str | None = None) -> list[str]:
    """Fetch model ids for a provider definition, optionally with another wire."""
    probe = ProviderConfig(
        name=provider.name, base_url=provider.base_url, api_key=provider.api_key,
        wire=wire or provider.wire, models=provider.models,
    )
    return LLMClient(probe).list_models()
