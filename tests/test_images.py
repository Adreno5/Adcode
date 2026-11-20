import struct
import zlib
from pathlib import Path

import pytest

from adcode import images
from adcode.llm.client import LLMClient, _image_parts
from infra.orchestrator.files import references_file as _has_file_id
from infra.orchestrator.files import file_reference_rejected as _file_id_rejected
from adcode.llm.usage import estimate_text_tokens, estimate_tokens


def sample_png(width: int = 2, height: int = 2) -> bytes:
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            pixels.extend((x * 40 % 256, y * 40 % 256, 128, 255))
    return images.encode_png(width, height, pixels)


def png_rows(data: bytes) -> tuple[int, int, bytes]:
    width, height = struct.unpack('>II', data[16:24])
    index = 8
    payload = b''
    while index + 8 <= len(data):
        length = struct.unpack('>I', data[index:index + 4])[0]
        tag = data[index + 4:index + 8]
        if tag == b'IDAT':
            payload += data[index + 8:index + 8 + length]
        index += 12 + length
    return width, height, zlib.decompress(payload)


def dib(width: int, height: int, rows: list[bytes], bits: int = 32) -> bytes:
    stride = ((width * bits + 31) // 32) * 4
    padded = [row + b'\x00' * (stride - len(row)) for row in rows]
    header = struct.pack('<IiiHHIIiiII', 40, width, height, 1, bits, 0, 0,
                         0, 0, 0, 0)
    return header + b''.join(padded)


def test_marker_helpers_keep_paths_and_spaces():
    marker = images.image_marker('sessions/images/a b.png')
    assert marker == '[image: sessions/images/a b.png]'
    text = f'look at {marker} and this'
    assert images.has_image_marker(text)
    assert images.strip_image_markers(text) == 'look at  and this'
    assert images.summarize_image_markers(text) == 'look at [image: a b.png] and this'
    assert images.iter_segments(text) == [
        ('text', 'look at '), ('image', 'sessions/images/a b.png'),
        ('text', ' and this')]
    assert images.count_image_markers(text) == 1


def test_png_encoder_roundtrip():
    data = sample_png(2, 2)
    assert images.sniff_media_type(data) == 'image/png'
    width, height, raw = png_rows(data)
    assert (width, height) == (2, 2)
    assert len(raw) == height * (width * 4 + 1)
    assert raw[0] == 0
    assert bytes(raw[1:5]) == bytes((0, 0, 128, 255))


def test_dib_conversion_flips_bottom_up_rows_and_fixes_alpha():
    # Rows are stored bottom-up; alpha is zero in both rows.
    bottom = bytes([255, 0, 0, 0, 0, 0, 255, 0])   # blue, red (BGRA)
    top = bytes([0, 0, 255, 0, 0, 255, 0, 0])      # red, green
    data = images.dib_to_png(dib(2, 2, [bottom, top]))
    assert data is not None
    width, height, raw = png_rows(data)
    assert (width, height) == (2, 2)
    # Top row first, alpha forced opaque.
    assert bytes(raw[1:9]) == bytes((255, 0, 0, 255, 0, 255, 0, 255))
    assert bytes(raw[10:18]) == bytes((0, 0, 255, 255, 255, 0, 0, 255))


def test_dib_conversion_keeps_real_alpha():
    bottom = bytes([0, 0, 255, 128, 0, 0, 0, 0])
    top = bytes([0, 0, 255, 64, 0, 0, 0, 0])
    data = images.dib_to_png(dib(2, 2, [bottom, top]))
    assert data is not None
    _, _, raw = png_rows(data)
    assert bytes(raw[1:9]) == bytes((255, 0, 0, 64, 0, 0, 0, 0))


def test_dib_rejects_garbage():
    assert images.dib_to_png(b'') is None
    assert images.dib_to_png(b'\x00' * 39) is None
    assert images.dib_to_png(struct.pack('<IiiHHIIiiII', 40, 2, 2, 1, 1, 0, 0,
                                         0, 0, 0, 0)) is None


def test_image_store_content_addressed(tmp_path):
    store = images.ImageStore(tmp_path / 'sessions' / 'images', tmp_path)
    data = sample_png()
    reference = store.save(data)
    assert reference.startswith('sessions/images/paste-')
    assert (tmp_path / reference).read_bytes() == data
    assert store.save(data) == reference
    with pytest.raises(ValueError):
        store.save(b'not an image')


def test_resolve_image_relative_and_url(tmp_path):
    store = images.ImageStore(tmp_path / 'images', tmp_path)
    reference = store.save(sample_png())
    resolved = images.resolve_image(reference, str(tmp_path))
    assert resolved is not None and resolved.path == tmp_path / reference
    assert resolved.media_type == 'image/png'
    assert images.resolve_image('images/missing.png', str(tmp_path)) is None
    remote = images.resolve_image('https://example.com/cat.png')
    assert remote is not None and remote.url.endswith('cat.png')


def test_is_image_url_only_matches_image_links():
    assert images.is_image_url('https://example.com/cat.PNG')
    assert images.is_image_url('https://example.com/a/b.jpeg?x=1#frag')
    assert not images.is_image_url('https://example.com/page')
    assert not images.is_image_url('ftp://example.com/cat.png')
    assert not images.is_image_url('sessions/images/cat.png')


def test_file_uri_to_path():
    assert images.file_uri_to_path('file:///C:/shots/a%20b.png') == 'C:/shots/a b.png'
    assert images.file_uri_to_path('file://localhost/tmp/a.png') == '/tmp/a.png'
    assert images.file_uri_to_path('"C:/shots/a.png"') == 'C:/shots/a.png'
    assert images.file_uri_to_path('https://example.com/a.png') == \
        'https://example.com/a.png'


def test_parts_for_each_wire(tmp_path):
    from infra.orchestrator.anthropic import AnthropicCompatClient
    from infra.orchestrator.openai import OpenAICompatClient

    store = images.ImageStore(tmp_path / 'images', tmp_path)
    reference = store.save(sample_png())
    text = f'look [image: {reference}] please'

    client = LLMClient(_provider('openai chat completions'))
    client.base_dir = str(tmp_path)
    prepared = client._prepared([{'role': 'user', 'content': text}])
    chat_parts = prepared[0]['content']
    assert chat_parts[0] == {'type': 'text', 'text': 'look '}
    assert chat_parts[1]['type'] == 'image_url'
    assert chat_parts[1]['image_url']['url'].startswith('data:image/png;base64,')
    assert chat_parts[2] == {'type': 'text', 'text': ' please'}

    # the sdk converters map the chat-shaped parts onto the native blocks
    responses = OpenAICompatClient._messages_to_responses_input(prepared)
    parts = responses[0]['content']
    assert [part['type'] for part in parts] == ['input_text', 'input_image',
                                                'input_text']
    assert parts[1]['image_url'].startswith('data:image/png;base64,')
    assert parts[1]['detail'] == 'auto'

    _, anthropic = AnthropicCompatClient._split_system(prepared)
    blocks = anthropic[0]['content']
    assert [block['type'] for block in blocks] == ['text', 'image', 'text']
    source = blocks[1]['source']
    assert source['type'] == 'base64'
    assert source['media_type'] == 'image/png'
    # the sdk converter inlines the bytes; equivalent to handing the sdk a path
    assert source['data'] and isinstance(source['data'], str)


def _provider(wire: str = 'openai chat completions') -> 'ProviderConfig':
    from adcode.config import ModelConfig, ProviderConfig
    return ProviderConfig(
        name='Test', base_url='http://localhost:1234/v1', api_key='k',
        wire=wire, models=[ModelConfig(id='test-model')])


def test_unresolved_marker_stays_text(tmp_path):
    client = LLMClient(_provider(), base_dir=str(tmp_path))
    text = 'see [image: images/missing.png]'
    # Nothing resolved, so the message stays unchanged instead of half-split.
    assert _image_parts(text, base_dir=str(tmp_path)) is None
    prepared = client._prepared([{'role': 'user', 'content': text}])
    assert prepared[0]['content'] == text


def test_has_file_id_scans_nested_payloads():
    assert _has_file_id({'input': [{'content': [
        {'type': 'input_image', 'file_id': 'file-1'}]}]})
    assert not _has_file_id({'input': [{'content': [
        {'type': 'input_image', 'file_id': ''}]}]})
    assert not _has_file_id({'input': 'file_id'})


def test_file_id_rejected_statuses():
    class Failure(Exception):
        def __init__(self, status: int) -> None:
            self.status_code = status

    for status in (400, 403, 404, 422):
        assert _file_id_rejected(Failure(status))
    for status in (0, 401, 408, 429, 500, 503):
        assert not _file_id_rejected(Failure(status))


def test_an_unsupported_scope_wins_over_cached_file_ids(tmp_path):
    from adcode.config import ProviderConfig

    store = images.ImageStore(tmp_path / 'images', tmp_path)
    resolved = images.resolve_image(store.save(sample_png()), str(tmp_path))
    assert resolved is not None
    client = LLMClient(ProviderConfig(name='Test', base_url='http://localhost/v1'),
                       file_cache=images.FileIdCache(tmp_path / 'files.json'))
    scope = client._file_scope()

    client.file_cache.set_id(
        scope, client._files._file_key(resolved.path, resolved.size),
        'file-cached')
    assert client._file_id(resolved) == 'file-cached'

    # Once the provider rejected the reference, the cached ids are not reused.
    client.file_cache.mark_unsupported(scope, 'nope')
    assert client._file_id(resolved) == ''


def test_url_image_needs_no_local_file(tmp_path):
    from infra.orchestrator.anthropic import AnthropicCompatClient
    from infra.orchestrator.openai import OpenAICompatClient

    client = LLMClient(_provider('openai responses'), base_dir=str(tmp_path))
    text = 'see [image: https://example.com/cat.png]'
    prepared = client._prepared([{'role': 'user', 'content': text}])
    parts = OpenAICompatClient._messages_to_responses_input(prepared)[0]['content']
    assert parts[1] == {'type': 'input_image',
                        'image_url': 'https://example.com/cat.png',
                        'detail': 'auto'}
    _, blocks_msg = AnthropicCompatClient._split_system(prepared)
    assert blocks_msg[0]['content'][1]['source'] == {
        'type': 'url', 'url': 'https://example.com/cat.png'}


# ── Files API ──────────────────────────────────────────────


class FakeFilesClient:
    """Stands in for the SDK client and its ``files`` resource."""

    def __init__(self, file_id: str = 'file-123', error: Exception | None = None):
        self.files = self
        self.file_id = file_id
        self.error = error
        self.uploads: list[tuple] = []

    def create(self, *, file, purpose):
        return self.upload(file=file, purpose=purpose)

    def upload(self, *, file, purpose=''):
        if self.error is not None:
            raise self.error
        self.uploads.append(file)
        return _FakeFile(self.file_id)


class _FakeFile:
    def __init__(self, file_id: str) -> None:
        self.id = file_id


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f'status {status_code}')
        self.status_code = status_code


def make_client(tmp_path, wire='anthropic messages', sdk=None) -> LLMClient:
    from adcode.config import ModelConfig, ProviderConfig

    provider = ProviderConfig(
        name='Files', base_url='http://localhost:9999/v1', api_key='k',
        wire=wire, models=[ModelConfig(id='harness-model')])
    return LLMClient(provider, ModelConfig(id='harness-model'),
                     base_dir=str(tmp_path),
                     file_cache=images.FileIdCache(tmp_path / 'files.json'),
                     sdk=sdk)


def test_file_id_cache_roundtrip(tmp_path):
    cache = images.FileIdCache(tmp_path / 'images' / 'files.json')
    assert cache.get_id('scope', 'key') == ''
    cache.set_id('scope', 'key', 'file-1')
    assert cache.get_id('scope', 'key') == 'file-1'
    assert not cache.unsupported('scope')
    cache.mark_unsupported('scope', 'no endpoint')
    reloaded = images.FileIdCache(tmp_path / 'images' / 'files.json')
    assert reloaded.get_id('scope', 'key') == 'file-1'
    assert reloaded.unsupported('scope')
    assert reloaded.get_id('other', 'key') == ''


def test_images_upload_once_and_are_referenced_by_id(tmp_path):
    from infra.orchestrator.anthropic import AnthropicCompatClient

    store = images.ImageStore(tmp_path / 'images', tmp_path)
    reference = store.save(sample_png())
    text = f'look [image: {reference}] please'
    sdk = FakeFilesClient('file-abc')
    client = make_client(tmp_path, sdk=sdk)

    def prepared() -> list[dict]:
        return client._prepared([{'role': 'user', 'content': text}])[0]['content']

    first = prepared()
    assert first[1] == {'type': 'image_url', 'image_url': {'file_id': 'file-abc'}}
    _, msgs = AnthropicCompatClient._split_system(
        [{'role': 'user', 'content': first}])
    assert msgs[0]['content'][1]['source'] == {'type': 'file',
                                               'file_id': 'file-abc'}
    second = prepared()
    assert second[1] == {'type': 'image_url', 'image_url': {'file_id': 'file-abc'}}
    assert len(sdk.uploads) == 1, 'the id is cached, not re-uploaded'
    name, data, mime = sdk.uploads[0]
    assert name.startswith('paste-') and name.endswith('.png')
    assert data == sample_png()
    assert mime == 'image/png'


def test_responses_use_file_id_too(tmp_path):
    from infra.orchestrator.openai import OpenAICompatClient

    store = images.ImageStore(tmp_path / 'images', tmp_path)
    reference = store.save(sample_png())
    sdk = FakeFilesClient('file-resp')
    client = make_client(tmp_path, wire='openai responses', sdk=sdk)

    prepared = client._prepared([{'role': 'user',
                                  'content': f'see [image: {reference}]'}])
    parts = OpenAICompatClient._messages_to_responses_input(
        prepared)[0]['content']
    assert parts[1] == {'type': 'input_image', 'file_id': 'file-resp',
                        'detail': 'auto'}


def test_chat_completions_never_uses_a_file_id(tmp_path):
    store = images.ImageStore(tmp_path / 'images', tmp_path)
    reference = store.save(sample_png())
    sdk = FakeFilesClient('file-chat')
    client = make_client(tmp_path, wire='openai chat completions', sdk=sdk)

    prepared = client._prepared([{'role': 'user',
                                  'content': f'see [image: {reference}]'}])
    parts = prepared[0]['content']
    assert parts[1]['image_url']['url'].startswith('data:image/png;base64,')
    assert sdk.uploads == [], 'chat completions has no file id to send'


def test_rejected_upload_falls_back_and_is_not_retried(tmp_path):
    store = images.ImageStore(tmp_path / 'images', tmp_path)
    reference = store.save(sample_png())
    warnings: list[str] = []
    sdk = FakeFilesClient(error=_StatusError(404))
    client = make_client(tmp_path, sdk=sdk)
    client.warn = warnings.append

    def inline_parts(text: str) -> list[dict]:
        prepared = client._prepared([{'role': 'user', 'content': text}])
        return prepared[0]['content']

    first = inline_parts(f'see [image: {reference}]')
    assert first[1]['image_url']['url'].startswith('data:image/png;base64,')
    assert warnings and 'Files API unavailable' in warnings[0]

    second = inline_parts(f'again [image: {reference}]')
    assert second[1]['image_url']['url'].startswith('data:image/png;base64,')
    assert len(sdk.uploads) == 0 and sdk.error is not None
    assert len(warnings) == 1, 'the warning is reported once, not per turn'
    # The provider stays marked so no later turn probes it again.
    assert client.file_cache.unsupported(client._file_scope())


def test_transient_upload_failure_is_retried(tmp_path):
    store = images.ImageStore(tmp_path / 'images', tmp_path)
    reference = store.save(sample_png())
    sdk = FakeFilesClient(error=_StatusError(503))
    client = make_client(tmp_path, sdk=sdk)
    client.warn = lambda text: None

    def parts(text: str) -> list[dict]:
        return client._prepared([{'role': 'user', 'content': text}])[0]['content']

    first = parts(f'see [image: {reference}]')
    assert first[1]['image_url']['url'].startswith('data:image/png;base64,')
    assert not client.file_cache.unsupported(client._file_scope())
    sdk.error = None
    second = parts(f'see [image: {reference}]')
    assert second[1] == {'type': 'image_url',
                         'image_url': {'file_id': 'file-123'}}


def test_token_estimate_counts_images():
    marker = '[image: sessions/images/paste-123.png]'
    with_image = estimate_tokens({'role': 'user', 'content': marker})
    assert with_image >= (estimate_text_tokens(marker)
                          + images.IMAGE_TOKEN_ESTIMATE)


# ── prompt attachment ──────────────────────────────────────


def make_app(tmp_path):
    import adcode.tui.app as app_module

    return app_module, app_module.AdCodeApp(
        cwd=str(tmp_path), resume=False,
        config_path=str(tmp_path / 'config.json'))


def close_app(app) -> None:
    app.input_thread.stop()
    app.store.save()


def test_alt_v_inserts_an_image_marker(tmp_path, monkeypatch):
    from adcode.tui.input import KeyEvent

    module, app = make_app(tmp_path)
    monkeypatch.setattr(module, 'read_clipboard_image', sample_png)
    try:
        app._handle_event(KeyEvent(kind='key', key='alt+v', alt=True))
        match = images.MARKER_RE.search(app.prompt.text)
        assert match is not None
        assert (tmp_path / match.group(1)).read_bytes() == sample_png()
        assert app.prompt.text.endswith('] ')
    finally:
        close_app(app)


def test_ctrl_v_falls_back_to_a_clipboard_image(tmp_path, monkeypatch):
    from adcode.tui.input import KeyEvent

    module, app = make_app(tmp_path)
    monkeypatch.setattr(module, 'get_clipboard', lambda: '')
    monkeypatch.setattr(module, 'read_clipboard_image', sample_png)
    try:
        app._handle_event(KeyEvent(kind='key', key='ctrl+v', ctrl=True))
        assert images.has_image_marker(app.prompt.text)
    finally:
        close_app(app)


def test_alt_v_reports_an_empty_clipboard(tmp_path, monkeypatch):
    from adcode.tui.input import KeyEvent

    module, app = make_app(tmp_path)
    monkeypatch.setattr(module, 'read_clipboard_image', lambda: None)
    monkeypatch.setattr(module, 'get_clipboard', lambda: '')
    try:
        app._handle_event(KeyEvent(kind='key', key='alt+v', alt=True))
        assert app.prompt.text == ''
        assert 'No image in the clipboard' in app.items[-1].text
    finally:
        close_app(app)


def test_alt_v_accepts_a_copied_file_path(tmp_path, monkeypatch):
    from adcode.tui.input import KeyEvent

    module, app = make_app(tmp_path)
    source = tmp_path / 'shot.png'
    source.write_bytes(sample_png())
    monkeypatch.setattr(module, 'read_clipboard_image', lambda: None)
    monkeypatch.setattr(module, 'get_clipboard', lambda: f'"{source}"')
    try:
        app._handle_event(KeyEvent(kind='key', key='alt+v', alt=True))
        assert images.has_image_marker(app.prompt.text)
        assert 'Attached' in app.items[-1].text
    finally:
        close_app(app)


def test_alt_v_accepts_a_copied_image_url(tmp_path, monkeypatch):
    from adcode.tui.input import KeyEvent

    module, app = make_app(tmp_path)
    monkeypatch.setattr(module, 'read_clipboard_image', lambda: None)
    monkeypatch.setattr(module, 'get_clipboard',
                        lambda: 'https://example.com/cat.png')
    try:
        app._handle_event(KeyEvent(kind='key', key='alt+v', alt=True))
        assert '[image: https://example.com/cat.png]' in app.prompt.text
    finally:
        close_app(app)


def select_model(app, vision: bool) -> None:
    from adcode.config import ModelConfig, ProviderConfig

    provider = ProviderConfig(name='Test', base_url='http://localhost/v1',
                              models=[ModelConfig(id='vision-model', vision=vision)])
    app.config.providers = [provider]
    app.config.select('Test', 'vision-model')


def test_attach_warns_when_the_model_is_not_vision_capable(tmp_path, monkeypatch):
    from adcode.tui.input import KeyEvent

    module, app = make_app(tmp_path)
    select_model(app, vision=False)
    monkeypatch.setattr(module, 'read_clipboard_image', sample_png)
    try:
        app._handle_event(KeyEvent(kind='key', key='alt+v', alt=True))
        assert 'not marked as vision-capable' in app.items[-1].text
        assert app.items[-1].level == 'warning'
    finally:
        close_app(app)


def test_attach_is_quiet_for_a_vision_model(tmp_path, monkeypatch):
    from adcode.tui.input import KeyEvent

    module, app = make_app(tmp_path)
    select_model(app, vision=True)
    monkeypatch.setattr(module, 'read_clipboard_image', sample_png)
    try:
        app._handle_event(KeyEvent(key='alt+v', kind='key', alt=True))
        assert 'Attached' in app.items[-1].text
        assert 'vision-capable' not in app.items[-1].text
    finally:
        close_app(app)


def test_image_command_copies_a_file_into_the_store(tmp_path):
    _module, app = make_app(tmp_path)
    source = tmp_path / 'shot.png'
    source.write_bytes(sample_png())
    try:
        assert app.attach_image_file(str(source))
        match = images.MARKER_RE.search(app.prompt.text)
        assert match is not None
        stored = tmp_path / match.group(1)
        assert stored.exists() and stored.read_bytes() == source.read_bytes()
        assert app.attach_image_file(str(tmp_path / 'missing.png')) is False
        assert 'Unsupported image' in app.items[-1].text
    finally:
        close_app(app)



# ── the model reading an image itself ──────────────────────


def sample_gif(width: int = 3, height: int = 5) -> bytes:
    return b'GIF89a' + struct.pack('<HH', width, height) + b'\x00' * 40


def sample_jpeg(width: int = 7, height: int = 9) -> bytes:
    # SOI, APP0, SOF0 (baseline), EOI: enough for the header reader.
    app0 = b'\xff\xe0' + struct.pack('>H', 16) + b'JFIF\x00' + b'\x00' * 9
    sof = b'\xff\xc0' + struct.pack('>HBHHB', 11, 8, height, width, 1) + b'\x01'
    return b'\xff\xd8' + app0 + sof + b'\xff\xd9'


def sample_webp(width: int = 11, height: int = 13) -> bytes:
    payload = b'\x2f' + struct.pack('<I', ((width - 1) & 0x3FFF)
                                    | (((height - 1) & 0x3FFF) << 14))
    chunk = b'VP8L' + struct.pack('<I', len(payload)) + payload
    chunk += b'\x00' * (-len(chunk) % 2)
    body = b'WEBP' + chunk
    return b'RIFF' + struct.pack('<I', len(body)) + body


def test_image_info_reads_every_supported_header():
    assert images.image_info(sample_png(2, 3)) == (2, 3)
    assert images.image_info(sample_gif(3, 5)) == (3, 5)
    assert images.image_info(sample_jpeg(7, 9)) == (7, 9)
    assert images.image_info(sample_webp(11, 13)) == (11, 13)
    assert images.image_info(b'not an image') is None
    assert images.image_info(b'') is None


def run_tool(name: str, args: dict, cwd) -> str:
    from adcode.tools.builtin import install_coding_tools
    from adcode.tools.registry import ToolRegistry
    from adcode.tools.runtime import ToolContext, pop_context, push_context

    registry = ToolRegistry()
    install_coding_tools(registry)
    token = push_context(ToolContext(cwd=str(cwd)))
    try:
        return registry.execute(name, args)[0]
    finally:
        pop_context(token)


def test_view_image_stores_the_file_and_returns_a_marker(tmp_path):
    (tmp_path / 'shot.png').write_bytes(sample_png(4, 6))
    text = run_tool('view_image', {'path': 'shot.png'}, tmp_path)
    assert 'Loaded shot.png (4x6 image/png' in text
    match = images.MARKER_RE.search(text)
    assert match is not None
    # the store lives under the app state dir; markers resolve anywhere
    stored = Path(match.group(1))
    assert stored.is_absolute() and stored.exists()
    assert stored.read_bytes() == (tmp_path / 'shot.png').read_bytes()
    # the image lives in the state directory -- not in the project
    from adcode.paths import images_dir
    assert str(images_dir()) in match.group(1)
    assert not (tmp_path / 'images').exists()


def test_view_image_deduplicates_the_same_bytes(tmp_path):
    (tmp_path / 'a.png').write_bytes(sample_png())
    (tmp_path / 'b.png').write_bytes(sample_png())
    first = images.MARKER_RE.search(
        run_tool('view_image', {'path': 'a.png'}, tmp_path))
    second = images.MARKER_RE.search(
        run_tool('view_image', {'path': 'b.png'}, tmp_path))
    assert first is not None and second is not None
    assert first.group(1) == second.group(1)


def test_view_image_rejects_what_it_cannot_show(tmp_path):
    (tmp_path / 'notes.txt').write_text('hello')
    assert run_tool('view_image', {'path': 'notes.txt'}, tmp_path).startswith(
        'Error: not a supported image')
    assert run_tool('view_image', {'path': 'missing.png'}, tmp_path).startswith(
        'Error: file not found')
    assert run_tool('view_image', {'path': ''}, tmp_path).startswith(
        'Error: path is required')


def test_view_image_passes_a_url_through(tmp_path):
    text = run_tool('view_image', {'path': 'https://example.com/cat.png'}, tmp_path)
    assert '[image: https://example.com/cat.png]' in text
    assert not (tmp_path / 'sessions').exists()


def test_vision_only_tools_are_hidden_from_text_models():
    from adcode.tools.builtin import install_coding_tools
    from adcode.tools.registry import ToolRegistry

    registry = ToolRegistry()
    install_coding_tools(registry)
    names = {schema['name'] for schema in registry.build_schema()}
    assert 'view_image' in names
    hidden = {schema['name'] for schema in registry.build_schema(vision=False)}
    assert 'view_image' not in hidden
