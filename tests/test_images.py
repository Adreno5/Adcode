import struct
import zlib

import pytest

from adcode import images
from adcode.llm.client import LLMClient
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


def test_parts_for_each_wire(tmp_path):
    store = images.ImageStore(tmp_path / 'images', tmp_path)
    reference = store.save(sample_png())
    text = f'look [image: {reference}] please'

    client = object.__new__(LLMClient)
    client.base_dir = str(tmp_path)
    chat_parts = client._chat_content(text)
    assert chat_parts is not None
    assert chat_parts[0] == {'type': 'text', 'text': 'look '}
    assert chat_parts[1]['type'] == 'image_url'
    assert chat_parts[1]['image_url']['url'].startswith('data:image/png;base64,')
    assert chat_parts[2] == {'type': 'text', 'text': ' please'}

    responses = LLMClient._to_responses_input(
        [{'role': 'user', 'content': text}], str(tmp_path))
    parts = responses[0]['content']
    assert [part['type'] for part in parts] == ['input_text', 'input_image',
                                                'input_text']
    assert parts[1]['image_url'].startswith('data:image/png;base64,')
    assert parts[1]['detail'] == 'auto'

    anthropic = LLMClient._to_anthropic_msgs(
        [{'role': 'user', 'content': text}], str(tmp_path))
    blocks = anthropic[0]['content']
    assert [block['type'] for block in blocks] == ['text', 'image', 'text']
    source = blocks[1]['source']
    assert source['type'] == 'base64'
    assert source['media_type'] == 'image/png'
    assert source['data'] == tmp_path / reference


def test_unresolved_marker_stays_text(tmp_path):
    client = object.__new__(LLMClient)
    client.base_dir = str(tmp_path)
    text = 'see [image: images/missing.png]'
    # Nothing resolved, so the message stays unchanged instead of half-split.
    assert client._chat_content(text) is None
    messages = client._openai_chat_messages(
        [{'role': 'user', 'content': text}])
    assert messages[0]['content'] == text


def test_url_image_needs_no_local_file(tmp_path):
    client = object.__new__(LLMClient)
    client.base_dir = str(tmp_path)
    text = 'see [image: https://example.com/cat.png]'
    parts = client._responses_content(text)
    assert parts[1] == {'type': 'input_image',
                        'image_url': 'https://example.com/cat.png',
                        'detail': 'auto'}
    blocks = client._anthropic_content(text)
    assert blocks[1]['source'] == {'type': 'url',
                                   'url': 'https://example.com/cat.png'}


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
    client = LLMClient(provider)
    client.base_dir = str(tmp_path)
    client.file_cache = images.FileIdCache(tmp_path / 'files.json')
    client._client = sdk if sdk is not None else FakeFilesClient()
    return client


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
    store = images.ImageStore(tmp_path / 'images', tmp_path)
    reference = store.save(sample_png())
    text = f'look [image: {reference}] please'
    sdk = FakeFilesClient('file-abc')
    client = make_client(tmp_path, sdk=sdk)

    first = client._anthropic_content(text)
    assert first[1]['source'] == {'type': 'file', 'file_id': 'file-abc'}
    assert 'base64' not in str(first)
    second = client._anthropic_content(text)
    assert second[1]['source'] == {'type': 'file', 'file_id': 'file-abc'}
    assert len(sdk.uploads) == 1, 'the id is cached, not re-uploaded'
    name, data, mime = sdk.uploads[0]
    assert name.startswith('paste-') and name.endswith('.png')
    assert data == sample_png()
    assert mime == 'image/png'


def test_responses_use_file_id_too(tmp_path):
    store = images.ImageStore(tmp_path / 'images', tmp_path)
    reference = store.save(sample_png())
    sdk = FakeFilesClient('file-resp')
    client = make_client(tmp_path, wire='openai responses', sdk=sdk)

    parts = client._responses_content(f'see [image: {reference}]')
    assert parts[1] == {'type': 'input_image', 'file_id': 'file-resp',
                        'detail': 'auto'}


def test_chat_completions_never_uses_a_file_id(tmp_path):
    store = images.ImageStore(tmp_path / 'images', tmp_path)
    reference = store.save(sample_png())
    sdk = FakeFilesClient('file-chat')
    client = make_client(tmp_path, wire='openai chat completions', sdk=sdk)

    parts = client._chat_content(f'see [image: {reference}]')
    assert parts[1]['image_url']['url'].startswith('data:image/png;base64,')
    assert sdk.uploads == [], 'chat completions has no file id to send'


def test_rejected_upload_falls_back_and_is_not_retried(tmp_path):
    store = images.ImageStore(tmp_path / 'images', tmp_path)
    reference = store.save(sample_png())
    warnings: list[str] = []
    sdk = FakeFilesClient(error=_StatusError(404))
    client = make_client(tmp_path, sdk=sdk)
    client.warn = warnings.append

    first = client._anthropic_content(f'see [image: {reference}]')
    assert first[1]['source']['type'] == 'base64'
    assert warnings and 'Files API unavailable' in warnings[0]

    second = client._anthropic_content(f'again [image: {reference}]')
    assert second[1]['source']['type'] == 'base64'
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

    first = client._anthropic_content(f'see [image: {reference}]')
    assert first[1]['source']['type'] == 'base64'
    assert not client.file_cache.unsupported(client._file_scope())
    sdk.error = None
    client._client = sdk
    second = client._anthropic_content(f'see [image: {reference}]')
    assert second[1]['source'] == {'type': 'file', 'file_id': 'file-123'}


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
    try:
        app._handle_event(KeyEvent(kind='key', key='alt+v', alt=True))
        assert app.prompt.text == ''
        assert 'No image in the clipboard' in app.items[-1].text
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
