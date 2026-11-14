import os
import time

from adcode.tools.registry import ToolRegistry, ToolSpec
from adcode.tools.runtime import ToolContext, pop_context, push_context


def test_read_write_edit_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    registry = ToolRegistry()
    result = registry.execute('write', {'path': 'a.txt', 'content': 'hello\nworld\n'})
    assert not result.error
    read = registry.execute('read', {'path': 'a.txt'}).text
    assert '1: hello' in read
    edit = registry.execute('edit', {'path': 'a.txt', 'old': 'world', 'new': 'there'})
    assert 'Replaced 1 occurrence' in edit.text
    assert 'there' in (tmp_path / 'a.txt').read_text()
    missing = registry.execute('edit', {'path': 'a.txt', 'old': 'nope', 'new': 'x'})
    assert missing.error


def test_glob_and_grep(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'a.py').write_text('def alpha():\n    return 1\n')
    (tmp_path / 'src' / 'b.txt').write_text('alpha beta\n')
    registry = ToolRegistry()
    globbed = registry.execute('glob', {'pattern': '**/*.py'}).text
    assert 'a.py' in globbed and 'src' in globbed
    matches = registry.execute('grep', {'pattern': 'alpha'}).text
    assert 'a.py' in matches and 'b.txt' in matches
    filtered = registry.execute('grep', {'pattern': 'alpha', 'include': '*.py'}).text
    assert 'a.py' in filtered and 'b.txt' not in filtered


def test_ls_and_fetch_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'dir').mkdir()
    (tmp_path / 'file.txt').write_text('x')
    registry = ToolRegistry()
    listing = registry.execute('ls', {'path': '.'}).text
    assert 'dir/' in listing and 'file.txt' in listing
    bad = registry.execute('fetch', {'url': 'not-a-url'})
    assert bad.text.startswith('Error')


def test_bash_execute_and_read(tmp_path):
    registry = ToolRegistry()
    command = 'echo hello' if os.name == 'nt' else 'echo hello'
    result = registry.execute('bash', {'sub': 'execute', 'command': command,
                                       'timeout': 10}).text
    assert 'hello' in result
    again = registry.execute('bash', {'sub': 'read'}).text
    assert again in ('(no output)', '(process still running, no new output)')


def test_bash_timeout():
    registry = ToolRegistry()
    command = ('powershell -NoProfile -Command "Start-Sleep -Seconds 3"'
               if os.name == 'nt' else 'sleep 3')
    started = time.time()
    result = registry.execute('bash', {'sub': 'execute', 'command': command,
                                       'timeout': 0.3}).text
    assert 'timed out' in result
    assert time.time() - started < 3


def test_custom_tool_and_diff_attachment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    registry = ToolRegistry()
    registry.register(ToolSpec('custom', 'custom', {'type': 'object'}, lambda args: 'ok'))
    assert registry.execute('custom', {}).text == 'ok'
    write = registry.execute('write', {'path': 'new.txt', 'content': 'one\n'})
    assert write.file_diff is not None
    assert write.file_diff.additions == 1
    assert '<file_diff' in write.text


def test_schemas_shapes():
    registry = ToolRegistry()
    openai = registry.schemas('openai chat completions')
    anthropic = registry.schemas('anthropic messages')
    assert openai[0]['type'] == 'function'
    assert 'input_schema' in anthropic[0]
    allowed = registry.schemas('openai chat completions', {'read'})
    assert [schema['function']['name'] for schema in allowed] == ['read']


def test_tool_context_drives_bash_interrupt(tmp_path):
    registry = ToolRegistry()
    context = ToolContext(cwd=str(tmp_path))
    token = push_context(context)
    try:
        command = 'echo hi' if os.name == 'nt' else 'echo hi'
        result = registry.execute('bash', {'sub': 'execute', 'command': command,
                                           'timeout': 10}).text
        assert 'hi' in result
    finally:
        pop_context(token)
