import os
import time

from adcode.tools.builtin import install_coding_tools, shutdown_shell_sessions
from adcode.tools.registry import AnnotatedTool, ToolRegistry
from adcode.tools.runtime import ToolContext, pop_context, push_context


def _registry():
    registry = ToolRegistry()
    install_coding_tools(registry)
    return registry


def test_read_write_edit_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    registry = _registry()
    token = push_context(ToolContext(cwd=str(tmp_path)))
    try:
        text, diff, error = registry.execute(
            'write', {'path': 'a.txt', 'content': 'hello\nworld\n'})
        assert not error
        read = registry.execute('read', {'path': 'a.txt'})[0]
        assert '1: hello' in read
        edit = registry.execute('edit', {'path': 'a.txt',
                                         'old': 'world', 'new': 'there'})
        assert 'Replaced 1 occurrence' in edit[0]
        assert 'there' in (tmp_path / 'a.txt').read_text()
        missing = registry.execute('edit', {'path': 'a.txt',
                                            'old': 'nope', 'new': 'x'})
        assert missing[2]
    finally:
        pop_context(token)


def test_glob_and_grep(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'a.py').write_text('def alpha():\n    return 1\n')
    (tmp_path / 'src' / 'b.txt').write_text('alpha beta\n')
    registry = _registry()
    token = push_context(ToolContext(cwd=str(tmp_path)))
    try:
        globbed = registry.execute('glob', {'pattern': '**/*.py'})[0]
        assert 'a.py' in globbed and 'src' in globbed
        matches = registry.execute('grep', {'pattern': 'alpha'})[0]
        assert 'a.py' in matches and 'b.txt' in matches
        filtered = registry.execute('grep',
                                    {'pattern': 'alpha', 'include': '*.py'})[0]
        assert 'a.py' in filtered and 'b.txt' not in filtered
    finally:
        pop_context(token)


def test_ls_and_fetch_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'dir').mkdir()
    (tmp_path / 'file.txt').write_text('x')
    registry = _registry()
    token = push_context(ToolContext(cwd=str(tmp_path)))
    try:
        listing = registry.execute('ls', {'path': '.'})[0]
        assert 'dir/' in listing and 'file.txt' in listing
        bad = registry.execute('fetch', {'url': 'not-a-url'})
        assert bad[0].startswith('Error')
    finally:
        pop_context(token)


def test_bash_execute_and_read(tmp_path):
    registry = _registry()
    token = push_context(ToolContext(cwd=str(tmp_path)))
    try:
        result = registry.execute('bash', {'sub': 'execute', 'command': 'echo hello',
                                           'timeout': 10})[0]
        assert 'hello' in result
        again = registry.execute('bash', {'sub': 'read'})[0]
        assert again in ('(no output)', '(process still running, no new output)')
    finally:
        pop_context(token)
        shutdown_shell_sessions()


def test_bash_timeout():
    registry = _registry()
    command = ('powershell -NoProfile -Command "Start-Sleep -Seconds 3"'
               if os.name == 'nt' else 'sleep 3')
    started = time.time()
    token = push_context(ToolContext(cwd='.'))
    try:
        result = registry.execute('bash', {'sub': 'execute', 'command': command,
                                           'timeout': 0.3})[0]
    finally:
        pop_context(token)
        shutdown_shell_sessions()
    assert 'timed out' in result
    assert time.time() - started < 3


def test_custom_tool_and_diff_attachment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    registry = _registry()
    registry.register(AnnotatedTool(
        name='custom', description='custom',
        parameters={'type': 'object', 'properties': {}},
        handler=lambda: 'ok'), source='skill')
    token = push_context(ToolContext(cwd=str(tmp_path)))
    try:
        text, diff, error = registry.execute('custom', {})
        assert text == 'ok' and not error
        write = registry.execute('write', {'path': 'new.txt', 'content': 'one\n'})
        assert write[1] is not None
        assert write[1].additions == 1
        assert '<file_diff' in write[0]
    finally:
        pop_context(token)


def test_schemas_and_vision_filter():
    registry = _registry()
    schemas = registry.build_schema()
    assert all(schema['name'] for schema in schemas)
    names = [schema['name'] for schema in registry.build_schema(vision=False)]
    assert 'view_image' not in names
    assert 'read' in names
    allowed = registry.build_schema(names=['read'])
    assert [schema['name'] for schema in allowed] == ['read']


def test_tool_context_drives_bash_interrupt(tmp_path):
    registry = _registry()
    context = ToolContext(cwd=str(tmp_path))
    token = push_context(context)
    try:
        result = registry.execute('bash', {'sub': 'execute', 'command': 'echo hi',
                                           'timeout': 10})[0]
        assert 'hi' in result
    finally:
        pop_context(token)
        shutdown_shell_sessions()
