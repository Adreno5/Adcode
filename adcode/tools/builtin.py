"""Built-in coding tools: files, search, shell, HTTP and diffs."""

from __future__ import annotations

import ctypes
import fnmatch
import glob as glob_module
import locale
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
import weakref
from dataclasses import dataclass, field
from typing import Any

from adcode.tools.runtime import current_context

_GREP_SKIP_DIRS = frozenset({
    '.git', '.hg', '.svn', 'node_modules', '__pycache__', '.venv', 'venv',
    'dist', 'build', 'out', 'target', '.idea', '.vscode', '.svelte-kit',
    '.next', '.nuxt', '.cache', 'coverage', '.pytest_cache', '.mypy_cache',
    '.ruff_cache',
})
_GREP_MAX_FILE = 10 * 1024 * 1024
_GREP_MAX_RESULTS = 200
_MAX_OUTPUT_LINES = 5_000
_rg_path: str | None | bool = False


def _read(args: dict[str, Any]) -> str:
    path = str(args['path'])
    offset = args.get('offset')
    limit = args.get('limit')
    if not os.path.isfile(path):
        return f'Error: file not found: {path}'
    try:
        with open(path, encoding='utf-8-sig', errors='replace') as stream:
            lines = stream.readlines()
    except Exception as error:
        return f'Error reading {path}: {error}'
    total = len(lines)
    if offset is not None:
        start = max(0, int(offset) - 1)
        lines = lines[start:]
    if limit is not None:
        lines = lines[:int(limit)]
    base = (max(0, int(offset) - 1) if offset is not None else 0) + 1
    out = ''.join(f'{number:6}: {line}' for number, line in enumerate(lines, base))
    if (offset is not None or limit is not None) and total > len(lines):
        out += f'\n(showing {len(lines)} of {total} lines)'
    return out or '(empty file)'


def _write(args: dict[str, Any]) -> str:
    path = str(args['path'])
    content = str(args['content'])
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    encoding = 'utf-8'
    try:
        if os.path.isfile(path):
            with open(path, 'rb') as existing:
                if existing.read(3) == b'\xef\xbb\xbf':
                    encoding = 'utf-8-sig'
        with open(path, 'w', encoding=encoding, newline='') as stream:
            stream.write(content)
    except Exception as error:
        return f'Error writing {path}: {error}'
    return f'Wrote {len(content.encode("utf-8"))} bytes to {path}'


def _edit(args: dict[str, Any]) -> str:
    path = str(args['path'])
    old = str(args['old'])
    new = str(args['new'])
    if not os.path.isfile(path):
        return f'Error: file not found: {path}'
    try:
        with open(path, 'rb') as probe:
            has_bom = probe.read(3) == b'\xef\xbb\xbf'
        encoding = 'utf-8-sig' if has_bom else 'utf-8'
        with open(path, encoding=encoding) as stream:
            content = stream.read()
    except Exception as error:
        return f'Error reading {path}: {error}'
    count = content.count(old)
    if count == 0:
        return f'Error: old_string not found in {path}'
    if count > 1:
        return (f'Error: found {count} matches of old_string in {path}; '
                f'provide more context to make it unique')
    new_content = content.replace(old, new, 1)
    try:
        with open(path, 'w', encoding=encoding, newline='') as stream:
            stream.write(new_content)
    except Exception as error:
        return f'Error writing {path}: {error}'
    return f'Replaced 1 occurrence in {path}'


def _glob(args: dict[str, Any]) -> str:
    pattern = str(args['pattern'])
    try:
        matches = glob_module.glob(pattern, recursive=True)
    except Exception as error:
        return f'Error in glob: {error}'
    if not matches:
        return '(no matches)'
    return '\n'.join(sorted(matches))


def _find_rg() -> str | None:
    global _rg_path
    if _rg_path is False:
        import shutil
        _rg_path = shutil.which('rg')
    return _rg_path


def _grep_include(path: str, include: str) -> bool:
    if fnmatch.fnmatch(os.path.basename(path), include):
        return True
    relative = os.path.relpath(path).replace('\\', '/')
    if fnmatch.fnmatch(relative, include):
        return True
    try:
        from pathlib import PurePath
        return PurePath(relative).match(include)
    except Exception:
        return False


def _grep_rg(rg: str, args: dict[str, Any]) -> str | None:
    command = [rg, '--no-heading', '--line-number', '--color', 'never',
               '--max-count', str(_GREP_MAX_RESULTS), str(args['pattern'])]
    if args.get('include'):
        command += ['-g', str(args['include'])]
    command += ['.']
    try:
        process = subprocess.run(command, capture_output=True, timeout=30)
    except Exception:
        return None
    if process.returncode == 2:
        return None
    if process.returncode == 1 or not process.stdout:
        return '(no matches)'
    text = process.stdout.decode('utf-8', errors='replace').rstrip('\n')
    return '\n'.join(text.split('\n')[:_GREP_MAX_RESULTS])


def _grep_python(args: dict[str, Any]) -> str:
    pattern = str(args['pattern'])
    include = str(args.get('include', '') or '')
    try:
        regex = re.compile(pattern)
    except re.error as error:
        return f'Invalid regex: {error}'
    files: list[str] = []
    for root, dirs, names in os.walk('.'):
        dirs[:] = [name for name in dirs
                   if not name.startswith('.') and name not in _GREP_SKIP_DIRS]
        for name in names:
            path = os.path.join(root, name)
            if include and not _grep_include(path, include):
                continue
            try:
                if os.path.getsize(path) > _GREP_MAX_FILE:
                    continue
            except OSError:
                continue
            files.append(path)

    def search(path: str) -> list[str]:
        local: list[str] = []
        try:
            with open(path, 'rb') as stream:
                data = stream.read()
            if b'\x00' in data:
                return local
            text = data.decode('utf-8', errors='replace')
            for number, line in enumerate(text.split('\n'), 1):
                if regex.search(line):
                    local.append(f'{path}:{number}: {line.rstrip()}')
                    if len(local) >= _GREP_MAX_RESULTS:
                        break
        except Exception:
            pass
        return local

    results: list[str] = []
    workers = min(32, (os.cpu_count() or 2) + 4)
    if len(files) > 1 and workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(search, path) for path in files]
            for future in as_completed(futures):
                results.extend(future.result())
                if len(results) >= _GREP_MAX_RESULTS:
                    break
    else:
        for path in files:
            results.extend(search(path))
    if not results:
        return '(no matches)'
    return '\n'.join(results[:_GREP_MAX_RESULTS])


def _grep(args: dict[str, Any]) -> str:
    rg = _find_rg()
    if rg:
        output = _grep_rg(rg, args)
        if output is not None:
            return output
    return _grep_python(args)


# ── persistent shell sessions ──────────────────────────────

@dataclass(eq=False)
class _BashSession:
    proc: subprocess.Popen[bytes] | None = None
    output: list[str] = field(default_factory=list)
    read_pos: int = 0
    workdir: str = '.'


_bash_local = threading.local()
_bash_sessions: weakref.WeakSet[_BashSession] = weakref.WeakSet()
_bash_sessions_lock = threading.RLock()


def _bash_session() -> _BashSession:
    session = getattr(_bash_local, 'session', None)
    if session is None:
        session = _BashSession()
        _bash_local.session = session
        with _bash_sessions_lock:
            _bash_sessions.add(session)
    return session


def _console_output_codepage() -> str:
    if os.name != 'nt':
        return ''
    try:
        code_page = ctypes.windll.kernel32.GetConsoleOutputCP()
        return f'cp{code_page}' if code_page else ''
    except Exception:
        return ''


def _decode_output(data: bytes) -> str:
    if data.startswith(b'\xef\xbb\xbf'):
        data = data[3:]
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        encoding = _console_output_codepage() or locale.getpreferredencoding(False) or 'utf-8'
        return data.decode(encoding, errors='replace')


def _close_process_pipes(proc: subprocess.Popen[bytes]) -> None:
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        if pipe is not None and not pipe.closed:
            try:
                pipe.close()
            except OSError:
                pass


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """Kill the shell and every child it spawned (shell=True on Windows)."""
    if os.name == 'nt':
        try:
            subprocess.run(
                ['taskkill', '/F', '/T', '/PID', str(proc.pid)],
                capture_output=True, timeout=5,
            )
            return
        except Exception:
            pass
    else:
        try:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass


def _bash_kill(session: _BashSession | None = None) -> None:
    active = session or _bash_session()
    if active.proc:
        proc = active.proc
        _kill_tree(proc)
        try:
            proc.wait(timeout=3)
        except Exception:
            pass
        _close_process_pipes(proc)
        active.proc = None


def shutdown_shell_sessions() -> None:
    with _bash_sessions_lock:
        sessions = list(_bash_sessions)
    for session in sessions:
        _bash_kill(session)


def _bash_execute(args: dict[str, Any]) -> str:
    session = _bash_session()
    _bash_kill(session)
    command = str(args['command'])
    session.workdir = str(args.get('workdir', '.') or '.')
    try:
        timeout = float(args.get('timeout', 48))
    except (TypeError, ValueError):
        timeout = 48.0
    session.output = []
    session.read_pos = 0
    context = current_context()
    live_sink = context.live_sink
    try:
        session.proc = subprocess.Popen(
            command, shell=True, cwd=session.workdir,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'},
            start_new_session=(os.name != 'nt'),
        )
    except Exception as error:
        session.proc = None
        return f'Error: {error}'
    proc = session.proc

    def reader() -> None:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, b''):
            try:
                decoded = _decode_output(line).rstrip('\r\n')
                session.output.append(decoded)
                if len(session.output) > _MAX_OUTPUT_LINES:
                    drop = len(session.output) - _MAX_OUTPUT_LINES
                    del session.output[:drop]
                    session.read_pos = max(0, session.read_pos - drop)
                # The sink is captured in this thread's closure: contextvars do
                # not propagate to new threads automatically.
                if live_sink is not None:
                    live_sink(decoded)
            except Exception:
                pass

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    interrupted = False
    deadline = time.monotonic() + max(0.1, timeout)
    while True:
        try:
            proc.wait(timeout=0.1)
            break
        except subprocess.TimeoutExpired:
            if context.interrupt.is_set():
                interrupted = True
                break
            if time.monotonic() >= deadline:
                session.output.append(
                    '<bash_tool_metadata>command execution timed out</bash_tool_metadata>')
                _kill_tree(proc)
                break
    thread.join(timeout=1)
    if interrupted:
        _kill_tree(proc)
        session.output.append('<bash_tool_metadata>command interrupted</bash_tool_metadata>')
    if proc.poll() is not None:
        _close_process_pipes(proc)
        if session.proc is proc:
            session.proc = None
    return _bash_read(args)


def _bash_read(args: dict[str, Any]) -> str:
    session = _bash_session()
    new_lines = session.output[session.read_pos:]
    session.read_pos = len(session.output)
    if not new_lines:
        if session.proc and session.proc.poll() is None:
            return '(process still running, no new output)'
        return '(no output)'
    return '\n'.join(new_lines)


def _bash_write(args: dict[str, Any]) -> str:
    session = _bash_session()
    if not session.proc or session.proc.stdin is None:
        return 'Error: no running process'
    text = str(args['input'])
    try:
        session.proc.stdin.write((text + '\n').encode('utf-8'))
        session.proc.stdin.flush()
        return f'Wrote {len(text)} bytes to stdin'
    except Exception as error:
        return f'Error writing to stdin: {error}'


def _bash_terminate(args: dict[str, Any]) -> str:
    session = _bash_session()
    if not session.proc:
        return '(no running process)'
    _bash_kill(session)
    return 'Process terminated'


_BASH_SUBS = {
    'execute': _bash_execute,
    'read': _bash_read,
    'write': _bash_write,
    'terminate': _bash_terminate,
}


def _bash(args: dict[str, Any]) -> str:
    sub = str(args.get('sub', 'execute'))
    function = _BASH_SUBS.get(sub)
    if function is None:
        return f'Error: unknown bash sub-command: {sub}'
    return function(args)


def _fetch(args: dict[str, Any]) -> str:
    url = str(args['url'])
    method = str(args.get('method', 'GET')).upper()
    headers = args.get('headers') or {}
    if not isinstance(headers, dict):
        headers = {}
    body = args.get('body') or None
    if isinstance(body, str):
        body = body.encode('utf-8')
    try:
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
            raw = response.read(2_000_000)
        text = raw.decode('utf-8', errors='replace')
    except urllib.error.HTTPError as error:
        status = error.code
        try:
            text = error.read().decode('utf-8', errors='replace')
        except Exception:
            text = ''
    except Exception as error:
        return f'Error: {error}'
    if len(text) > 4_000:
        text = text[:4_000] + '\n...(truncated)'
    return f'[{status}]\n{text}'


def _diff(args: dict[str, Any]) -> str:
    from adcode.diff import git_worktree_diff
    file_diff = git_worktree_diff('.', str(args.get('path', '') or ''))
    if file_diff is None:
        return '(no working-tree changes)'
    return file_diff.unified or '(no textual diff)'


def _ls(args: dict[str, Any]) -> str:
    path = str(args.get('path', '.') or '.')
    try:
        entries = sorted(os.listdir(path))
    except Exception as error:
        return f'Error listing {path}: {error}'
    if not entries:
        return '(empty directory)'
    lines = []
    for name in entries:
        full = os.path.join(path, name)
        suffix = '/' if os.path.isdir(full) else ''
        lines.append(f'{name}{suffix}')
    return '\n'.join(lines)


def builtin_tools() -> list[dict[str, Any]]:
    return [
        {
            'name': 'read',
            'description': ('Read a file from the filesystem. Returns content with line numbers. '
                            'Use offset/limit for large files.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'path': {'type': 'string', 'description': 'File path to read'},
                    'offset': {'type': 'integer', 'description': '1-based first line'},
                    'limit': {'type': 'integer', 'description': 'Maximum lines to return'},
                },
                'required': ['path'],
            },
            'func': _read,
            'read_only': True,
        },
        {
            'name': 'write',
            'description': 'Create or overwrite a file with the given content.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'path': {'type': 'string', 'description': 'File path to write'},
                    'content': {'type': 'string', 'description': 'Full file content'},
                },
                'required': ['path', 'content'],
            },
            'func': _write,
        },
        {
            'name': 'edit',
            'description': ('Replace an exact string in a file. The old string must appear '
                            'exactly once in the file.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'path': {'type': 'string'},
                    'old': {'type': 'string', 'description': 'Exact text to find'},
                    'new': {'type': 'string', 'description': 'Replacement text'},
                },
                'required': ['path', 'old', 'new'],
            },
            'func': _edit,
        },
        {
            'name': 'glob',
            'description': 'Find files matching a glob pattern. Supports ** recursion.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'pattern': {'type': 'string', 'description': 'e.g. "src/**/*.py"'},
                },
                'required': ['pattern'],
            },
            'func': _glob,
            'read_only': True,
        },
        {
            'name': 'grep',
            'description': 'Search file contents with a regular expression.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'pattern': {'type': 'string', 'description': 'Regex to search for'},
                    'include': {'type': 'string', 'description': 'File filter, e.g. "*.py"'},
                },
                'required': ['pattern'],
            },
            'func': _grep,
            'read_only': True,
        },
        {
            'name': 'bash',
            'description': (
                'Persistent shell session with sub-commands:\n'
                + ('- Platform: Windows cmd.exe. Use cd/dir, or invoke powershell '
                   '-NoProfile -Command explicitly; do not use Unix pwd/ls.\n'
                   if os.name == 'nt' else '- Platform: POSIX shell.\n')
                + '- execute: run a command (kills the previous one). Default timeout 48s; '
                  'a timeout is reported with <bash_tool_metadata>.\n'
                + '- read: incremental output since the last read.\n'
                + '- write: send input to stdin (a newline is appended).\n'
                + '- terminate: kill the running process.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'sub': {'type': 'string', 'enum': ['execute', 'read', 'write', 'terminate']},
                    'command': {'type': 'string', 'description': 'Shell command (execute)'},
                    'workdir': {'type': 'string', 'description': 'Working directory (execute)'},
                    'timeout': {'type': 'integer', 'description': 'Timeout seconds (execute)'},
                    'input': {'type': 'string', 'description': 'Stdin text (write)'},
                },
                'required': ['sub'],
            },
            'func': _bash,
        },
        {
            'name': 'fetch',
            'description': 'Make an HTTP request and return the status and body.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'url': {'type': 'string'},
                    'method': {'type': 'string', 'description': 'GET, POST, PUT, DELETE'},
                    'headers': {'type': 'object'},
                    'body': {'type': 'string'},
                },
                'required': ['url'],
            },
            'func': _fetch,
            'read_only': True,
        },
        {
            'name': 'diff',
            'description': 'Show the current Git working-tree diff, optionally for one path.',
            'parameters': {
                'type': 'object',
                'properties': {'path': {'type': 'string', 'description': 'Optional path'}},
                'required': [],
            },
            'func': _diff,
            'read_only': True,
        },
        {
            'name': 'ls',
            'description': 'List directory entries.',
            'parameters': {
                'type': 'object',
                'properties': {'path': {'type': 'string', 'description': 'Directory (default .)'}},
                'required': [],
            },
            'func': _ls,
            'read_only': True,
        },
    ]
