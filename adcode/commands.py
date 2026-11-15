"""Slash commands and their tab completion."""

from __future__ import annotations

import os
import shlex
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from adcode.config import REASONING_EFFORTS
from adcode.diff import git_worktree_diff
from adcode.theme import ACCENT_PRESETS, THEMES
from adcode.tui.dialogs import DialogItem
from adcode.tui.provider_ui import ProviderWizard


@dataclass
class Command:
    name: str
    description: str
    usage: str
    handler: Callable[[Any, str], None]
    subcommands: list[tuple[str, str]] = field(default_factory=list)


COMMANDS: list[Command] = []


def command(name: str, description: str, usage: str = '',
            subcommands: list[tuple[str, str]] | None = None):
    def decorate(function: Callable[[Any, str], None]) -> Callable[[Any, str], None]:
        COMMANDS.append(Command(
            name=name, description=description, usage=usage or name,
            handler=function, subcommands=subcommands or [],
        ))
        return function
    return decorate


def find(name: str) -> Command | None:
    return next((item for item in COMMANDS if item.name == name), None)


def dispatch(app: Any, text: str) -> None:
    head, _, tail = text.partition(' ')
    item = find(head)
    if item is None:
        app.notify(f'Unknown command: {head} · type /help', 'warning')
        return
    try:
        item.handler(app, tail.strip())
    except Exception as error:
        app.notify(f'Command failed: {error}', 'error')


def completion_candidates(text: str) -> list[tuple[str, str, str]]:
    """Return (completed_text, label, description) for the slash palette."""
    stripped = text.lstrip()
    if not stripped.startswith('/'):
        return []
    head_end = stripped.find(' ')
    if head_end == -1:
        token = stripped
        candidates: list[tuple[str, str, str]] = []
        for item in COMMANDS:
            if item.name.startswith(token):
                candidates.append((item.name + ' ', item.name, item.description))
        return candidates
    head = stripped[:head_end]
    item = find(head)
    if item is None or not item.subcommands:
        return []
    tail = stripped[head_end + 1:]
    if ' ' in tail:
        return []
    candidates = []
    for sub, description in item.subcommands:
        if sub.startswith(tail):
            candidates.append((f'{head} {sub} ', sub, description))
    return candidates


# ── general ────────────────────────────────────────────────

@command('/help', 'Show commands and key bindings')
def cmd_help(app: Any, args: str) -> None:
    rows = [f'{item.usage or item.name:<22} {item.description}' for item in COMMANDS]
    keys = [
        'enter              send · shift+enter / alt+enter newline',
        '↑ / ↓              history, or navigate the / command palette',
        'tab                complete the selected command',
        'esc                interrupt the current turn / close overlays',
        'ctrl+c             copy selection, clear input, or press twice to exit',
        'ctrl+v / shift+ins paste text, or attach a copied screenshot',
        'alt+v              attach the clipboard image to the message',
        'ctrl+a select all · ctrl+z undo · ctrl+y redo',
        'pgup / pgdn        scroll the transcript · mouse wheel scrolls too',
        'mouse drag        select text in the transcript',
        'ctrl+home / end    jump in the input',
        'ctrl+p             open the command palette',
        'ctrl+q             quit',
    ]
    app.open_message('Help', '**Commands**\n' + '\n'.join(rows)
                     + '\n\n**Keys**\n' + '\n'.join(keys))


@command('/image', 'Attach an image file or URL to your next message',
         '/image <path|url>')
def cmd_image(app: Any, args: str) -> None:
    app.attach_image_file(args)


@command('/clear', 'Clear the conversation history')
def cmd_clear(app: Any, args: str) -> None:
    app.clear_conversation()


@command('/new', 'Start a new conversation')
def cmd_new(app: Any, args: str) -> None:
    app.clear_conversation()


@command('/config', 'Open the settings browser')
def cmd_config(app: Any, args: str) -> None:
    app.open_settings()


@command('/quit', 'Exit AdCode')
@command('/exit', 'Exit AdCode')
def cmd_quit(app: Any, args: str) -> None:
    app.request_exit()


# ── model / provider ───────────────────────────────────────

@command('/model', 'Switch model or edit capabilities', subcommands=[
    ('list', 'Choose a model'),
    ('edit', 'Edit the selected model capabilities'),
])
def cmd_model(app: Any, args: str) -> None:
    sub = args.split(' ', 1)[0] if args else 'list'
    if sub == 'edit':
        selected = app.config.selected()
        if selected is None:
            app.notify('No model selected.', 'warning')
            return
        ProviderWizard(app).open_model_editor(selected[0], selected[1])
        return
    items: list[DialogItem] = []
    values: list[tuple[Any, Any]] = []
    for provider in app.config.providers:
        for model in provider.models:
            bits = [f'ctx {model.context_window // 1000}k']
            if model.vision:
                bits.append('vision')
            if model.reasoning_style != 'none':
                bits.append('reasoning')
            items.append(DialogItem(
                (provider.name, model.id), model.id,
                f'{provider.name} · ' + ' · '.join(bits)))
            values.append((provider, model))
    if not items:
        app.notify('No models configured. Use /provider add.', 'warning')
        return

    def choose(value: Any) -> None:
        provider_name, model_id = value
        provider = app.config.provider_by_name(provider_name)
        if provider is None:
            return
        model = provider.model(model_id)
        if model is not None:
            app.select_model(provider, model)
    app.open_select('Select model', items, choose, width=86)


@command('/provider', 'Manage API providers', subcommands=[
    ('add', 'Add a provider and fetch its models'),
    ('edit', 'Edit a provider and its models'),
    ('remove', 'Remove a provider'),
    ('list', 'List configured providers'),
])
def cmd_provider(app: Any, args: str) -> None:
    sub = args.split(' ', 1)[0] if args else 'list'
    wizard = ProviderWizard(app)
    if sub == 'add':
        wizard.run_add()
        return
    if sub == 'edit':
        wizard.run_edit()
        return
    if sub == 'remove':
        wizard.run_remove()
        return
    lines = []
    selected = app.config.selected()
    selected_name = selected[0].name if selected is not None else ''
    for provider in app.config.providers:
        marker = '· selected' if provider.name == selected_name else ''
        lines.append(f'**{provider.name}**  ({provider.wire})  '
                     f'{len(provider.models)} models {marker}')
    app.open_message('Providers', '\n'.join(lines) or 'No providers configured.')


# ── reasoning and context ──────────────────────────────────

@command('/effort', 'Set the reasoning effort', usage='/effort <level>',
         subcommands=[(value, f'Set effort to {value}') for value in REASONING_EFFORTS])
def cmd_effort(app: Any, args: str) -> None:
    value = args.strip().lower()
    if not value:
        items = [DialogItem(level, level, '') for level in REASONING_EFFORTS]

        def choose(level: Any) -> None:
            app.config.reasoning.effort = str(level)
            app.config_changed()
            app.notify(f'Reasoning effort: {level}', 'success')
        app.open_select('Reasoning effort', items, choose)
        return
    if value not in REASONING_EFFORTS:
        app.notify(f'Usage: /effort {"|".join(REASONING_EFFORTS)}', 'warning')
        return
    app.config.reasoning.effort = value
    app.config_changed()
    app.notify(f'Reasoning effort: {value}', 'success')


@command('/thinking', 'Set the thinking display mode', usage='/thinking <mode>',
         subcommands=[(mode, f'Thinking display: {mode}')
                      for mode in ('hidden', 'label', 'partial', 'full')])
def cmd_thinking(app: Any, args: str) -> None:
    value = args.strip().lower()
    modes = ('hidden', 'label', 'partial', 'full')
    if value not in modes:
        items = [DialogItem(mode, mode, '') for mode in modes]

        def choose(mode: Any) -> None:
            app.config.thinking.mode = str(mode)
            app.config_changed()
            app.notify(f'Thinking display: {mode}', 'success')
        app.open_select('Thinking display', items, choose)
        return
    app.config.thinking.mode = value
    app.config_changed()
    app.notify(f'Thinking display: {value}', 'success')


@command('/max_tokens', 'Set the maximum output tokens', usage='/max_tokens <n>')
def cmd_max_tokens(app: Any, args: str) -> None:
    value = args.strip()
    if not value:
        def submit(text: str) -> None:
            cmd_max_tokens(app, text)
        app.open_text_prompt('Max output tokens', 'Tokens per request',
                             str(app.config.max_output_tokens), submit)
        return
    try:
        number = int(value)
    except ValueError:
        app.notify('Usage: /max_tokens <n>', 'warning')
        return
    if number < 256 or number > 2_000_000:
        app.notify('Value must be between 256 and 2000000.', 'warning')
        return
    app.config.max_output_tokens = number
    app.config.context.reserved_output = number
    app.config_changed()
    app.notify(f'Max output tokens: {number}', 'success')


@command('/compact', 'Compact older turns into a summary')
def cmd_compact(app: Any, args: str) -> None:
    if app.loop is None:
        app.notify('No provider configured.', 'warning')
        return
    app.loop.request_compact()
    app.notify('Compaction queued.', 'info')


@command('/context', 'Show context token usage')
def cmd_context(app: Any, args: str) -> None:
    if app.loop is None:
        app.notify('No provider configured.', 'warning')
        return
    snap = app.loop.snapshot()
    loop = app.loop
    pct = round(100 * snap.context_tokens / snap.context_window) if snap.context_window else 0
    text = (
        f'**Context**\n'
        f'provider: {snap.provider}\n'
        f'model: {snap.model}\n'
        f'window: {snap.context_window:,} tokens\n'
        f'estimated context: {snap.context_tokens:,} tokens ({pct}%)\n'
        f'committed messages: {len(loop.context.committed)}\n'
        f'draft messages: {len(loop.context.draft)}\n'
        f'archived messages: {len(loop.context.archived)}\n'
        f'prefix generation: {loop.context.stable_prefix.generation}\n'
        f'auto compact: {"on" if app.config.context.auto_compact else "off"} '
        f'at {app.config.context.compact_threshold}%'
    )
    app.open_message('Context', text)


@command('/cache', 'Show the last provider usage record')
def cmd_cache(app: Any, args: str) -> None:
    if app.loop is None or not app.loop.usage_records:
        app.notify('No provider usage evidence yet.', 'info')
        return
    usage = app.loop.usage_records[-1]
    hit = usage.get('cache_hit_rate')
    hit_text = 'unknown' if hit is None else f'{hit:.1%}'
    text = (
        f'input: {usage.get("input_tokens")}\n'
        f'cached input: {usage.get("cached_input_tokens")}\n'
        f'uncached input: {usage.get("uncached_input_tokens")}\n'
        f'output: {usage.get("output_tokens")}\n'
        f'cache hit rate: {hit_text}\n'
        f'prefix generation: {usage.get("prefix_generation")}'
    )
    app.open_message('Cache usage', text)


@command('/prefix', 'Show the stable prompt prefix identity')
def cmd_prefix(app: Any, args: str) -> None:
    if app.loop is None:
        app.notify('No provider configured.', 'warning')
        return
    prefix = app.loop.context.stable_prefix
    from adcode.llm.usage import estimate_tokens
    tokens = estimate_tokens({'system': prefix.system, 'tools': prefix.tools})
    app.open_message(
        'Stable prefix',
        f'generation: {prefix.generation}\n'
        f'model: {prefix.model}\n'
        f'provider: {prefix.provider}\n'
        f'stable tokens: ~{tokens}\n\n'
        f'`{prefix.fingerprint}`')


# ── goals ──────────────────────────────────────────────────

@command('/goal', 'Manage durable goals', usage='/goal <objective>', subcommands=[
    ('start', 'Start a goal'), ('status', 'Show the active goal'),
    ('list', 'List goals'), ('pause', 'Pause the active goal'),
    ('resume', 'Resume a paused goal'), ('complete', 'Complete the active goal'),
    ('blocked', 'Mark the active goal blocked'), ('cancel', 'Cancel the active goal'),
])
def cmd_goal(app: Any, args: str) -> None:
    store = app.goals
    parts = args.split(' ', 1)
    sub = parts[0] if parts else ''
    value = parts[1] if len(parts) > 1 else ''
    known = {'start', 'status', 'list', 'pause', 'resume', 'complete', 'blocked', 'cancel'}
    if sub not in known:
        value = args.strip()
        sub = 'start' if value else 'status'
    session = app.loop.session_name if app.loop is not None else ''
    if sub == 'start':
        if not value.strip():
            app.notify('Usage: /goal <objective>', 'warning')
            return
        goal = store.create(value.strip(), session)
        app.notify(f'Goal {goal.id} active.', 'success')
        if app.loop is not None:
            app.loop.submit(value.strip())
        return
    if sub == 'list':
        if not store.goals:
            app.notify('No goals saved.', 'info')
            return
        rows = [f'{goal.id}  {goal.status}  {goal.objective}' for goal in
                reversed(store.goals[-20:])]
        app.open_message('Goals', '\n'.join(rows))
        return
    if sub == 'status':
        goal = store.active() or (store.goals[-1] if store.goals else None)
        if goal is None:
            app.notify('No goals saved.', 'info')
        else:
            app.open_message('Goal', f'id: {goal.id}\nstatus: {goal.status}\n'
                                     f'objective: {goal.objective}\n'
                                     f'result: {goal.result or "(none)"}')
        return
    if sub == 'resume':
        paused = next((goal for goal in reversed(store.goals)
                       if goal.status == 'paused'), None)
        if paused is None:
            app.notify('No paused goal to resume.', 'info')
            return
        current = store.active()
        if current is not None and current.id != paused.id:
            store.update(current, status='paused')
        store.update(paused, status='active')
        app.notify(f'Goal {paused.id} resumed.', 'success')
        if app.loop is not None:
            app.loop.submit(f'Resume goal {paused.id}: {paused.objective}')
        return
    goal = store.active()
    if goal is None:
        app.notify('There is no active goal.', 'info')
        return
    status = {'pause': 'paused', 'complete': 'completed',
              'blocked': 'blocked', 'cancel': 'cancelled'}[sub]
    store.update(goal, status=status,  # type: ignore[arg-type]
                 result=value.strip() or None)
    app.notify(f'Goal {goal.id}: {status}', 'success')


# ── sub-agents ─────────────────────────────────────────────

@command('/agents', 'Inspect and configure sub-agents', subcommands=[
    ('list', 'Show the agent tree'), ('max', 'Set the concurrent limit'),
    ('depth', 'Set the delegation depth'), ('wait', 'Wait for agent activity'),
    ('interrupt', 'Interrupt an agent'), ('types', 'List agent types'),
    ('teams', 'List agent teams'),
])
def cmd_agents(app: Any, args: str) -> None:
    parts = args.split()
    sub = parts[0] if parts else 'list'
    if sub == 'types':
        lines = [f'{name}: {agent_type.description or "(no description)"}'
                 for name, agent_type in sorted(app.config.agent_types.items())]
        app.open_message('Agent types', '\n'.join(lines))
        return
    if sub == 'teams':
        lines = [f'{name}: {", ".join(members)}'
                 for name, members in sorted(app.config.agent_teams.items())]
        app.open_message('Agent teams', '\n'.join(lines) or 'No teams configured.')
        return
    if app.loop is None:
        app.notify('No provider configured.', 'warning')
        return
    manager = app.loop.agents
    if sub == 'list':
        rows = manager.list_agents()
        if not rows:
            app.notify('No agents running.', 'info')
            return
        lines = []
        for row in rows:
            status = row.get('agent_status')
            if isinstance(status, dict):
                status = next(iter(status))
            lines.append(f'{row["agent_name"]}  ({row.get("agent_type", "")})  {status}')
        app.open_message('Agents', '\n'.join(lines))
        return
    if sub in ('max', 'depth', 'wait'):
        if len(parts) < 2:
            current = {'max': app.config.agents.max_concurrent,
                       'depth': app.config.agents.max_depth,
                       'wait': app.config.agents.wait_timeout_ms}[sub]
            app.notify(f'{sub} = {current}', 'info')
            return
        try:
            value = int(parts[1])
        except ValueError:
            app.notify('Value must be an integer.', 'warning')
            return
        if sub == 'max':
            app.config.agents.max_concurrent = max(1, min(128, value))
        elif sub == 'depth':
            app.config.agents.max_depth = max(1, min(32, value))
        else:
            app.config.agents.wait_timeout_ms = max(0, min(3_600_000, value))

            def wait() -> None:
                result = manager.wait_for_activity(app.config.agents.wait_timeout_ms)
                app.post(lambda: app.notify(str(result), 'info'))

            app.notify('Waiting for agent activity…', 'info')
            threading.Thread(target=wait, name='adcode-agent-wait', daemon=True).start()
            return
        app.config_changed()
        app.notify(f'agents {sub} set to {value}.', 'success')
        return
    if sub == 'interrupt':
        if len(parts) > 1:
            try:
                previous = manager.interrupt_agent(parts[1])
                app.notify(f'Interrupted {parts[1]} (was {previous}).', 'info')
            except Exception as error:
                app.notify(str(error), 'error')
            return
        rows = [row for row in manager.list_agents()
                if row['agent_name'] != 'root']
        items = [DialogItem(row['agent_name'], row['agent_name'],
                            str(row.get('agent_status'))) for row in rows]
        if not items:
            app.notify('No agents to interrupt.', 'info')
            return

        def choose(name: Any) -> None:
            try:
                previous = manager.interrupt_agent(str(name))
                app.notify(f'Interrupted {name} (was {previous}).', 'info')
            except Exception as error:
                app.notify(str(error), 'error')
        app.open_select('Interrupt agent', items, choose)
        return
    app.notify('/agents list|max|depth|wait|interrupt|types|teams', 'info')


# ── MCP ────────────────────────────────────────────────────

@command('/mcp', 'Manage MCP servers', subcommands=[
    ('add', 'Add an MCP server'), ('edit', 'Edit an MCP server'),
    ('remove', 'Remove an MCP server'), ('list', 'List MCP servers'),
    ('doctor', 'Show MCP diagnostics'), ('refresh', 'Re-scan MCP tools'),
])
def cmd_mcp(app: Any, args: str) -> None:
    sub = args.split()[0] if args.split() else 'list'
    manager = app.mcp
    if sub == 'list':
        rows = manager.diagnostics()
        if not rows:
            app.notify('No MCP servers configured.', 'info')
            return
        lines = []
        for row in rows:
            state = 'connected' if row.get('connected') else 'failed'
            lines.append(f'{row["name"]}: {state} · {row.get("transport")} · '
                         f'{row.get("tool_count", 0)} tools')
        app.open_message('MCP servers', '\n'.join(lines))
        return
    if sub == 'doctor':
        rows = manager.diagnostics()
        if not rows:
            app.notify('No MCP servers have been initialized.', 'info')
            return
        lines = []
        for row in rows:
            lines.append(f'{row["name"]}: connected={row.get("connected")} '
                         f'transport={row.get("transport")} '
                         f'protocol={row.get("protocol_version") or "?"} '
                         f'tools={row.get("tool_count", 0)}')
            if row.get('last_error'):
                lines.append(f'  error: {row["last_error"]}')
            for line in row.get('stderr_tail', [])[-3:]:
                lines.append(f'  stderr: {line}')
        app.open_message('MCP diagnostics', '\n'.join(lines))
        return
    if sub == 'refresh':
        result = manager.refresh()
        app.refresh_tools()
        app.notify('MCP refresh: ' + ', '.join(
            f'{name}={state}' for name, state in result.items()), 'info')
        return
    if sub == 'add':
        items = [
            DialogItem('stdio', 'stdio', 'Spawn a local MCP server process'),
            DialogItem('http', 'Streamable HTTP', 'POST JSON-RPC with SSE responses'),
            DialogItem('sse', 'SSE', 'Legacy HTTP event-stream transport'),
        ]
        app.open_select('MCP transport', items,
                        lambda transport: _mcp_add_transport(app, str(transport)))
        return
    if sub == 'remove':
        names = manager.server_names or [
            str(server.get('name')) for server in app.config.mcp_servers]
        if not names:
            app.notify('No MCP servers to remove.', 'info')
            return

        def choose(name: Any) -> None:
            manager.remove_server(str(name))
            app.config.mcp_servers = [
                server for server in app.config.mcp_servers
                if str(server.get('name')) != str(name)]
            app.save_config()
            app.refresh_tools()
            app.notify(f'Removed MCP server {name}.', 'info')
        app.open_select('Remove MCP server',
                        [DialogItem(name, name) for name in names], choose)
        return
    if sub == 'edit':
        names = manager.server_names
        if not names:
            app.notify('No MCP servers to edit.', 'info')
            return

        def choose(name: Any) -> None:
            config = next((server for server in app.config.mcp_servers
                           if str(server.get('name')) == str(name)), None)
            if config is None:
                app.notify('Server config not found.', 'warning')
                return
            transport = config.get('transport', 'stdio')
            if transport in ('http', 'sse'):
                def submit(url: str) -> None:
                    _mcp_replace(app, config, {'url': url.strip()})
                app.open_text_prompt('Edit MCP server', 'URL',
                                     str(config.get('url', '')), submit)
            else:
                def submit(command: str) -> None:
                    _mcp_replace(app, config, {'command': command.strip()})
                app.open_text_prompt('Edit MCP server', 'Command',
                                     str(config.get('command', '')), submit)
        app.open_select('Edit MCP server',
                        [DialogItem(name, name) for name in names], choose)
        return
    app.notify('/mcp add|edit|remove|list|doctor|refresh', 'info')


def _mcp_add_transport(app: Any, transport: str) -> None:
    def ask_name(name: str) -> None:
        if not name.strip():
            app.notify('Server name is required.', 'warning')
            return
        if transport == 'stdio':
            def ask_command(command: str) -> None:
                if not command.strip():
                    app.notify('Command is required.', 'warning')
                    return

                def ask_args(args_text: str) -> None:
                    try:
                        # POSIX mode strips Windows backslashes from paths.
                        arguments = shlex.split(args_text, posix=os.name != 'nt')
                    except ValueError:
                        arguments = args_text.split()
                    config = {'name': name.strip(), 'transport': 'stdio',
                              'command': command.strip(), 'args': arguments}
                    _mcp_connect(app, config)
                app.open_text_prompt('MCP stdio · args', 'Arguments', '', ask_args)
            app.open_text_prompt('MCP stdio · command', 'Command', '', ask_command)
        else:
            def ask_url(url: str) -> None:
                if not url.strip():
                    app.notify('URL is required.', 'warning')
                    return
                config = {'name': name.strip(), 'transport': transport,
                          'url': url.strip()}
                _mcp_connect(app, config)
            app.open_text_prompt(f'MCP {transport} · URL', 'Server URL', '', ask_url)
    app.open_text_prompt('MCP server name', 'Name', '', ask_name)


def _mcp_connect(app: Any, config: dict[str, Any]) -> None:
    app.notify(f'Connecting MCP {config["name"]}…', 'info')

    def worker() -> None:
        ok = app.mcp.add_server(config, app._mcp_status)
        def done() -> None:
            if ok:
                app.config.mcp_servers = [
                    server for server in app.config.mcp_servers
                    if server.get('name') != config['name']] + [config]
                app.save_config()
                app.refresh_tools()
            else:
                app.notify(f'MCP {config["name"]} did not connect.', 'error')
        app.post(done)
    threading.Thread(target=worker, name='adcode-mcp-add', daemon=True).start()


def _mcp_replace(app: Any, config: dict[str, Any], changes: dict[str, Any]) -> None:
    name = str(config.get('name'))
    app.mcp.remove_server(name)
    updated = dict(config)
    updated.update(changes)
    _mcp_connect(app, updated)


# ── skills ─────────────────────────────────────────────────

@command('/skills', 'List and reload installed skills', subcommands=[
    ('list', 'List installed skills'), ('reload', 'Rescan the skill directories'),
])
def cmd_skills(app: Any, args: str) -> None:
    sub = args.split()[0] if args.split() else 'list'
    if sub == 'reload':
        app.skills.scan()
        app.refresh_tools()
        app.notify(f'Scanned {len(app.skills.skills)} skill(s).', 'success')
        return
    skills = app.skills.list()
    if not skills:
        app.notify('No skills found in ~/.agents/skills.', 'info')
        return

    def choose(name: Any) -> None:
        skill = app.skills.get(str(name))
        if skill is None:
            return
        body = skill.body.strip()
        if len(body) > 4_000:
            body = body[:4_000] + '\n…'
        app.open_message(f'Skill · {skill.name}',
                         skill.description + '\n\n' + body)
    app.open_select('Skills', [DialogItem(skill.name, skill.name,
                                          skill.description) for skill in skills],
                    choose, width=86)


# ── sessions, rewind, fork ─────────────────────────────────

@command('/sessions', 'Browse and load saved sessions')
def cmd_sessions(app: Any, args: str) -> None:
    rows = app.session_store.list()
    if not rows:
        app.notify('No saved sessions.', 'info')
        return

    def choose(name: Any) -> None:
        app.load_session(str(name))
    app.open_select(
        'Sessions',
        [DialogItem(row['name'], row['title'] or row['name'],
                    f'{row["updated"]} · {row["messages"]} messages')
         for row in rows],
        choose, width=86)


@command('/save', 'Save the current session', usage='/save [name]')
def cmd_save(app: Any, args: str) -> None:
    if app.loop is None:
        app.notify('No active session.', 'warning')
        return
    name = args.strip()
    if not name:
        import time as _time
        def submit(value: str) -> None:
            cmd_save(app, value)
        app.open_text_prompt('Save session', 'Name',
                             app.loop.session_name or _time.strftime('%Y%m%d_%H%M%S'),
                             submit)
        return
    app.loop.save_as(name)
    app.notify(f'Session saved as {name}.', 'success')


@command('/load', 'Load a saved session', usage='/load [name]')
def cmd_load(app: Any, args: str) -> None:
    name = args.strip()
    if not name:
        cmd_sessions(app, '')
        return
    app.load_session(name)


@command('/rewind', 'Roll the conversation back to an earlier turn')
def cmd_rewind(app: Any, args: str) -> None:
    _pick_turn(app, prefill=False)


@command('/edit', 'Rewind to a turn and edit its message')
def cmd_edit(app: Any, args: str) -> None:
    _pick_turn(app, prefill=True)


def _pick_turn(app: Any, prefill: bool) -> None:
    if app.loop is None:
        app.notify('No active conversation.', 'warning')
        return
    items = [item for item in app.items if item.kind == 'user']
    if not items:
        app.notify('No user turns to rewind to.', 'info')
        return

    def choose(turn: Any) -> None:
        text = app.loop.rewind(int(turn))
        if prefill:
            app.set_prompt(text)
            app.notify(f'Rewound to turn {turn}; edit the message and submit.', 'info')
        else:
            app.notify(f'Rewound to before turn {turn}.', 'info')
    app.open_select(
        'Rewind conversation',
        [DialogItem(item.turn, f'Turn {item.turn}',
                    (item.text.splitlines()[0] if item.text else '')[:80])
         for item in items],
        choose, width=86)


@command('/fork', 'Fork the conversation into a new session', usage='/fork [name]')
def cmd_fork(app: Any, args: str) -> None:
    if app.loop is None:
        app.notify('No active conversation.', 'warning')
        return
    name = app.loop.fork(args.strip())
    app.notify(f'Forked into {name}.', 'success')


@command('/btw', 'Ask an aside question without touching the task', usage='/btw <question>')
def cmd_btw(app: Any, args: str) -> None:
    if app.loop is None:
        app.notify('No provider configured.', 'warning')
        return
    question = args.strip()
    if question:
        app.loop.submit(question, 'aside')
        return
    app.btw_mode = not app.btw_mode
    app.notify('Aside mode ' + ('on: the next message is a side question.'
                                if app.btw_mode else 'off.'), 'info')


# ── misc ───────────────────────────────────────────────────

@command('/diff', 'Show the Git working-tree diff', usage='/diff [path]')
def cmd_diff(app: Any, args: str) -> None:
    file_diff = git_worktree_diff(getattr(app, 'cwd', '.') or '.', args.strip())
    if app.loop is not None:
        app.loop.add_diff_item(f'git diff {args.strip()}'.strip(), file_diff)
    else:
        app.notify(file_diff.unified if file_diff else 'No working-tree changes.',
                   'info')


@command('/theme', 'Change the canvas theme or accent colour')
def cmd_theme(app: Any, args: str) -> None:
    items = []
    for name in THEMES:
        items.append(DialogItem(('theme', name), f'Theme · {name}', ''))
    for label, value in ACCENT_PRESETS:
        items.append(DialogItem(('accent', value), f'Accent · {label}', value))

    def choose(value: Any) -> None:
        kind, setting = value
        if kind == 'theme':
            app.config.ui.theme = str(setting)
        else:
            app.config.ui.accent = str(setting)
        app.config_changed()
        app.notify(f'Theme updated ({kind}: {setting}).', 'success')
    app.open_select('Appearance', items, choose, width=72)
