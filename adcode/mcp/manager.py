"""MCP server lifecycle and tool registration."""

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from adcode.mcp.client import MCPClient, MCPError, build_transport
from adcode.tools.registry import ToolRegistry, ToolSpec

StatusCallback = Callable[[str, str, str], None]

_SAFE_NAME = re.compile(r'[^A-Za-z0-9_-]+')


def _sanitize(value: str, fallback: str = 'server') -> str:
    cleaned = _SAFE_NAME.sub('_', value.strip()).strip('_')
    return cleaned[:48] or fallback


class MCPManager:
    """Connects MCP servers, scans their tools and exposes them to the agent."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry
        self.clients: dict[str, MCPClient] = {}
        self.configs: dict[str, dict[str, Any]] = {}
        self._tool_map: dict[str, tuple[str, str]] = {}
        self._lock = threading.RLock()

    def set_registry(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self._rebuild_tools()

    # ── lifecycle ───────────────────────────────────────────
    def _make_client(self, config: dict[str, Any]) -> MCPClient:
        name = str(config.get('name') or '').strip()
        if not name:
            raise MCPError('server name is required')
        transport = build_transport(config)
        return MCPClient(
            name, transport,
            init_timeout=float(config.get('init_timeout', 15) or 15),
            request_timeout=float(config.get('request_timeout', 60) or 60),
        )

    def add_server(self, config: dict[str, Any],
                   status: StatusCallback | None = None) -> bool:
        name = str(config.get('name') or '').strip()
        if not name:
            return False
        if name in self.clients:
            return False
        if status is not None:
            status(name, 'connecting', '')
        try:
            client = self._make_client(config)
            if hasattr(client.transport, 'start'):
                client.transport.start()  # type: ignore[attr-defined]
            ok = client.connect()
        except Exception as error:
            self.configs[name] = dict(config)
            if status is not None:
                status(name, 'failed', str(error))
            return False
        self.clients[name] = client
        self.configs[name] = dict(config)
        self._rebuild_tools()
        if status is not None:
            if ok:
                status(name, 'ready', f'{len(client.tools)} tools')
            else:
                status(name, 'failed', client.transport.last_error or 'connection failed')
        if not ok:
            self.clients.pop(name, None)
            client.close()
            self._rebuild_tools()
        return ok

    def load(self, configs: list[dict[str, Any]],
             status: StatusCallback | None = None) -> int:
        pending: list[dict[str, Any]] = []
        for config in configs:
            name = str(config.get('name') or '').strip()
            if not name or name in self.clients:
                continue
            self.configs[name] = dict(config)
            pending.append(dict(config))
        if not pending:
            return 0

        def worker(config: dict[str, Any]) -> tuple[str, bool, str]:
            name = str(config.get('name'))
            try:
                client = self._make_client(config)
                if hasattr(client.transport, 'start'):
                    client.transport.start()  # type: ignore[attr-defined]
                ok = client.connect()
                with self._lock:
                    if ok:
                        self.clients[name] = client
                    else:
                        client.close()
                return name, ok, client.transport.last_error
            except Exception as error:
                return name, False, str(error)

        results: dict[str, tuple[bool, str]] = {}
        for config in pending:
            if status is not None:
                status(str(config.get('name')), 'connecting', '')
        with ThreadPoolExecutor(max_workers=max(1, len(pending))) as pool:
            futures = [pool.submit(worker, config) for config in pending]
            for future in as_completed(futures):
                name, ok, error = future.result()
                results[name] = (ok, error)
                if status is not None:
                    if ok:
                        client = self.clients.get(name)
                        count = len(client.tools) if client else 0
                        status(name, 'ready', f'{count} tools')
                    else:
                        status(name, 'failed', error)
        self._rebuild_tools()
        return sum(1 for ok, _ in results.values() if ok)

    def remove_server(self, name: str) -> bool:
        client = self.clients.pop(name, None)
        self.configs.pop(name, None)
        if client is None:
            return False
        client.close()
        self._rebuild_tools()
        return True

    def refresh(self, name: str | None = None) -> dict[str, str]:
        """Re-scan tools on one server or all of them."""
        targets = [name] if name else list(self.clients)
        result: dict[str, str] = {}
        for target in targets:
            client = self.clients.get(target)
            if client is None:
                result[target] = 'not connected'
                continue
            try:
                ok = client.scan_tools()
                result[target] = 'ok' if ok else (client.transport.last_error or 'failed')
            except Exception as error:
                result[target] = str(error)
        self._rebuild_tools()
        return result

    def shutdown(self) -> None:
        for name in list(self.clients):
            self.remove_server(name)

    # ── tool registration ───────────────────────────────────
    def _rebuild_tools(self) -> None:
        if self.registry is None:
            return
        with self._lock:
            self.registry.unregister_source_prefix('mcp:')
            self._tool_map.clear()
            specs: list[ToolSpec] = []
            for server_name, client in self.clients.items():
                used: set[str] = set()
                for tool in client.tools:
                    original = str(tool['name'])
                    base = _sanitize(original, 'tool')
                    safe = base
                    counter = 1
                    while safe in used:
                        safe = f'{base}_{counter}'
                        counter += 1
                    used.add(safe)
                    full_name = f'mcp__{_sanitize(server_name)}__{safe}'
                    while full_name in self._tool_map:
                        full_name += '_'
                    self._tool_map[full_name] = (server_name, original)

                    def handler(args: dict[str, Any], _server: str = server_name,
                                _tool: str = original) -> str:
                        client = self.clients.get(_server)
                        if client is None:
                            return f'Error: MCP server {_server} is not connected'
                        return client.call_tool(_tool, args)

                    description = str(tool.get('description') or '')
                    specs.append(ToolSpec(
                        name=full_name,
                        description=f'[{server_name}] {description}',
                        parameters=tool.get('input_schema') or {
                            'type': 'object', 'properties': {}},
                        handler=handler,
                        source=f'mcp:{server_name}',
                    ))
            self.registry.register_many(specs)

    # ── invocation ──────────────────────────────────────────
    def execute(self, full_name: str, arguments: dict[str, Any]) -> str:
        info = self._tool_map.get(full_name)
        if info is None:
            return f'Error: unknown MCP tool: {full_name}'
        server_name, tool_name = info
        client = self.clients.get(server_name)
        if client is None:
            return f'Error: MCP server {server_name} is not connected'
        return client.call_tool(tool_name, arguments)

    def tool_display(self, full_name: str) -> str:
        info = self._tool_map.get(full_name)
        if info is not None:
            return f'{info[0]}/{info[1]}'
        return full_name

    # ── introspection ───────────────────────────────────────
    @property
    def server_names(self) -> list[str]:
        return list(self.clients)

    @property
    def all_tools(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for full_name, (server, original) in self._tool_map.items():
            client = self.clients.get(server)
            if client is None:
                continue
            tool = next((item for item in client.tools if item['name'] == original), None)
            if tool is None:
                continue
            result.append({
                'name': full_name,
                'description': f'[{server}] {tool.get("description", "")}',
                'parameters': tool.get('input_schema') or {
                    'type': 'object', 'properties': {}},
            })
        return result

    def get_client(self, name: str) -> MCPClient | None:
        return self.clients.get(name)

    def diagnostics(self) -> list[dict[str, Any]]:
        rows = [client.diagnostics() for client in self.clients.values()]
        known = set(self.clients)
        for name, config in self.configs.items():
            if name in known:
                continue
            rows.append({
                'name': name, 'transport': config.get('transport', 'stdio'),
                'connected': False, 'tool_count': 0,
                'last_error': 'not connected', 'protocol_version': '',
                'init_timeout': config.get('init_timeout', 15),
                'request_timeout': config.get('request_timeout', 60),
            })
        return rows


_manager: MCPManager | None = None
_manager_lock = threading.Lock()


def get_mcp_manager(registry: ToolRegistry | None = None) -> MCPManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = MCPManager(registry)
        elif registry is not None and _manager.registry is not registry:
            _manager.set_registry(registry)
        return _manager
