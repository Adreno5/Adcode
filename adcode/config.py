"""Application configuration.

The config file lives in ``~/.adcode/config.json`` (user-level, shared across
projects) and stores providers (with encrypted API keys), per-model
capabilities, and every user-facing behaviour that ``/config`` can edit.  Files
written by AdCode 1.x are migrated on load without losing provider credentials.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adcode.crypto import decrypt_secret, encrypt_secret

CONFIG_DIR = Path.home() / '.adcode'
CONFIG_PATH = CONFIG_DIR / 'config.json'
# AdCode 2.x kept the file next to the package; migrate it on first load.
LEGACY_CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config.json'
CONFIG_VERSION = 2

API_WIRES = (
    'openai chat completions',
    'openai responses',
    'anthropic messages',
)
REASONING_EFFORTS = ('low', 'medium', 'high', 'xhigh', 'max', 'ultra')
THINKING_MODES = ('hidden', 'label', 'partial', 'full')
THINKING_ANIMATIONS = ('fixed', 'rotating')
THINKING_STYLES = ('normal', 'gradient')
ELLIPSIS_STYLES = ('none', 'label', 'count')
TOOL_MODES = ('hidden', 'call', 'result', 'both')
RESULT_MODES = ('partial', 'full')
THEMES = ('dark', 'light')


def _now() -> str:
    import time
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def _bounded_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _choice(value: Any, choices: tuple[str, ...], default: str) -> str:
    return value if isinstance(value, str) and value in choices else default


def _bool(value: Any, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _str(value: Any, default: str) -> str:
    return value if isinstance(value, str) else default


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


# ── models and providers ───────────────────────────────────


@dataclass
class ModelConfig:
    id: str
    name: str = ''
    vision: bool = False
    context_window: int = 128_000
    max_output_tokens: int = 8_192
    reasoning: bool = False
    reasoning_style: str = 'none'      # none | effort | toggle
    reasoning_efforts: list[str] = field(default_factory=list)
    tool_call: bool = True
    source: str = 'manual'            # manual | models.dev

    def display_name(self) -> str:
        return self.name or self.id

    @classmethod
    def from_dict(cls, raw: Any) -> 'ModelConfig | None':
        if isinstance(raw, str):
            return cls(id=raw)
        if not isinstance(raw, dict):
            return None
        model_id = str(raw.get('id') or raw.get('name') or '').strip()
        if not model_id:
            return None
        efforts = raw.get('reasoning_efforts') or []
        if not isinstance(efforts, list):
            efforts = []
        style = _choice(raw.get('reasoning_style'), ('none', 'effort', 'toggle'), 'none')
        reasoning = bool(raw.get('reasoning')) or style != 'none'
        return cls(
            id=model_id,
            name=str(raw.get('name') or ''),
            vision=bool(raw.get('vision')),
            context_window=_bounded_int(raw.get('context_window'), 1_024,
                                        10_000_000, 128_000),
            max_output_tokens=_bounded_int(raw.get('max_output_tokens'), 256,
                                           2_000_000, 8_192),
            reasoning=reasoning,
            reasoning_style=style if reasoning else 'none',
            reasoning_efforts=[str(item) for item in efforts if isinstance(item, str)],
            tool_call=bool(raw.get('tool_call', True)),
            source=_str(raw.get('source'), 'manual'),
        )

    def apply_metadata(self, metadata: dict[str, Any]) -> None:
        self.vision = bool(metadata.get('vision', self.vision))
        self.context_window = _bounded_int(
            metadata.get('context_window'), 1_024, 10_000_000, self.context_window)
        self.max_output_tokens = _bounded_int(
            metadata.get('max_output_tokens'), 256, 2_000_000, self.max_output_tokens)
        self.reasoning = bool(metadata.get('reasoning', self.reasoning))
        self.reasoning_style = _choice(
            metadata.get('reasoning_style'), ('none', 'effort', 'toggle'),
            self.reasoning_style)
        efforts = metadata.get('reasoning_efforts')
        if isinstance(efforts, (list, tuple)) and efforts:
            self.reasoning_efforts = [str(item) for item in efforts]
        if not self.reasoning:
            self.reasoning_style = 'none'
            self.reasoning_efforts = []
        self.tool_call = bool(metadata.get('tool_call', self.tool_call))
        self.source = _str(metadata.get('source'), self.source)


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key: str = ''
    wire: str = 'openai chat completions'
    models: list[ModelConfig] = field(default_factory=list)
    extra_headers: dict[str, str] = field(default_factory=dict)
    reasoning_extra_body: dict[str, Any] = field(default_factory=dict)

    def model(self, model_id: str) -> ModelConfig | None:
        return next((model for model in self.models if model.id == model_id), None)

    @classmethod
    def from_dict(cls, raw: Any) -> 'ProviderConfig | None':
        if not isinstance(raw, dict):
            return None
        name = str(raw.get('name') or '').strip()
        base_url = str(raw.get('base_url') or '').strip()
        if not name or not base_url:
            return None
        models: list[ModelConfig] = []
        raw_models = raw.get('models')
        if isinstance(raw_models, (list, tuple)):
            for item in raw_models:
                model = ModelConfig.from_dict(item)
                if model is not None and not any(
                        existing.id == model.id for existing in models):
                    models.append(model)
        if not models:
            models = [ModelConfig(id='default')]
        return cls(
            name=name,
            base_url=base_url,
            api_key=decrypt_secret(str(raw.get('api_key') or '')),
            wire=_choice(raw.get('wire'), API_WIRES, 'openai chat completions'),
            models=models,
            extra_headers={
                str(key): str(value)
                for key, value in _mapping(raw.get('extra_headers')).items()
            },
            reasoning_extra_body=_mapping(raw.get('reasoning_extra_body')),
        )


# ── display settings ───────────────────────────────────────


@dataclass
class ThinkingDisplay:
    mode: str = 'label'
    label: str = 'Thinking'
    animation: str = 'rotating'
    fixed_suffix: str = '...'
    rotation: list[str] = field(default_factory=lambda: ['...', '.', '..'])
    rotation_ms: list[int] = field(default_factory=lambda: [400, 400, 400])
    from_index: int = -5
    to_index: int = -1
    style: str = 'gradient'
    gradient_top_alpha: int = 40
    ellipsis: str = 'label'
    ellipsis_label: str = '...'
    show_completed: bool = True

    @classmethod
    def from_dict(cls, raw: Any) -> 'ThinkingDisplay':
        raw = raw if isinstance(raw, dict) else {}
        rotation = raw.get('rotation')
        if not isinstance(rotation, list) or not rotation:
            rotation = ['...', '.', '..']
        rotation_ms = raw.get('rotation_ms')
        if not isinstance(rotation_ms, list) or len(rotation_ms) != len(rotation):
            rotation_ms = [400] * len(rotation)
        return cls(
            mode=_choice(raw.get('mode'), THINKING_MODES, 'label'),
            label=_str(raw.get('label'), 'Thinking') or 'Thinking',
            animation=_choice(raw.get('animation'), THINKING_ANIMATIONS, 'rotating'),
            fixed_suffix=_str(raw.get('fixed_suffix'), '...'),
            rotation=[str(item) for item in rotation],
            rotation_ms=[_bounded_int(item, 50, 5_000, 400) for item in rotation_ms],
            from_index=_bounded_int(raw.get('from_index'), -500, -1, -5),
            to_index=_bounded_int(raw.get('to_index'), -500, -1, -1),
            style=_choice(raw.get('style'), THINKING_STYLES, 'gradient'),
            gradient_top_alpha=_bounded_int(raw.get('gradient_top_alpha'), 0, 255, 40),
            ellipsis=_choice(raw.get('ellipsis'), ELLIPSIS_STYLES, 'label'),
            ellipsis_label=_str(raw.get('ellipsis_label'), '...'),
            show_completed=_bool(raw.get('show_completed'), True),
        )


@dataclass
class ToolDisplay:
    mode: str = 'both'
    show_args: bool = True
    result_mode: str = 'partial'
    from_index: int = -5
    to_index: int = -1
    style: str = 'normal'
    gradient_top_alpha: int = 40
    ellipsis: str = 'label'
    ellipsis_label: str = '...'
    show_diff: bool = True
    max_result_lines: int = 0      # 0 = unlimited in full mode

    @classmethod
    def from_dict(cls, raw: Any) -> 'ToolDisplay':
        raw = raw if isinstance(raw, dict) else {}
        return cls(
            mode=_choice(raw.get('mode'), TOOL_MODES, 'both'),
            show_args=_bool(raw.get('show_args'), True),
            result_mode=_choice(raw.get('result_mode'), RESULT_MODES, 'partial'),
            from_index=_bounded_int(raw.get('from_index'), -500, -1, -5),
            to_index=_bounded_int(raw.get('to_index'), -500, -1, -1),
            style=_choice(raw.get('style'), THINKING_STYLES, 'normal'),
            gradient_top_alpha=_bounded_int(raw.get('gradient_top_alpha'), 0, 255, 40),
            ellipsis=_choice(raw.get('ellipsis'), ELLIPSIS_STYLES, 'label'),
            ellipsis_label=_str(raw.get('ellipsis_label'), '...'),
            show_diff=_bool(raw.get('show_diff'), True),
            max_result_lines=_bounded_int(raw.get('max_result_lines'), 0, 10_000, 0),
        )


@dataclass
class ContextSettings:
    auto_compact: bool = True
    compact_threshold: int = 85        # percent of the usable window
    keep_recent_turns: int = 2
    reserved_output: int = 8_192
    compaction_reasoning_effort: str = 'low'

    @classmethod
    def from_dict(cls, raw: Any) -> 'ContextSettings':
        raw = raw if isinstance(raw, dict) else {}
        return cls(
            auto_compact=_bool(raw.get('auto_compact'), True),
            compact_threshold=_bounded_int(raw.get('compact_threshold'), 40, 99, 85),
            keep_recent_turns=_bounded_int(raw.get('keep_recent_turns'), 0, 20, 2),
            reserved_output=_bounded_int(raw.get('reserved_output'), 256, 2_000_000, 8_192),
            compaction_reasoning_effort=_choice(
                raw.get('compaction_reasoning_effort'), REASONING_EFFORTS, 'low'),
        )


@dataclass
class ReasoningSettings:
    effort: str = 'medium'
    force: bool = False

    @classmethod
    def from_dict(cls, raw: Any) -> 'ReasoningSettings':
        raw = raw if isinstance(raw, dict) else {}
        return cls(
            effort=_choice(raw.get('effort'), REASONING_EFFORTS, 'medium'),
            force=_bool(raw.get('force'), False),
        )


@dataclass
class AgentsSettings:
    enabled: bool = True
    team_enabled: bool = True
    max_concurrent: int = 10
    max_depth: int = 4
    wait_timeout_ms: int = 30_000
    default_fork_turns: str = 'all'
    show_activity: bool = True
    show_output: bool = False
    default_type: str = 'general'
    system_suffix: str = ''

    @classmethod
    def from_dict(cls, raw: Any) -> 'AgentsSettings':
        raw = raw if isinstance(raw, dict) else {}
        fork = raw.get('default_fork_turns', 'all')
        if not isinstance(fork, str) or not fork:
            fork = 'all'
        return cls(
            enabled=_bool(raw.get('enabled'), True),
            team_enabled=_bool(raw.get('team_enabled'), True),
            max_concurrent=_bounded_int(raw.get('max_concurrent'), 1, 128, 10),
            max_depth=_bounded_int(raw.get('max_depth'), 1, 32, 4),
            wait_timeout_ms=_bounded_int(raw.get('wait_timeout_ms'), 0, 3_600_000, 30_000),
            default_fork_turns=fork,
            show_activity=_bool(raw.get('show_activity'), True),
            show_output=_bool(raw.get('show_output'), False),
            default_type=_str(raw.get('default_type'), 'general') or 'general',
            system_suffix=_str(raw.get('system_suffix'), ''),
        )


@dataclass
class SkillsSettings:
    enabled: bool = True
    include_project: bool = True
    extra_paths: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: Any) -> 'SkillsSettings':
        raw = raw if isinstance(raw, dict) else {}
        paths = raw.get('extra_paths')
        if not isinstance(paths, list):
            paths = []
        return cls(
            enabled=_bool(raw.get('enabled'), True),
            include_project=_bool(raw.get('include_project'), True),
            extra_paths=[str(item) for item in paths],
        )


@dataclass
class UiSettings:
    theme: str = 'dark'
    accent: str = '#8fa7d6'
    mouse_scroll_lines: int = 3
    show_header: bool = True
    show_token_status: bool = True
    show_turn_stats: bool = True
    double_click_ms: int = 350
    fps: int = 60

    @classmethod
    def from_dict(cls, raw: Any) -> 'UiSettings':
        raw = raw if isinstance(raw, dict) else {}
        return cls(
            theme=_choice(raw.get('theme'), THEMES, 'dark'),
            accent=_str(raw.get('accent'), '#8fa7d6') or '#8fa7d6',
            mouse_scroll_lines=_bounded_int(raw.get('mouse_scroll_lines'), 1, 30, 3),
            show_header=_bool(raw.get('show_header'), True),
            show_token_status=_bool(raw.get('show_token_status'), True),
            show_turn_stats=_bool(raw.get('show_turn_stats'), True),
            double_click_ms=_bounded_int(raw.get('double_click_ms'), 100, 2_000, 350),
            fps=_bounded_int(raw.get('fps'), 10, 240, 60),
        )


@dataclass
class AgentTypeConfig:
    name: str
    description: str = ''
    system: str = ''
    model: str = ''
    tools: list[str] = field(default_factory=list)
    max_turns: int = 0

    @classmethod
    def from_dict(cls, name: str, raw: Any) -> 'AgentTypeConfig':
        raw = raw if isinstance(raw, dict) else {}
        tools = raw.get('tools')
        if not isinstance(tools, list):
            tools = []
        return cls(
            name=name,
            description=_str(raw.get('description'), ''),
            system=_str(raw.get('system'), ''),
            model=_str(raw.get('model'), ''),
            tools=[str(item) for item in tools],
            max_turns=_bounded_int(raw.get('max_turns'), 0, 10_000, 0),
        )


DEFAULT_AGENT_TYPES: dict[str, dict[str, Any]] = {
    'general': {
        'description': 'General-purpose worker with the full tool set.',
        'system': '',
    },
    'explorer': {
        'description': 'Read-only investigator for searching and reading code.',
        'system': (
            'You are an exploration specialist. Use read, glob, grep and diff '
            'to investigate and report precise findings. Do not modify files.'
        ),
        'tools': ['read', 'glob', 'grep', 'bash', 'fetch'],
    },
    'reviewer': {
        'description': 'Reviews changes for correctness and missing tests.',
        'system': (
            'You are a code reviewer. Inspect the requested changes, verify '
            'behaviour against the code, and report concrete defects with file '
            'and line references. Do not rewrite unrelated code.'
        ),
        'tools': ['read', 'glob', 'grep', 'bash', 'diff'],
    },
    'worker': {
        'description': 'Focused implementer for a well-scoped change.',
        'system': (
            'You are an implementation worker. Make the smallest correct change, '
            'verify it with the available tools, and report exactly what changed.'
        ),
    },
}

DEFAULT_AGENT_TEAMS: dict[str, list[str]] = {
    'explore': ['explorer', 'explorer'],
    'build': ['worker', 'reviewer'],
    'review': ['reviewer', 'explorer'],
}


@dataclass
class AppConfig:
    version: int = CONFIG_VERSION
    providers: list[ProviderConfig] = field(default_factory=list)
    selected_model: str = ''
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    thinking: ThinkingDisplay = field(default_factory=ThinkingDisplay)
    tools: ToolDisplay = field(default_factory=ToolDisplay)
    context: ContextSettings = field(default_factory=ContextSettings)
    reasoning: ReasoningSettings = field(default_factory=ReasoningSettings)
    agents: AgentsSettings = field(default_factory=AgentsSettings)
    skills: SkillsSettings = field(default_factory=SkillsSettings)
    ui: UiSettings = field(default_factory=UiSettings)
    max_output_tokens: int = 8_192
    agent_types: dict[str, AgentTypeConfig] = field(default_factory=dict)
    agent_teams: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.agent_types:
            self.agent_types = {
                name: AgentTypeConfig.from_dict(name, raw)
                for name, raw in DEFAULT_AGENT_TYPES.items()
            }
        if not self.agent_teams:
            self.agent_teams = copy.deepcopy(DEFAULT_AGENT_TEAMS)

    # ── selection helpers ───────────────────────────────────
    def provider_by_name(self, name: str) -> ProviderConfig | None:
        return next((provider for provider in self.providers if provider.name == name), None)

    def selected(self) -> tuple[ProviderConfig, ModelConfig] | None:
        if self.selected_model and '::' in self.selected_model:
            # Model ids never contain '::', provider names may.
            provider_name, model_id = self.selected_model.rsplit('::', 1)
            provider = self.provider_by_name(provider_name)
            if provider is not None:
                model = provider.model(model_id)
                if model is not None:
                    return provider, model
        for provider in self.providers:
            if provider.models:
                return provider, provider.models[0]
        return None

    def select(self, provider_name: str, model_id: str) -> None:
        self.selected_model = f'{provider_name}::{model_id}'

    def remove_provider(self, name: str) -> None:
        self.providers = [provider for provider in self.providers if provider.name != name]
        selected = self.selected()
        if selected is None:
            self.selected_model = ''
            return
        provider, model = selected
        self.select(provider.name, model.id)


def migrate_legacy(raw: dict[str, Any]) -> AppConfig:
    """Convert an AdCode 1.x config payload into the current schema."""
    config = AppConfig()
    providers: list[ProviderConfig] = []
    for item in _list_of_dicts(raw.get('providers')):
        provider = ProviderConfig.from_dict(item)
        if provider is not None:
            providers.append(provider)
    config.providers = providers
    config.selected_model = _str(raw.get('selected_model'), '')
    config.mcp_servers = [
        server for server in _list_of_dicts(raw.get('mcp_servers'))
        if str(server.get('name') or '').strip()
    ]

    # Older releases stored verbose_mode, reasoning_effort and the raw ANSI
    # theme sequence directly on the config object.
    verbose = raw.get('verbose_mode')
    if verbose == 'full':
        config.thinking.mode = 'full'
        config.tools.mode = 'both'
        config.tools.result_mode = 'full'
    elif verbose == 'limited':
        config.thinking.mode = 'partial'
        config.tools.mode = 'both'
        config.tools.result_mode = 'partial'
    elif verbose == 'files_only':
        config.thinking.mode = 'partial'
        config.tools.mode = 'call'
    elif verbose == 'mini':
        config.thinking.mode = 'label'
        config.tools.mode = 'call'

    effort = raw.get('reasoning_effort')
    if isinstance(effort, str) and effort in REASONING_EFFORTS:
        config.reasoning.effort = effort

    config.agents.max_concurrent = _bounded_int(raw.get('max_subagents'), 1, 128, 10)
    config.agents.max_depth = _bounded_int(raw.get('max_agent_depth'), 1, 32, 4)
    config.agents.wait_timeout_ms = _bounded_int(
        raw.get('agent_wait_timeout_ms'), 0, 3_600_000, 30_000)
    max_output = _bounded_int(raw.get('max_output_tokens'), 256, 2_000_000, 8_192)
    config.max_output_tokens = max_output
    config.context.reserved_output = max_output

    theme = raw.get('theme')
    if isinstance(theme, str) and theme.startswith('\x1b['):
        config.ui.accent = '#c8cbd1' if '37m' in theme else config.ui.accent

    for model in (model for provider in config.providers for model in provider.models):
        if model.max_output_tokens == 8_192:
            model.max_output_tokens = min(max_output, 2_000_000)
    return config


def _config_version(raw: dict[str, Any]) -> int:
    try:
        return int(raw.get('version') or 0)
    except (TypeError, ValueError):
        return 0


def config_from_dict(raw: dict[str, Any]) -> AppConfig:
    if not isinstance(raw, dict):
        return AppConfig()
    if _config_version(raw) < CONFIG_VERSION and 'thinking' not in raw:
        return migrate_legacy(raw)
    config = AppConfig()
    config.version = CONFIG_VERSION
    config.providers = [
        provider for item in _list_of_dicts(raw.get('providers'))
        if (provider := ProviderConfig.from_dict(item)) is not None
    ]
    config.selected_model = _str(raw.get('selected_model'), '')
    config.mcp_servers = [
        server for server in _list_of_dicts(raw.get('mcp_servers'))
        if str(server.get('name') or '').strip()
    ]
    config.thinking = ThinkingDisplay.from_dict(raw.get('thinking'))
    config.tools = ToolDisplay.from_dict(raw.get('tools'))
    config.context = ContextSettings.from_dict(raw.get('context'))
    config.reasoning = ReasoningSettings.from_dict(raw.get('reasoning'))
    config.agents = AgentsSettings.from_dict(raw.get('agents'))
    config.skills = SkillsSettings.from_dict(raw.get('skills'))
    config.ui = UiSettings.from_dict(raw.get('ui'))
    config.max_output_tokens = _bounded_int(
        raw.get('max_output_tokens'), 256, 2_000_000, 8_192)
    types = raw.get('agent_types')
    if isinstance(types, dict) and types:
        config.agent_types = {
            str(name): AgentTypeConfig.from_dict(str(name), value)
            for name, value in types.items()
        }
    teams = raw.get('agent_teams')
    if isinstance(teams, dict) and teams:
        config.agent_teams = {
            str(name): [str(item) for item in members if isinstance(item, str)]
            for name, members in teams.items() if isinstance(members, list)
        }
    return config


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    # Dataclasses with plain data fields serialize recursively; API keys are
    # encrypted in a second pass so they never touch disk in plaintext.
    from dataclasses import asdict
    data = asdict(config)
    data['version'] = CONFIG_VERSION
    for provider in data.get('providers', []):
        key = provider.get('api_key', '')
        if key and not key.startswith('gAAAAA'):
            provider['api_key'] = encrypt_secret(key)
    return data


class ConfigStore:
    """Loads, migrates, and atomically saves the config file."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path is not None else CONFIG_PATH
        self.config = AppConfig()

    def load(self) -> AppConfig:
        raw, legacy = self._read_payload()
        self.config = config_from_dict(raw if isinstance(raw, dict) else {})
        self._enrich_from_cached_catalog()
        if legacy is not None and self.save():
            try:
                legacy.unlink()
            except OSError:
                pass
        return self.config

    def _read_payload(self) -> tuple[Any, Path | None]:
        """Read the config payload, falling back to the pre-``~/.adcode`` file."""
        if (not self.path.exists() and self.path == CONFIG_PATH
                and LEGACY_CONFIG_PATH.exists()):
            try:
                raw = json.loads(LEGACY_CONFIG_PATH.read_text(encoding='utf-8-sig'))
                return raw, LEGACY_CONFIG_PATH
            except (OSError, ValueError):
                pass
        try:
            return json.loads(self.path.read_text(encoding='utf-8-sig')), None
        except (OSError, ValueError):
            return {}, None

    def _enrich_from_cached_catalog(self) -> None:
        """Fill missing model capabilities from an already-downloaded catalogue."""
        try:
            from adcode.models_dev import load_cached_catalog
            catalog = load_cached_catalog()
        except Exception:
            return
        if catalog is None:
            return
        for provider in self.config.providers:
            for model in provider.models:
                if model.source != 'manual':
                    continue
                metadata = catalog.enrich(model.id, provider.name)
                if metadata is not None:
                    model.apply_metadata(metadata)

    def save(self) -> bool:
        payload = config_to_dict(self.config)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(
                prefix=f'.{self.path.name}.', suffix='.tmp', dir=str(self.path.parent))
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                    json.dump(payload, stream, ensure_ascii=False, indent=4)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp_name, self.path)
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
        except OSError:
            # An unwritable config location must not take the agent down; the
            # in-memory configuration remains usable for this run.
            return False
        return True


def get_by_path(config: Any, path: str) -> Any:
    current = config
    for part in path.split('.'):
        current = getattr(current, part)
    return current


def set_by_path(config: Any, path: str, value: Any) -> None:
    parts = path.split('.')
    current = config
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], value)
