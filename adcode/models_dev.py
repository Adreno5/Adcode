"""Model metadata from the models.dev catalogue.

When a provider is added, the fetched model ids are matched against the
models.dev database.  Vision support, context window, maximum output tokens,
tool calling and the available reasoning-effort values are copied into the
provider's model entries so the agent can configure requests precisely.

The catalogue is cached on disk for a day.  Network failures are silent: the
provider flow simply keeps the defaults and lets the user edit them.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CATALOG_URL = 'https://models.dev/api.json'
CACHE_TTL_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class CatalogModel:
    id: str
    name: str = ''
    provider_id: str = ''
    provider_name: str = ''
    vision: bool = False
    context_window: int = 128_000
    max_output_tokens: int = 8_192
    reasoning: bool = False
    reasoning_style: str = 'none'      # none | effort | toggle
    reasoning_efforts: tuple[str, ...] = ()
    tool_call: bool = True
    temperature: bool = True
    canonical_id: str = ''
    release_date: str = ''

    def as_metadata(self) -> dict[str, Any]:
        return {
            'vision': self.vision,
            'context_window': self.context_window,
            'max_output_tokens': self.max_output_tokens,
            'reasoning': self.reasoning,
            'reasoning_style': self.reasoning_style,
            'reasoning_efforts': list(self.reasoning_efforts),
            'tool_call': self.tool_call,
            'source': 'models.dev',
        }


def _coerce_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def parse_catalog(raw: dict[str, Any]) -> list[CatalogModel]:
    """Flatten the provider-keyed models.dev payload into model records."""
    result: list[CatalogModel] = []
    for provider_id, provider in raw.items():
        if not isinstance(provider, dict):
            continue
        provider_name = str(provider.get('name') or provider_id)
        models = provider.get('models') or {}
        if not isinstance(models, dict):
            continue
        for model_id, model in models.items():
            if not isinstance(model, dict):
                continue
            modalities = model.get('modalities')
            if not isinstance(modalities, dict):
                modalities = {}
            inputs = modalities.get('input')
            if not isinstance(inputs, list):
                inputs = []
            reasoning_options = model.get('reasoning_options')
            if not isinstance(reasoning_options, list):
                reasoning_options = []
            style = 'none'
            efforts: tuple[str, ...] = ()
            if model.get('reasoning'):
                style = 'toggle'
                for option in reasoning_options:
                    if not isinstance(option, dict):
                        continue
                    if option.get('type') == 'effort':
                        style = 'effort'
                        values = option.get('values')
                        if not isinstance(values, list):
                            values = []
                        efforts = tuple(str(value) for value in values if value)
                        break
                if style == 'toggle' and not efforts:
                    efforts = ('low', 'high')
            limits = model.get('limit')
            if not isinstance(limits, dict):
                limits = {}
            result.append(CatalogModel(
                id=str(model.get('id') or model_id),
                name=str(model.get('name') or model_id),
                provider_id=str(provider_id),
                provider_name=provider_name,
                vision='image' in inputs,
                context_window=_coerce_int(limits.get('context'), 128_000),
                max_output_tokens=_coerce_int(limits.get('output'), 8_192),
                reasoning=bool(model.get('reasoning')),
                reasoning_style=style,
                reasoning_efforts=efforts,
                tool_call=bool(model.get('tool_call', True)),
                temperature=bool(model.get('temperature', True)),
                canonical_id=str(model.get('canonical_model_id') or ''),
                release_date=str(model.get('release_date') or ''),
            ))
    return result


def _normalize(value: str) -> str:
    return value.strip().lower()


def _strip_date_suffix(value: str) -> str:
    """Drop a trailing ``-YYYYMMDD`` snapshot suffix from a model id."""
    if len(value) > 9 and value[-9] == '-' and value[-8:].isdigit():
        return value[:-9]
    return value


class ModelCatalog:
    """An in-memory index over models.dev with loose id matching."""

    def __init__(self, models: list[CatalogModel] | None = None) -> None:
        self.models = models or []
        self._by_id: dict[str, list[CatalogModel]] = {}
        self._by_canonical: dict[str, list[CatalogModel]] = {}
        self._rebuild()

    def _rebuild(self) -> None:
        self._by_id.clear()
        self._by_canonical.clear()
        for model in self.models:
            self._by_id.setdefault(_normalize(model.id), []).append(model)
            if model.canonical_id:
                self._by_canonical.setdefault(_normalize(model.canonical_id), []).append(model)

    def __len__(self) -> int:
        return len(self.models)

    def find(self, model_id: str, provider_hint: str = '') -> CatalogModel | None:
        key = _normalize(model_id)
        if not key:
            return None
        candidates: list[CatalogModel] = []
        for variant in (key, _strip_date_suffix(key), key.split('/')[-1]):
            candidates = self._by_id.get(variant, [])
            if candidates:
                break
        if not candidates:
            canonical = self._by_canonical.get(key, [])
            if canonical:
                candidates = canonical
        if not candidates:
            # Loose match: catalogue id is a prefix or suffix of the model id.
            base = _strip_date_suffix(key.split('/')[-1])
            loose: list[tuple[int, CatalogModel]] = []
            for model in self.models:
                candidate = _normalize(model.id)
                if not candidate:
                    continue
                if candidate == base or base.startswith(candidate + '-') or \
                        candidate.startswith(base + '-'):
                    loose.append((len(candidate), model))
            if loose:
                loose.sort(key=lambda pair: pair[0], reverse=True)
                candidates = [pair[1] for pair in loose]
        if not candidates:
            return None
        hint = _normalize(provider_hint)
        if hint:
            for model in candidates:
                if hint in (_normalize(model.provider_id), _normalize(model.provider_name)):
                    return model
            for model in candidates:
                if hint in _normalize(model.provider_id) or hint in _normalize(model.provider_name):
                    return model
        return max(candidates, key=lambda model: model.release_date)

    def enrich(self, model_id: str, provider_hint: str = '') -> dict[str, Any] | None:
        model = self.find(model_id, provider_hint)
        return model.as_metadata() if model else None

    def to_json(self) -> dict[str, Any]:
        return {'models': [model.__dict__ for model in self.models]}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> 'ModelCatalog':
        if not isinstance(raw, dict):
            return cls([])
        items = raw.get('models')
        if not isinstance(items, list):
            return cls([])
        models = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                models.append(CatalogModel(
                    id=str(item['id']), name=str(item.get('name', '')),
                    provider_id=str(item.get('provider_id', '')),
                    provider_name=str(item.get('provider_name', '')),
                    vision=bool(item.get('vision')),
                    context_window=int(item.get('context_window', 128_000)),
                    max_output_tokens=int(item.get('max_output_tokens', 8_192)),
                    reasoning=bool(item.get('reasoning')),
                    reasoning_style=str(item.get('reasoning_style') or 'none'),
                    reasoning_efforts=tuple(str(value) for value in
                                            item.get('reasoning_efforts') or ()
                                            if isinstance(value, str)),
                    tool_call=bool(item.get('tool_call', True)),
                    temperature=bool(item.get('temperature', True)),
                    canonical_id=str(item.get('canonical_id') or ''),
                    release_date=str(item.get('release_date') or ''),
                ))
            except (KeyError, TypeError, ValueError):
                continue
        return cls(models)


def cache_path() -> Path:
    base = Path.home() / '.adcode'
    return base / 'models_dev.json'


_lock = threading.RLock()
_catalog: ModelCatalog | None = None
_fetch_thread: threading.Thread | None = None
_fetch_error = ''


def _read_cache() -> tuple[ModelCatalog | None, float]:
    path = cache_path()
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None, 0.0
    if not isinstance(raw, dict):
        return None, 0.0
    try:
        fetched_at = float(raw.get('fetched_at', 0))
    except (TypeError, ValueError):
        fetched_at = 0.0
    try:
        catalog = ModelCatalog.from_json(raw)
    except (TypeError, ValueError, AttributeError):
        return None, 0.0
    return catalog, fetched_at


def _write_cache(catalog: ModelCatalog) -> None:
    path = cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'fetched_at': time.time(), **catalog.to_json()}
        fd, temp_name = tempfile.mkstemp(prefix='.models_dev.', suffix='.tmp',
                                         dir=str(path.parent))
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(payload, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    except OSError:
        pass


def fetch_catalog(timeout: float = 25.0) -> ModelCatalog | None:
    """Download and parse models.dev; returns None on any failure."""
    global _fetch_error
    request = urllib.request.Request(CATALOG_URL, headers={
        'User-Agent': 'AdCode/2.0 (+https://github.com/adcode)',
        'Accept': 'application/json',
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode('utf-8'))
        models = parse_catalog(raw)
        if not models:
            return None
        catalog = ModelCatalog(models)
        _write_cache(catalog)
        _fetch_error = ''
        return catalog
    except Exception as error:
        _fetch_error = str(error)
        return None


def load_cached_catalog() -> ModelCatalog | None:
    """Read the on-disk catalogue without touching the network."""
    with _lock:
        if _catalog is not None:
            return _catalog if _catalog.models else None
    cached, _ = _read_cache()
    return cached if cached is not None and cached.models else None


def get_catalog(wait: bool = False) -> ModelCatalog:
    """Return the shared catalogue, loading cache first and fetching if stale."""
    global _catalog, _fetch_thread
    with _lock:
        if _catalog is not None and _catalog.models:
            return _catalog
        cached, fetched_at = _read_cache()
        if cached is not None and cached.models:
            _catalog = cached
        stale = _catalog is None or time.time() - fetched_at > CACHE_TTL_SECONDS
        if stale and (_fetch_thread is None or not _fetch_thread.is_alive()):
            def _worker() -> None:
                global _catalog, _fetch_thread
                try:
                    fetched = fetch_catalog()
                    if fetched is not None:
                        with _lock:
                            _catalog = fetched
                finally:
                    # Allow a later retry if the download failed.
                    with _lock:
                        _fetch_thread = None
            _fetch_thread = threading.Thread(target=_worker, name='models-dev',
                                             daemon=True)
            _fetch_thread.start()
        if wait and (_catalog is None or not _catalog.models):
            fetched = fetch_catalog()
            if fetched is not None:
                _catalog = fetched
        return _catalog or ModelCatalog()


def catalog_error() -> str:
    return _fetch_error


def wait_for_fetch(timeout: float = 25.0) -> ModelCatalog | None:
    thread = _fetch_thread
    if thread is not None:
        thread.join(timeout)
    with _lock:
        return _catalog if _catalog and _catalog.models else None
