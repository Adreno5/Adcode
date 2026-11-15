"""Provider setup and model management dialogs."""

from __future__ import annotations

import threading
from typing import Any, Callable

from adcode.config import ModelConfig, ProviderConfig
from adcode.llm.client import fetch_model_ids
from adcode.models_dev import get_catalog, wait_for_fetch
from adcode.theme import Palette
from adcode.tui.canvas import BOLD, Canvas, text_width, truncate_text
from adcode.tui.dialogs import (
    DialogItem, MessageDialog, MultiSelectDialog, Overlay,
    _draw_backdrop, _draw_frame, _draw_hint,
    _panel_geometry,
)
from adcode.tui.input import KeyEvent


WIRE_LABELS = [
    ('openai chat completions', 'OpenAI Chat Completions',
     'Standard /v1/chat/completions streaming'),
    ('openai responses', 'OpenAI Responses',
     'Native Responses API with reasoning summaries'),
    ('anthropic messages', 'Anthropic Messages',
     'Claude Messages API with thinking blocks'),
]


def normalize_base_url(raw: str) -> str:
    url = raw.strip()
    if not url:
        return ''
    if url.endswith('#'):
        return url[:-1]
    if url.rstrip('/').endswith('/v1'):
        return url
    if url.endswith('/'):
        return url + 'v1'
    return url + '/v1'


def _valid_url(url: str) -> bool:
    from urllib.parse import urlparse
    try:
        parts = urlparse(url)
        return bool(parts.scheme) and bool(parts.netloc)
    except Exception:
        return False


class ModelEditorOverlay(Overlay):
    """Edit a single model's capability metadata."""

    def __init__(self, provider: ProviderConfig, model: ModelConfig,
                 on_save: Callable[[], None],
                 on_cancel: Callable[[], None] | None = None) -> None:
        super().__init__()
        self.title = f'Model · {model.id}'
        self.provider = provider
        self.model = model
        self.on_save = on_save
        self.on_cancel = on_cancel
        self.index = 0
        self.editing = False
        self.edit_text = ''
        self._fields = [
            ('context_window', 'Context window', 'number'),
            ('max_output_tokens', 'Max output tokens', 'number'),
            ('vision', 'Vision input', 'bool'),
            ('tool_call', 'Tool calling', 'bool'),
            ('reasoning_style', 'Reasoning style', 'choice'),
            ('reasoning_efforts', 'Reasoning efforts', 'list'),
            ('source', 'Metadata source', 'readonly'),
        ]
        self.edit_values = {
            'reasoning_style': ['none', 'effort', 'toggle'],
        }

    def _value(self, key: str) -> Any:
        return getattr(self.model, key)

    def _display(self, key: str) -> str:
        value = self._value(key)
        if isinstance(value, bool):
            return 'on' if value else 'off'
        if isinstance(value, list):
            return ', '.join(value) or '(none)'
        return str(value)

    def handle(self, event: KeyEvent) -> bool:
        if self.editing:
            if event.kind == 'text':
                self.edit_text += event.text
                return True
            if event.kind == 'paste':
                self.edit_text += event.text.replace('\n', ' ')
                return True
            if event.kind == 'key':
                if event.key == 'esc':
                    self.editing = False
                elif event.key == 'backspace':
                    self.edit_text = self.edit_text[:-1]
                elif event.key == 'enter':
                    self._apply_edit()
            return True
        if event.kind == 'mouse':
            return True
        if event.kind == 'key':
            if event.key == 'esc':
                self.closed = True
                if self.on_cancel is not None:
                    self.on_cancel()
                return True
            if event.key == 'up':
                self.index = (self.index - 1) % len(self._fields)
                return True
            if event.key == 'down':
                self.index = (self.index + 1) % len(self._fields)
                return True
            if event.key == 'ctrl+s':
                self.closed = True
                self.on_save()
                return True
            if event.key in ('enter', 'right'):
                key, _label, kind = self._fields[self.index]
                if kind == 'bool':
                    setattr(self.model, key, not bool(self._value(key)))
                    self.model.source = 'custom'
                elif kind in ('number', 'list'):
                    self.editing = True
                    self.edit_text = self._display(key) if kind == 'number' else \
                        ','.join(self.model.reasoning_efforts)
                elif kind == 'choice':
                    options = self.edit_values[key]
                    current = str(self._value(key))
                    next_value = options[(options.index(current) + 1) % len(options)] \
                        if current in options else options[0]
                    self.model.reasoning_style = next_value
                    self.model.reasoning = next_value != 'none'
                    self.model.source = 'custom'
                return True
            if event.key == 'left' and self._fields[self.index][2] == 'choice':
                key, _label, _kind = self._fields[self.index]
                options = self.edit_values[key]
                current = str(self._value(key))
                next_value = options[(options.index(current) - 1) % len(options)] \
                    if current in options else options[0]
                self.model.reasoning_style = next_value
                self.model.reasoning = next_value != 'none'
                self.model.source = 'custom'
                return True
        return True

    def _apply_edit(self) -> None:
        key, _label, kind = self._fields[self.index]
        text = self.edit_text.strip()
        if kind == 'number':
            try:
                value = int(text)
            except ValueError:
                # Keep the editor open so the value can be corrected.
                return
            if key == 'context_window':
                value = max(1_024, min(10_000_000, value))
            else:
                value = max(256, min(2_000_000, value))
            setattr(self.model, key, value)
            self.model.source = 'custom'
        elif kind == 'list':
            values = [item.strip() for item in text.split(',') if item.strip()]
            self.model.reasoning_efforts = values
            self.model.source = 'custom'
        self.edit_text = ''
        self.editing = False

    def render(self, canvas: Canvas, width: int, height: int,
               palette: Palette, now: float) -> None:
        x, y, w, h = _panel_geometry(width, height, 72,
                                     desired_height=len(self._fields) + 6)
        _draw_backdrop(canvas, width, height, palette)
        _draw_frame(canvas, x, y, w, h, palette, self.title)
        for index, (key, label, kind) in enumerate(self._fields):
            row_y = y + 2 + index
            selected = index == self.index
            if selected:
                canvas.fill_rect(x + 1, row_y, w - 2, 1, palette.selection_bg)
            bg = palette.selection_bg if selected else None
            marker = '› ' if selected else '  '
            canvas.draw_text(x + 2, row_y, marker,
                             palette.accent if selected else palette.muted, bg)
            canvas.draw_text(x + 4, row_y, label,
                             palette.selection_fg if selected else palette.text,
                             bg, BOLD if selected else 0)
            value = self._display(key)
            if self.editing and index == self.index:
                value = self.edit_text + '█'
            value_x = x + w - 3 - text_width(value)
            canvas.draw_text(max(x + 30, value_x), row_y,
                             truncate_text(value, max(8, x + w - 4 - max(x + 30, value_x))),
                             palette.subtext, bg)
        _draw_hint(canvas, x, y + h - 2, w,
                   '↑↓ move · enter toggle/edit · ←→ change · ctrl+s save · esc close',
                   palette)


class ProviderWizard:
    def __init__(self, app: Any) -> None:
        self.app = app

    # ── add flow ────────────────────────────────────────────
    def run_add(self, first_run: bool = False) -> None:
        self._ask_base_url(first_run)

    def _ask_base_url(self, first_run: bool) -> None:
        def submit(value: str) -> None:
            url = normalize_base_url(value)
            if not _valid_url(url):
                self.app.notify('Invalid URL: include a scheme and host.', 'error')
                self._ask_base_url(first_run)
                return
            self._ask_key(url, first_run)
        self.app.open_text_prompt(
            'New provider · 1/4', 'Base URL', '',
            submit, hint='End with # to use the URL exactly as typed.',
        )

    def _ask_key(self, base_url: str, first_run: bool) -> None:
        def submit(value: str) -> None:
            self._ask_name(base_url, value, first_run)
        self.app.open_text_prompt(
            'New provider · 2/4', 'API key', '',
            submit, hint='Stored encrypted in the OS keyring. Empty is allowed '
                         'for local servers.', password=True,
        )

    def _ask_name(self, base_url: str, api_key: str, first_run: bool) -> None:
        from urllib.parse import urlparse
        host = urlparse(base_url).hostname or 'provider'
        default = host.split('.')[0][:24] or 'provider'

        def submit(value: str) -> None:
            name = value.strip() or default
            self._ask_wire(base_url, api_key, name, first_run)
        self.app.open_text_prompt(
            'New provider · 3/4', 'Display name', default, submit,
        )

    def _ask_wire(self, base_url: str, api_key: str, name: str,
                  first_run: bool) -> None:
        items = [DialogItem(wire, label, description)
                 for wire, label, description in WIRE_LABELS]
        self.app.open_select(
            'New provider · 4/4 · API structure', items,
            lambda wire: self._fetch_models(base_url, api_key, name, str(wire),
                                            first_run),
        )

    def _fetch_models(self, base_url: str, api_key: str, name: str, wire: str,
                      first_run: bool, fetch_wire: str | None = None) -> None:
        probe = ProviderConfig(name=name, base_url=base_url, api_key=api_key,
                               wire=fetch_wire or wire, models=[])
        message = MessageDialog(
            'Loading models', f'Fetching the model list from\n{base_url}\n\n'
                              'This may take a few seconds.')
        self.app.open_overlay(message)

        def worker() -> None:
            models: list[str] = []
            error = ''
            try:
                models = fetch_model_ids(probe)
            except Exception as exc:
                error = str(exc)
            catalog = get_catalog()
            if not catalog.models:
                catalog = wait_for_fetch(8) or catalog

            def done() -> None:
                if self.app.overlay is message:
                    self.app.overlay = None
                if error or not models:
                    self._fetch_failed(base_url, api_key, name, wire, error,
                                       fetch_wire or wire)
                    return
                self._choose_models(base_url, api_key, name, wire, models,
                                    catalog, first_run)
            self.app.post(done)

        threading.Thread(target=worker, name='adcode-model-fetch',
                         daemon=True).start()

    def _fetch_failed(self, base_url: str, api_key: str, name: str, wire: str,
                      error: str, current_fetch_wire: str) -> None:
        options = [
            DialogItem('retry', 'Retry', 'Try the same wire again'),
            DialogItem('manual', 'Enter models manually',
                       'Type ids separated by commas'),
        ]
        if current_fetch_wire != 'openai chat completions':
            options.insert(1, DialogItem('openai chat completions', 'Fetch with OpenAI',
                                         'Probe /models using an OpenAI client'))
        if current_fetch_wire != 'anthropic messages':
            options.insert(1, DialogItem('anthropic messages', 'Fetch with Anthropic',
                                         'Probe the Anthropic models endpoint'))
        options.append(DialogItem('cancel', 'Cancel setup', 'Return to the prompt'))

        def choose(value: Any) -> None:
            if value == 'retry':
                self._fetch_models(base_url, api_key, name, wire, False,
                                   current_fetch_wire)
            elif value in ('openai chat completions', 'anthropic messages'):
                self._fetch_models(base_url, api_key, name, wire, False, str(value))
            elif value == 'manual':
                self._manual_models(base_url, api_key, name, wire)
            else:
                self.app.notify('Provider setup cancelled.', 'warning')
        detail = error[:120] if error else 'No model list was returned.'
        options[0].description = detail
        self.app.open_select('Could not fetch models', options, choose)

    def _manual_models(self, base_url: str, api_key: str, name: str,
                       wire: str) -> None:
        def submit(value: str) -> None:
            models = [item.strip() for item in value.split(',') if item.strip()]
            if not models:
                self.app.notify('Enter at least one model id.', 'error')
                self._manual_models(base_url, api_key, name, wire)
                return
            catalog = get_catalog()
            self._choose_models(base_url, api_key, name, wire, models, catalog,
                                False)
        self.app.open_text_prompt(
            'Manual models', 'Model ids (comma separated)', '', submit,
            hint='Metadata is filled from models.dev when a match exists.',
        )

    def _choose_models(self, base_url: str, api_key: str, name: str, wire: str,
                       models: list[str], catalog: Any, first_run: bool) -> None:
        items: list[DialogItem] = []
        for model_id in models:
            metadata = catalog.enrich(model_id, name)
            description = ''
            if metadata:
                bits = [f'ctx {_short_number(metadata["context_window"])}']
                if metadata.get('vision'):
                    bits.append('vision')
                if metadata.get('reasoning_efforts'):
                    bits.append('effort: ' + '/'.join(metadata['reasoning_efforts']))
                description = ' · '.join(bits)
            else:
                description = 'no models.dev match'
            items.append(DialogItem(model_id, model_id, description, False))
        dialog = MultiSelectDialog(
            f'{name} · select models', items,
            lambda selected: self._save_provider(base_url, api_key, name, wire,
                                                 list(selected), catalog, first_run),
            allow_manual=True, manual_label='Add a custom model id',
            width=86,
        )
        self.app.open_overlay(dialog)

    def _save_provider(self, base_url: str, api_key: str, name: str, wire: str,
                       model_ids: list[str], catalog: Any, first_run: bool) -> None:
        if not model_ids:
            self.app.notify('No models selected. Provider was not added.', 'warning')
            return
        existing = self.app.config.provider_by_name(name)
        if existing is not None:
            name = self._unique_name(name)
        models: list[ModelConfig] = []
        for model_id in model_ids:
            model = ModelConfig(id=model_id)
            metadata = catalog.enrich(model_id, name)
            if metadata:
                model.apply_metadata(metadata)
            models.append(model)
        provider = ProviderConfig(name=name, base_url=base_url, api_key=api_key,
                                  wire=wire, models=models)
        self.app.config.providers.append(provider)
        self.app.config.select(provider.name, models[0].id)
        self.app.save_config()
        self.app.notify(
            f'Provider {name} added · {len(models)} model(s) · '
            f'wire {wire}', 'success')
        self.app.after_provider_change()
        self._offer_model_edit(provider, models[0])

    def _unique_name(self, name: str) -> str:
        existing = {provider.name for provider in self.app.config.providers}
        candidate = name
        index = 2
        while candidate in existing:
            candidate = f'{name} {index}'
            index += 1
        return candidate

    def _offer_model_edit(self, provider: ProviderConfig, model: ModelConfig) -> None:
        items = [
            DialogItem('edit', f'Review {model.id}', 'Context window, vision, reasoning'),
            DialogItem('done', 'Done', 'Start using this provider'),
        ]

        def choose(value: Any) -> None:
            if value == 'edit':
                self.open_model_editor(provider, model)
            else:
                self.app.notify('Ready.', 'info')
        self.app.open_select('Provider added', items, choose)

    def open_model_editor(self, provider: ProviderConfig, model: ModelConfig) -> None:
        def save() -> None:
            self.app.save_config()
            self.app.notify(f'Updated {provider.name} / {model.id}.', 'success')
            if self.app.loop is not None and \
                    self.app.loop.provider.name == provider.name and \
                    self.app.loop.model.id == model.id:
                self.app.loop.set_model(provider, model)
        self.app.open_overlay(ModelEditorOverlay(provider, model, save))

    # ── edit flow ───────────────────────────────────────────
    def run_edit(self) -> None:
        providers = self.app.config.providers
        if not providers:
            self.app.notify('No providers configured.', 'warning')
            return
        items = [DialogItem(provider.name, provider.name,
                            f'{provider.wire} · {len(provider.models)} models')
                 for provider in providers]

        def choose(name: Any) -> None:
            provider = self.app.config.provider_by_name(str(name))
            if provider is not None:
                self._edit_details(provider)
        self.app.open_select('Edit provider', items, choose)

    def _edit_details(self, provider: ProviderConfig, step: int = 0) -> None:
        if step == 0:
            def submit(value: str) -> None:
                url = normalize_base_url(value) if value.strip() else provider.base_url
                if not _valid_url(url):
                    self.app.notify('Invalid URL.', 'error')
                    self._edit_details(provider, 0)
                    return
                provider.base_url = url
                self._edit_details(provider, 1)
            self.app.open_text_prompt('Edit provider · 1/4', 'Base URL',
                                      provider.base_url, submit)
            return
        if step == 1:
            def submit(value: str) -> None:
                if value.strip():
                    provider.api_key = value
                self._edit_details(provider, 2)
            self.app.open_text_prompt('Edit provider · 2/4', 'API key',
                                      '', submit, password=True,
                                      hint='Leave empty to keep the current key.')
            return
        if step == 2:
            def submit(value: str) -> None:
                name = value.strip()
                if name and name != provider.name:
                    old_name = provider.name
                    if self.app.config.provider_by_name(name) is not None:
                        name = self._unique_name(name)
                    provider.name = name
                    # Keep the selection pointing at the renamed provider.
                    selected = self.app.config.selected_model
                    if '::' in selected and selected.rsplit('::', 1)[0] == old_name:
                        self.app.config.select(name, selected.rsplit('::', 1)[1])
                self._edit_details(provider, 3)
            self.app.open_text_prompt('Edit provider · 3/4', 'Display name',
                                      provider.name, submit)
            return
        items = [DialogItem(wire, label, description)
                 for wire, label, description in WIRE_LABELS]
        self.app.open_select(
            'Edit provider · 4/4 · API structure', items,
            lambda wire: self._finish_edit(provider, str(wire)))

    def _finish_edit(self, provider: ProviderConfig, wire: str) -> None:
        provider.wire = wire
        self.app.save_config()
        self.app.after_provider_change()
        self.app.notify(f'Provider {provider.name} updated.', 'success')
        self._manage_models(provider)

    def _manage_models(self, provider: ProviderConfig) -> None:
        items = [DialogItem(model.id, model.id,
                            f'ctx {_short_number(model.context_window)} · '
                            f'out {_short_number(model.max_output_tokens)}')
                 for model in provider.models]
        dialog = MultiSelectDialog(
            f'{provider.name} · models', items,
            lambda selected: self._apply_models(provider, list(selected)),
            allow_manual=True, manual_label='Add a custom model id', width=86,
        )
        for item, model in zip(dialog.items, provider.models):
            item.selected = True
        self.app.open_overlay(dialog)

    def _apply_models(self, provider: ProviderConfig, model_ids: list[str]) -> None:
        if not model_ids:
            self.app.notify('A provider needs at least one model.', 'warning')
            return
        catalog = get_catalog()
        models: list[ModelConfig] = []
        for model_id in model_ids:
            existing = provider.model(model_id)
            if existing is not None:
                models.append(existing)
                continue
            model = ModelConfig(id=model_id)
            metadata = catalog.enrich(model_id, provider.name)
            if metadata:
                model.apply_metadata(metadata)
            models.append(model)
        provider.models = models
        selected = self.app.config.selected()
        if selected is not None and selected[0].name == provider.name and \
                provider.model(selected[1].id) is None:
            self.app.config.select(provider.name, models[0].id)
        self.app.save_config()
        self.app.after_provider_change()
        self.app.notify(f'{provider.name} now has {len(models)} model(s).', 'success')
        self._offer_model_edit(provider, models[0])

    def run_remove(self) -> None:
        providers = self.app.config.providers
        if not providers:
            self.app.notify('No providers configured.', 'warning')
            return
        items = [DialogItem(provider.name, provider.name,
                            f'{provider.wire} · {len(provider.models)} models')
                 for provider in providers]

        def choose(name: Any) -> None:
            def confirm(ok: bool) -> None:
                if not ok:
                    return
                self.app.config.remove_provider(str(name))
                self.app.save_config()
                self.app.after_provider_change()
                self.app.notify(f'Removed provider {name}.', 'info')
            self.app.open_confirm('Remove provider', f'Remove **{name}**?', confirm)
        self.app.open_select('Remove provider', items, choose)


def _short_number(value: int) -> str:
    if value >= 1_000_000:
        return f'{value / 1_000_000:.1f}M'
    if value >= 1_000:
        return f'{value / 1_000:.0f}k'
    return str(value)
