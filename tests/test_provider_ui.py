from adcode.config import AppConfig, ModelConfig, ProviderConfig
from adcode.models_dev import CatalogModel, ModelCatalog
from adcode.tui.provider_ui import ProviderWizard, normalize_base_url


class FakeApp:
    def __init__(self, config):
        self.config = config
        self.loop = None
        self.saved = 0
        self.notices = []
        self.selects = []
        self.prompts = []

    def save_config(self):
        self.saved += 1

    def notify(self, text, level='info'):
        self.notices.append((level, text))

    def after_provider_change(self):
        pass

    def open_select(self, title, items, on_select, width=72):
        self.selects.append((title, items, on_select))

    def open_text_prompt(self, *args, **kwargs):
        self.prompts.append((args, kwargs))

    def open_overlay(self, overlay):
        pass


def catalog():
    return ModelCatalog([
        CatalogModel(
            id='m1', name='Model One', provider_id='acme', provider_name='Acme',
            vision=True, context_window=200_000, max_output_tokens=32_000,
            reasoning=True, reasoning_style='effort',
            reasoning_efforts=('low', 'high', 'max'), release_date='2026-01-01'),
    ])


def test_save_provider_enriches_models():
    cfg = AppConfig()
    app = FakeApp(cfg)
    wizard = ProviderWizard(app)
    wizard._save_provider('https://api.acme.test/v1', 'key', 'Acme',
                          'openai chat completions', ['m1', 'custom'],
                          catalog(), True)
    provider = cfg.provider_by_name('Acme')
    assert provider is not None
    assert provider.api_key == 'key'
    assert [model.id for model in provider.models] == ['m1', 'custom']
    model = provider.model('m1')
    assert model.vision
    assert model.context_window == 200_000
    assert model.reasoning_efforts == ['low', 'high', 'max']
    assert provider.model('custom').context_window == 128_000
    assert cfg.selected()[0].name == 'Acme'
    assert app.saved == 1


def test_save_provider_renames_duplicate():
    cfg = AppConfig()
    cfg.providers = [ProviderConfig(name='Acme', base_url='http://x',
                                    models=[ModelConfig(id='old')])]
    app = FakeApp(cfg)
    wizard = ProviderWizard(app)
    wizard._save_provider('http://y', 'k', 'Acme', 'openai chat completions',
                          ['m1'], catalog(), True)
    names = [provider.name for provider in cfg.providers]
    assert 'Acme' in names and 'Acme 2' in names
    assert cfg.selected()[0].name == 'Acme 2'


def test_apply_models_preserves_existing_and_adds(monkeypatch):
    import adcode.tui.provider_ui as provider_ui
    monkeypatch.setattr(provider_ui, 'get_catalog', catalog)
    cfg = AppConfig()
    provider = ProviderConfig(name='P', base_url='http://x',
                              models=[ModelConfig(id='old', vision=False)])
    cfg.providers = [provider]
    cfg.select('P', 'old')
    app = FakeApp(cfg)
    wizard = ProviderWizard(app)
    wizard._apply_models(provider, ['old', 'm1'])
    assert [model.id for model in provider.models] == ['old', 'm1']
    assert provider.model('old').vision is False
    assert provider.model('m1').vision is True
    assert app.saved == 1


def test_apply_models_requires_one():
    cfg = AppConfig()
    provider = ProviderConfig(name='P', base_url='http://x',
                              models=[ModelConfig(id='old')])
    cfg.providers = [provider]
    cfg.select('P', 'old')
    app = FakeApp(cfg)
    ProviderWizard(app)._apply_models(provider, [])
    assert [model.id for model in provider.models] == ['old']
    assert any(level == 'warning' for level, _ in app.notices)


def test_normalize_url():
    assert normalize_base_url('https://x.test') == 'https://x.test/v1'
    assert normalize_base_url('https://x.test/api#') == 'https://x.test/api'
