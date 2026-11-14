from adcode.models_dev import CatalogModel, ModelCatalog, parse_catalog


RAW = {
    'acme': {
        'id': 'acme', 'name': 'Acme', 'models': {
            'acme-reasoner': {
                'id': 'acme-reasoner', 'name': 'Acme Reasoner',
                'reasoning': True,
                'reasoning_options': [
                    {'type': 'toggle'},
                    {'type': 'effort', 'values': ['low', 'high', 'max']},
                ],
                'tool_call': True, 'temperature': True,
                'modalities': {'input': ['text', 'image'], 'output': ['text']},
                'limit': {'context': 200_000, 'output': 32_000},
                'release_date': '2026-01-01',
            },
            'acme-fast-20260101': {
                'id': 'acme-fast-20260101', 'name': 'Acme Fast',
                'reasoning': True,
                'reasoning_options': [{'type': 'toggle'}],
                'modalities': {'input': ['text'], 'output': ['text']},
                'limit': {'context': 64_000, 'output': 8_000},
                'release_date': '2026-01-01',
            },
        },
    },
    'other': {
        'id': 'other', 'name': 'Other', 'models': {
            'acme-reasoner': {
                'id': 'acme-reasoner', 'name': 'Acme Reasoner (other)',
                'reasoning': False, 'modalities': {'input': ['text']},
                'limit': {'context': 10_000, 'output': 1_000},
                'release_date': '2025-01-01',
            },
        },
    },
}


def test_parse_catalog():
    models = parse_catalog(RAW)
    assert len(models) == 3
    reasoner = next(model for model in models
                    if model.provider_id == 'acme' and model.id == 'acme-reasoner')
    assert reasoner.vision
    assert reasoner.context_window == 200_000
    assert reasoner.max_output_tokens == 32_000
    assert reasoner.reasoning_style == 'effort'
    assert reasoner.reasoning_efforts == ('low', 'high', 'max')


def test_find_exact_and_provider_hint():
    catalog = ModelCatalog(parse_catalog(RAW))
    model = catalog.find('acme-reasoner', 'other')
    assert model.provider_id == 'other'
    model = catalog.find('acme-reasoner', 'acme')
    assert model.provider_id == 'acme'


def test_find_strips_date_suffix_and_loose_match():
    catalog = ModelCatalog(parse_catalog(RAW))
    assert catalog.find('acme-fast') is not None
    assert catalog.find('acme-reasoner-20260202') is not None
    assert catalog.find('nonexistent') is None


def test_enrich_and_roundtrip():
    catalog = ModelCatalog(parse_catalog(RAW))
    metadata = catalog.enrich('acme-reasoner', 'acme')
    assert metadata['vision'] is True
    assert metadata['source'] == 'models.dev'
    clone = ModelCatalog.from_json(catalog.to_json())
    assert len(clone) == len(catalog)
    assert clone.find('acme-reasoner', 'acme').context_window == 200_000


def test_catalog_model_metadata_shape():
    model = CatalogModel(id='x', vision=True, context_window=1,
                         max_output_tokens=2, reasoning=True,
                         reasoning_style='effort', reasoning_efforts=('low',))
    data = model.as_metadata()
    assert data['reasoning_efforts'] == ['low']
    assert data['vision'] is True
