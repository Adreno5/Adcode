"""Whole-transcript regression gate for the render-path optimisations.

``tests/fixtures/transcript_golden.pkl.gz`` was captured from the renderer
*before* the wrap/markdown fast paths landed: every span of every visual row
(text, fg, bg, attrs, alpha) for three transcript sizes at 120 columns.  The
optimised pipeline must reproduce it exactly -- a changed glyph, a changed
style run, or a changed row break shows up here even when a spot test would
still pass.
"""

from __future__ import annotations

import gzip
import pickle
import time
from pathlib import Path

import pytest

from adcode.config import AppConfig
from adcode.session import TranscriptItem
from adcode.theme import build_palette
from adcode.tui.transcript import LiveRender, TranscriptRenderer

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'transcript_golden.pkl.gz'

DOC = ('## Heading\n\nSome **bold** text and `code` here with prose that wraps.\n'
       '- bullet one with a few words\n- bullet two `x = 1`\n\n'
       '```python\ndef f(a, b):\n    return a + b\n```\n\n'
       'Trailing paragraph with [a link](https://example.com) and *emphasis*.\n')


def build_items(n: int) -> list[TranscriptItem]:
    out: list[TranscriptItem] = []
    for i in range(n):
        kind = ('user', 'assistant', 'thinking', 'tool', 'notice')[i % 5]
        if kind == 'assistant':
            out.append(TranscriptItem(kind=kind, text=DOC + f'\npara {i}\n'))
        elif kind == 'tool':
            out.append(TranscriptItem(kind=kind, name='bash',
                                      display='bash(pytest)',
                                      args={'command': 'pytest -q'},
                                      output='\n'.join(f'line {j} of output'
                                                       for j in range(15)),
                                      status='done'))
        elif kind == 'thinking':
            out.append(TranscriptItem(kind=kind, text=f'thinking {i} ' * 6,
                                      duration=1.2))
        else:
            out.append(TranscriptItem(kind=kind, text=f'item {i} ' * 5,
                                      level='info'))
    return out


def serialise(lines) -> list[list[tuple]]:
    return [[(s.text, str(s.fg), str(s.bg), s.attrs, s.alpha) for s in line]
            for line in lines]


@pytest.fixture(scope='module')
def golden() -> dict:
    with gzip.open(FIXTURE, 'rb') as handle:
        return pickle.load(handle)


@pytest.mark.parametrize('count', [50, 200, 500])
def test_transcript_matches_pre_optimisation_golden(golden, count):
    renderer = TranscriptRenderer()
    lines = renderer.render(build_items(count), LiveRender(phase='idle'), 120,
                            AppConfig(), 0, time.time(), build_palette('dark'))
    got = serialise(lines)
    want = golden[count]
    if got != want:
        first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b),
                     min(len(got), len(want)))
        pytest.fail(f'row {first} differs:\n'
                    f'  got  {got[first] if first < len(got) else "<missing>"}\n'
                    f'  want {want[first] if first < len(want) else "<missing>"}')
    assert len(got) == len(want)
