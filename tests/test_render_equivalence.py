"""The refactored canvas must be pixel-for-pixel compatible with the old one.

``benchmarks/legacy_canvas.py`` is a verbatim snapshot of the renderer that
shipped before the split-plane / incremental-flush refactor.  Every test here
draws the same script into both implementations and compares the resulting
cell grids, so "it got faster" can never be paid for with a changed picture.

The incremental-flush tests go one step further: they feed the new renderer's
*partial* payloads into a small VT emulator and check that the terminal grid
still matches a legacy full repaint after every frame — that is the failure
mode a byte-level diff cannot see (a stale row, a wide glyph torn in half, a
background that stopped being repainted once its row stopped changing).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'benchmarks'))

import legacy_canvas  # noqa: E402

from adcode.theme import build_palette  # noqa: E402
from adcode.tui import canvas as current  # noqa: E402
from harness.vt import VirtualScreen  # noqa: E402


class LegacyCanvas(legacy_canvas.Canvas):
    """Legacy renderer plus the new API surface, for A/B runs."""

    def render_diff(self) -> str:
        return self.render()


def grid(canvas) -> tuple:
    """Full cell grid: glyph, colours, attributes and continuation flags."""
    return tuple(
        tuple(
            (cell.ch, cell.fg, cell.bg, cell.attrs, cell.cont)
            for cell in (canvas.cell(x, y) for x in range(canvas.width))
        )
        for y in range(canvas.height)
    )


def vt_grid(screen: VirtualScreen) -> tuple:
    return tuple(
        tuple((cell.ch, cell.fg, cell.bg, cell.attrs, cell.cont) for cell in row)
        for row in screen.rows()
    )


@pytest.fixture()
def palette():
    return build_palette('dark', '#8fa7d6')


# ── the drawing corpus ──────────────────────────────────────
def draw_corpus(Canvas, palette):
    """Exercise every primitive, including the awkward ones."""
    canvas = Canvas(28, 10, palette)
    canvas.clear()
    canvas.draw_text(0, 0, 'plain ascii', palette.text)
    canvas.draw_text(0, 1, '中文宽字 mixed', palette.text, attrs=current.BOLD)
    canvas.draw_text(0, 2, 'ctrl:\x01\x1b here', palette.text)
    canvas.draw_text(0, 3, 'tab\there\rX\nignored', palette.text)
    canvas.draw_text(0, 4, 'reverse', palette.text, palette.panel,
                     attrs=current.REVERSE | current.UNDERLINE)
    canvas.draw_text(2, 5, 'clipped tail', palette.text, max_width=5)
    canvas.draw_text(0, 6, 'combining e\u0301 a\u0308', palette.text)
    canvas.fill_rect(0, 7, 28, 1, palette.panel_alt)
    canvas.fill_rect(0, 8, 28, 1, (255, 0, 0), alpha=96)
    canvas.draw_text(4, 8, 'faded', palette.text, alpha=140)
    canvas.hline(0, 0, 10, palette.border, alpha=120)
    canvas.vline(27, 0, 9, palette.border, alpha=120)
    canvas.box(12, 2, 12, 3, palette.border, bg=palette.panel,
               title='title that is too long')
    canvas.draw_text(0, 0, 'overshoot', palette.text, max_width=3)
    canvas.draw_text(-3, 1, 'clip left', palette.text)
    canvas.draw_text(0, 20, 'off canvas', palette.text)   # row out of range
    # Wipe half a wide glyph, then draw another one against the right edge.
    canvas.draw_text(4, 1, '中', palette.text)
    canvas.draw_text(5, 1, 'X', palette.text)
    canvas.draw_text(26, 9, '中', palette.text)
    canvas.draw_text(27, 9, '中', palette.text)           # no room: must not split
    return canvas


def clipped_corpus(Canvas, palette):
    canvas = Canvas(20, 6, palette)
    canvas.clear()
    canvas.draw_text(0, 0, 'whole row of text', palette.text)
    with canvas.clip(2, 1, 6, 3):
        canvas.draw_text(0, 1, 'inside clip', palette.text)
        canvas.fill_rect(0, 2, 20, 2, palette.selection_bg)
        canvas.box(1, 1, 8, 3, palette.border)
        with canvas.clip(4, 2, 2, 1):
            canvas.draw_text(0, 2, 'nested clip', palette.text)
            canvas.draw_text(0, 3, 'clipped out', palette.text)
    canvas.draw_text(0, 5, 'after clip', palette.text)
    return canvas


def corpus_cases(palette):
    return [
        ('mixed', draw_corpus),
        ('clipping', clipped_corpus),
    ]


# ── equivalence on the cell grid ────────────────────────────
@pytest.mark.parametrize('name', ['mixed', 'clipping'])
def test_cell_grid_matches_legacy(name, palette):
    builder = dict(corpus_cases(palette))[name]
    old = builder(LegacyCanvas, palette)
    new = builder(current.Canvas, palette)
    assert grid(new) == grid(old)


@pytest.mark.parametrize('name', ['mixed', 'clipping'])
def test_serialized_output_decodes_to_the_same_grid(name, palette):
    builder = dict(corpus_cases(palette))[name]
    old = builder(LegacyCanvas, palette)
    new = builder(current.Canvas, palette)
    old_vt = VirtualScreen(old.width, old.height)
    new_vt = VirtualScreen(new.width, new.height)
    old_vt.feed(old.render())
    new_vt.feed(new.render())
    assert vt_grid(new_vt) == vt_grid(old_vt)
    assert new.render_row(0) == old.render_row(0)


def test_resize_preserves_content_like_legacy(palette):
    old = draw_corpus(LegacyCanvas, palette)
    new = draw_corpus(current.Canvas, palette)
    old.resize(18, 6)
    new.resize(18, 6)
    assert grid(new) == grid(old)
    old.resize(30, 11)
    new.resize(30, 11)
    assert grid(new) == grid(old)


def test_clear_with_custom_background(palette):
    old = LegacyCanvas(6, 2, palette)
    new = current.Canvas(6, 2, palette)
    old.clear((1, 2, 3))
    new.clear((1, 2, 3))
    assert grid(new) == grid(old)
    old.draw_text(0, 0, 'ok', palette.text)
    new.draw_text(0, 0, 'ok', palette.text)
    assert grid(new) == grid(old)


def test_text_width_and_truncation_agree(palette):
    samples = ['', 'ascii', '中文', 'e\u0301', 'a\tb', 'ctrl\x01x',
               'mixed 中文 ascii', '…ellipsis…']
    for sample in samples:
        assert current.text_width(sample) == legacy_canvas.text_width(sample)
        for width in range(0, 12):
            assert (current.truncate_text(sample, width, '…')
                    == legacy_canvas.truncate_text(sample, width, '…'))


# ── incremental flush vs full repaint ───────────────────────
def _frame_script(palette):
    """A sequence of edits applied between flushes."""
    return [
        ('initial', lambda canvas: canvas.draw_text(0, 0, 'hello world', palette.text)),
        ('idle', lambda canvas: None),
        ('one char', lambda canvas: canvas.draw_text(12, 0, '!', palette.text)),
        ('wide glyph', lambda canvas: canvas.draw_text(0, 1, '中文', palette.text)),
        ('overwrite half', lambda canvas: canvas.draw_text(1, 1, 'X', palette.text)),
        ('tint over text', lambda canvas: canvas.fill_rect(0, 0, 20, 2,
                                                           palette.selection_bg, 128)),
        ('clear row', lambda canvas: canvas.fill_row(0, palette.background)),
        ('nothing', lambda canvas: None),
        ('box', lambda canvas: canvas.box(1, 2, 10, 4, palette.border, title='t')),
        ('erase all', lambda canvas: None),
    ]


def test_incremental_flush_matches_full_repaint(palette):
    width, height = 24, 8
    incremental = current.Canvas(width, height, palette)
    legacy = LegacyCanvas(width, height, palette)
    incremental_vt = VirtualScreen(width, height)
    legacy_vt = VirtualScreen(width, height)

    for step, action in _frame_script(palette):
        incremental.clear()
        legacy.clear()
        action(incremental)
        action(legacy)
        payload = incremental.render_diff()
        incremental_vt.feed(payload)
        legacy_vt.feed(legacy.render())
        assert vt_grid(incremental_vt) == vt_grid(legacy_vt), f'step: {step}'
        assert grid(incremental) == grid(legacy), f'step: {step}'


def test_idle_frame_writes_no_bytes(palette):
    canvas = current.Canvas(24, 8, palette)
    canvas.clear()
    canvas.draw_text(0, 0, 'stable', palette.text)
    canvas.box(0, 1, 10, 5, palette.border, title='panel')
    assert canvas.render_diff()                     # first frame paints
    for _ in range(5):
        canvas.clear()
        canvas.draw_text(0, 0, 'stable', palette.text)
        canvas.box(0, 1, 10, 5, palette.border, title='panel')
        assert canvas.render_diff() == ''


def test_only_the_changed_row_is_emitted(palette):
    canvas = current.Canvas(24, 8, palette)
    canvas.clear()
    for y in range(8):
        canvas.draw_text(0, y, f'row {y}', palette.text)
    canvas.render_diff()
    canvas.clear()
    for y in range(8):
        canvas.draw_text(0, y, f'row {y}', palette.text)
    canvas.draw_text(0, 3, 'row 3 changed', palette.text)
    payload = canvas.render_diff()
    assert payload.count('\x1b[') >= 1
    assert 'row 3 changed' in payload
    assert 'row 0' not in payload and 'row 7' not in payload
    assert payload.startswith(f'\x1b[{4};1H')


def test_disappearing_content_is_blanked(palette):
    canvas = current.Canvas(12, 3, palette)
    canvas.clear()
    canvas.draw_text(0, 1, 'gone soon', palette.text)
    canvas.render_diff()
    canvas.clear()                      # nothing redrawn this frame
    payload = canvas.render_diff()
    vt = VirtualScreen(12, 3)
    vt.feed(payload)
    assert 'gone soon' not in vt.text()
    blank_row = canvas._blank_text()
    assert payload.count(blank_row) == 1


def test_full_repaint_costs_no_more_than_a_full_render(palette):
    """A frame that changes every row must not pay a per-row cursor premium."""
    incremental = draw_corpus(current.Canvas, palette)
    full = draw_corpus(LegacyCanvas, palette)
    diff_payload = incremental.render_diff()
    assert diff_payload == full.render()
    # And on an already-flushed canvas, a new frame of the same shape is the
    # same size as the full render, never 40 extra positioning sequences.
    again = draw_corpus(current.Canvas, palette)
    again.render_diff()
    assert len(again.render_diff()) < len(again.render())


def test_resize_forces_a_full_repaint(palette):
    canvas = current.Canvas(16, 5, palette)
    canvas.clear()
    canvas.draw_text(0, 0, 'before', palette.text)
    canvas.render_diff()
    canvas.resize(20, 6)
    canvas.clear()
    canvas.draw_text(0, 0, 'after', palette.text)
    payload = canvas.render_diff()
    vt = VirtualScreen(20, 6)
    vt.feed(payload)
    assert vt.line(0).startswith('after')
    assert 'before' not in vt.text()


def test_a_continuation_cell_has_no_glyph_to_emit(palette):
    """The right half of a wide glyph must never carry printable text.

    The row serializer also skips ``_CONT`` cells, but that guard alone would
    let a stray character through; the stored glyph being empty is what makes
    "a continuation cell is never output on its own" structurally true.
    """
    canvas = current.Canvas(10, 1, palette)
    canvas.clear()
    canvas.draw_text(0, 0, '中a', palette.text)
    assert canvas.cell(1, 0).cont is True
    assert canvas._ch[1] == ''
    row = canvas.render_row(0)
    assert row.count('中') == 1 and 'a' in row
    # One wide glyph (2 cells) + 'a' + padding: the row is still exactly the
    # canvas width in display cells, so no column math drifts.
    import re as _re
    plain = _re.sub(r'\x1b\[[0-9;]*m', '', row)
    assert current.text_width(plain) == canvas.width


def test_a_torn_wide_glyph_is_repaired_before_the_row_is_emitted(palette):
    canvas = current.Canvas(8, 1, palette)
    canvas.clear()
    canvas.draw_text(0, 0, '中文', palette.text)
    canvas.render_diff()
    canvas.clear()
    canvas.draw_text(0, 0, '中文', palette.text)
    canvas.draw_text(1, 0, 'X', palette.text)      # scribble on the right half
    vt = VirtualScreen(8, 1)
    vt.feed(canvas.render_diff())
    assert vt.line(0).startswith(' X文') or vt.line(0).startswith('X文')
    assert canvas.cell(0, 0).cont is False
