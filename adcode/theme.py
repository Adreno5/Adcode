"""Colour palette and semantic style tokens for the AdCode terminal UI.

The interface keeps the theme colour to a minimum: content is layered with
light grey and white, and the accent colour appears only as small markers
(``>``, ``|``, selected rows).  Palettes are plain RGB tuples so the canvas can
blend them with an alpha channel like a small software renderer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

RGB = tuple[int, int, int]

# ── colour helpers ─────────────────────────────────────────


def parse_hex(value: str, fallback: RGB) -> RGB:
    """Parse ``#rrggbb`` (or ``rrggbb``) into an RGB tuple."""
    text = value.strip().lstrip('#')
    if len(text) == 3:
        text = ''.join(ch * 2 for ch in text)
    if len(text) != 6:
        return fallback
    try:
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    except ValueError:
        return fallback


def to_hex(color: RGB) -> str:
    return '#{:02x}{:02x}{:02x}'.format(*color)


def mix(source: RGB, target: RGB, amount: float) -> RGB:
    """Blend ``source`` towards ``target`` by ``amount`` (0..1)."""
    amount = max(0.0, min(1.0, amount))
    return (
        round(source[0] + (target[0] - source[0]) * amount),
        round(source[1] + (target[1] - source[1]) * amount),
        round(source[2] + (target[2] - source[2]) * amount),
    )


def blend_alpha(foreground: RGB, background: RGB, alpha: int) -> RGB:
    """Composite ``foreground`` over ``background`` using an 8-bit alpha."""
    if alpha >= 255:
        return foreground
    if alpha <= 0:
        return background
    return mix(background, foreground, alpha / 255.0)


@dataclass(frozen=True)
class Palette:
    name: str
    background: RGB
    panel: RGB
    panel_alt: RGB
    border: RGB
    text: RGB
    subtext: RGB
    muted: RGB
    accent: RGB
    success: RGB
    error: RGB
    warning: RGB
    info: RGB
    diff_add: RGB
    diff_del: RGB
    selection_bg: RGB
    selection_fg: RGB
    status_bg: RGB

    def with_accent(self, accent: RGB) -> 'Palette':
        return replace(self, accent=accent)


DARK_BASE = Palette(
    name='dark',
    background=(13, 14, 17),
    panel=(21, 23, 27),
    panel_alt=(28, 30, 36),
    border=(45, 48, 55),
    text=(232, 233, 236),
    subtext=(158, 162, 170),
    muted=(96, 101, 110),
    accent=(143, 167, 214),
    success=(126, 178, 134),
    error=(207, 122, 122),
    warning=(214, 176, 112),
    info=(128, 160, 196),
    diff_add=(126, 178, 134),
    diff_del=(207, 122, 122),
    selection_bg=(38, 43, 54),
    selection_fg=(245, 246, 248),
    status_bg=(18, 20, 24),
)

LIGHT_BASE = Palette(
    name='light',
    background=(250, 250, 248),
    panel=(243, 243, 240),
    panel_alt=(235, 235, 232),
    border=(205, 205, 200),
    text=(32, 33, 36),
    subtext=(96, 99, 105),
    muted=(140, 143, 150),
    accent=(58, 104, 168),
    success=(56, 128, 70),
    error=(176, 58, 58),
    warning=(150, 108, 32),
    info=(52, 108, 160),
    diff_add=(56, 128, 70),
    diff_del=(176, 58, 58),
    selection_bg=(225, 231, 240),
    selection_fg=(20, 21, 24),
    status_bg=(246, 246, 243),
)

THEMES = {'dark': DARK_BASE, 'light': LIGHT_BASE}

DEFAULT_ACCENT = (143, 167, 214)

ACCENT_PRESETS: list[tuple[str, str]] = [
    ('Slate blue', '#8fa7d6'),
    ('Sand', '#d7b78f'),
    ('Sage', '#9dbf9d'),
    ('Rose', '#d29aa6'),
    ('Violet', '#b3a1d6'),
    ('Teal', '#86bfc0'),
    ('Amber', '#d9b06c'),
    ('Mono', '#c8cbd1'),
]


def build_palette(theme: str = 'dark', accent: str = '') -> Palette:
    base = THEMES.get(theme, DARK_BASE)
    if accent:
        base = base.with_accent(parse_hex(accent, base.accent))
    return base


# Semantic colour roles used by spans and widgets.
ROLES = (
    'background', 'panel', 'panel_alt', 'border', 'text', 'subtext', 'muted',
    'accent', 'success', 'error', 'warning', 'info', 'diff_add', 'diff_del',
    'selection_bg', 'selection_fg', 'status_bg',
)


def resolve_role(palette: Palette, role: str | RGB | None,
                 default: str = 'text') -> RGB:
    if role is None:
        role = default
    if isinstance(role, tuple):
        return role
    return getattr(palette, role, palette.text)
