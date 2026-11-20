"""Renaming a session in the real application."""

from __future__ import annotations

import pytest


def run_turn(harness, text: str) -> None:
    """Type a message, send it, and wait for the model answer to finish."""
    requests = len(harness.requests)
    harness.type_text(text)
    harness.press('enter')
    harness.wait_until(lambda: len(harness.requests) > requests, timeout=20.0,
                       description=f'the request for {text!r}')
    harness.wait_idle(20.0)


def command(harness, text: str, done, description: str = '') -> None:
    """Send a slash command and wait for its effect."""
    harness.type_text(text)
    harness.press('enter')
    harness.wait_until(done, timeout=15.0,
                       description=description or f'{text} to take effect')


def notices(harness) -> list[str]:
    return [text for kind, text in harness.transcript_texts() if kind == 'notice']


@pytest.fixture()
def live_run(harness_factory):
    """A real run with one finished turn, so the session exists on disk."""
    harness = harness_factory()
    run_turn(harness, 'hi')
    app = harness.app
    assert app is not None and app.loop is not None
    assert app.loop.session_name
    return harness


def test_rename_moves_the_session_to_the_new_name(live_run):
    harness = live_run
    app = harness.app
    old_name = app.loop.session_name

    command(harness, '/rename demo',
            lambda: any('renamed' in text for text in notices(harness)),
            'the session to be renamed')

    assert app.loop.session_name == 'demo'
    names = [row['name'] for row in app.session_store.list()]
    # the session was moved, not copied: the old name is gone
    assert 'demo' in names
    assert old_name not in names


def test_rename_without_a_name_asks_for_one(live_run):
    harness = live_run
    app = harness.app

    command(harness, '/rename', lambda: app.overlay is not None,
            'the rename prompt')
    assert type(app.overlay).__name__ == 'TextPrompt'

    harness.type_text('from-prompt')
    harness.press('enter')
    harness.wait_until(lambda: app.loop.session_name == 'from-prompt',
                       timeout=15.0, description='the session to be renamed')
    assert 'from-prompt' in [row['name'] for row in app.session_store.list()]


def test_the_settings_command_opens_the_settings_browser(live_run):
    harness = live_run
    app = harness.app

    command(harness, '/settings', lambda: app.overlay is not None,
            'the settings browser')
    assert type(app.overlay).__name__ == 'SettingsOverlay'
    harness.press('esc')
    harness.wait_until(lambda: app.overlay is None, timeout=10.0,
                       description='the settings to close')


def test_the_language_switches_the_interface(live_run):
    harness = live_run
    app = harness.app

    command(harness, '/settings', lambda: app.overlay is not None,
            'the settings browser')
    harness.wait_for_text('Recents')

    # The pinned section comes first: theme, language, accent, ...
    harness.press('down')
    harness.press('down')
    harness.wait_until(
        lambda: getattr(harness.overlay, 'row_key', '') == 'recent:ui.locale',
        timeout=5.0, description='the Language row to be selected')
    harness.press('enter')
    harness.wait_until(
        lambda: 'recent:ui.locale'
        in getattr(harness.overlay, 'expanded', []),
        timeout=5.0, description='the Language item to expand')
    harness.press('down')
    harness.wait_until(
        lambda: getattr(harness.overlay, 'row_key', '')
        == 'recent:ui.locale::en',
        timeout=5.0, description='the English option to be highlighted')
    harness.press('down')
    harness.press('enter')

    harness.wait_until(
        lambda: app.config.ui.locale == 'zh', timeout=10.0,
        description='the language to be switched to Chinese')
    harness.wait_for_text('常用设置')
    assert harness.workspace.config_json()['ui']['locale'] == 'zh'
