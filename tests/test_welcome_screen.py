"""The welcome pane over a real run: fresh start, first message, /clear."""

from __future__ import annotations

LOGO_NEEDLE = '▀'


def test_fresh_start_shows_the_logo_and_a_tip(harness):
    harness.wait_until(
        lambda: 'try /' in harness.screen_text(),
        timeout=10.0, description='the welcome tip to appear')
    assert LOGO_NEEDLE in harness.screen_text()
    # A fresh session has no content and no stale session title.
    assert harness.transcript_items() == []
    assert 'AdCode  ·  ' not in harness.screen_text().splitlines()[0]


def test_the_first_message_replaces_the_welcome(harness):
    harness.submit('hello')
    text = harness.screen_text()
    assert 'try /' not in text
    assert LOGO_NEEDLE not in text


def test_clear_restores_the_welcome_with_no_notice(harness):
    harness.submit('hello')
    harness.run_command('/clear')
    harness.wait_until(
        lambda: 'try /' in harness.screen_text(),
        timeout=10.0, description='the welcome screen to return')
    assert LOGO_NEEDLE in harness.screen_text()
    kinds = [item['kind'] for item in harness.transcript_items()]
    assert 'notice' not in kinds
    # The header bar is clean: no stale session title after the clear.
    assert 'AdCode  ·  ' not in harness.screen_text().splitlines()[0]
