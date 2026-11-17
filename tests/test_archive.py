"""Archived sessions: read-only, hidden, and restorable.

Archiving is a property of the session row, so it survives the engine saving
the same session again.  An archived session is hidden from the list and from
auto-resume, refuses new messages and the commands that work on it, and comes
back with /respawn or disappears for good with /perish.  The rest of the
application -- /new, /config, /sessions, /help -- keeps working.
"""

from __future__ import annotations

import time

import pytest

from adcode.session import ArchivedSessionStore, SessionPayload
from adcode.tui.input import KeyEvent


def key(name: str) -> KeyEvent:
    return KeyEvent(kind='key', key=name)


def modifier(name: str, action: str) -> KeyEvent:
    return KeyEvent(kind='modifier', key=name, action=action)


@pytest.fixture()
def open_store(tmp_path):
    """A store on a temporary database, closed when the test is over."""
    created: list[ArchivedSessionStore] = []

    def make() -> ArchivedSessionStore:
        instance = ArchivedSessionStore(
            db_path=str(tmp_path / 'sessions' / 'agent.db'))
        created.append(instance)
        return instance

    yield make
    for instance in created:
        instance.store.close()


def payload(name: str, text: str = 'hello') -> SessionPayload:
    return SessionPayload(name=name, title=f'title {name}', messages=[
        {'role': 'user', 'content': text},
        {'role': 'assistant', 'content': 'hi'},
    ])


# ── the store ──────────────────────────────────────────────
def test_archiving_hides_a_session_from_the_list(open_store):
    sessions = open_store()
    sessions.save(payload('alpha'))
    sessions.save(payload('beta'))
    assert {row['name'] for row in sessions.list()} == {'alpha', 'beta'}

    assert sessions.archive('alpha') is True
    assert {row['name'] for row in sessions.list()} == {'beta'}
    assert sessions.is_archived('alpha')

    listed = sessions.list(include_archived=True)
    flags = {row['name']: row['archived'] for row in listed}
    assert flags == {'alpha': True, 'beta': False}


def test_an_archived_session_is_never_resumed(open_store):
    sessions = open_store()
    sessions.save(payload('old'))
    sessions.save(payload('new'))
    assert sessions.latest() is not None
    # The checkpoint is what a run resumes; without one the newest live
    # session is used, and the archived one is not a candidate.
    sessions.archive('new')
    assert sessions.latest() is not None
    assert sessions.latest().name == 'old'


def test_archiving_survives_the_engine_saving_again(open_store):
    sessions = open_store()
    sessions.save(payload('alpha'))
    sessions.archive('alpha')
    # The engine writes the payload again on every turn: the row and its
    # messages are replaced, the archive flag is not.
    sessions.save(payload('alpha', text='second turn'))
    assert sessions.is_archived('alpha')
    assert sessions.load('alpha').messages[0]['content'] == 'second turn'


def test_respawn_restores_the_session(open_store):
    sessions = open_store()
    sessions.save(payload('alpha'))
    sessions.archive('alpha')
    assert sessions.unarchive('alpha') is True
    assert not sessions.is_archived('alpha')
    assert [row['name'] for row in sessions.list()] == ['alpha']


def test_deleting_a_session_drops_its_archive_too(open_store):
    sessions = open_store()
    sessions.save(payload('alpha'))
    sessions.archive('alpha')
    assert sessions.delete('alpha') is True
    assert sessions.is_archived('alpha') is False
    assert sessions.load('alpha') is None
    # The name is free again, so a new session may take it.
    assert sessions.unique_name('alpha') == 'alpha'


def test_archiving_a_session_that_is_not_there_is_refused(open_store):
    sessions = open_store()
    assert sessions.archive('nope') is False
    assert sessions.is_archived('nope') is False


# ── the application ────────────────────────────────────────
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
    return [text for kind, text in harness.transcript_texts()
            if kind == 'notice']


@pytest.fixture()
def archived_run(harness_factory):
    """A real run with one finished turn, so the session exists on disk."""
    harness = harness_factory()
    run_turn(harness, 'hi')
    app = harness.app
    assert app is not None and app.loop is not None
    assert app.loop.session_name
    return harness


def test_archive_makes_the_session_read_only(archived_run):
    harness = archived_run
    app = harness.app
    name = app.loop.session_name

    command(harness, '/archive', lambda: app.session_archived,
            'the session to be archived')
    assert app.session_store.is_archived(name)
    assert any('Archived' in text for text in notices(harness))
    # Hidden from the list a new run would resume from.
    assert name not in [row['name'] for row in app.session_store.list()]

    # A message is refused, and it stays in the prompt for after /respawn.
    requests = len(harness.requests)
    app.status_hint = ''            # the archive command set one of its own
    harness.type_text('can you hear me')
    harness.press('enter')
    harness.wait_until(lambda: app.status_hint != '', timeout=10.0,
                       description='the refusal to show')
    assert len(harness.requests) == requests
    assert app.prompt.text == 'can you hear me'
    assert 'archived' in app.status_hint


def test_the_commands_that_act_on_the_session_are_refused(archived_run):
    """Only what works on this conversation is refused, not the whole app."""
    harness = archived_run
    app = harness.app
    name = app.loop.session_name
    command(harness, '/archive', lambda: app.session_archived,
            'the session to be archived')
    requests = len(harness.requests)

    def refusals() -> int:
        return len([text for text in notices(harness) if 'is archived' in text])

    for text in ('/model', '/btw 2+2', '/compact', '/save copy', '/rewind'):
        seen = refusals()
        command(harness, text,
                lambda seen=seen: refusals() > seen,
                f'{text} to be refused')

    # None of them ran: no model picker, no aside turn, no second session.
    assert app.overlay is None
    assert len(harness.requests) == requests
    assert app.loop.session_name == name
    assert [row['name'] for row in app.session_store.list()] == []

def test_the_app_commands_keep_working_while_archived(archived_run):
    harness = archived_run
    app = harness.app
    command(harness, '/archive', lambda: app.session_archived,
            'the session to be archived')

    command(harness, '/help', lambda: app.overlay is not None,
            'the help dialog')
    assert type(app.overlay).__name__ == 'MessageDialog'
    harness.press('esc')
    harness.wait_until(lambda: app.overlay is None, timeout=10.0,
                       description='the help dialog to close')

    command(harness, '/sessions', lambda: app.overlay is not None,
            'the sessions browser')
    harness.press('esc')
    harness.wait_until(lambda: app.overlay is None, timeout=10.0,
                       description='the sessions browser to close')

    command(harness, '/config', lambda: app.overlay is not None,
            'the settings browser')
    assert type(app.overlay).__name__ == 'SettingsOverlay'
    harness.press('esc')
    harness.wait_until(lambda: app.overlay is None, timeout=10.0,
                       description='the settings to close')

    assert not [text for text in notices(harness) if 'is archived' in text]

def test_new_starts_a_live_session_and_leaves_the_archived_one(archived_run):
    harness = archived_run
    app = harness.app
    name = app.loop.session_name
    command(harness, '/archive', lambda: app.session_archived,
            'the session to be archived')

    command(harness, '/new',
            lambda: not app.session_archived and app.loop.session_name == '',
            'the new conversation')

    run_turn(harness, 'a fresh start')
    assert app.session_archived is False
    assert app.loop.session_name not in ('', name)
    # The archived session is untouched: still on disk, still archived.
    assert app.session_store.is_archived(name)
    assert app.session_store.is_archived(app.loop.session_name) is False


def test_respawn_restores_the_session(archived_run):
    harness = archived_run
    app = harness.app
    command(harness, '/archive', lambda: app.session_archived,
            'the session to be archived')

    command(harness, '/respawn', lambda: not app.session_archived,
            'the session to be restored')
    assert any('is back' in text for text in notices(harness))

    # Messages are accepted again.
    run_turn(harness, 'after respawn')
    assert any(kind == 'assistant' and 'after respawn' in text
               for kind, text in harness.transcript_texts())


def test_perish_deletes_the_session_for_good(archived_run):
    harness = archived_run
    app = harness.app
    name = app.loop.session_name
    command(harness, '/archive', lambda: app.session_archived,
            'the session to be archived')

    command(harness, '/perish', lambda: app.overlay is not None,
            'the confirmation dialog')
    assert type(app.overlay).__name__ == 'ConfirmDialog'
    harness.press('enter')                 # confirm
    harness.wait_until(
        lambda: (app.session_store.load(name) is None
                 and app.session_archived is False
                 and app.loop.session_name == ''),
        timeout=15.0, description='the session to be deleted')

    assert app.session_store.is_archived(name) is False
    assert app.session_archived is False
    assert app.loop.session_name == ''
    # The conversation is gone; only the notice about the deletion is left.
    assert [item.kind for item in app.items
            if item.kind in ('user', 'assistant', 'tool')] == []
    # The next message starts a new, live session.  (Its name may be the same
    # timestamp again, which is fine: what matters is that it is not archived
    # and holds only the new conversation.)
    run_turn(harness, 'fresh start')
    assert app.session_archived is False
    assert app.session_store.is_archived(app.loop.session_name) is False
    assert [row['name'] for row in app.session_store.list()] == [
        app.loop.session_name]


# ── the sessions browser ───────────────────────────────────
def test_delete_in_the_browser_archives_the_highlighted_session(archived_run):
    harness = archived_run
    app = harness.app
    name = app.loop.session_name
    app.session_store.save(payload('zzz_other'))

    app.open_sessions_browser()
    dialog = app.overlay
    assert dialog is not None
    labels = [item.value for item in dialog.items]
    assert name in labels
    dialog.index = labels.index(name)

    app._handle_event(key('delete'))
    assert app.session_store.is_archived(name)
    assert app.session_archived is True
    assert name not in [item.value for item in app.overlay.items]


def test_alt_in_the_browser_reveals_the_archived_rows(archived_run):
    harness = archived_run
    app = harness.app
    name = app.loop.session_name
    app.session_store.archive(name)
    app._sync_archive_state()

    app.open_sessions_browser()
    dialog = app.overlay
    assert dialog is not None
    assert name not in [item.value for item in dialog.items]

    app._handle_event(modifier('alt', 'press'))
    items = {item.value: item for item in app.overlay.items}
    assert name in items
    assert items[name].dim is True
    assert 'archived' in items[name].description

    app._handle_event(modifier('alt', 'press'))
    assert name not in [item.value for item in app.overlay.items]


def test_the_terminals_own_alt_key_reveals_the_archived_rows(archived_run):
    """The whole path: kitty Alt bytes -> parser -> modifier event -> dialog."""
    harness = archived_run
    app = harness.app
    name = app.loop.session_name
    app.session_store.archive(name)
    app._sync_archive_state()

    harness.press('ctrl+h')
    harness.wait_until(lambda: app.overlay is not None, timeout=10.0,
                       description='the sessions browser')
    assert name not in [item.value for item in app.overlay.items]

    # Flag 8 + flag 2: LEFT_ALT (57443) pressed.  This is what a terminal that
    # speaks the kitty keyboard protocol sends when the key goes down.
    harness.input.feed('\x1b[57443;3:1u')
    harness.wait_until(
        lambda: name in [item.value for item in app.overlay.items],
        timeout=10.0, description='the archived row to appear')
    # Releasing it does not hide them again: the press is the gesture.
    harness.input.feed('\x1b[57443;1:3u')
    time.sleep(0.2)
    assert name in [item.value for item in app.overlay.items]
