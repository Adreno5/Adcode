"""Renaming a session: the command, the app, and the saved rows."""

from __future__ import annotations

import pytest

from adcode.commands import COMMANDS, dispatch, find
from adcode.session import ArchivedSessionStore, SessionPayload


class FakeLoop:
    """The little the rename command needs from the agent loop."""

    def __init__(self, name: str = 'old-name') -> None:
        self.session_name = name
        self.renamed: list[str] = []

    def rename_session(self, name: str) -> str:
        self.renamed.append(name)
        self.session_name = name
        return name


class FakeApp:
    def __init__(self, loop=None) -> None:
        self.config = type('Config', (), {})()
        self.loop = loop
        self.notices: list[tuple[str, str]] = []
        self.prompts: list[tuple] = []
        self.renamed: list[str] = []

    def notify(self, text, level='info'):
        self.notices.append((level, text))

    def open_text_prompt(self, title, label, initial, submit, **kwargs):
        self.prompts.append((title, label, initial, submit))

    def open_message(self, title, message):
        pass

    def open_select(self, *args, **kwargs):
        pass

    def config_changed(self):
        pass

    def rename_session(self, name):
        self.renamed.append(name)
        return True


def test_the_rename_command_is_registered():
    command = find('/rename')
    assert command is not None
    assert command.name in {item.name for item in COMMANDS}


def test_rename_with_a_name_renames_at_once():
    app = FakeApp(FakeLoop())
    dispatch(app, '/rename demo')
    assert app.renamed == ['demo']


def test_rename_without_a_name_asks_for_one():
    app = FakeApp(FakeLoop('current'))
    dispatch(app, '/rename')
    assert app.renamed == []
    assert app.prompts
    title, label, initial, submit = app.prompts[0]
    # the field starts empty: the new name is typed, not edited
    assert initial == ''
    # the answer of the prompt renames the session
    submit('from-prompt')
    assert app.renamed == ['from-prompt']


def test_the_settings_command_opens_the_same_browser():
    app = FakeApp()
    opened: list[str] = []
    app.open_settings = lambda: opened.append('settings')
    dispatch(app, '/settings')
    assert opened == ['settings']


def test_a_renamed_session_is_moved_not_copied(tmp_path):
    store = ArchivedSessionStore(db_path=str(tmp_path / 'agent.db'))
    try:
        store.save(SessionPayload(name='old-name', title='old', messages=[
            {'role': 'user', 'content': 'hello'},
        ]))
        assert {row['name'] for row in store.list()} == {'old-name'}
        store.delete('old-name')
        assert store.list() == []
    finally:
        store.store.close()


def test_the_archived_session_refuses_a_rename():
    from adcode.commands import ARCHIVED_REFUSAL
    app = FakeApp(FakeLoop())
    app.session_archived = True
    dispatch(app, '/rename demo')
    assert app.renamed == []
    assert any(ARCHIVED_REFUSAL in text for _, text in app.notices)
