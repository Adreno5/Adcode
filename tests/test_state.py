"""The state lives in the database -- and never in the project.

An agent working in a repository must not leave its session state there.
The rules the app follows:

* the state directory is never inside the project (even when the
  environment points it there);
* the structured state -- the todo plan, the goals, the wire cache, the
  prediction state, the configuration, the audit trail -- is in one
  database;
* the files an older version dropped into the project are taken over and
  removed.
"""

from __future__ import annotations

import json

from adcode import paths
from adcode.state import (KEY_GOALS, KEY_WIRES, documents,
                          import_legacy_document, state_store)


def test_the_state_leaves_the_project(tmp_path, monkeypatch):
    """A state directory inside the project is not used."""
    project = tmp_path / 'project'
    project.mkdir()
    inside = project / '.adcode'
    inside.mkdir()
    monkeypatch.setenv('ADCODE_HOME', str(inside))
    paths.set_app_dir(None)
    paths.set_project_root(str(project))

    assert paths.state_moved() is True
    assert paths.state_dir() != inside
    assert paths.db_path().is_relative_to(paths.state_dir())


def test_a_repository_is_never_a_state_directory(tmp_path, monkeypatch):
    """A checkout is for code; session state does not belong in one.

    ``ADCODE_HOME=$HOME/myproject`` is an easy accident to write, and it
    puts the database and the log into the repository the agent is
    working on.
    """
    checkout = tmp_path / 'checkout'
    (checkout / '.git').mkdir(parents=True)
    monkeypatch.setenv('ADCODE_HOME', str(checkout))
    paths.set_app_dir(None)
    paths.set_project_root('')

    assert paths.state_dir() != checkout
    assert paths.state_moved() is True
    assert paths.db_path().is_relative_to(paths.state_dir())


def test_the_state_stays_where_it_was_told(tmp_path, monkeypatch):
    """A state directory outside the project is used as it is."""
    outside = tmp_path / 'home'
    outside.mkdir()
    monkeypatch.setenv('ADCODE_HOME', str(outside))
    paths.set_app_dir(None)
    paths.set_project_root(str(tmp_path / 'project'))

    assert paths.state_moved() is False
    assert paths.state_dir() == outside


def test_the_documents_are_in_the_database():
    store = state_store()
    store.save_document('a.document', {'value': 1})
    assert store.load_document('a.document') == {'value': 1}
    assert 'a.document' in store.document_keys()
    assert store.delete_document('a.document') is True
    assert store.load_document('a.document') is None


def test_the_audit_trail_is_in_the_database():
    store = state_store()
    before = store.audit_count()
    store.append_audit({'kind': 'audit.tool_call', 'tool': 'read', 'ts': 1.0})
    events = store.audit_tail(10)
    assert events, 'the trail must hold the event'
    assert events[-1]['tool'] == 'read'
    assert store.audit_count() == before + 1


def test_a_legacy_file_is_taken_over(tmp_path):
    """A json file an older version wrote is read once and removed.

    The file was in the project -- that was the mistake -- so that is
    where the take-over looks for it.
    """
    paths.set_project_root(str(tmp_path))
    legacy = tmp_path / 'wires.json'
    legacy.write_text(json.dumps({'scope': {'broken': 'working'}}),
                      encoding='utf-8')
    import_legacy_document(KEY_WIRES, 'wires.json')
    assert not legacy.exists()
    assert documents().load(KEY_WIRES) is not None


def test_the_cli_opens_the_log_in_the_state_directory(tmp_path, monkeypatch):
    """The log file is opened after the project is known.

    Otherwise the file lands in the configured state directory before the
    app had any idea which directory it is working in -- which is how an
    empty log file ends up in the repository.
    """
    import adcode.cli as cli
    from adcode.logging_setup import configure_logging as real

    seen: dict = {}

    def spy(*args, **kwargs):
        seen['target'] = real(*args, **kwargs)
        return seen['target']

    monkeypatch.setattr('adcode.logging_setup.configure_logging', spy)
    monkeypatch.chdir(tmp_path)
    try:
        cli.main(['--help'])
    except SystemExit:
        pass
    target = str(seen.get('target', ''))
    assert target, 'the log file must be opened'
    assert str(paths.state_dir()) in target, target
    assert not (tmp_path / 'adcode.log').exists()


def test_the_state_directory_holds_no_json_state(tmp_path):
    """Only the things that are really files stay files."""
    paths.set_project_root(str(tmp_path))
    store = state_store()
    store.save_document(KEY_GOALS, {'goals': []})
    assert documents().load(KEY_GOALS) == {'goals': []}
    leftovers = [path.name for path in paths.state_dir().glob('*.json')]
    assert leftovers == [], leftovers
