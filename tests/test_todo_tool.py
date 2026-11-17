"""The todo tool: the harness plan store, exposed to the model."""

from __future__ import annotations

import asyncio
import json

import pytest

from adcode.tools.registry import ManagedToolRegistry
from adcode.tools.runtime import ToolContext, pop_context, push_context
from adcode.tools.todo import (TODO_FILE, install_todo_tool, reset_todo_stores,
                               todo_store)


@pytest.fixture(autouse=True)
def _fresh_stores():
    reset_todo_stores()
    yield
    reset_todo_stores()


def _registry() -> ManagedToolRegistry:
    registry = ManagedToolRegistry()
    install_todo_tool(registry)
    return registry


def _call(registry, args, cwd):
    """Run the tool the way the engine does: worker thread, tool context set."""
    async def call():
        token = push_context(ToolContext(cwd=str(cwd)))
        try:
            return await asyncio.to_thread(registry.execute, 'todo', args)
        finally:
            pop_context(token)
    return asyncio.run(call())


def _json(text):
    return json.loads(text)


def test_the_tool_is_registered_with_a_model_facing_schema():
    registry = _registry()
    assert registry.get('todo') is not None
    assert registry.source_of('todo') == 'builtin'
    schema = {entry['name']: entry for entry in registry.build_schema()}
    assert schema['todo']['parameters']['required'] == ['action']
    actions = schema['todo']['parameters']['properties']['action']['enum']
    assert {'create', 'update', 'list', 'complete'} <= set(actions)


def test_create_then_update_tracks_progress(tmp_path):
    registry = _registry()
    text, _, error = _call(registry, {
        'action': 'create',
        'items': [{'title': 'read the code'}, {'title': 'write the fix'}],
    }, tmp_path)
    assert not error, text
    view = _json(text)
    assert [item['title'] for item in view['items']] == ['read the code',
                                                        'write the fix']
    assert view['open'] == 2 and view['status'] == 'executing'
    assert (tmp_path / TODO_FILE).exists()

    text, _, error = _call(registry, {
        'action': 'update', 'id': 'item-00', 'status': 'completed'}, tmp_path)
    assert not error, text
    view = _json(text)
    assert view['progress'] == 0.5 and view['done'] == 1
    assert view['current'] == 'write the fix'

    text, _, error = _call(registry, {'action': 'list'}, tmp_path)
    assert not error, text
    assert _json(text)['items'][0]['status'] == 'completed'


def test_items_may_arrive_as_a_json_string(tmp_path):
    """Models sometimes hand over the list as text; accept it."""
    registry = _registry()
    text, _, error = _call(registry, {
        'action': 'create',
        'items': json.dumps([{'title': 'one'}, {'title': 'two'}]),
    }, tmp_path)
    assert not error, text
    assert _json(text)['open'] == 2


def test_actions_without_a_plan_explain_themselves(tmp_path):
    registry = _registry()
    text, _, error = _call(registry, {'action': 'update', 'id': 'item-00',
                                      'status': 'completed'}, tmp_path)
    assert error is True
    assert text.startswith('Error:') and 'create' in text


def test_create_without_items_is_refused(tmp_path):
    registry = _registry()
    text, _, error = _call(registry, {'action': 'create', 'items': []}, tmp_path)
    assert error is True and 'items' in text


def test_a_bad_item_id_lists_the_real_ones(tmp_path):
    registry = _registry()
    _call(registry, {'action': 'create', 'items': [{'title': 'a'}]}, tmp_path)
    text, _, error = _call(registry, {'action': 'update', 'id': 'item-99',
                                      'status': 'completed'}, tmp_path)
    assert error is True and 'item-00' in text
    assert todo_store(str(tmp_path)).active_plan().items[0].status == 'pending'


def test_an_unknown_status_is_refused(tmp_path):
    registry = _registry()
    _call(registry, {'action': 'create', 'items': [{'title': 'a'}]}, tmp_path)
    text, _, error = _call(registry, {'action': 'update', 'id': 'item-00',
                                      'status': 'banana'}, tmp_path)
    assert error is True and 'banana' in text


def test_add_remove_and_complete(tmp_path):
    registry = _registry()
    _call(registry, {'action': 'create', 'items': [{'title': 'a'}]}, tmp_path)
    text, _, error = _call(registry, {'action': 'add', 'title': 'b'}, tmp_path)
    assert not error, text
    assert [item['id'] for item in _json(text)['items']] == ['item-00', 'item-01']

    text, _, error = _call(registry, {'action': 'remove', 'id': 'item-00'}, tmp_path)
    assert not error, text
    assert [item['id'] for item in _json(text)['items']] == ['item-01']

    text, _, error = _call(registry, {'action': 'complete', 'summary': 'done'}, tmp_path)
    assert not error, text
    assert _json(text)['status'] == 'completed'
    # nothing left to show in the panel
    assert todo_store(str(tmp_path)).active_plan() is None


def test_the_list_survives_a_restart(tmp_path):
    registry = _registry()
    _call(registry, {'action': 'create', 'items': [{'title': 'a'}, {'title': 'b'}]},
          tmp_path)
    reset_todo_stores()                      # a fresh process would have no cache
    plan = todo_store(str(tmp_path)).active_plan()
    assert plan is not None
    assert [item.title for item in plan.items] == ['a', 'b']


def test_the_old_flat_file_becomes_the_active_plan(tmp_path):
    """Users upgrading from the previous tool keep their list."""
    (tmp_path / TODO_FILE).write_text(json.dumps([
        {'text': 'old one', 'status': 'pending'},
        {'text': 'old two', 'status': 'completed'},
    ]), encoding='utf-8')
    plan = todo_store(str(tmp_path)).active_plan()
    assert plan is not None
    assert [item.title for item in plan.items] == ['old one', 'old two']
    assert plan.items[1].status == 'completed'


def test_each_project_keeps_its_own_list(tmp_path):
    one = tmp_path / 'one'
    two = tmp_path / 'two'
    one.mkdir()
    two.mkdir()
    registry = _registry()
    _call(registry, {'action': 'create', 'items': [{'title': 'only here'}]}, one)
    assert todo_store(str(one)).active_plan() is not None
    assert todo_store(str(two)).active_plan() is None
    assert todo_store(str(one)) is todo_store(str(one)), 'one store per project'
