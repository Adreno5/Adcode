"""The todo tool: the plan store from the sdk, adapted to adcode.

``harness.todo`` owns the model now -- plans, items, progress and change
events.  This module only adapts it to the shape adcode's tools use
(``handler(**args) -> str``, ``Error: ...`` text on failure) and hands the UI
the same store instance, so a panel and the model never disagree.

The plan lives in the **state database**, one document per project, so it
survives a restart and every agent in the session (including sub-agents,
which share the tool registry) works the same list -- without a
``.harness_todo.json`` appearing in the project the agent is working on.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

from harness.todo import TodoPlan, TodoStore

from adcode.state import KEY_TODO, state_store
from adcode.tools.registry import AnnotatedTool
from adcode.tools.runtime import current_context

TODO_FILE = '.harness_todo.json'

_STORES: dict[str, TodoStore] = {}
_STORES_LOCK = threading.Lock()


def _key(cwd: str = '') -> str:
    base = cwd or current_context().cwd or '.'
    return str(Path(base).resolve())


def _document_key(root: str) -> str:
    """The document key for a project: one plan per project."""
    return f'{KEY_TODO}:{root}'


def _import_legacy(root: str) -> None:
    """Take over a plan an older version left in the project.

    The file used to be ``.harness_todo.json`` in the project root.  It is
    read once, put into the database, and removed -- a repository is for
    code, and that file was never code.

    The old file may hold either shape (a document, or a bare list of
    items), so the store itself does the reading: it knows how a flat list
    becomes a plan.
    """
    legacy = Path(root) / TODO_FILE
    if not legacy.exists():
        return
    try:
        document = TodoStore(legacy).to_dict()
    except Exception:      # noqa: BLE001 - an unreadable file stays put
        return
    try:
        documents = state_store().documents
        if documents.load(_document_key(root)) is None:
            documents.save(_document_key(root), document)
        legacy.unlink()
    except Exception:      # noqa: BLE001
        pass


def todo_store(cwd: str = '') -> TodoStore:
    """The store for a project; one instance per directory, shared by all."""
    key = _key(cwd)
    with _STORES_LOCK:
        store = _STORES.get(key)
        if store is None:
            # An older version left the plan in the project root; take it
            # over before the store reads the database.
            _import_legacy(key)
            store = TodoStore(backend=state_store().documents,
                              key=_document_key(key))
            _STORES[key] = store
        return store


def reset_todo_stores() -> None:
    """Forget the cached stores (tests and session switches)."""
    with _STORES_LOCK:
        _STORES.clear()


def active_plan(cwd: str = '') -> Optional[TodoPlan]:
    """The plan the UI should show: the newest one still being worked on."""
    try:
        return todo_store(cwd).active_plan()
    except Exception:      # noqa: BLE001 - a UI must not crash on a bad file
        return None


def todo_revision(cwd: str = '') -> int:
    try:
        return todo_store(cwd).revision
    except Exception:      # noqa: BLE001
        return -1


def _plan_or_active(store: TodoStore, plan_id: str) -> Optional[TodoPlan]:
    if plan_id:
        return store.get_plan(plan_id)
    return store.active_plan()


def _parse_items(raw: Any) -> list[dict[str, Any]]:
    """Items may arrive as a list, or as a JSON string the model wrote."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    if not isinstance(raw, list):
        return []
    items: list[dict[str, Any]] = []
    for entry in raw:
        if isinstance(entry, str):
            items.append({'title': entry})
        elif isinstance(entry, dict):
            items.append(dict(entry))
    return items


def _view(plan: TodoPlan) -> dict[str, Any]:
    """The compact shape the model (and the UI) reads."""
    current = plan.current
    return {
        'plan_id': plan.plan_id,
        'status': plan.status,
        'progress': round(plan.progress, 3),
        'done': plan.done_count,
        'open': plan.open_count,
        'current': current.title if current is not None else '',
        'items': [{'id': item.id, 'title': item.title, 'status': item.status}
                  for item in plan.items],
    }


def _ids(plan: TodoPlan) -> str:
    return ', '.join(item.id for item in plan.items) or '(none)'


def _todo(**args: Any) -> str:
    action = str(args.get('action') or 'list').strip().lower()
    store = todo_store()
    plan_id = str(args.get('plan_id') or '').strip()

    if action == 'create':
        items = _parse_items(args.get('items'))
        if not items:
            return ('Error: create needs items, e.g. '
                    'items=[{"title": "read the code"}, {"title": "fix it"}]')
        plan = store.create_plan(items)
        return json.dumps(_view(plan), ensure_ascii=False)

    plan = _plan_or_active(store, plan_id)
    if plan is None:
        return ('Error: there is no todo plan yet; call todo with '
                'action="create" and an items list first')

    if action == 'list':
        return json.dumps(_view(plan), ensure_ascii=False)

    if action == 'add':
        title = str(args.get('title') or args.get('text') or '').strip()
        if not title:
            return 'Error: add needs a title'
        item = store.add_item(plan.plan_id, title,
                              description=str(args.get('description') or ''),
                              status=_status(args) or 'pending')
        if item is None:
            return f'Error: no such plan: {plan.plan_id}'
        return json.dumps(_view(store.get_plan(plan.plan_id)), ensure_ascii=False)

    if action == 'update':
        item_id = str(args.get('id') or '').strip()
        if not item_id:
            return f'Error: update needs an item id; ids: {_ids(plan)}'
        fields: dict[str, Any] = {}
        status = _status(args)
        if status:
            fields['status'] = status
        for key in ('title', 'description', 'result', 'error'):
            value = args.get(key)
            if value is not None:
                fields[key] = str(value)
        if not fields:
            return 'Error: update needs at least one of status/title/description/result/error'
        try:
            item = store.update_item(plan.plan_id, item_id, **fields)
        except ValueError as error:
            return f'Error: {error}'
        if item is None:
            return f'Error: no item {item_id}; ids: {_ids(plan)}'
        return json.dumps(_view(store.get_plan(plan.plan_id)), ensure_ascii=False)

    if action == 'remove':
        item_id = str(args.get('id') or '').strip()
        if not item_id:
            return f'Error: remove needs an item id; ids: {_ids(plan)}'
        if not store.remove_item(plan.plan_id, item_id):
            return f'Error: no item {item_id}; ids: {_ids(plan)}'
        return json.dumps(_view(store.get_plan(plan.plan_id)), ensure_ascii=False)

    if action == 'complete':
        store.set_completed(plan.plan_id, str(args.get('summary') or ''))
        return json.dumps(_view(store.get_plan(plan.plan_id)), ensure_ascii=False)

    if action == 'fail':
        store.set_failed(plan.plan_id, str(args.get('error') or args.get('summary') or ''))
        return json.dumps(_view(store.get_plan(plan.plan_id)), ensure_ascii=False)

    return ('Error: unknown action '
            f'{action!r} (create/add/update/remove/list/complete/fail)')


def _status(args: dict[str, Any]) -> str:
    return str(args.get('status') or '').strip()


def todo_tool() -> AnnotatedTool:
    """The ``todo`` tool: a visible checklist for multi-step work."""
    return AnnotatedTool(
        name='todo',
        description=(
            'Plan and track multi-step work in a todo list the user can see. '
            'create a plan with items, then update each item as you work: '
            'pending -> in_progress -> completed (or failed/skipped). '
            'Item ids are item-00, item-01, ... and stay stable. '
            'Use it for any task with more than two or three steps, and keep '
            'exactly one item in_progress at a time.'),
        parameters={
            'type': 'object',
            'properties': {
                'action': {'type': 'string',
                           'enum': ['create', 'add', 'update', 'remove',
                                    'list', 'complete', 'fail']},
                'items': {
                    'type': 'array',
                    'description': 'create: the steps, in order',
                    'items': {'type': 'object', 'properties': {
                        'title': {'type': 'string'},
                        'description': {'type': 'string'},
                        'priority': {'type': 'integer'},
                    }, 'required': ['title']},
                },
                'id': {'type': 'string',
                       'description': 'update/remove: the item id (item-00, ...)'},
                'title': {'type': 'string', 'description': 'add: the step'},
                'description': {'type': 'string'},
                'status': {
                    'type': 'string',
                    'enum': ['pending', 'in_progress', 'completed', 'failed',
                             'skipped'],
                },
                'result': {'type': 'string',
                           'description': 'what that step produced'},
                'error': {'type': 'string'},
                'summary': {'type': 'string', 'description': 'complete: the wrap-up'},
                'plan_id': {'type': 'string',
                            'description': 'defaults to the active plan'},
            },
            'required': ['action'],
        },
        handler=_todo,
    )


def install_todo_tool(registry) -> list[str]:
    """Register the todo tool on a ManagedToolRegistry; return its names."""
    tool = todo_tool()
    registry.register(tool, replace=True, source='builtin')
    return [tool.name]
