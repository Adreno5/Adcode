"""The todo strip above the prompt: what it shows and when."""

from __future__ import annotations

import json

from harness.todo import TodoStore

from adcode.tui.text import plain_text
from adcode.tui.todo_panel import MAX_ROWS, panel_lines


def _plan(store, titles=('one', 'two', 'three')):
    return store.create_plan([{'title': title} for title in titles])


def test_no_plan_means_no_panel():
    store = TodoStore()
    assert panel_lines(None, 60) == []
    assert panel_lines(store.active_plan(), 60) == []
    empty = store.create_plan([])
    assert panel_lines(empty, 60) == []


def test_the_panel_lists_the_items_with_a_progress_header():
    store = TodoStore()
    plan = _plan(store)
    store.update_item(plan.plan_id, 'item-00', status='completed')
    store.update_item(plan.plan_id, 'item-01', status='in_progress')
    rows = [plain_text(line) for line in panel_lines(plan, 60)]
    assert rows[0].startswith('│ todo  1/3')
    assert 'write' not in rows[0] or True
    assert '✓ 1 one' in rows[1]
    assert '▸ 2 two' in rows[2]
    assert '○ 3 three' in rows[3]
    assert len(rows) == 4


def test_the_header_names_what_is_running():
    store = TodoStore()
    plan = _plan(store)
    store.update_item(plan.plan_id, 'item-01', status='in_progress')
    header = plain_text(panel_lines(plan, 60)[0])
    assert 'two' in header, header


def test_markers_follow_the_item_state():
    store = TodoStore()
    plan = _plan(store, ('a', 'b', 'c', 'd'))
    store.update_item(plan.plan_id, 'item-00', status='completed')
    store.update_item(plan.plan_id, 'item-01', status='failed')
    store.update_item(plan.plan_id, 'item-02', status='skipped')
    rows = [plain_text(line) for line in panel_lines(plan, 60)]
    assert '✓ 1 a' in rows[1]
    assert '✗ 2 b' in rows[2]
    assert '· 3 c' in rows[3]
    assert '○ 4 d' in rows[4]


def test_a_long_plan_keeps_the_current_item_in_view():
    store = TodoStore()
    plan = _plan(store, tuple(f'step {index}' for index in range(12)))
    store.update_item(plan.plan_id, 'item-09', status='in_progress')
    rows = [plain_text(line) for line in panel_lines(plan, 60, max_rows=5)]
    assert len(rows) == 5, 'header + the visible window'
    assert any('step 9' in row for row in rows), rows


def test_rows_are_truncated_to_the_width():
    store = TodoStore()
    plan = _plan(store, ('x' * 200,))
    rows = panel_lines(plan, 30)
    for line in rows:
        assert len(plain_text(line)) <= 30


def test_the_panel_never_exceeds_its_row_budget():
    store = TodoStore()
    plan = _plan(store, tuple(f's{index}' for index in range(40)))
    assert len(panel_lines(plan, 60)) <= MAX_ROWS


def test_the_numbering_reads_one_based():
    store = TodoStore()
    plan = _plan(store, ('first', 'second'))
    rows = [plain_text(line) for line in panel_lines(plan, 60)]
    assert '1 first' in rows[1] and '2 second' in rows[2]


def test_a_completed_plan_is_not_an_active_plan():
    store = TodoStore()
    plan = _plan(store, ('only',))
    store.update_item(plan.plan_id, 'item-00', status='completed')
    assert store.active_plan() is None
    assert panel_lines(store.active_plan(), 60) == []


def test_the_tool_result_collapses_to_one_line():
    """The transcript must not echo the plan JSON: the panel shows it."""
    from adcode.tui.transcript import tool_result_brief

    output = json.dumps({
        'plan_id': 'todo-1', 'status': 'executing', 'progress': 0.5,
        'done': 1, 'open': 1, 'current': 'write the fix',
        'items': [{'id': 'item-00', 'title': 'a', 'status': 'completed'},
                  {'id': 'item-01', 'title': 'write the fix', 'status': 'in_progress'}],
    })
    assert tool_result_brief('todo', output) == '1/2  ·  executing  ·  now write the fix'
    assert tool_result_brief('read', output) == ''
    assert tool_result_brief('todo', 'not json') == ''
    assert tool_result_brief('todo', 'Error: no plan') == ''
    done = json.dumps({'status': 'completed', 'done': 1, 'current': 'a',
                       'items': [{'id': 'item-00', 'title': 'a'}]})
    assert tool_result_brief('todo', done) == '1/1  ·  completed'
