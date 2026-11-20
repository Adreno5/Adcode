"""The todo panel above the prompt, driven by real tool calls.

The model asks for the ``todo`` tool through the fake gateway; the production
registry runs it, the production app reads the same store and paints the strip.
"""

from __future__ import annotations

from bench.script import ScriptedResponder, ToolCall, Turn


def _prompt_row(frame) -> int:
    """The input bar: the last row carrying the prompt marker.

    Transcript echoes of user messages have a gutter (``" > text"``), so the
    marker with no leading space is the input bar itself.
    """
    for y in range(len(frame.lines) - 1, -1, -1):
        if frame.lines[y].startswith('>'):
            return y
    return -1


def _panel_rows(frame) -> list[int]:
    """The todo block: the contiguous styled rows directly above the prompt."""
    row = _prompt_row(frame) - 1
    rows: list[int] = []
    while row >= 0 and frame.line(row).startswith('│'):
        rows.append(row)
        row -= 1
    return sorted(rows)


def _create(items):
    return Turn(tool_calls=[ToolCall(name='todo', args={
        'action': 'create', 'items': items}, id='call_todo_create')])


def _update(item_id, status):
    return Turn(tool_calls=[ToolCall(name='todo', args={
        'action': 'update', 'id': item_id, 'status': status},
        id='call_todo_update')])


def test_the_panel_appears_above_the_prompt_and_follows_updates(harness):
    harness.gateway.responder = ScriptedResponder([
        _create([{'title': 'read the collab code'}, {'title': 'write the fix'}]),
        _update('item-00', 'completed'),
        Turn(text='Plan created.'),
    ])

    harness.submit('plan the work', expect_requests=3, timeout=30.0)
    harness.wait_idle(20.0)
    frame = harness.wait_for_text('write the fix')

    # the plan lives in the state database -- the project stays clean
    assert not (harness.workspace.project / '.harness_todo.json').exists()
    from adcode.tools.todo import todo_store
    plan = todo_store(str(harness.workspace.project)).active_plan()
    assert plan is not None
    assert [item.title for item in plan.items] == ['read the collab code',
                                                   'write the fix']

    # the panel sits directly above the input bar, carrying the real state
    rows = _panel_rows(frame)
    panel = [frame.line(row) for row in rows]
    assert rows and rows[-1] == _prompt_row(frame) - 1, (rows, frame.text())
    assert panel[0].startswith('│ todo'), panel
    assert '1/2' in panel[0], panel
    assert any('✓ 1 read the collab code' in line for line in panel), panel
    assert any('○ 2 write the fix' in line for line in panel), panel
    # the transcript keeps one readable line instead of the plan JSON
    assert '"plan_id"' not in frame.text()
    assert '1/2  ·  executing' in frame.text(), frame.text()
    assert not frame.vt_mismatch, 'serialized ANSI must match the canvas'


def test_the_panel_disappears_once_the_plan_is_complete(harness):
    harness.gateway.responder = ScriptedResponder([
        _create([{'title': 'only step'}]),
        Turn(tool_calls=[ToolCall(name='todo', args={
            'action': 'update', 'id': 'item-00', 'status': 'completed'},
            id='call_todo_done')]),
        Turn(tool_calls=[ToolCall(name='todo', args={
            'action': 'complete', 'summary': 'done'}, id='call_todo_complete')]),
        Turn(text='All done.'),
    ])

    harness.submit('do it', expect_requests=4, timeout=30.0)
    harness.wait_idle(20.0)
    harness.wait_until(
        lambda: harness.screen.latest is not None
        and not _panel_rows(harness.screen.latest),
        timeout=15.0, description='the todo panel to go away')

    frame = harness.screen.latest
    assert frame is not None
    assert _panel_rows(frame) == [], frame.text()
    assert not frame.vt_mismatch


def test_the_panel_keeps_the_prompt_visible_on_a_small_screen(harness):
    """A long list may not push the transcript or the input off the screen."""
    items = [{'title': f'step {index}'} for index in range(12)]
    harness.gateway.responder = ScriptedResponder([
        _create(items),
        Turn(text='Many steps.'),
    ])

    harness.submit('plan a lot', expect_requests=2, timeout=30.0)
    harness.wait_idle(20.0)
    frame = harness.wait_for_text('step 0')

    prompt_row = _prompt_row(frame)
    assert prompt_row >= frame.height - 3, (prompt_row, frame.height)
    rows = _panel_rows(frame)
    assert 0 < len(rows) <= 8, rows
    # the capped list keeps the newest visible state readable, not just the head
    assert any('step' in frame.line(row) for row in rows), rows
    assert not frame.vt_mismatch
