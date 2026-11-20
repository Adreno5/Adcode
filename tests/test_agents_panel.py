"""The sub-agent strip: which rows it shows and how it words them."""

from __future__ import annotations

from adcode.tui.agents_panel import (ACTIVE, MAX_ROWS, format_elapsed,
                                     panel_lines)
from adcode.tui.text import plain_text


def _rows():
    return [
        {'name': 'scout', 'status': 'running', 'elapsed': 12,
         'step': 'grep todo_store', 'tail': 'the panel renders the view model'},
        {'name': 'worker', 'status': 'completed', 'elapsed': 95,
         'step': 'gone', 'tail': 'gone too'},
        {'name': 'root', 'status': 'running', 'elapsed': 200,
         'step': 'spawn_agent', 'tail': 'should never show'},
    ]


def test_no_agents_means_no_panel():
    assert panel_lines([], 80) == []
    assert panel_lines([{'name': 'root', 'status': 'running'}], 80) == []


def test_the_header_counts_running_and_done():
    header = plain_text(panel_lines(_rows(), 80)[0])
    assert header == '│ agents  1 running · 1 done', header


def test_a_running_row_carries_step_and_tail():
    rows = [plain_text(line) for line in panel_lines(_rows(), 90)]
    scout = next(row for row in rows if 'scout' in row)
    assert '▸ scout' in scout
    assert 'running' in scout
    assert '12s' in scout
    assert 'grep todo_store' in scout
    assert 'the panel renders the view model' in scout


def test_a_finished_row_keeps_its_state_and_drops_the_details():
    rows = [plain_text(line) for line in panel_lines(_rows(), 90)]
    worker = next(row for row in rows if 'worker' in row)
    assert '✓ worker done 1m35s' in worker, worker
    assert 'gone' not in worker, worker


def test_the_root_agent_is_never_listed():
    text = '\n'.join(plain_text(line) for line in panel_lines(_rows(), 90))
    assert 'should never show' not in text


def test_running_agents_come_first():
    rows = [
        {'name': 'late', 'status': 'completed', 'elapsed': 5, 'started': 10.0},
        {'name': 'early', 'status': 'completed', 'elapsed': 5, 'started': 1.0},
        {'name': 'live', 'status': 'running', 'elapsed': 1, 'started': 20.0},
    ]
    lines = [plain_text(line) for line in panel_lines(rows, 80)]
    assert 'live' in lines[1]
    # newest finished next to the live one
    assert 'late' in lines[2] and 'early' in lines[3]


def test_the_list_is_capped_and_says_how_many_are_hidden():
    rows = [{'name': f'a{index}', 'status': 'running', 'elapsed': 1,
             'started': float(index)} for index in range(12)]
    lines = panel_lines(rows, 80, max_rows=4)
    assert len(lines) == 4, 'header + three rows'
    assert 'more' in plain_text(lines[-1]), plain_text(lines[-1])


def test_rows_fit_the_given_width():
    rows = [{'name': 'scout' * 8, 'status': 'running', 'elapsed': 12,
             'step': 'x' * 200, 'tail': 'y' * 200}]
    for line in panel_lines(rows, 40):
        assert len(plain_text(line)) <= 40


def test_states_are_worded_and_marked():
    rows = [
        {'name': 'a', 'status': 'errored', 'elapsed': 3},
        {'name': 'b', 'status': 'interrupted', 'elapsed': 4},
        {'name': 'c', 'status': 'pending_init', 'elapsed': 1},
    ]
    text = '\n'.join(plain_text(line) for line in panel_lines(rows, 80))
    assert '✗ a failed 3s' in text
    assert '· b stopped 4s' in text
    assert '○ c starting 1s' in text


def test_elapsed_formatting():
    assert format_elapsed(0) == '0s'
    assert format_elapsed(12.9) == '12s'
    assert format_elapsed(95) == '1m35s'
    assert format_elapsed(3600 + 120) == '1h02m'
    assert format_elapsed(None) == ''


def test_active_is_the_set_the_app_watches():
    assert ACTIVE == ('running', 'pending_init')
    assert MAX_ROWS >= 4


def test_the_spawn_receipt_collapses_to_one_line():
    """The panel already shows the agent; the JSON receipt is noise."""
    import json

    from adcode.tui.transcript import tool_result_brief

    receipt = json.dumps({'agent': 'scout', 'status': 'running',
                          'parent': 'root', 'depth': 1})
    assert tool_result_brief('spawn_agent', receipt) == 'scout  ·  running  ·  depth 1'
    assert tool_result_brief('spawn_agent', 'Error: too many agents') == ''
    assert tool_result_brief('list_agents', '[{"name": "scout"}]') == ''
