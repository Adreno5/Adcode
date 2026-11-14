import pytest

from adcode.llm.context import (
    ConversationContext, StablePrefix, stream_compact, structured_summary,
)
from adcode.llm.usage import TokenBudget, TokenBudgetExceeded, estimate_tokens


def make_context(messages=None):
    prefix = StablePrefix.create(
        provider='P', endpoint='http://x', model='m', system='sys', tools=[],
    )
    return ConversationContext(prefix, list(messages or []))


def test_stage_commit_rollback():
    context = make_context()
    context.stage({'role': 'user', 'content': 'hi'})
    assert context.draft
    context.rollback()
    assert context.committed == []
    context.stage({'role': 'user', 'content': 'hi'})
    context.commit()
    assert context.committed[0]['content'] == 'hi'
    assert context.draft == []


def test_prefix_fingerprint_changes_with_tools():
    a = StablePrefix.create(provider='P', endpoint='http://x', model='m',
                            system='s', tools=[])
    b = StablePrefix.create(provider='P', endpoint='http://x', model='m',
                            system='s', tools=[{'name': 'read'}])
    assert a.fingerprint != b.fingerprint


def test_compaction_plan_and_apply():
    messages = []
    for index in range(6):
        messages.append({'role': 'user', 'content': f'u{index} ' + 'x' * 400})
        messages.append({'role': 'assistant', 'content': f'a{index} ' + 'y' * 400})
    context = make_context(messages)
    plan = context.compaction_plan(target_tokens=400, keep_recent_turns=1)
    assert plan is not None
    assert plan.removed
    assert plan.kept_messages
    summary = {'role': 'user', 'content': '<conversation_summary>sum</conversation_summary>'}
    removed = context.apply_compaction(plan, summary)
    assert removed == len(plan.removed)
    assert context.committed[0]['content'].startswith('<conversation_summary>')
    assert context.stable_prefix.generation == 2
    assert len(context.archived) == removed


def test_stream_compact_uses_stream_and_falls_back():
    messages = [{'role': 'user', 'content': f'u{i} ' + 'x' * 400} for i in range(8)]
    context = make_context(messages)
    removed = stream_compact(
        context, 300,
        lambda messages, **kwargs: iter([
            {'kind': 'text', 'content': 'STREAMED SUMMARY'},
            {'kind': 'finish', 'reason': 'stop'},
        ]),
        keep_recent_turns=1,
    )
    assert removed > 0
    assert 'STREAMED SUMMARY' in str(context.committed)

    messages = [{'role': 'user', 'content': f'u{i} ' + 'x' * 400} for i in range(8)]
    context = make_context(messages)

    def failing(messages, **kwargs):
        raise RuntimeError('no network')

    removed = stream_compact(context, 300, failing, keep_recent_turns=1)
    assert removed > 0
    assert '<conversation_summary>' in context.committed[0]['content']


def test_structured_summary_bounds():
    summary = structured_summary(
        [{'role': 'user', 'content': 'Fix the bug ' + 'z' * 2000}], max_tokens=100)
    assert summary['role'] == 'user'
    assert '<conversation_summary>' in summary['content']
    assert len(summary['content']) < 2000


def test_enforce_budget_trims_and_raises():
    context = make_context([
        {'role': 'user', 'content': 'x' * 100},
        {'role': 'tool', 'content': 'y' * 50_000},
    ])
    budget = TokenBudget(context_window=4_000, reserved_output=500,
                         safety_margin=100, stable_tokens=100)
    result = context.enforce_budget(budget)
    assert result['committed_tokens'] <= budget.usable()
    tiny = make_context([{'role': 'user', 'content': 'x' * 100_000}])
    with pytest.raises(TokenBudgetExceeded):
        tiny.enforce_budget(TokenBudget(context_window=1_000, reserved_output=100,
                                        safety_margin=100))


def test_needs_compaction_threshold():
    context = make_context([{'role': 'user', 'content': 'x' * 4000}])
    budget = TokenBudget(context_window=2_000, reserved_output=100,
                         safety_margin=100, stable_tokens=0)
    assert context.needs_compaction(budget, 50)
    assert not context.needs_compaction(budget, 101)


def test_estimate_tokens_scales():
    assert estimate_tokens('abcd') >= 1
    assert estimate_tokens('x' * 400) > estimate_tokens('x' * 40)
