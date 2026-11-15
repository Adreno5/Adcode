# adcode/llm/context.py
# Conversation state, re-exported from the engine sdk.
#
# ManagedConversation carries the same draft/commit/rollback semantics, the
# same stable-prefix fingerprint and the same compaction machinery the loop
# historically owned.  The alias keeps the historical import path stable.

from harness.conversation import (
    COMPACTION_SYSTEM,
    INTERRUPTED_TOOL_RESULT,
    PROMPT_VERSION,
    CompactionPlan,
    ManagedConversation,
    StablePrefix,
    canonical_json,
    messages_need_repair,
    normalized_endpoint,
    repair_messages,
    stable_tools,
    stream_compact,
    structured_summary,
)
from harness.usage import TokenBudget, TokenBudgetExceeded, estimate_tokens

ConversationContext = ManagedConversation

__all__ = [
    'COMPACTION_SYSTEM', 'INTERRUPTED_TOOL_RESULT', 'PROMPT_VERSION',
    'CompactionPlan', 'ConversationContext', 'ManagedConversation',
    'StablePrefix', 'TokenBudget', 'TokenBudgetExceeded', 'canonical_json',
    'estimate_tokens', 'messages_need_repair', 'normalized_endpoint',
    'repair_messages', 'stable_tools', 'stream_compact', 'structured_summary',
]
