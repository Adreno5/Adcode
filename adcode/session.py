# adcode/session.py
# Session persistence, re-exported from the engine sdk.
#
# The sdk's transcript module carries the same payload format (version 2) and
# the same store semantics: atomic JSON writes, the __last__ checkpoint,
# rewind and fork helpers.  This shim keeps the historical import path stable.

from harness.transcript import (
    LAST_SESSION,
    RESERVED_SESSION_NAMES,
    SESSION_VERSION,
    SessionPayload,
    SessionStore,
    TranscriptItem,
    group_turns,
    new_session_name,
)

__all__ = [
    'LAST_SESSION', 'RESERVED_SESSION_NAMES', 'SESSION_VERSION',
    'SessionPayload', 'SessionStore', 'TranscriptItem', 'group_turns',
    'new_session_name',
]
