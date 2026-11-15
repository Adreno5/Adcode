# adcode/session.py
# Session persistence, re-exported from the engine sdk.
#
# Two routes share one payload format (version 2) and one set of helpers:
# DbSessionStore persists the conversation thread in the sqlite database
# (the default the app wires), SessionStore keeps the json-file route for
# tooling.  This shim keeps the historical import path stable.

from harness.transcript import (
    DbSessionStore,
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
    'DbSessionStore', 'LAST_SESSION', 'RESERVED_SESSION_NAMES',
    'SESSION_VERSION', 'SessionPayload', 'SessionStore', 'TranscriptItem',
    'group_turns', 'new_session_name',
]
