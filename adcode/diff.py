# adcode/diff.py
# File snapshots, unified diffs and Git work-tree inspection, re-exported
# from the engine sdk.

from harness.filediff import (
    MAX_DIFF_LINES,
    MAX_SNAPSHOT_BYTES,
    DiffLine,
    FileDiff,
    FileSnapshot,
    diff_lines,
    format_diff_for_model,
    git_worktree_diff,
    make_file_diff,
    snapshot_file,
)

__all__ = [
    'MAX_DIFF_LINES', 'MAX_SNAPSHOT_BYTES', 'DiffLine', 'FileDiff',
    'FileSnapshot', 'diff_lines', 'format_diff_for_model',
    'git_worktree_diff', 'make_file_diff', 'snapshot_file',
]
