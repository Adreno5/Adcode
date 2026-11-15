"""Clipboard access with cross-platform fallbacks.

``pyperclip`` covers Windows and macOS, and X11/Wayland when the matching
helper program happens to be installed.  On a bare Linux box none of them
exist, and the failure was silent: ``ctrl+v`` pasted nothing at all and the
user had no way of telling a broken clipboard from an empty one.

Every platform therefore gets a list of mechanisms and the first one that
works wins.  A copy made inside AdCode is also kept in memory, so pasting it
back always works even when the machine has no clipboard tool at all.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading

_IS_WIN = os.name == 'nt'
_IS_MAC = sys.platform == 'darwin'

_lock = threading.RLock()
_cached: str = ''
_tk_root: object | None = None


def _run(command: list[str], stdin: str | None = None,
         timeout: float = 5.0) -> str | None:
    """Run a clipboard helper; ``None`` when it is missing or failed."""
    try:
        result = subprocess.run(
            command, input=stdin, capture_output=True, text=True,
            encoding='utf-8', errors='replace', timeout=timeout, check=False)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _strip_trailing_newline(text: str) -> str:
    """Drop the newline a clipboard helper adds to the end of the text.

    A trailing newline inside the prompt would submit the message, so only
    the very last line break is removed.
    """
    if text.endswith('\r\n'):
        return text[:-2]
    if text.endswith('\n'):
        return text[:-1]
    return text


# mechanisms


def _read_pyperclip() -> str | None:
    try:
        import pyperclip
        return pyperclip.paste()
    except Exception:
        return None


def _write_pyperclip(text: str) -> bool:
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        return False


def _read_commands() -> list[list[str]]:
    if _IS_WIN:
        return [['powershell', '-NoProfile', '-Command', 'Get-Clipboard -Raw']]
    if _IS_MAC:
        return [['pbpaste']]
    return [
        ['wl-paste', '--no-newline'],
        ['wl-paste'],
        ['xclip', '-selection', 'clipboard', '-out'],
        ['xsel', '--clipboard', '--output'],
    ]


def _write_commands() -> list[list[str]]:
    if _IS_WIN:
        return [['clip.exe']]
    if _IS_MAC:
        return [['pbcopy']]
    return [
        ['wl-copy'],
        ['xclip', '-selection', 'clipboard'],
        ['xsel', '--clipboard', '--input'],
    ]


def _tk() -> object | None:
    """A hidden Tk root used as the last resort on X11 desktops.

    The root is kept alive: X11 hands the clipboard to the process that owns
    it, so destroying the root would empty the clipboard right away.
    """
    global _tk_root
    with _lock:
        if _tk_root is not None:
            return _tk_root
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
    except Exception:
        return None
    with _lock:
        _tk_root = root
    return root


def _read_tk() -> str | None:
    root = _tk()
    if root is None:
        return None
    try:
        root.update()
        return root.clipboard_get()
    except Exception:
        return None


def _write_tk(text: str) -> bool:
    root = _tk()
    if root is None:
        return False
    try:
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        return True
    except Exception:
        return False


# public API


def get_clipboard() -> str:
    """Return the clipboard text, or an empty string when unavailable."""
    text = _read_pyperclip()
    if text is None:
        for command in _read_commands():
            if not shutil.which(command[0]):
                continue
            text = _run(command)
            if text is not None:
                break
    if text is None:
        text = _read_tk()
    if text is None:
        # No clipboard mechanism at all: fall back to what this process
        # copied, so a copy made inside AdCode can still be pasted back.
        with _lock:
            return _cached
    return _strip_trailing_newline(text)


def set_clipboard(text: str) -> bool:
    """Put ``text`` on the clipboard; ``False`` when no mechanism worked."""
    global _cached
    with _lock:
        _cached = text
    if _write_pyperclip(text):
        return True
    for command in _write_commands():
        if not shutil.which(command[0]):
            continue
        result = _run(command, stdin=text)
        if result is not None:
            return True
    return _write_tk(text)


# Every binding that means "paste from the clipboard".  ``shift+insert`` is
# the traditional terminal shortcut and ``ctrl+shift+v`` the one several
# terminals send for a paste.
PASTE_KEYS = ('ctrl+v', 'ctrl+shift+v', 'shift+insert')


def is_paste_key(key: str) -> bool:
    """True when the key asks for the clipboard content."""
    return key in PASTE_KEYS
