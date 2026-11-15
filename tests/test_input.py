from __future__ import annotations

import os
import time

import pytest

from adcode.tui.input import KeyEvent, KeyParser

windows_only = pytest.mark.skipif(
    os.name != 'nt', reason='Windows console input records')


def parse(text: str) -> list[KeyEvent]:
    parser = KeyParser()
    return parser.feed(text)


def test_plain_text():
    events = parse('abc')
    assert [event.text for event in events] == ['a', 'b', 'c']


def test_control_keys():
    events = parse('\x03\x01\x7f\t')
    assert [event.key for event in events] == ['ctrl+c', 'ctrl+a', 'backspace', 'tab']


def test_arrows_and_modified():
    events = parse('\x1b[A\x1b[B\x1b[C\x1b[D')
    assert [event.key for event in events] == ['up', 'down', 'right', 'left']
    events = parse('\x1b[1;5C')
    assert events[0].key == 'ctrl+right'
    assert events[0].ctrl


def test_shift_tab_and_delete():
    assert parse('\x1b[Z')[0].key == 'shift+tab'
    assert parse('\x1b[3~')[0].key == 'delete'
    assert parse('\x1b[5~')[0].key == 'pgup'


def test_alt_char():
    events = parse('\x1bx')
    assert events[0].key == 'alt+x'
    assert events[0].alt


def test_mouse_wheel_and_click():
    events = parse('\x1b[<64;10;5M')
    assert events[0].kind == 'mouse'
    assert events[0].action == 'wheel_up'
    assert (events[0].x, events[0].y) == (10, 5)
    events = parse('\x1b[<0;3;4M')
    assert events[0].action == 'press'
    events = parse('\x1b[<0;3;4m')
    assert events[0].action == 'release'


def test_bracketed_paste():
    events = parse('\x1b[200~hello\nworld\x1b[201~')
    assert events[0].kind == 'paste'
    assert events[0].text == 'hello\nworld'


def test_incomplete_sequence_waits():
    parser = KeyParser()
    assert parser.feed('\x1b[') == []
    events = parser.feed('A')
    assert events[0].key == 'up'


def test_lone_escape_flushes():
    parser = KeyParser()
    assert parser.feed('\x1b') == []
    events = parser.flush_pending()
    assert events[0].key == 'esc'


def test_surrogate_sequence_is_ignored():
    parser = KeyParser()
    assert parser.feed('\x1b]0;title\x07') == []


# ── split / truncated control sequences ────────────────────
# A terminal may deliver one report in several writes.  The tail must never be
# typed into the prompt, and a truncated report must not fire a stray Esc
# (which would interrupt a running turn).

def test_split_mouse_report_still_scrolls():
    parser = KeyParser()
    assert parser.feed('\x1b[<64;1', now=0.0) == []
    events = parser.feed('0;5M', now=0.05)
    assert [event.action for event in events] == ['wheel_up']
    assert (events[0].x, events[0].y) == (10, 5)


def test_truncated_mouse_report_never_becomes_text():
    parser = KeyParser()
    assert parser.feed('\x1b[<64;10', now=0.0) == []
    assert parser.flush_due(now=0.2) == []          # still waiting
    assert parser.flush_due(now=0.5) == []          # abandoned silently
    assert parser.feed(';5M', now=0.6) == []


def test_escape_tail_after_committed_esc_is_swallowed():
    parser = KeyParser()
    assert parser.feed('\x1b', now=0.0) == []
    events = parser.flush_due(now=0.2)
    assert [event.key for event in events] == ['esc']
    # The rest of the report that followed the Esc must not reach the prompt.
    events = parser.feed('[<64;10;5M', now=0.25)
    assert [event.kind for event in events] == []


def test_typing_after_escape_is_not_swallowed():
    parser = KeyParser()
    parser.feed('\x1b', now=0.0)
    parser.flush_due(now=0.2)
    events = parser.feed('[hello', now=0.25)
    assert ''.join(event.text for event in events) == '[hello'


@pytest.mark.parametrize('typed', ['/', '3', ';', '?', '=', 'text'])
def test_printables_after_escape_reach_the_prompt(typed):
    """Esc closes a dialog; the next key typed there must not vanish."""
    parser = KeyParser()
    parser.feed('\x1b', now=0.0)
    assert parser.flush_due(now=0.2)[0].key == 'esc'
    events = parser.feed(typed, now=0.25)
    assert ''.join(event.text for event in events) == typed


def test_split_report_parameters_after_escape_are_swallowed():
    parser = KeyParser()
    parser.feed('\x1b', now=0.0)
    parser.flush_due(now=0.2)
    assert parser.feed('1;5', now=0.25) == []
    assert parser.feed('A', now=0.3) == []
    assert parser.feed('q', now=0.35)[0].text == 'q'   # typing resumes


def test_escape_flush_waits_for_its_window():
    parser = KeyParser()
    assert parser.feed('\x1b', now=0.0) == []
    assert parser.flush_due(now=0.02) == []         # too early for Esc
    assert parser.flush_due(now=0.2)[0].key == 'esc'


def test_streaming_paste_is_not_dropped():
    parser = KeyParser()
    assert parser.feed('\x1b[200~hel', now=0.0) == []
    assert parser.feed('lo', now=0.05) == []
    events = parser.feed('\x1b[201~', now=0.06)
    assert events[0].kind == 'paste' and events[0].text == 'hello'


def test_paste_without_terminator_is_delivered():
    parser = KeyParser()
    parser.feed('\x1b[200~partial', now=0.0)
    events = parser.flush_pending(now=20.0)
    assert events[0].kind == 'paste' and events[0].text == 'partial'


# ── Windows console records ────────────────────────────────
# msvcrt reads through ReadConsole, which filters mouse records, so conhost's
# wheel has to be taken from the console input buffer directly.

def _record(event_type: int):
    import adcode.tui.input as input_module
    record = input_module._INPUT_RECORD()
    record.EventType = event_type
    return record


def _wheel_record(delta: int, flags: int = 0x0004, control: int = 0):
    record = _record(0x0002)
    mouse = record.Event.MouseEvent
    mouse.dwMousePosition.X = 20
    mouse.dwMousePosition.Y = 12
    mouse.dwButtonState = (delta & 0xFFFF) << 16
    mouse.dwControlKeyState = control
    mouse.dwEventFlags = flags
    return record


@windows_only
def test_windows_wheel_record_conversion():
    import adcode.tui.input as input_module

    event = input_module._win_wheel_event(_wheel_record(120).Event.MouseEvent)
    assert event is not None
    assert event.kind == 'mouse'
    assert event.action == 'wheel_up'
    assert (event.x, event.y) == (21, 13)

    event = input_module._win_wheel_event(
        _wheel_record(-120, control=0x000C).Event.MouseEvent)
    assert event is not None
    assert event.action == 'wheel_down'
    assert event.ctrl and not event.alt

    # Horizontal wheels, plain moves and zero deltas are not vertical scrolls.
    assert input_module._win_wheel_event(
        _wheel_record(120, flags=0x0008).Event.MouseEvent) is None
    assert input_module._win_wheel_event(
        _wheel_record(0).Event.MouseEvent) is None
    assert input_module._win_wheel_event(
        _wheel_record(120, flags=0x0001).Event.MouseEvent) is None


@windows_only
def test_windows_drain_converts_mouse_and_leaves_key_down(monkeypatch):
    import adcode.tui.input as input_module

    key_up = _record(0x0001)
    key_up.Event.KeyEvent.bKeyDown = False
    key_down = _record(0x0001)
    key_down.Event.KeyEvent.bKeyDown = True
    pending = [_record(0x0010), _wheel_record(120), key_up, key_down]

    monkeypatch.setattr(input_module, '_win_console_input', lambda: 1)
    monkeypatch.setattr(
        input_module, '_win_peek_record',
        lambda handle: pending[0] if pending else None)
    monkeypatch.setattr(
        input_module, '_win_read_record',
        lambda handle: pending.pop(0) if pending else None)

    events: list[KeyEvent] = []
    input_module._win_drain_records(events)

    assert [event.action for event in events] == ['wheel_up']
    # The key-down record stays for msvcrt; focus and key-up were consumed.
    assert [record.EventType for record in pending] == [0x0001]
    assert pending[0].Event.KeyEvent.bKeyDown


@windows_only
def test_windows_reader_survives_a_split_mouse_report(monkeypatch):
    """A wheel report split by a long gap must not type its tail."""
    import queue as queue_module

    import adcode.tui.input as input_module

    chars: list[str] = []
    monkeypatch.setattr(input_module, '_win_drain_records',
                        lambda events: None)
    monkeypatch.setattr(input_module, '_win_read_char',
                        lambda: chars.pop(0))
    monkeypatch.setattr(input_module, '_win_input_ready', lambda: bool(chars))
    monkeypatch.setattr(input_module, '_win_wait_input', lambda seconds: None)

    delivered: 'queue_module.Queue[KeyEvent]' = queue_module.Queue()
    reader = input_module.InputThread(delivered)
    reader.start()
    try:
        chars.append('\x1b')
        time.sleep(0.15)          # longer than the Esc window
        # The rest of the report arrives one character at a time.
        for char in '[<64;10;5M':
            chars.append(char)
            time.sleep(0.005)
        time.sleep(0.05)
    finally:
        reader.stop()

    events: list[KeyEvent] = []
    while True:
        try:
            events.append(delivered.get_nowait())
        except queue_module.Empty:
            break
    assert [event.kind for event in events] == ['key', 'mouse']
    assert events[0].key == 'esc'
    assert events[1].action == 'wheel_up'


