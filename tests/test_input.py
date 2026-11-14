from adcode.tui.input import KeyEvent, KeyParser


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
