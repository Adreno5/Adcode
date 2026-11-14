from adcode.tui.canvas import BOLD, Canvas, char_width, text_width, truncate_text


def test_char_width():
    assert char_width('a') == 1
    assert char_width('中') == 2
    assert char_width('́') == 0
    assert text_width('a中') == 3


def test_truncate_text():
    assert truncate_text('abcdef', 4) == 'abcd'
    assert truncate_text('abcdef', 4, '…') == 'abc…'
    assert truncate_text('中文测试', 5, '…') == '中文…'


def test_alpha_fill(palette):
    canvas = Canvas(4, 1, palette)
    canvas.clear()
    base = canvas.cell(0, 0).bg
    canvas.fill_rect(0, 0, 4, 1, (255, 0, 0), alpha=128)
    blended = canvas.cell(0, 0).bg
    assert blended != base
    assert blended[0] > base[0]
    assert blended[1] < 128


def test_text_alpha_blends_toward_background(palette):
    canvas = Canvas(4, 1, palette)
    canvas.clear()
    canvas.draw_text(0, 0, 'X', (255, 255, 255), alpha=64)
    cell = canvas.cell(0, 0)
    assert cell.ch == 'X'
    assert cell.fg != (255, 255, 255)
    assert cell.fg[0] > 60


def test_wide_char_continuation(palette):
    canvas = Canvas(4, 1, palette)
    canvas.clear()
    canvas.draw_text(0, 0, '中a', palette.text)
    assert canvas.cell(0, 0).ch == '中'
    assert canvas.cell(1, 0).cont is True
    assert canvas.cell(2, 0).ch == 'a'
    frame = canvas.render()
    assert '中' in frame


def test_overwriting_half_a_wide_char_repairs_it(palette):
    canvas = Canvas(4, 1, palette)
    canvas.clear()
    canvas.draw_text(0, 0, '中', palette.text)
    canvas.draw_text(1, 0, 'X', palette.text)
    assert canvas.cell(0, 0).ch == ' '
    assert canvas.cell(1, 0).cont is False
    assert canvas.cell(1, 0).ch == 'X'


def test_clipping(palette):
    canvas = Canvas(6, 2, palette)
    canvas.clear()
    with canvas.clip(0, 0, 2, 1):
        canvas.draw_text(0, 0, 'abcdef', palette.text)
        canvas.draw_text(0, 1, 'hidden', palette.text)
    assert canvas.cell(1, 0).ch == 'b'
    assert canvas.cell(2, 0).ch == ' '
    assert canvas.cell(0, 1).ch == ' '


def test_box_and_render(palette):
    canvas = Canvas(10, 4, palette)
    canvas.clear()
    canvas.box(0, 0, 6, 4, palette.border, bg=palette.panel, title='Hi')
    frame = canvas.render()
    assert frame.startswith('\x1b[H')
    assert 'Hi' in frame
    assert '┌' in frame


def test_attributes_serialized(palette):
    canvas = Canvas(3, 1, palette)
    canvas.clear()
    canvas.draw_text(0, 0, 'B', palette.text, attrs=BOLD)
    assert '\x1b[' in canvas.render()
