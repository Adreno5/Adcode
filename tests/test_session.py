from adcode.session import (
    LAST_SESSION, SessionPayload, SessionStore, TranscriptItem,
)


def _payload(name='s1', messages=None, items=None):
    return SessionPayload(
        name=name, title='Title', provider='P', model='m', turn=1,
        messages=messages or [{'role': 'user', 'content': 'hello'}],
        items=items or [TranscriptItem(kind='user', turn=1, text='hello')],
    )


def test_save_load_list(tmp_path):
    store = SessionStore(tmp_path)
    store.save(_payload())
    loaded = store.load('s1')
    assert loaded is not None
    assert loaded.title == 'Title'
    assert loaded.items[0].text == 'hello'
    rows = store.list()
    assert rows and rows[0]['name'] == 's1'


def test_last_checkpoint_and_empty_clear(tmp_path):
    store = SessionStore(tmp_path)
    store.save(_payload('real'), LAST_SESSION)
    latest = store.latest()
    assert latest is not None and latest.name == 'real'
    store.save(SessionPayload(name=''), LAST_SESSION)
    assert store.latest() is None


def test_rewind_truncates_messages_and_items(tmp_path):
    messages = [
        {'role': 'user', 'content': 'first'},
        {'role': 'assistant', 'content': 'answer one'},
        {'role': 'user', 'content': 'second'},
        {'role': 'assistant', 'content': 'answer two'},
    ]
    items = [
        TranscriptItem(kind='user', turn=1, text='first'),
        TranscriptItem(kind='assistant', turn=1, text='answer one'),
        TranscriptItem(kind='user', turn=2, text='second'),
        TranscriptItem(kind='assistant', turn=2, text='answer two'),
    ]
    payload = SessionPayload(name='s', messages=messages, items=items, turn=2)
    SessionStore.rewind(payload, 2)
    assert [message['content'] for message in payload.messages] == [
        'first', 'answer one']
    assert [item.turn for item in payload.items] == [1, 1]
    assert payload.turn == 1


def test_turn_starts_ignore_injected_blocks():
    messages = [
        {'role': 'user', 'content': 'real'},
        {'role': 'user', 'content': '<subagent_updates>noise</subagent_updates>'},
        {'role': 'user', 'content': '<conversation_summary>sum</conversation_summary>'},
        {'role': 'user', 'content': 'real two'},
    ]
    assert SessionStore.turn_starts(messages) == [0, 3]


def test_transcript_item_roundtrip():
    item = TranscriptItem(kind='tool', turn=2, name='read', args={'path': 'x'},
                          status='done', output='data', diff={'path': 'x'})
    clone = TranscriptItem.from_dict(item.to_dict())
    assert clone == item
    assert TranscriptItem.from_dict({'bad': True}) is None
