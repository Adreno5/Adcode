from adcode.session import (
    DbSessionStore, LAST_SESSION, SessionPayload, SessionStore,
    TranscriptItem,
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


# ── the sqlite route ───────────────────────────────────────


def _db(tmp_path):
    return DbSessionStore(db_path=tmp_path / 'sessions' / 'adcode.db')


def test_db_roundtrip_and_rows(tmp_path):
    store = _db(tmp_path)
    payload = SessionPayload(name='s', title='t', provider='P', model='m',
                             messages=[
                                 {'role': 'user', 'content': 'hi',
                                  'reasoning_content': 'th'},
                                 {'role': 'assistant', 'content': None,
                                  'tool_calls': [{'id': 'c1', 'type': 'function',
                                                  'function': {'name': 'edit',
                                                               'arguments': '{}'}}]}],
                             items=[TranscriptItem(kind='user', turn=1, text='hi')],
                             turn=1, prefix_generation=4)
    store.save(payload)
    loaded = store.load('s')
    assert loaded is not None and loaded.name == 's'
    assert loaded.messages[0]['reasoning_content'] == 'th'
    assert loaded.messages[1]['tool_calls'][0]['id'] == 'c1'
    assert loaded.prefix_generation == 4
    # a second save replaces the rows instead of appending
    payload.messages = payload.messages[:1]
    store.save(payload)
    assert len(store.load('s').messages) == 1


def test_db_names_listing_and_delete(tmp_path):
    store = _db(tmp_path)
    store.save(SessionPayload(name='alpha', messages=[
        {'role': 'user', 'content': 'x'}]))
    assert store.unique_name('alpha') == 'alpha_2'
    assert [row['name'] for row in store.list()] == ['alpha']
    assert store.delete('alpha') is True and store.delete('alpha') is False
    assert store.list() == []


def test_db_latest_checkpoint_semantics(tmp_path):
    store = _db(tmp_path)
    assert store.latest() is None
    store.save(SessionPayload(name='old', messages=[
        {'role': 'user', 'content': 'hi'}]), as_name='old')
    assert store.latest().name == 'old'
    store.save(SessionPayload(name=LAST_SESSION), as_name=LAST_SESSION)
    assert store.latest() is None, 'a cleared checkpoint must not resurrect'
    store.save(SessionPayload(name='next', messages=[
        {'role': 'user', 'content': 'hi'}]), as_name='next')
    store.save(SessionPayload(name='next', messages=[
        {'role': 'user', 'content': 'hi'}]), as_name=LAST_SESSION)
    # the checkpoint keeps the session's own name, like the json route
    assert store.latest().name == 'next'


def test_db_rewind_helpers(tmp_path):
    payload = SessionPayload(name='s', messages=[
        {'role': 'user', 'content': 'first'},
        {'role': 'assistant', 'content': 'ok'},
        {'role': 'user', 'content': 'second'},
    ])
    starts = DbSessionStore.turn_starts(payload.messages)
    assert starts == [0, 2]
    rewound = DbSessionStore.rewind(payload, 2)
    assert [m['content'] for m in rewound.messages] == ['first', 'ok']
