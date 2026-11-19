
from adcode.commands import (
    COMMANDS, command_for, completion_candidates, dispatch, find,
)
from adcode.config import AppConfig
from adcode.tui.provider_ui import normalize_base_url


class FakeApp:
    def __init__(self):
        self.config = AppConfig()
        self.cwd = '.'
        self.loop = None
        self.notices = []
        self.messages = []
        self.cleared = 0
        self.btw_mode = False

    def notify(self, text, level='info'):
        self.notices.append((level, text))

    def open_message(self, title, message):
        self.messages.append((title, message))

    def open_select(self, *args, **kwargs):
        self.messages.append(('select', args))

    def open_text_prompt(self, *args, **kwargs):
        self.messages.append(('text', args))

    def open_settings(self):
        self.messages.append(('settings',))

    def clear_conversation(self):
        self.cleared += 1

    def config_changed(self):
        pass

    def request_exit(self):
        self.messages.append(('exit',))


def test_command_registry():
    names = {command.name for command in COMMANDS}
    for expected in ('/help', '/config', '/provider', '/mcp', '/skills',
                     '/rewind', '/fork', '/edit', '/btw', '/goal', '/agents'):
        assert expected in names
    assert find('/provider').subcommands


def test_completion_top_level():
    candidates = completion_candidates('/')
    assert candidates
    assert all(candidate[0].startswith('/') for candidate in candidates)
    provider = completion_candidates('/prov')
    assert provider[0][0] == '/provider '


def test_completion_subcommands():
    candidates = completion_candidates('/provider ')
    assert {candidate[1] for candidate in candidates} == {
        'add', 'edit', 'remove', 'list'}
    assert completion_candidates('/provider add x') == []
    assert completion_candidates('hello') == []


def test_a_message_that_starts_with_a_path_is_not_a_command():
    # the whole input used to be swallowed as a command, so the model never
    # saw a message that began with a slash
    text = '/home/darkstar/temp_workspace/CODE_STYLE.md review this file'
    assert command_for(text) is None
    assert command_for('/nope') is None
    assert command_for('/') is None
    assert command_for('hello /help') is None
    assert command_for('') is None

def test_a_registered_name_is_still_a_command():
    assert command_for('/help').name == '/help'
    assert command_for('  /help  ').name == '/help'
    assert command_for('/config extra').name == '/config'
    # a multi-line paste whose first word is a command stays a command
    assert command_for('/help\nmore text').name == '/help'

def test_dispatch_effort():
    app = FakeApp()
    dispatch(app, '/effort high')
    assert app.config.reasoning.effort == 'high'
    assert any('Reasoning effort' in text for _, text in app.notices)


def test_dispatch_clear_and_unknown():
    app = FakeApp()
    dispatch(app, '/clear')
    assert app.cleared == 1
    dispatch(app, '/doesnotexist')
    assert any(level == 'warning' for level, _ in app.notices)


def test_dispatch_help_and_max_tokens():
    app = FakeApp()
    dispatch(app, '/help')
    assert app.messages and app.messages[-1][0] == 'Help'
    dispatch(app, '/max_tokens 2048')
    assert app.config.max_output_tokens == 2048
    dispatch(app, '/max_tokens abc')
    assert any('Usage' in text for _, text in app.notices)


def test_permissions_shows_and_sets_the_level(tmp_path):
    """``/permissions`` is the switch for the project's access level."""
    from adcode import paths

    paths.set_project_root(str(tmp_path))
    app = FakeApp()
    app.config.security.access = 'readonly'

    dispatch(app, '/permissions')
    assert app.messages and app.messages[0][0] == 'select', app.messages

    dispatch(app, '/permissions workspace')
    root = str(tmp_path.resolve())
    assert app.config.security.access_by_project.get(root) == 'workspace'
    assert app.notices, app.notices

    dispatch(app, '/permissions default')
    assert root not in app.config.security.access_by_project

    dispatch(app, '/permissions nonsense')
    assert any('nonsense' in text for _, text in app.notices), app.notices


def test_normalize_base_url():
    assert normalize_base_url('https://api.x.com') == 'https://api.x.com/v1'
    assert normalize_base_url('https://api.x.com/') == 'https://api.x.com/v1'
    assert normalize_base_url('https://api.x.com/v1') == 'https://api.x.com/v1'
    assert normalize_base_url('http://localhost:8080#') == 'http://localhost:8080'
