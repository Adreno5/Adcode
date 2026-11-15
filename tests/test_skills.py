from pathlib import Path

from adcode.config import SkillsSettings
from adcode.skills import SkillRegistry, parse_frontmatter


def write_skill(root: Path, name: str, body: str, description: str = 'A skill'):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / 'SKILL.md').write_text(
        f'---\nname: {name}\ndescription: {description}\n---\n\n{body}',
        encoding='utf-8',
    )


def test_parse_frontmatter():
    metadata, body = parse_frontmatter(
        '---\nname: demo\ndescription: "Quoted: yes"\n---\n\n# Body\n')
    assert metadata['name'] == 'demo'
    assert metadata['description'] == 'Quoted: yes'
    assert '# Body' in body


def test_parse_block_description():
    metadata, _ = parse_frontmatter(
        '---\nname: demo\ndescription: >\n  line one\n  line two\n---\nbody')
    assert metadata['description'] == 'line one line two'


def test_registry_scan_and_tools(tmp_path):
    root = tmp_path / 'skills'
    write_skill(root, 'alpha', 'Alpha body', 'First skill')
    write_skill(root, 'beta', 'Beta body', 'Second skill')
    nested = root / 'org' / 'gamma'
    nested.mkdir(parents=True)
    (nested / 'SKILL.md').write_text(
        '---\nname: gamma\ndescription: Nested\n---\nGamma body', encoding='utf-8')
    settings = SkillsSettings(enabled=True, include_project=False,
                              extra_paths=[str(root)])
    registry = SkillRegistry(settings).scan()
    assert {skill.name for skill in registry.list()} >= {'alpha', 'beta', 'gamma'}
    preview = registry.preview_prompt()
    assert 'alpha: First skill' in preview
    specs = {spec.name: spec for spec in registry.tool_specs()}
    assert set(specs) == {'skill_list', 'skill_load'}
    assert 'Alpha body' in specs['skill_load'].handler(name='alpha')
    assert 'unknown skill' in specs['skill_load'].handler(name='nope')
    assert 'beta' in specs['skill_list'].handler(query='sec')


def test_disabled_registry_has_no_tools(tmp_path):
    registry = SkillRegistry(SkillsSettings(enabled=False, extra_paths=[])).scan()
    assert registry.list() == []
    assert registry.tool_specs() == []
    assert registry.preview_prompt() == ''
