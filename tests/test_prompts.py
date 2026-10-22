"""Prompts (`llmlm.prompts`): one file per version, searched in a list of directories (the
built-in one, then those a consumer registers or passes); a change is a copy to the next version."""

import pytest

from llmlm import prompts

FRONT = (
    '---\nid = "{id}"\nversion = {version}\nstage = "greet"\n'
    'output_schema = "greeting_reply"\nconfig_section = "greeting"\n---\n'
)


def write_prompt(directory, prompt_id="greeting", version=1, body="Say {word}.\n"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prompt_id}_v{version}.md"
    path.write_text(FRONT.format(id=prompt_id, version=version) + body)
    return path


def test_a_prompt_renders_its_placeholders_and_declines_a_missing_one(tmp_path):
    write_prompt(tmp_path)
    prompt = prompts.load("greeting", directories=[tmp_path])
    assert prompt.render({"word": "hello"}) == "Say hello.\n"
    assert prompt.placeholders() == ["word"]
    assert len(prompt.sha256) == 64
    with pytest.raises(prompts.PromptError, match="missing variables"):
        prompt.render({})


def test_load_takes_the_newest_version_or_the_one_asked_for(tmp_path):
    write_prompt(tmp_path, version=1)
    write_prompt(tmp_path, version=2, body="Say {word} twice.\n")
    assert prompts.load("greeting", directories=[tmp_path]).version == 2
    assert prompts.load("greeting", version=1, directories=[tmp_path]).version == 1
    assert prompts.load("greeting_v1", directories=[tmp_path]).version == 1
    with pytest.raises(prompts.PromptError, match="no version 9"):
        prompts.load("greeting", version=9, directories=[tmp_path])
    with pytest.raises(prompts.PromptError, match="no prompt named 'absent'"):
        prompts.load("absent", directories=[tmp_path])


def test_a_file_whose_name_and_front_matter_disagree_is_declined(tmp_path):
    (tmp_path / "greeting_v7.md").write_text(FRONT.format(id="greeting", version=1) + "x\n")
    with pytest.raises(prompts.PromptError, match="greeting_v7.md"):
        prompts.load("greeting", directories=[tmp_path])


def test_front_matter_must_be_complete_and_known(tmp_path):
    (tmp_path / "a_v1.md").write_text('---\nid = "a"\nversion = 1\n---\nbody\n')
    with pytest.raises(prompts.PromptError, match="missing"):
        prompts.load("a", directories=[tmp_path])
    (tmp_path / "b_v1.md").write_text(
        FRONT.format(id="b", version=1).replace('stage = "greet"', 'stage = "g"\nextra = 1') + "x"
    )
    with pytest.raises(prompts.PromptError, match="unknown front matter keys"):
        prompts.load("b", directories=[tmp_path])
    (tmp_path / "c_v1.md").write_text("no front matter")
    with pytest.raises(prompts.PromptError, match="no TOML front matter"):
        prompts.load("c", directories=[tmp_path])


def test_registered_directories_are_searched_and_llmlm_ships_no_prompts(tmp_path, monkeypatch):
    monkeypatch.setattr(prompts, "_registered", [])
    assert prompts.search_directories() == []
    write_prompt(tmp_path / "mine")
    with pytest.raises(prompts.PromptError):
        prompts.load("greeting")
    prompts.register_directory(tmp_path / "mine")
    prompts.register_directory(tmp_path / "mine")  # a repeat is ignored
    assert prompts.search_directories() == [tmp_path / "mine"]
    assert prompts.load("greeting").path.parent == tmp_path / "mine"
    assert [
        prompt.id for prompt in prompts.list_prompts() if prompt.path.parent == tmp_path / "mine"
    ] == ["greeting"]


def test_several_directories_are_searched_and_a_released_version_is_never_shadowed(tmp_path):
    write_prompt(tmp_path / "one", version=1)
    write_prompt(tmp_path / "two", prompt_id="other", version=1)
    both = [tmp_path / "one", tmp_path / "two"]
    assert {prompt.id for prompt in prompts.list_prompts(both)} == {"greeting", "other"}
    write_prompt(tmp_path / "two", version=1, body="Different {word}.\n")
    with pytest.raises(prompts.PromptError, match="in two places"):
        prompts.load("greeting", directories=both)


def test_expected_output_tokens_is_optional_front_matter(tmp_path):
    write_prompt(tmp_path)
    assert prompts.load("greeting", directories=[tmp_path]).expected_output_tokens is None
    front = FRONT.format(id="sized", version=1)
    sized = front[: -len("---\n")] + "expected_output_tokens = 150\n---\n"
    (tmp_path / "sized_v1.md").write_text(sized + "Say {word}.\n")
    assert prompts.load("sized", directories=[tmp_path]).expected_output_tokens == 150
