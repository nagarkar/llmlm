"""Request parameters are model dependent and checked up front: a model with no row in
MODEL_PROFILES gets the plain request and a warning; a `temperature` configured for a model that
takes none is a config error naming `[section].temperature`. The consumer names the sections."""

import pytest

from llmlm import llm

SECTIONS = ("cleanup", "digest")


def test_each_verified_model_has_one_exact_row_and_no_wildcard():
    names = [name for name, _ in llm.MODEL_PROFILES]
    assert len(names) == len(set(names)) and not any("*" in name for name in names)
    assert llm.model_profile("gemini/gemini-2.5-flash").temperature == llm.ANY
    assert llm.model_profile("gemini/gemini-2.5-flash").json_mode is True
    assert llm.model_profile("gemini/gemini-3.1-flash-lite").temperature == llm.VENDOR_DEFAULT_ONLY
    assert llm.model_profile("gemini/gemini-2.5-flash-lite-preview") is llm.UNKNOWN_PROFILE


def test_an_unlisted_model_gets_the_plain_request_profile():
    profile = llm.model_profile("vendor/no-row-yet")
    assert profile.temperature == llm.VENDOR_DEFAULT_ONLY and profile.json_mode is False


@pytest.mark.parametrize("section", SECTIONS)
@pytest.mark.parametrize("model", ["gemini/gemini-3.1-flash-lite", "vendor/no-row-yet"])
def test_a_temperature_for_a_model_that_takes_none_is_a_config_error(section, model):
    config = {section: {"model": model, "temperature": 0.2}}
    with pytest.raises(ValueError, match=rf"\[{section}\]\.temperature.*config\.toml"):
        llm.check_temperatures(config, SECTIONS)


def test_a_temperature_for_a_model_that_takes_one_is_kept():
    llm.check_temperatures(
        {"cleanup": {"model": "gemini/gemini-2.5-flash", "temperature": 0.3}}, SECTIONS
    )
    llm.check_temperatures({"cleanup": {"model": "vendor/x"}, "other": {}}, SECTIONS)


def test_a_model_with_no_row_is_named_with_its_section():
    warnings = llm.unprofiled_models(
        {
            "digest": {"model": "vendor/no-row-yet"},
            "cleanup": {"model": "gemini/gemini-3.1-flash-lite"},
            "arbiter": {"model": "jev-latest"},  # not in the sections the caller names
        },
        SECTIONS,
    )
    assert len(warnings) == 1
    assert "[digest].model" in warnings[0] and "vendor/no-row-yet" in warnings[0]
    assert "MODEL_PROFILES" in warnings[0] and "plain request" in warnings[0]


def test_a_section_with_no_model_or_absent_is_not_a_warning():
    assert llm.unprofiled_models({"cleanup": {}}, SECTIONS) == []


def test_the_openai_models_the_tools_run_on_take_no_temperature_and_json_mode():
    for model in ("openai/gpt-5.5", "openai/gpt-5.6", "openai/gpt-5.6-sol"):
        profile = llm.model_profile(model)
        assert profile.temperature == llm.VENDOR_DEFAULT_ONLY and profile.json_mode is True


def test_gamelms_openai_roles_get_a_strict_schema_and_may_take_an_effort():
    # verified live 2026-10-22: strict json_schema accepted, reasoning_effort accepted
    for model in ("openai/gpt-5.5", "openai/gpt-5.6-sol"):
        profile = llm.model_profile(model)
        assert profile.json_schema is True and profile.reasoning_effort is True
