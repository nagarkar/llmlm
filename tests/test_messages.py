"""`Client.complete_messages` (`llmlm.llm`): a call made from raw messages, with content blocks and
`cache_control` breakpoints, through the same cache, price, caps and ledger as `complete`."""

import json
from types import SimpleNamespace

import pytest

from llmlm import llm
from llmlm.prices import Prices

OPUS = "anthropic/claude-opus-5"
GEMINI = "gemini/gemini-2.5-flash"


def _prices(tmp_path):
    return Prices(
        tmp_path / "prices.json",
        fetch=lambda: {
            OPUS: {
                "input_cost_per_token": 5e-6,
                "output_cost_per_token": 2.5e-5,
                "cache_read_input_token_cost": 5e-7,
                "cache_creation_input_token_cost": 6.25e-6,
            }
        },
    )


class Recorder:
    name = "recorder"

    def __init__(self, reply="a beat", **fields):
        self.reply, self.fields, self.sent = reply, fields, []

    def send(self, model, messages, params, *, tag):
        self.sent.append((model, messages, params))
        return llm.RawReply(self.reply, 100, 20, None, **self.fields)


def _client(tmp_path, transport, config=None):
    config = config or {"writing": {"model": OPUS, "max_tokens": 8000}, "logs": {}}
    return llm.Client(tmp_path / "root", config, transport=transport, prices=_prices(tmp_path))


def _blocks(cache=True):
    system = [{"type": "text", "text": "system law"}]
    if cache:
        system[0]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "text", "text": "the beat"}]},
    ]


def test_the_cache_key_ignores_cache_control_and_includes_only_the_knobs_that_are_set():
    plain = llm.messages_sha256(OPUS, None, _blocks(cache=False))
    assert plain == llm.messages_sha256(OPUS, None, _blocks(cache=True))
    assert plain != llm.messages_sha256(OPUS, None, _blocks(), max_tokens=8000)
    assert llm.messages_sha256(OPUS, None, _blocks(), reasoning_effort=None) == plain


def test_a_profile_without_sampling_gets_no_temperature_and_effort_and_max_tokens_are_sent(
    tmp_path,
):
    transport = Recorder()
    reply = _client(tmp_path, transport).complete_messages(
        "writing", _blocks(), tag="beat-1", max_tokens=8000, reasoning_effort="high"
    )
    _, messages, params = transport.sent[0]
    assert "temperature" not in params
    assert params["max_tokens"] == 8000 and params["reasoning_effort"] == "high"
    assert messages[0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert reply.raw == "a beat" and reply.parsed is None and reply.cached is False


def test_an_identical_call_is_served_from_the_cache_and_a_changed_knob_is_not(tmp_path):
    transport = Recorder()
    client = _client(tmp_path, transport)
    client.complete_messages("writing", _blocks(), tag="b", max_tokens=8000)
    again = client.complete_messages("writing", _blocks(cache=False), tag="b", max_tokens=8000)
    assert again.cached is True and len(transport.sent) == 1
    client.complete_messages("writing", _blocks(), tag="b", max_tokens=4000)
    assert len(transport.sent) == 2


def test_cache_tokens_are_priced_and_reach_the_ledger_line(tmp_path):
    transport = Recorder(cache_read=1000, cache_write=500, finish_reason="stop")
    client = _client(tmp_path, transport)
    reply = client.complete_messages("writing", _blocks(), tag="b")
    expected = round(100 * 5e-6 + 20 * 2.5e-5 + 1000 * 5e-7 + 500 * 6.25e-6, 6)
    assert reply.cost_usd == expected
    lines = (tmp_path / "root/logs/usage.jsonl").read_text().splitlines()
    assert len(lines) == 1
    line = json.loads(lines[0])
    assert line["cache_read"] == 1000 and line["cache_write"] == 500
    assert line["tokens_in"] == 100 and line["cost_usd"] == expected


def test_a_parse_function_turns_the_text_into_parsed_and_a_value_error_is_a_bad_reply(tmp_path):
    client = _client(tmp_path, Recorder('{"a": 1}'))
    reply = client.complete_messages("writing", _blocks(), tag="b", parse=json.loads)
    assert reply.parsed == {"a": 1}
    broken = _client(tmp_path / "other", Recorder("not json"))
    with pytest.raises(llm.BadReply):
        broken.complete_messages("writing", _blocks(), tag="b", parse=json.loads)


def test_the_spend_cap_stops_a_messages_call_too(tmp_path):
    config = {"writing": {"model": OPUS}, "llm": {"max_usd_per_run": 0.0001}, "logs": {}}
    client = _client(tmp_path, Recorder(), config)
    client.complete_messages("writing", _blocks(), tag="one")
    other = [{"role": "user", "content": "a different beat"}]
    with pytest.raises(llm.SpendCapReached):
        client.complete_messages("writing", other, tag="two")


def test_an_unset_stage_is_not_configured(tmp_path):
    with pytest.raises(llm.NotConfigured, match="writing"):
        _client(tmp_path, Recorder(), {"logs": {}}).complete_messages("writing", _blocks(), tag="b")


def _response(finish, content="text", prompt=10, completion=2, cached=0, created=0):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=cached, cache_creation_tokens=created
            ),
        ),
    )


def test_the_transport_nets_cache_tokens_out_of_the_prompt_count_and_names_a_refusal():
    transport = llm.LiteLLMTransport(
        completion=lambda **kw: _response("stop", prompt=1600, cached=1000, created=500),
        cost=lambda response: 0.01,
    )
    raw = transport.send(OPUS, [], {}, tag="t")
    assert (raw.tokens_in, raw.cache_read, raw.cache_write) == (100, 1000, 500)
    refusing = llm.LiteLLMTransport(
        completion=lambda **kw: _response("content_filter"), cost=lambda response: 0.0
    )
    with pytest.raises(llm.Refused):
        refusing.send(OPUS, [], {}, tag="t")


def test_the_new_profile_rows_say_what_authorlm_verified():
    assert llm.model_profile(OPUS).prompt_cache == "anthropic"
    assert llm.model_profile(OPUS).temperature == llm.VENDOR_DEFAULT_ONLY
    assert llm.model_profile("anthropic/claude-haiku-4-5").temperature == llm.ANY
    assert llm.model_profile(GEMINI).prompt_cache is None


def test_messages_hash_is_the_key_the_call_will_use_after_the_profile_has_its_say(tmp_path):
    client = _client(tmp_path, Recorder())
    key = client.messages_hash("writing", _blocks(), max_tokens=8000, reasoning_effort="high")
    assert key == llm.messages_sha256(
        OPUS, None, _blocks(), max_tokens=8000, reasoning_effort="high"
    )
    client.complete_messages(
        "writing", _blocks(), tag="b", max_tokens=8000, reasoning_effort="high"
    )
    assert (tmp_path / "root" / "llm-cache" / f"{key}.json").exists()


def test_an_endpoint_effort_is_passed_through_not_checked_against_litellms_map(tmp_path):
    transport = Recorder()
    client = _client(tmp_path, transport)
    client.complete_messages(
        "writing",
        _blocks(),
        tag="b",
        model="openai/stub-writer",
        reasoning_effort="high",
        api_base="http://127.0.0.1:1/v1",
    )
    _, _, params = transport.sent[0]
    assert params["allowed_openai_params"] == ["reasoning_effort"]
    client.complete_messages("writing", _blocks(), tag="c", reasoning_effort="high")
    assert "allowed_openai_params" not in transport.sent[1][2]
