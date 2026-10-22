"""The model-call layer (`llmlm.llm`): every call leaves a full provenance record, identical
payloads never bill twice, replies are parsed strictly, the spend caps stop a run cleanly, and the
cache key is what it always was. Offline: the scripted transport serves replies from a directory,
or a stub transport answers; no test calls a model or the network."""

import json
import re
from datetime import timedelta
from pathlib import Path

import pytest

from llmlm import clock, llm, prompts
from llmlm.llm import Client
from llmlm.prices import Prices

FIXTURES = Path(__file__).parent / "fixtures"
FLASH_LITE = "gemini/gemini-2.5-flash-lite"
GEMINI_3 = "gemini/gemini-3.1-flash-lite"
CONFIG = {"greeting": {"model": FLASH_LITE, "temperature": 0.2}, "llm": {}, "logs": {}}


def validate(schema_name, reply):
    """Stands in for a consumer's schema validator: a greeting has one string `greeting`."""
    assert schema_name == "greeting_reply"
    if not isinstance(reply.get("greeting"), str):
        raise ValueError("greeting must be a string")


def greeting_schema(schema_name):
    """Stands in for a consumer's schema loader (sent as json_schema to a model that honours it)."""
    assert schema_name == "greeting_reply"
    return {"type": "object", "required": ["greeting"]}


@pytest.fixture
def prompt(tmp_path):
    directory = tmp_path / "prompts"
    directory.mkdir()
    (directory / "greeting_v1.md").write_text(
        '---\nid = "greeting"\nversion = 1\nstage = "greet"\n'
        'output_schema = "greeting_reply"\nconfig_section = "greeting"\n---\nSay {word}.\n'
    )
    return prompts.load("greeting", directories=[directory])


def script(tmp_path, tag, reply, *, tokens=(100, 20), stage="greet"):
    directory = tmp_path / "scripts" / stage
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{tag}.json").write_text(
        json.dumps({"reply": reply, "tokens_in": tokens[0], "tokens_out": tokens[1]})
    )
    return tmp_path / "scripts"


def prices(tmp_path):
    """A price cache a test owns: the Gemini rows corpuslm verified, never the network."""
    return Prices(
        tmp_path / "prices.json",
        fetch=lambda: {
            "gemini/gemini-2.5-flash-lite": {
                "input_cost_per_token": 1e-7,
                "output_cost_per_token": 4e-7,
            },
            "gemini/gemini-3.1-flash-lite": {
                "input_cost_per_token": 2.5e-7,
                "output_cost_per_token": 1.5e-6,
            },
        },
    )


def client(tmp_path, scripts=None, config=CONFIG, **kwargs):
    root = tmp_path / "root"
    transport = llm.ScriptedTransport(scripts or tmp_path / "scripts")
    return Client(
        root,
        config,
        transport=transport,
        validator=validate,
        schema_loader=greeting_schema,
        prices=prices(tmp_path),
        **kwargs,
    )


def usage_lines(tmp_path):
    path = tmp_path / "root" / "logs" / "usage.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def invocation_logs(tmp_path):
    return sorted((tmp_path / "root" / "logs" / "runs").glob("*.log"))


# ---- the cache key ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "temperature", "messages", "expected"),
    [
        # Values captured from ytlm's llm.py BEFORE it moved here. A stored cache of model
        # replies stays valid only while these hold byte for byte.
        (
            "gemini/gemini-2.5-flash",
            0.2,
            [{"role": "user", "content": 'Héllo {x} "q"\n wörld'}],
            "d73a86d4e2f79ff2ab990fc9d336e04a9fee1ee77ed93b980ea2f997329be228",
        ),
        (
            GEMINI_3,
            None,
            [{"role": "user", "content": "plain"}],
            "9c9a23fddd706a15af009bdc4890905439a9f6699641b4b3ec4d2eb3c6ec3b40",
        ),
        (
            "gemini/gemini-2.5-flash",
            0.2,
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": 'Say é中 "hi"\n'},
            ],
            "62fb7ee971681267d6df173b04d9ee7aa2aa4220bdf9b55f196d9399be35194b",
        ),
    ],
)
def test_the_cache_key_is_byte_for_byte_what_it_always_was(model, temperature, messages, expected):
    assert llm.payload_sha256(model, temperature, messages) == expected


def test_a_cache_entry_stored_before_the_move_is_served_without_a_call(prompt, tmp_path):
    """tests/fixtures/llm_cache/<hash>.json is named by a hash the old code computed for this
    prompt rendered with word=hello, on the model below with no temperature."""

    class Forbidden:
        name = "forbidden"

        def send(self, *args, **kwargs):
            raise AssertionError("a cached payload must not reach the vendor")

    config = {"greeting": {"model": GEMINI_3}, "llm": {}, "logs": {}}
    caller = Client(
        tmp_path / "root",
        config,
        transport=Forbidden(),
        validator=validate,
        schema_loader=greeting_schema,
        cache_dir=FIXTURES / "llm_cache",
        prices=prices(tmp_path),
    )
    reply = caller.complete("greet", prompt, {"word": "hello"}, tag="old-cache")
    assert reply.cached is True and reply.cost_usd == 0.0
    assert reply.parsed == {"greeting": "hello"}
    assert (
        reply.payload_sha256 == "c719d1fc0c2b9a19f6a977762801b088a09907ae545634f6c5cb090464098edb"
    )
    assert caller.calls == 0


# ---- provenance, the ledger, the cache ------------------------------------------------------


def test_a_call_records_full_provenance_and_the_ledger(prompt, tmp_path):
    script(tmp_path, "t0", json.dumps({"greeting": "hello"}))
    reply = client(tmp_path).complete("greet", prompt, {"word": "hello"}, tag="t0")
    assert reply.parsed == {"greeting": "hello"}
    provenance = reply.provenance()
    assert set(provenance) == {
        "prompt_id",
        "prompt_version",
        "prompt_sha256",
        "model",
        "profile",
        "payload_sha256",
        "requested_at",
        "tokens_in",
        "tokens_out",
        "cost_usd",
        "cached",
        "attempts",
    }
    assert provenance["prompt_id"] == "greeting" and provenance["model"] == FLASH_LITE
    assert provenance["tokens_in"] == 100 and provenance["cached"] is False
    assert provenance["profile"] == "gemini-2.5-flash-lite"
    (line,) = usage_lines(tmp_path)
    assert line["kind"] == "api" and line["stage"] == "greet" and line["outcome"] == "ok"
    assert line["cost_usd"] == pytest.approx(100 / 1e6 * 0.10 + 20 / 1e6 * 0.40)
    (log,) = invocation_logs(tmp_path)
    assert "[greet]" in log.read_text() and "Say hello." in log.read_text()


def test_identical_payloads_never_bill_twice(prompt, tmp_path):
    scripts = script(tmp_path, "t0", json.dumps({"greeting": "hello"}))
    caller = client(tmp_path)
    first = caller.complete("greet", prompt, {"word": "hello"}, tag="t0")
    (scripts / "greet" / "t0.json").unlink()  # a second call can only succeed from the cache
    second = caller.complete("greet", prompt, {"word": "hello"}, tag="t0")
    assert second.parsed == first.parsed and second.cached is True
    assert second.payload_sha256 == first.payload_sha256
    assert [line["cached"] for line in usage_lines(tmp_path)] == [False, True]
    assert usage_lines(tmp_path)[1]["cost_usd"] == 0
    assert len(list((tmp_path / "root" / "llm-cache").glob("*.json"))) == 1


def test_the_cache_file_holds_the_documented_fields(prompt, tmp_path):
    script(tmp_path, "t0", json.dumps({"greeting": "hello"}))
    reply = client(tmp_path).complete("greet", prompt, {"word": "hello"}, tag="t0")
    entry = json.loads(
        (tmp_path / "root" / "llm-cache" / f"{reply.payload_sha256}.json").read_text()
    )
    assert list(entry) == [
        "model",
        "temperature",
        "messages",
        "reply",
        "tokens_in",
        "tokens_out",
        "cost_usd",
        "requested_at",
    ]
    assert entry["temperature"] == 0.2
    assert entry["messages"] == [{"role": "user", "content": "Say hello.\n"}]


def test_a_model_that_takes_no_temperature_is_sent_none(prompt, tmp_path):
    script(tmp_path, "t0", json.dumps({"greeting": "hello"}))
    config = {"greeting": {"model": GEMINI_3, "temperature": 0.9}, "llm": {}, "logs": {}}
    reply = client(tmp_path, config=config).complete("greet", prompt, {"word": "hello"}, tag="t0")
    assert reply.payload_sha256 == llm.payload_sha256(
        GEMINI_3, None, [{"role": "user", "content": "Say hello.\n"}]
    )


def test_the_client_keeps_everything_under_the_root_it_is_given(prompt, tmp_path):
    script(tmp_path, "t0", json.dumps({"greeting": "hello"}))
    client(tmp_path).complete("greet", prompt, {"word": "hello"}, tag="t0")
    top = {path.name for path in (tmp_path / "root").iterdir()}
    assert top == {"llm-cache", "logs"}


# ---- strict parsing -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    ["not json at all", json.dumps(["a list"]), json.dumps({"greeting": 3})],
)
def test_a_reply_that_is_not_the_schema_fails_with_the_raw_reply_logged(prompt, tmp_path, reply):
    script(tmp_path, "bad", reply)
    with pytest.raises(llm.BadReply) as error:
        client(tmp_path).complete("greet", prompt, {"word": "x"}, tag="bad")
    assert error.value.attempts == 2 and error.value.raw == reply
    (log,) = invocation_logs(tmp_path)
    assert reply in log.read_text()
    assert [line["outcome"] for line in usage_lines(tmp_path)] == ["bad_reply", "bad_reply"]
    assert list((tmp_path / "root").glob("llm-cache/*.json")) == []  # never cached


def test_a_fenced_reply_parses_and_a_check_failure_is_a_bad_reply_never_cached(prompt, tmp_path):
    script(tmp_path, "fenced", "```json\n" + json.dumps({"greeting": "hi"}) + "\n```")
    caller = client(tmp_path)
    assert caller.complete("greet", prompt, {"word": "a"}, tag="fenced").parsed == {
        "greeting": "hi"
    }

    def wrong(parsed):
        raise ValueError("wrong for this call")

    script(tmp_path, "checked", json.dumps({"greeting": "hi"}))
    with pytest.raises(llm.BadReply, match="wrong for this call"):
        caller.complete("greet", prompt, {"word": "b"}, tag="checked", check=wrong)
    assert len(list((tmp_path / "root" / "llm-cache").glob("*.json"))) == 1


def test_the_scripted_transport_names_the_missing_script(prompt, tmp_path):
    with pytest.raises(llm.NotConfigured, match="no reply at"):
        client(tmp_path, scripts=tmp_path / "nowhere").complete(
            "greet", prompt, {"word": "x"}, tag="absent"
        )


def test_a_section_or_model_that_is_absent_is_not_configured(prompt, tmp_path):
    with pytest.raises(llm.NotConfigured, match=r"\[greeting\] is absent"):
        client(tmp_path, config={"llm": {}, "logs": {}}).complete(
            "greet", prompt, {"word": "x"}, tag="t"
        )
    with pytest.raises(llm.NotConfigured, match=r"\[greeting\].model is not set"):
        client(tmp_path, config={"greeting": {"temperature": 0.1}}).complete(
            "greet", prompt, {"word": "x"}, tag="t"
        )


def test_a_real_transport_with_no_vendor_key_declines_before_any_call(
    prompt, tmp_path, monkeypatch
):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    caller = Client(
        tmp_path / "root",
        CONFIG,
        transport=llm.LiteLLMTransport(
            completion=lambda **keyword_arguments: pytest.fail("called the vendor")
        ),
        validator=validate,
        schema_loader=greeting_schema,
        prices=prices(tmp_path),
    )
    with pytest.raises(llm.NotConfigured, match="GEMINI_API_KEY is not set"):
        caller.complete("greet", prompt, {"word": "x"}, tag="t")


def test_the_default_transport_is_the_vendor_unless_scripts_are_set(tmp_path, monkeypatch):
    assert llm.SCRIPTS is None
    assert llm.default_transport().name == "litellm"
    monkeypatch.setattr(llm, "SCRIPTS", tmp_path)
    assert llm.default_transport().name == "scripted"


# ---- the invocation's log -------------------------------------------------------------------


def test_a_verb_outside_a_run_gets_its_own_log_file_never_overwritten(prompt, tmp_path):
    script(tmp_path, "t0", json.dumps({"greeting": "hi"}))
    for word in ("first", "second"):
        client(tmp_path, verb="dev-gate").complete("greet", prompt, {"word": word}, tag="t0")
    logs = invocation_logs(tmp_path)
    assert len(logs) == 2
    for log in logs:
        assert re.fullmatch(r"\d{8}T\d{6}Z-dev-gate-[0-9a-f]{6}\.log", log.name), log.name
    texts = [log.read_text() for log in logs]
    assert sorted("Say first." in text for text in texts) == [False, True]
    assert sorted("Say second." in text for text in texts) == [False, True]


def test_a_run_points_the_client_at_its_own_log_file(prompt, tmp_path):
    script(tmp_path, "t0", json.dumps({"greeting": "hi"}))
    caller = client(tmp_path)
    caller.log_path = tmp_path / "run.log"
    caller.log_path.write_text("")
    caller.complete("greet", prompt, {"word": "x"}, tag="t0")
    assert "[greet]" in caller.log_path.read_text()
    assert invocation_logs(tmp_path) == []


def test_a_client_that_never_calls_a_model_writes_no_log(tmp_path):
    client(tmp_path, verb="status")
    assert invocation_logs(tmp_path) == []


# ---- the spend caps -------------------------------------------------------------------------


def two_expensive_scripts(tmp_path):
    for script_number in range(2):
        script(
            tmp_path,
            f"t{script_number}",
            json.dumps({"greeting": "hi"}),
            tokens=(2_000_000, 2_000_000),
        )


def test_the_run_cap_stops_after_the_call_that_crossed_it(prompt, tmp_path):
    two_expensive_scripts(tmp_path)
    config = {**CONFIG, "llm": {"max_usd_per_run": 0.5}}
    caller = client(tmp_path, config=config)
    caller.complete("greet", prompt, {"word": "a"}, tag="t0")
    assert caller.spent_usd > 0.5
    with pytest.raises(llm.SpendCapReached) as error:
        caller.complete("greet", prompt, {"word": "b"}, tag="t1")
    assert error.value.cap == llm.RUN_CAP and "per-run" in str(error.value)
    assert len(usage_lines(tmp_path)) == 1  # the declined call never happened


def write_usage(tmp_path, usd, *, days_ago=0, kind="api", cached=False):
    line = {
        "at": clock.iso(clock.utc_now() - timedelta(days=days_ago)),
        "stage": "greet",
        "prompt_id": "greeting",
        "model": FLASH_LITE,
        "tokens_in": 1,
        "tokens_out": 1,
        "cost_usd": usd,
        "kind": kind,
        "cached": cached,
        "tag": "earlier",
        "outcome": "ok",
    }
    path = tmp_path / "root" / "logs" / "usage.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as usage_file:
        usage_file.write(json.dumps(line) + "\n")


def test_the_day_cap_counts_todays_billed_lines_and_names_itself(prompt, tmp_path):
    script(tmp_path, "t0", json.dumps({"greeting": "hi"}))
    write_usage(tmp_path, 6.0)
    write_usage(tmp_path, 4.0)
    write_usage(tmp_path, 50.0, days_ago=1)  # yesterday does not count
    write_usage(tmp_path, 50.0, kind="chat")  # a chat-road line is never summed
    write_usage(tmp_path, 50.0, cached=True)  # a cache hit was not billed
    config = {**CONFIG, "llm": {"max_usd_per_day": 10.0}}
    with pytest.raises(llm.SpendCapReached) as error:
        client(tmp_path, config=config).complete("greet", prompt, {"word": "x"}, tag="t0")
    assert error.value.cap == llm.DAY_CAP and "per-day" in str(error.value)


def test_yesterdays_spend_does_not_count_today(prompt, tmp_path):
    script(tmp_path, "t0", json.dumps({"greeting": "hi"}))
    write_usage(tmp_path, 50.0, days_ago=1)
    config = {**CONFIG, "llm": {"max_usd_per_day": 10.0}}
    caller = client(tmp_path, config=config)
    assert caller.complete("greet", prompt, {"word": "x"}, tag="t0").cached is False
    assert caller.spent_today() == pytest.approx(caller.spent_usd)


def test_a_day_cap_without_the_usage_ledger_declines_to_spend(prompt, tmp_path):
    script(tmp_path, "t0", json.dumps({"greeting": "hi"}))
    config = {**CONFIG, "llm": {"max_usd_per_day": 10.0}, "logs": {"usage": False}}
    with pytest.raises(llm.SpendCapReached, match=r"\[logs\]\.usage") as error:
        client(tmp_path, config=config).complete("greet", prompt, {"word": "x"}, tag="t0")
    assert error.value.cap == llm.DAY_CAP


def test_an_unpriced_model_under_a_cap_stops_instead_of_guessing(prompt, tmp_path):
    script(tmp_path, "t0", json.dumps({"greeting": "hi"}))
    config = {
        "greeting": {"model": "gemini/gemini-no-price-yet"},
        "llm": {"max_usd_per_run": 1.0},
        "logs": {},
    }
    with pytest.raises(llm.SpendCapReached, match="no price in the table"):
        client(tmp_path, config=config).complete("greet", prompt, {"word": "x"}, tag="t0")


def test_an_unknown_models_cost_is_null_never_an_estimate(tmp_path):
    table = prices(tmp_path)
    assert llm.price_usd("gemini/gemini-no-price-yet", 1000, 1000, prices=table) is None
    assert llm.price_usd(FLASH_LITE, 1_000_000, 1_000_000, prices=table) == 0.5
    assert llm.price_usd("jev-latest", 1_000_000, 5, prices=table) == 0.042  # an override


def test_a_transport_with_no_price_is_priced_from_the_cache(prompt, tmp_path):
    """A vendor reply litellm cannot price is priced from the cache, so the ledger line and the
    caps see a figure; a model neither knows stays null."""

    class Unpriced:
        name = "unpriced"

        def send(self, model, messages, params, *, tag):
            return llm.RawReply('{"greeting": "hi"}', 1_000_000, 0, None)

    caller = Client(
        tmp_path / "root", CONFIG, transport=Unpriced(), validator=validate, prices=prices(tmp_path)
    )
    assert caller.complete("greet", prompt, {"word": "x"}, tag="t").cost_usd == 0.1
    (line,) = usage_lines(tmp_path)
    assert line["cost_usd"] == 0.1


# ---- the estimate ---------------------------------------------------------------------------


def test_an_estimate_prices_the_payloads_at_the_configured_model_and_cached_ones_are_free(
    prompt, tmp_path
):
    script(tmp_path, "t0", json.dumps({"greeting": "hi"}))
    caller = client(tmp_path)
    caller.complete("greet", prompt, {"word": "hello"}, tag="t0")  # now cached
    counted = []

    def count(model, messages):
        counted.append(model)
        return 1_000_000

    import llmlm.estimate as estimate_mod

    cheap = caller.estimate(prompt, [{"word": "hello"}, {"word": "new"}], expected_output_tokens=0)
    assert cheap.calls == 2 and cheap.cached == 1 and cheap.model == FLASH_LITE
    # the same payloads at a dearer model cost more: the estimate follows the configured model
    dear = Client(
        tmp_path / "root",
        {**CONFIG, "greeting": {"model": GEMINI_3}},
        transport=llm.ScriptedTransport(tmp_path / "scripts"),
        validator=validate,
        prices=prices(tmp_path),
    )
    exact = estimate_mod.estimate(
        GEMINI_3,
        [([{"role": "user", "content": "Say new.\n"}], "")],
        expected_output_tokens=1_000_000,
        prices=prices(tmp_path),
        count_tokens=count,
    )
    assert exact.tokens_in == 1_000_000 and exact.cost_usd == pytest.approx(0.25 + 1.5)
    assert counted == [GEMINI_3] and exact.tokenizer == "injected"
    assert dear.estimate(prompt, [{"word": "new"}]).cost_usd > cheap.cost_usd


def test_an_estimate_for_an_unknown_model_counts_tokens_but_prices_null(tmp_path):
    import llmlm.estimate as estimate_mod

    out = estimate_mod.estimate(
        "vendor/unknown",
        [([{"role": "user", "content": "x" * 400}], None)],
        expected_output_tokens=10,
        prices=prices(tmp_path),
        count_tokens=estimate_mod.approximate_tokens,
    )
    assert out.cost_usd is None and out.tokens_in == 100 and out.tokens_out == 10


# ---- replay ---------------------------------------------------------------------------------


def test_replay_serves_a_cached_reply_without_the_payload_and_names_a_miss(prompt, tmp_path):
    script(tmp_path, "t0", json.dumps({"greeting": "hi"}))
    caller = client(tmp_path)
    first = caller.complete("greet", prompt, {"word": "x"}, tag="t0")
    replayed = caller.replay("greet", prompt, first.payload_sha256, tag="t0")
    assert replayed.parsed == first.parsed and replayed.cached is True
    assert replayed.profile == "replay" and replayed.model == FLASH_LITE
    with pytest.raises(llm.CacheMiss, match="no cached reply for payload 0123456789ab"):
        caller.replay("greet", prompt, "0123456789abcdef", tag="t0")
    with pytest.raises(llm.CacheMiss, match="payload \\?"):
        caller.replay("greet", prompt, "", tag="t0")


def test_payload_hash_is_the_cache_key_of_the_call_before_it_is_made(prompt, tmp_path):
    caller = client(tmp_path)
    key = caller.payload_hash(prompt, {"word": "hello"})
    messages = [{"role": "user", "content": prompt.render({"word": "hello"})}]
    assert key == llm.payload_sha256(FLASH_LITE, 0.2, messages)


def test_a_sections_max_tokens_reaches_the_transport_and_not_the_cache_key(prompt, tmp_path):
    seen = []

    class Recorder:
        name = "recorder"

        def send(self, model, messages, params, *, tag):
            seen.append(params)
            return llm.RawReply('{"greeting": "hi"}', 1, 1, 0.0)

    config = {
        "greeting": {"model": FLASH_LITE, "temperature": 0.2, "max_tokens": 8000},
        "llm": {},
        "logs": {},
    }
    capped = client(tmp_path, config=config)
    capped.transport = Recorder()
    capped.complete("greeting", prompt, {"word": "x"}, tag="t")
    assert seen[0]["max_tokens"] == 8000
    plain = client(tmp_path)
    assert plain.payload_hash(prompt, {"word": "x"}) == capped.payload_hash(prompt, {"word": "x"})
