"""The TypeSafe transport (`src/llmlm/typesafe.py`): only a 429 or 529 is retried, and said on
stderr; a 5xx fails at once; a 4xx or a body that is not JSON is a TypeSafeError that never echoes
the request. `post` and `sleep` are injected, so nothing goes out."""

from types import SimpleNamespace

import pytest

from llmlm import llm, typesafe
from llmlm.prices import Prices


def _post(responses, calls):
    def post(url, *, json, headers, timeout):
        calls.append(json)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return post


def _resp(status, body=None, headers=None):
    return SimpleNamespace(status_code=status, headers=headers or {}, json=lambda: body or {})


def test_typesafe_retries_429_and_529_only_and_says_so(capsys):
    calls, sleeps = [], []
    transport = typesafe.TypeSafeTransport(
        post=_post([_resp(429), _resp(529), _resp(200, {"ok": 1})], calls), sleep=sleeps.append
    )
    assert transport.send({"q": 1}, tag="v-amb-0001") == {"ok": 1}
    assert len(calls) == 3 and len(sleeps) == 2
    assert capsys.readouterr().err.count("retry") == 2


@pytest.mark.parametrize("status", [500, 502, 503])
def test_typesafe_fails_a_5xx_at_once(status):
    calls, sleeps = [], []
    transport = typesafe.TypeSafeTransport(post=_post([_resp(status)], calls), sleep=sleeps.append)
    with pytest.raises(llm.VendorError, match=str(status)):
        transport.send({"q": 1}, tag="v-amb-0001")
    assert len(calls) == 1 and sleeps == []


@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_a_4xx_is_a_typesafe_error_naming_the_tag_and_not_the_body(status):
    calls = []
    transport = typesafe.TypeSafeTransport(
        post=_post([_resp(status, {"echo": "the transcript text"})], calls),
        sleep=lambda seconds: None,
    )
    with pytest.raises(typesafe.TypeSafeError) as raised:
        transport.send({"q": 1}, tag="v-amb-0002")
    assert "v-amb-0002" in str(raised.value) and "transcript text" not in str(raised.value)
    assert len(calls) == 1


def test_a_reply_that_is_not_json_is_a_typesafe_error():
    def not_json():
        raise ValueError("no json")

    response = SimpleNamespace(status_code=200, headers={}, json=not_json)
    transport = typesafe.TypeSafeTransport(post=_post([response], []), sleep=lambda seconds: None)
    with pytest.raises(typesafe.TypeSafeError, match="not JSON"):
        transport.send({"q": 1}, tag="v-amb-0003")


def test_the_key_goes_in_the_authorization_header_only(monkeypatch):
    monkeypatch.setenv(typesafe.KEY_ENV, "test-key")
    seen = []

    def post(url, *, json, headers, timeout):
        seen.append((url, json, headers))
        return _resp(200, {"ok": 1})

    typesafe.TypeSafeTransport(post=post).send({"q": 1}, tag="t")
    ((url, body, headers),) = seen
    assert headers["Authorization"] == "Bearer test-key"
    assert "test-key" not in url and "test-key" not in str(body)


def _prices(tmp_path):
    return Prices(tmp_path / "prices.json", fetch=lambda: {})


def test_a_call_costs_its_input_tokens_at_the_jev_row_and_nothing_for_output(tmp_path):
    reply = {"usage": {"input_tokens": 1_000_000, "output_tokens": 500}}
    assert typesafe.cost_usd("jev-latest", reply, prices=_prices(tmp_path)) == 0.042
    assert typesafe.cost_usd("jev-unpriced", reply, prices=_prices(tmp_path)) is None


def test_the_ledger_line_is_an_arbiter_api_line_and_a_cached_answer_is_free(tmp_path):
    reply = {"model": "jev-1.13.0", "usage": {"input_tokens": 770, "output_tokens": 39}}
    line = typesafe.usage_line(
        "jev-latest", reply, "v-amb-1", prompt_id="arbiter", prices=_prices(tmp_path)
    )
    assert line == {
        "stage": "arbiter",
        "prompt_id": "arbiter",
        "model": "jev-1.13.0",
        "tokens_in": 770,
        "tokens_out": 39,
        "cost_usd": round(770 / 1e6 * 0.042, 6),
        "kind": "api",
        "cached": False,
        "tag": "v-amb-1",
        "outcome": "ok",
    }
    cached = typesafe.usage_line(
        "jev-latest", reply, "v-amb-1", prompt_id="arbiter", cached=True, prices=_prices(tmp_path)
    )
    assert cached["cost_usd"] == 0.0 and cached["cached"] is True
