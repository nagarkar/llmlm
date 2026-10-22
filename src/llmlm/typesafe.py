"""TypeSafe: the one HTTP client for Jev, TypeSafe's System One model (`llmlm[typesafe]`).

`TypeSafeTransport.send` is one POST per question. Only a 429 or 529 is retried (at most
`llm.MAX_ATTEMPTS` attempts, honouring `retry-after`), every retry says so on stderr, and any
other failure fails the step: a 5xx is a `VendorError`; a 4xx, or a body that is not JSON, is a
`TypeSafeError` that never echoes the request. `post` and `sleep` are injectable, so the rule is
tested offline. The key is read from `TYPESAFE_API_KEY`; a tool loads its own `.env` first.

TypeSafe bills input tokens only, so a call's ledger line (`usage_line`) carries the cost of the
reply's `input_tokens` at the `jev-latest` row of the price cache (`llmlm.prices`, an override with
its source and date), never an estimate: a model with no price is a null cost.

Never imports: usage, estimate. Needs the `typesafe` extra (`httpx`), imported at the first call.

Terms:
- **Jev**: TypeSafe's System One model, named by the alias `jev-latest` to follow its newest
  release.
"""

from __future__ import annotations

import os
import time

from . import llm
from .prices import Prices
from .prices import default as default_prices

KEY_ENV = "TYPESAFE_API_KEY"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
STAGE = "arbiter"


class TypeSafeError(RuntimeError):
    """TypeSafe answered, but not with a usable reply (a 4xx, or a body that is not JSON). A
    tool turns it into its own error for the item."""


class TypeSafeTransport:
    name = "typesafe"

    def __init__(self, *, post=None, sleep=time.sleep):
        self._post = post
        self._sleep = sleep

    def send(self, body: dict, *, tag: str) -> dict:
        post = self._post
        network_error: type[BaseException] = OSError
        if post is None:
            import httpx

            post, network_error = httpx.post, httpx.HTTPError
        headers = {
            "Authorization": f"Bearer {os.environ.get(KEY_ENV, '')}",
            "Content-Type": "application/json",
        }
        for attempt in range(1, llm.MAX_ATTEMPTS + 1):
            try:
                response = post(ENDPOINT, json=body, headers=headers, timeout=llm.REQUEST_TIMEOUT_S)
            except network_error as error:
                raise llm.VendorError(f"typesafe: {type(error).__name__} for {tag}") from error
            if response.status_code in llm.RETRY_STATUSES:
                if attempt == llm.MAX_ATTEMPTS:
                    raise llm.VendorError(f"typesafe: HTTP {response.status_code} for {tag}")
                wait_header = str(response.headers.get("retry-after", ""))
                wait = (
                    float(wait_header)
                    if wait_header.replace(".", "", 1).isdigit()
                    else llm.RETRY_SLEEP_S * attempt
                )
                llm.say_retry("typesafe", response.status_code, tag, attempt, wait)
                self._sleep(wait)
                continue
            if response.status_code >= 500:
                raise llm.VendorError(f"typesafe: HTTP {response.status_code} for {tag}")
            if response.status_code != 200:
                # the body is not echoed: it may carry the request
                raise TypeSafeError(f"typesafe: HTTP {response.status_code} for {tag}")
            try:
                return response.json()
            except ValueError as error:
                raise TypeSafeError(f"typesafe: reply for {tag} is not JSON") from error
        raise llm.VendorError(f"typesafe: no reply for {tag}")


def cost_usd(model: str, reply: dict, *, prices: Prices | None = None) -> float | None:
    """The cost of a reply: its `usage.input_tokens` at the model's input rate (TypeSafe bills no
    output); null when the price cache has no row for the model."""
    tokens_in = int((reply.get("usage") or {}).get("input_tokens", 0) or 0)
    return (prices or default_prices()).price_usd(model, tokens_in, 0)


def usage_line(
    model: str,
    reply: dict,
    tag: str,
    *,
    prompt_id: str,
    cached: bool = False,
    outcome: str = "ok",
    prices: Prices | None = None,
) -> dict:
    """The `usage.jsonl` event of one TypeSafe call (`llmlm.ledger.Ledger.usage`), stage
    `arbiter`, kind `api`; a cached answer costs nothing."""
    usage = reply.get("usage") or {}
    return {
        "stage": STAGE,
        "prompt_id": prompt_id,
        "model": str(reply.get("model") or model),
        "tokens_in": int(usage.get("input_tokens", 0) or 0),
        "tokens_out": int(usage.get("output_tokens", 0) or 0),
        "cost_usd": 0.0 if cached else cost_usd(model, reply, prices=prices),
        "kind": "api",
        "cached": cached,
        "tag": tag,
        "outcome": outcome,
    }
