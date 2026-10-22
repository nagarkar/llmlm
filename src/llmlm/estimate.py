"""Estimate: what a run would cost before it spends, from the payloads it would send.

`estimate(model, payloads, ...)` renders nothing itself: the caller passes the messages each call
would send (a tool renders its own prompts). Each payload is counted with litellm's tokenizer for
that model (`count_tokens`, injectable; without litellm the count is approximate and the
estimate says so), the output size is what the prompt's front matter declares
(`expected_output_tokens`), and both are priced at the model's cached rates
(`llmlm.prices`). A payload already in the reply cache costs nothing. An unknown model's
cost is None, never a guess: the estimate still counts its tokens.

This replaces per-tool constants (a dollars-per-video figure) that ignored the configured model.

Never imports: llm, ledger, usage.

Terms:
- **estimate**: calls, cached calls, tokens in and out, and the cost at the configured model, for
  a run that has not happened.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .prices import Prices

APPROXIMATE_CHARS_PER_TOKEN = 4


@dataclass(frozen=True, slots=True)
class Estimate:
    model: str
    calls: int
    cached: int
    tokens_in: int
    tokens_out: int
    cost_usd: float | None
    tokenizer: str  # "litellm" or "approximate"

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "calls": self.calls,
            "cached": self.cached,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
            "tokenizer": self.tokenizer,
        }


def approximate_tokens(model: str, messages: list[dict]) -> int:
    text = "".join(str(message.get("content", "")) for message in messages)
    return max(1, len(text) // APPROXIMATE_CHARS_PER_TOKEN)


def litellm_tokens(model: str, messages: list[dict]) -> int:
    import litellm

    litellm.suppress_debug_info = True
    return int(litellm.token_counter(model=model, messages=messages))


def default_counter() -> tuple[Callable[[str, list[dict]], int], str]:
    try:
        import litellm  # noqa: F401
    except ImportError:
        return approximate_tokens, "approximate"
    return litellm_tokens, "litellm"


def estimate(
    model: str,
    payloads: Iterable[tuple[list[dict], str | None]],
    *,
    expected_output_tokens: int,
    prices: Prices,
    cache_dir: Path | None = None,
    count_tokens: Callable[[str, list[dict]], int] | None = None,
) -> Estimate:
    """`payloads` are (messages, payload_sha256) pairs; a pair whose hash has a file under
    `cache_dir` is cached and costs nothing. `expected_output_tokens` is per uncached call."""
    counter, tokenizer = (count_tokens, "injected") if count_tokens else default_counter()
    calls = cached = tokens_in = 0
    for messages, payload_hash in payloads:
        calls += 1
        if cache_dir is not None and payload_hash and (cache_dir / f"{payload_hash}.json").exists():
            cached += 1
            continue
        try:
            tokens_in += counter(model, messages)
        except Exception:  # noqa: BLE001 - a tokenizer that lacks the model falls back
            tokens_in += approximate_tokens(model, messages)
            tokenizer = "approximate"
    billed = calls - cached
    tokens_out = billed * int(expected_output_tokens)
    cost = prices.price_usd(model, tokens_in, tokens_out) if billed else 0.0
    return Estimate(model, calls, cached, tokens_in, tokens_out, cost, tokenizer)
