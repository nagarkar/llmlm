"""LLM: the model transport, reply cache, usage ledger lines and spend caps.

LiteLLM runs in-process; the vendor is the model string's prefix and reads its own key from the
environment. Importing this module does not import litellm until a real call is made.

Every call: renders a prompt (`llmlm.prompts`), consults the cache
(`<root>/llm-cache/<sha256>.json`, keyed by model, temperature and messages), calls the vendor
only on a miss, parses the reply strictly against the prompt's declared schema (two attempts,
then BadReply with the raw reply in the invocation's log), writes one `usage.jsonl` line
(`llmlm.ledger`), and returns a Reply whose `provenance()` carries prompt id and sha, model,
profile, payload sha, time, tokens, cost and whether it was cached. The spend cap sums cost
across the run and raises SpendCapReached BEFORE the call that would exceed
`[llm].max_usd_per_run`, or `[llm].max_usd_per_day` counting today's (UTC) billed lines already
in `<root>/logs/usage.jsonl`.

`Client(root, config, ...)` keeps everything it writes under `root`: the cache, `logs/` and the
invocation's log file. The caller passes the directory; the client knows no workspace. The
caller also passes the `validator` for its prompts' output schemas and the `schema_loader` that
hands a model honouring `json_schema` the declared schema; without a validator a reply is
checked only for being one JSON object, and without a loader such a model gets `json_object`.
A reply's cost is the transport's own figure when it has one, else the price cache's
(`llmlm.prices`); an unknown model's cost is null, never an estimate. `Client.estimate` prices
a run before it spends (`llmlm.estimate`).

`MODEL_PROFILES` records, per verified model, what a request may carry; the request is shaped
from it BEFORE sending and never changed after. A model whose row says `json_schema` gets the
prompt's declared reply schema as `response_format` (the vendor constrains the reply to it);
otherwise a `json_mode` row gets `json_object` (any JSON object). A model with no row gets the
plain request (no temperature, no JSON mode) and `unprofiled_models` names it. A `temperature`
configured for a model that takes none is a config error (`check_temperatures`). A vendor that
rejects a parameter anyway fails the call with a VendorError naming the row to add. Only a 429
or 529 is retried, at most MAX_ATTEMPTS times with a growing pause, and every retry says so on
stderr; any other failure fails the step. An unknown model's cost is null, never an estimate.

Every payload and reply is written, tagged with its stage, into the one log file of the
invocation: the file `log_path` names, else this client's own
`logs/runs/<UTC stamp>-<verb>-<short id>.log`, made at its first block and never overwritten.
The scripted transport serves replies from `<dir>/<stage>/<tag>.json` so a pipeline runs offline.

The cache key is `payload_sha256(model, temperature, messages)` and must never change: stored
replies from earlier tools stay valid only while it is byte for byte what it was
(`tests/test_llm.py::test_the_cache_key_is_byte_for_byte_what_it_always_was`).

Never imports: usage.

A tool that builds its own messages (content blocks with `cache_control` breakpoints, authorlm's
drafting) calls `Client.complete_messages` instead: the same cache (keyed by `messages_sha256`),
caps, price and ledger, with `max_tokens`, `reasoning_effort` and `thinking_budget` as call knobs,
cache reads and writes priced and recorded, and `Refused` for a model that declines.

Terms:
- **payload**: the rendered messages of one model call; its sha256 with the model and
  temperature is the cache key.
- **spend cap**: `[llm].max_usd_per_run` (per run) or `[llm].max_usd_per_day` (per day) in a
  tool's settings: the client raises `SpendCapReached` before a call that would exceed it. A source
  it stops is recorded failed with a reason that starts `budget` (the stored name).
- **provenance**: what an artifact records about how it was made: prompt, model, cost, time.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import UTC
from pathlib import Path
from typing import Any, Protocol

from . import clock as clock_mod
from . import prices as prices_mod
from .estimate import Estimate, estimate
from .files import write_atomic
from .ledger import Ledger
from .ledger import read as ledger_read
from .prompts import Prompt

CACHE_DIR = "llm-cache"
LOGS_DIR = "logs"
RUN_LOGS_DIR = "runs"  # <root>/logs/runs/: one log file per invocation, never overwritten
REQUEST_TIMEOUT_S = 180  # a batch is a few thousand tokens; never wait forever
RETRY_SLEEP_S = 5  # the first pause; attempt n waits n x this
MAX_ATTEMPTS = 3  # one call plus two retries, and only for RETRY_STATUSES
RETRY_STATUSES = (429, 529)  # rate limited, overloaded; nothing else is retried


class BadReply(RuntimeError):
    """The model's reply did not parse or did not match the schema, twice."""

    def __init__(self, message: str, attempts: int, raw: str):
        super().__init__(message)
        self.attempts = attempts
        self.raw = raw


class VendorError(RuntimeError):
    """The vendor or the network failed (timeout, 5xx, a rate limit that
    outlasted the retries): the step fails, the stage records it, and the run
    goes on to the next video."""


class Refused(RuntimeError):
    """The model declined to answer (Anthropic `stop_reason: refusal`, which litellm maps to
    `finish_reason: content_filter`). An HTTP 200, so it must never pass for an empty reply."""


class TruncatedReply(RuntimeError):
    """The vendor cut the reply at the token limit; a failure, never a partial artifact."""


RUN_CAP = "[llm].max_usd_per_run"
DAY_CAP = "[llm].max_usd_per_day"


class SpendCapReached(RuntimeError):
    """A spend cap was reached; no further calls. `cap` names which one
    (RUN_CAP or DAY_CAP), so the board can say."""

    def __init__(self, message: str, cap_name: str = RUN_CAP):
        super().__init__(message)
        self.cap = cap_name


class CacheMiss(RuntimeError):
    """A replay asked for a payload the cache no longer holds."""


class NotConfigured(RuntimeError):
    pass


# ---- model profiles (authorlm's table pattern) ------------------------

ANY = "any"
VENDOR_DEFAULT_ONLY = "vendor-default-only"


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    temperature: str  # ANY | VENDOR_DEFAULT_ONLY
    json_mode: bool  # vendor honours response_format json_object
    # vendor honours response_format json_schema: the reply is constrained to the prompt's
    # declared schema at the vendor, not only checked after (verified live per model)
    json_schema: bool = False
    # `reasoning_effort` may be sent (False for a model that 400s on it)
    reasoning_effort: bool = True
    # "anthropic": block-level `cache_control` is copied into the provider request, so attaching
    # breakpoints is worth it; None: nothing to attach (a vendor that caches server-side still does)
    prompt_cache: str | None = None


CLAUDE_5 = ModelProfile("claude-5", VENDOR_DEFAULT_ONLY, False, prompt_cache="anthropic")

# Rows only for contracts verified live or by litellm's own model map.
# Longest matching prefix wins; a model with no row gets UNKNOWN_PROFILE.
MODEL_PROFILES: tuple[tuple[str, ModelProfile], ...] = (
    # One row per model actually used or verified (never a wildcard).
    ("gemini/gemini-2.5-flash-lite", ModelProfile("gemini-2.5-flash-lite", ANY, True)),
    ("gemini/gemini-2.5-flash", ModelProfile("gemini-2.5-flash", ANY, True)),
    # Verified live 2026-09-21: Gemini 3 warns that temperature is slated for
    # removal; the request omits it (config cannot force a parameter
    # the vendor is retiring).
    # Verified live 2026-10-17 (nagarkar/ytlm#123): every reply schema the tools declare is
    # accepted as json_schema, and a prompt that asks for the wrong shape still gets the
    # declared one (with json_object it got the wrong shape).
    (
        "gemini/gemini-3.1-flash-lite",
        ModelProfile("gemini-3.1-flash-lite", VENDOR_DEFAULT_ONLY, True, json_schema=True),
    ),
    # gamelm's judge and critic and supplylm's author and propose run on these; reasoning models
    # take no temperature (verified live in gamelm and supplylm). Verified live 2026-10-22
    # (gamelm's OpenAI roles onto llmlm): gpt-5.5 and gpt-5.6-sol accept a strict json_schema and
    # `reasoning_effort`, and 400 on any temperature while reasoning is on.
    ("openai/gpt-5.5", ModelProfile("gpt-5.5", VENDOR_DEFAULT_ONLY, True, json_schema=True)),
    ("openai/gpt-5.6", ModelProfile("gpt-5.6", VENDOR_DEFAULT_ONLY, True)),
    (
        "openai/gpt-5.6-sol",
        ModelProfile("gpt-5.6-sol", VENDOR_DEFAULT_ONLY, True, json_schema=True),
    ),
    # authorlm's rows (nagarkar/llmlm#1, step 7). The Claude 5 family removed sampling parameters:
    # litellm's map says supports_sampling_params is false for these, and its client raises rather
    # than send one. Enumerated, never globbed: claude-haiku-4-5 still takes a temperature.
    ("anthropic/claude-sonnet-5", CLAUDE_5),
    ("anthropic/claude-fable-5", CLAUDE_5),
    ("anthropic/claude-opus-5", CLAUDE_5),
    (
        "anthropic/claude-haiku-4-5",
        ModelProfile("claude-haiku-4-5", ANY, False, prompt_cache="anthropic"),
    ),
    # Proven live 2026-08-30 (authorlm): 400s server-side on any temperature but 1. OpenAI caches
    # prefixes server-side, so there is no breakpoint to attach.
    ("openai/gpt-5.6-luna", ModelProfile("gpt-5.6-luna", VENDOR_DEFAULT_ONLY, False)),
)
# A model with no row gets the plain request (metalm coding.md, "LLM
# calls"): no temperature and no JSON mode, nothing the vendor may reject.
UNKNOWN_PROFILE = ModelProfile("unknown", VENDOR_DEFAULT_ONLY, False)


def model_profile(model: str) -> ModelProfile:
    for name, profile in MODEL_PROFILES:
        if model == name:
            return profile
    return UNKNOWN_PROFILE


def unprofiled_models(config: dict, sections: Iterable[str]) -> list[str]:
    """One warning per configuration section in `sections` (the sections whose `model` is
    called through this module; the consumer names them) whose model has no MODEL_PROFILES
    row."""
    out = []
    for section in sections:
        model = (config.get(section) or {}).get("model")
        if model and model_profile(model) is UNKNOWN_PROFILE:
            out.append(
                f"[{section}].model {model} has no MODEL_PROFILES row (llmlm/llm.py); "
                "it gets the plain request (no temperature, no JSON mode)"
            )
    return out


def check_temperatures(config: dict, sections: Iterable[str]) -> None:
    """A `temperature` set for a model that takes none (its row says
    VENDOR_DEFAULT_ONLY, or it has no row) is a config error naming the key:
    the request would silently not carry it."""
    for section in sections:
        values = config.get(section) or {}
        if "temperature" not in values or "model" not in values:
            continue
        model = values["model"]
        profile = model_profile(model)
        if profile.temperature == VENDOR_DEFAULT_ONLY:
            why = (
                "has no MODEL_PROFILES row, so it gets the plain request"
                if profile is UNKNOWN_PROFILE
                else "takes no temperature (its MODEL_PROFILES row says vendor-default-only)"
            )
            raise ValueError(
                f"config.toml: [{section}].temperature is set, but {model} {why}; "
                f"remove [{section}].temperature from config.toml"
            )


def rejected_params(error: BaseException, params: dict) -> list[str]:
    """The shaped parameters a vendor error names (temperature, JSON mode)."""
    text = str(error).lower()
    return [
        parameter_name
        for parameter_name in ("temperature", "response_format")
        if parameter_name in params and parameter_name in text
    ]


def profile_row(model: str, params: dict, rejected: list[str]) -> str:
    """The MODEL_PROFILES row that would have shaped this request right."""
    temperature = (
        "ANY"
        if "temperature" in params and "temperature" not in rejected
        else "VENDOR_DEFAULT_ONLY"
    )
    response_format = params.get("response_format") or {}
    if response_format.get("type") == "json_schema":
        # json_schema rejected: try json_object on the next row; kept, it stays on
        if "response_format" in rejected:
            return f'("{model}", ModelProfile("{model.split("/", 1)[-1]}", {temperature}, True))'
        name = model.split("/", 1)[-1]
        return f'("{model}", ModelProfile("{name}", {temperature}, True, json_schema=True))'
    json_mode = "response_format" in params and "response_format" not in rejected
    name = model.split("/", 1)[-1]
    return f'("{model}", ModelProfile("{name}", {temperature}, {json_mode}))'


def retry_status(error: BaseException) -> int | None:
    """429 or 529 when `error` is a rate limit or an overload, else None.
    Read from the exception's status code, then from its text (vendors
    wrap these differently)."""
    code = getattr(error, "status_code", None)
    if code in RETRY_STATUSES:
        return int(code)
    text = str(error).lower()
    if "429" in text or "rate limit" in text or "resource_exhausted" in text:
        return 429
    if "529" in text or "overloaded" in text:
        return 529
    return None


def say_retry(service: str, status: int, tag: str, attempt: int, wait: float) -> None:
    """Every retry says so on stderr (metalm operations.md)."""
    print(
        f"retry: {service} answered {status} for {tag}; attempt {attempt + 1} of "
        f"{MAX_ATTEMPTS} in {wait:g}s",
        file=sys.stderr,
    )


# ---- transports ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RawReply:
    """`tokens_in` is net of the cache reads and writes (litellm folds them into prompt_tokens)."""

    text: str
    tokens_in: int
    tokens_out: int
    cost_usd: float | None
    cache_read: int = 0
    cache_write: int = 0
    finish_reason: str = "stop"


class Transport(Protocol):
    name: str

    def send(self, model: str, messages: list[dict], params: dict, *, tag: str) -> RawReply: ...


def _cache_fields(cache_read: int, cache_write: int) -> dict:
    """The ledger line's optional cache counters, present only when the vendor reported some."""
    return {
        name: value
        for name, value in (("cache_read", cache_read), ("cache_write", cache_write))
        if value
    }


def _usage_int(holder: Any, name: str) -> int:
    """One usage counter off litellm's object (or a plain dict), 0 when absent: every provider
    fills a different subset."""
    if holder is None:
        return 0
    value = holder.get(name) if isinstance(holder, dict) else getattr(holder, name, 0)
    return int(value or 0)


class LiteLLMTransport:
    """The vendor adapter. `completion`, `cost` and `sleep` are injectable so
    the retry rule is tested offline; by default they are litellm's."""

    name = "litellm"

    def __init__(self, *, completion=None, cost=None, sleep=time.sleep):
        self._completion = completion
        self._cost = cost
        self._sleep = sleep

    def send(self, model: str, messages: list[dict], params: dict, *, tag: str) -> RawReply:
        completion, cost_of = self._completion, self._cost
        if completion is None or cost_of is None:
            import litellm

            litellm.suppress_debug_info = True
            completion = completion or litellm.completion
            cost_of = cost_of or (
                lambda response: litellm.completion_cost(completion_response=response)
            )
        params = {"timeout": REQUEST_TIMEOUT_S, **params}
        response: Any = None
        attempt = 1
        while attempt <= MAX_ATTEMPTS:
            try:
                response = completion(model=model, messages=messages, **params)
                break
            except Exception as error:  # noqa: BLE001 - vendor errors are many; classify by status
                text = str(error).splitlines()[0][:200] if str(error) else type(error).__name__
                status = retry_status(error)
                if status is not None and attempt < MAX_ATTEMPTS:
                    wait = RETRY_SLEEP_S * attempt
                    say_retry(model, status, tag, attempt, wait)
                    self._sleep(wait)
                    attempt += 1
                    continue
                rejected = rejected_params(error, params)
                if rejected:
                    # The request is never re-shaped after the vendor answers; the row is wrong.
                    raise VendorError(
                        f"{model} rejected {', '.join(rejected)} ({text}); the request "
                        f"was shaped from a wrong MODEL_PROFILES row. Add or correct the "
                        f"row in llmlm/llm.py: {profile_row(model, params, rejected)}"
                    ) from error
                raise VendorError(f"{model}: {text}") from error
        if response is None:
            raise VendorError(f"{model}: no reply after {MAX_ATTEMPTS} attempts")
        usage = getattr(response, "usage", None)
        finish = getattr(response.choices[0], "finish_reason", None)
        if finish == "length":
            # A reply cut off by the token limit is a failure, never a partial artifact.
            raise TruncatedReply("reply cut off by the token limit (finish_reason=length)")
        if finish == "content_filter":
            raise Refused(f"{model} refused to answer (finish_reason=content_filter)")
        try:
            cost = float(cost_of(response))
        except Exception:  # noqa: BLE001 - unknown price table → null, never an estimate
            cost = None
        details = getattr(usage, "prompt_tokens_details", None)
        cache_read = _usage_int(details, "cached_tokens")
        cache_write = _usage_int(details, "cache_creation_tokens")
        return RawReply(
            text=response.choices[0].message.content or "",
            tokens_in=max(0, _usage_int(usage, "prompt_tokens") - cache_read - cache_write),
            tokens_out=_usage_int(usage, "completion_tokens"),
            cost_usd=cost,
            cache_read=cache_read,
            cache_write=cache_write,
            finish_reason=finish or "stop",
        )


class ScriptedTransport:
    """Scripted tier: `<dir>/<stage>/<tag>.json` = {"reply": str, "tokens_in", "tokens_out"}.
    Cost is computed from the price cache so the spend cap is exercised."""

    name = "scripted"

    def __init__(self, directory: Path, *, prices: prices_mod.Prices | None = None):
        self.directory = Path(directory)
        self.prices = prices

    def send(self, model: str, messages: list[dict], params: dict, *, tag: str) -> RawReply:
        stage = params.get("_stage", "")
        path = self.directory / stage / f"{tag}.json"
        if not path.exists():
            raise NotConfigured(f"scripted transport: no reply at {path}")
        body = json.loads(path.read_text(encoding="utf-8"))
        tokens_in = int(body.get("tokens_in", 0))
        tokens_out = int(body.get("tokens_out", 0))
        cache_read = int(body.get("cache_read", 0))
        cache_write = int(body.get("cache_write", 0))
        prices = self.prices or prices_mod.default()
        return RawReply(
            text=body["reply"],
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=prices.price_usd(model, tokens_in, tokens_out, cache_read, cache_write),
            cache_read=cache_read,
            cache_write=cache_write,
            finish_reason=str(body.get("finish_reason", "stop")),
        )


VENDOR_KEY_ENV = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "xai": "XAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def vendor_key_env(model: str) -> str:
    return VENDOR_KEY_ENV.get(model.split("/", 1)[0] if "/" in model else "", "")


# The directory of scripted replies. Set only by the test suite, in-process (monkeypatch), to a
# directory its test wrote; a real run leaves it None and reads no environment switch for it.
SCRIPTS: Path | None = None


def default_transport() -> Transport:
    return ScriptedTransport(SCRIPTS) if SCRIPTS is not None else LiteLLMTransport()


def price_usd(
    model: str, tokens_in: int, tokens_out: int, *, prices: prices_mod.Prices | None = None
) -> float | None:
    """The price cache's figure for a call; None for a model it does not know."""
    return (prices or prices_mod.default()).price_usd(model, tokens_in, tokens_out)


# ---- the client ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Reply:
    parsed: Any
    raw: str
    prompt: Prompt
    model: str
    profile: str
    payload_sha256: str
    requested_at: str
    tokens_in: int
    tokens_out: int
    cost_usd: float | None
    cached: bool
    attempts: int = 1
    extra: dict = field(default_factory=dict)
    cache_read: int = 0
    cache_write: int = 0
    finish_reason: str = "stop"

    def provenance(self) -> dict:
        return {
            "prompt_id": self.prompt.id,
            "prompt_version": self.prompt.version,
            "prompt_sha256": self.prompt.sha256,
            "model": self.model,
            "profile": self.profile,
            "payload_sha256": self.payload_sha256,
            "requested_at": self.requested_at,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
            "cached": self.cached,
            "attempts": self.attempts,
        }


def payload_sha256(model: str, temperature: float | None, messages: list[dict]) -> str:
    key = json.dumps(
        {"model": model, "temperature": temperature, "messages": messages},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def messages_sha256(
    model: str, temperature: float | None, messages: list[dict], **params: Any
) -> str:
    """The reply-cache key of a call made from raw messages (`Client.complete_messages`): the
    model, temperature and the messages' text with every block-level `cache_control` removed
    (flipping caching on or off must not re-record a fixture), plus the other knobs that change
    the reply (`max_tokens`, `reasoning_effort`, `thinking_budget`), each keyed only when set."""
    key: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": without_cache_control(messages),
    }
    knobs = {name: value for name, value in params.items() if value is not None}
    if knobs:
        key["params"] = knobs
    return hashlib.sha256(
        json.dumps(key, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def without_cache_control(messages: list[dict]) -> list[dict]:
    """`messages` with every block-level `cache_control` key removed; a no-op for string content."""
    if not any(isinstance(message.get("content"), list) for message in messages):
        return messages
    stripped = copy.deepcopy(messages)
    for message in stripped:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block.pop("cache_control", None)
    return stripped


def flat_text(messages: list[dict]) -> str:
    """The messages' text, one string (block lists joined), for the invocation's log."""
    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            parts.append("\n".join(str(block.get("text", "")) for block in content))
        else:
            parts.append(str(content))
    return "\n".join(parts)


def parse_json_object(text: str) -> Any:
    """Strict: the reply is one JSON object, optionally fenced."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("reply is not a JSON object")
    return parsed


class Client:
    def __init__(
        self,
        root: Path,
        config: dict,
        *,
        transport: Transport | None = None,
        validator=None,
        schema_loader=None,
        cache_dir: Path | None = None,
        clock: clock_mod.Clock = clock_mod.utc_now,
        verb: str = "call",
        prices: prices_mod.Prices | None = None,
    ):
        self.root = Path(root)
        self.clock = clock
        # The invocation's log : a run sets it to
        # its RunLog file; otherwise the first block makes this client's own.
        self.verb = verb
        self.log_path: Path | None = None
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir else self.root / CACHE_DIR
        self.prices = prices or prices_mod.default()
        self.transport = transport or default_transport()
        if isinstance(self.transport, ScriptedTransport) and self.transport.prices is None:
            self.transport.prices = self.prices
        # callable(schema_name, obj) → raises ValueError on mismatch (the strict parse)
        self.validator = validator
        # callable(schema_name) → the schema dict sent as json_schema, for a model whose row
        # says the vendor honours it
        self.schema_loader = schema_loader
        self.ledger = Ledger(self.root / LOGS_DIR, config.get("logs") or {}, clock=clock)
        self.max_usd = (config.get("llm") or {}).get("max_usd_per_run")
        self.max_usd_day = (config.get("llm") or {}).get("max_usd_per_day")
        self.spent_usd = 0.0
        self.calls = 0
        self._spent_before_today: float | None = None  # read once, at the first billed call

    # -- the spend caps --
    def spent_today(self) -> float:
        """Today's (UTC) billed spend: the usage ledger's lines from before
        this client's first billed call, plus what this client has spent."""
        if self._spent_before_today is None:
            today = clock_mod.iso(self.clock())[:10]
            path = self.root / LOGS_DIR / "usage.jsonl"
            total = 0.0
            for log_path in [
                path.with_name(f"{path.name}.{generation}") for generation in (2, 1)
            ] + [path]:
                for entry in ledger_read(log_path):
                    if (
                        str(entry.get("at", "")).startswith(today)
                        and entry.get("kind") == "api"
                        and not entry.get("cached")
                    ):
                        total += float(entry.get("cost_usd") or 0.0)
            self._spent_before_today = total
        return self._spent_before_today + self.spent_usd

    def check_spend_cap(self, stage: str, tag: str) -> None:
        """Raise SpendCapReached, naming the cap, before a billed call that a
        cap no longer allows."""
        if self.max_usd is not None and self.spent_usd >= float(self.max_usd):
            raise SpendCapReached(
                f"spent ${self.spent_usd:.4f} of the ${float(self.max_usd):.2f} "
                f"per-run cap ({RUN_CAP}); {stage} for {tag} not attempted",
                RUN_CAP,
            )
        if self.max_usd_day is None:
            return
        if not self.ledger.enabled.get("usage", False):
            raise SpendCapReached(
                f"the per-day cap ({DAY_CAP}) is counted from logs/usage.jsonl, and "
                f"[logs].usage is false in config.toml; {stage} for {tag} not attempted",
                DAY_CAP,
            )
        today = self.spent_today()
        if today >= float(self.max_usd_day):
            raise SpendCapReached(
                f"spent ${today:.4f} today (UTC) of the ${float(self.max_usd_day):.2f} "
                f"per-day cap ({DAY_CAP}); {stage} for {tag} not attempted",
                DAY_CAP,
            )

    @property
    def capped(self) -> bool:
        return self.max_usd is not None or self.max_usd_day is not None

    # -- config helpers --
    def stage_config(self, prompt: Prompt) -> dict:
        section = self.config.get(prompt.config_section)
        if not section:
            raise NotConfigured(
                f"[{prompt.config_section}] is absent from config.toml; "
                f"the {prompt.stage} stage is off"
            )
        if "model" not in section:
            raise NotConfigured(f"[{prompt.config_section}].model is not set")
        return section

    def _payload(
        self, prompt: Prompt, variables: dict[str, str], model: str | None
    ) -> tuple[str, ModelProfile, float | None, list[dict], str]:
        """(model, profile, temperature, messages, payload hash) of a call: what `complete` sends
        and what the cache is keyed by."""
        section = self.stage_config(prompt)
        model = model or section["model"]
        profile = model_profile(model)
        temperature = section.get("temperature", 0.2)
        if profile.temperature == VENDOR_DEFAULT_ONLY:
            temperature = None
        messages = [{"role": "user", "content": prompt.render(variables)}]
        return model, profile, temperature, messages, payload_sha256(model, temperature, messages)

    def payload_hash(
        self, prompt: Prompt, variables: dict[str, str], *, model: str | None = None
    ) -> str:
        """The reply-cache key (and the ledger's payload sha256) of a call, before it is made."""
        return self._payload(prompt, variables, model)[4]

    def lookup(
        self,
        stage: str,
        prompt: Prompt,
        variables: dict[str, str],
        *,
        check=None,
    ) -> Reply | None:
        """The cached reply `complete` would serve for these variables, or None when the call
        would reach the model. Reads the cache only: no call, no ledger entry, no file written. A
        cached reply that fails `check` or the schema is None too: a run would not serve it."""
        model, profile, _temperature, _messages, payload_hash = self._payload(
            prompt, variables, None
        )
        cache_path = self.cache_dir / f"{payload_hash}.json"
        if not cache_path.exists():
            return None
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        try:
            parsed = self._parse(
                prompt, cached["reply"], stage, payload_hash, attempts=1, check=check
            )
        except BadReply:
            return None
        return Reply(
            parsed=parsed,
            raw=cached["reply"],
            prompt=prompt,
            model=model,
            profile=profile.name,
            payload_sha256=payload_hash,
            requested_at="",
            tokens_in=int(cached.get("tokens_in", 0)),
            tokens_out=int(cached.get("tokens_out", 0)),
            cost_usd=0.0,
            cached=True,
        )

    # -- the call --
    def complete(
        self,
        stage: str,
        prompt: Prompt,
        variables: dict[str, str],
        *,
        tag: str,
        model: str | None = None,
        check=None,
    ) -> Reply:
        """`check(parsed)` raises ValueError for a reply that parses and
        matches the schema but is wrong for this call (an index outside the
        batch); it runs BEFORE the reply is cached, so a bad reply is never
        served from cache."""
        model, profile, temperature, messages, payload_hash = self._payload(
            prompt, variables, model
        )
        requested_at = clock_mod.iso(self.clock())
        cache_path = self.cache_dir / f"{payload_hash}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            parsed = self._parse(
                prompt, cached["reply"], stage, payload_hash, attempts=1, check=check
            )
            reply = Reply(
                parsed=parsed,
                raw=cached["reply"],
                prompt=prompt,
                model=model,
                profile=profile.name,
                payload_sha256=payload_hash,
                requested_at=requested_at,
                tokens_in=int(cached.get("tokens_in", 0)),
                tokens_out=int(cached.get("tokens_out", 0)),
                cost_usd=0.0,
                cached=True,
            )
            self._ledger(stage, prompt, reply, tag)
            return reply
        key_env = vendor_key_env(model)
        if self.transport.name == "litellm" and key_env and not os.environ.get(key_env):
            # docs/design/testing.md: offline is asserted. A cache hit above needs no key; a
            # miss with no key means the stage is skipped, recorded as such,
            # never attempted against the vendor.
            raise NotConfigured(f"{key_env} is not set; {stage} is off")
        self.check_spend_cap(stage, tag)
        params: dict = {"_stage": stage}
        if temperature is not None:
            params["temperature"] = temperature
        if profile.json_schema and prompt.output_schema and self.schema_loader is not None:
            # The vendor constrains the reply to the declared schema; the strict parse below still
            # checks it. Not part of the payload hash, so the cache is unaffected.
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": prompt.output_schema,
                    "schema": self.schema_loader(prompt.output_schema),
                    "strict": True,
                },
            }
        elif profile.json_mode:
            params["response_format"] = {"type": "json_object"}
        max_tokens = (self.config.get(prompt.config_section) or {}).get("max_tokens")
        if max_tokens:
            # a vendor that needs a reply limit named (Anthropic) gets the section's; not part of
            # the payload hash, so the cache is unaffected
            params["max_tokens"] = int(max_tokens)
        reasoning_effort = (self.config.get(prompt.config_section) or {}).get("reasoning_effort")
        if reasoning_effort and profile.reasoning_effort:
            # a reasoning model's effort (low, medium, high) comes from the section, like
            # max_tokens: a tuning knob, not part of the payload hash
            params["reasoning_effort"] = str(reasoning_effort)
        timeout = (self.config.get(prompt.config_section) or {}).get("timeout_seconds")
        if timeout and self.transport.name == "litellm":
            # a stage whose replies are long (a whole conversation's line speakers) waits longer;
            # not part of the payload hash, so the cache is unaffected
            params["timeout"] = float(timeout)
        raw: RawReply | None = None
        attempts = 0
        last_error = ""
        parsed = None
        while attempts < 2:
            attempts += 1
            started = time.monotonic()
            raw = self.transport.send(
                model,
                messages,
                {
                    parameter_name: parameter_value
                    for parameter_name, parameter_value in params.items()
                    if parameter_name != "_stage" or self.transport.name == "scripted"
                },
                tag=tag,
            )
            elapsed = time.monotonic() - started
            if raw.cost_usd is None:
                # the transport had no price for it; the price cache may
                priced = self.prices.price_usd(
                    model, raw.tokens_in, raw.tokens_out, raw.cache_read, raw.cache_write
                )
                if priced is not None:
                    raw = replace(raw, cost_usd=priced)
            self.calls += 1
            self.spent_usd += raw.cost_usd or 0.0
            self._log(stage, tag, payload_hash, messages, raw, elapsed, attempts)
            if self.capped and raw.cost_usd is None:
                # A run must not be able to outspend its cap and a cost is never estimated, so
                # an unpriced model under a cap stops here.
                self._ledger_raw(stage, prompt, model, raw, tag, "bad_reply")
                cap_name = RUN_CAP if self.max_usd is not None else DAY_CAP
                raise SpendCapReached(
                    f"{model} has no price in the table, so the spend caps cannot "
                    f"be enforced; add a row to llmlm/prices.py OVERRIDES or remove {cap_name}",
                    cap_name,
                )
            try:
                parsed = self._parse(
                    prompt, raw.text, stage, payload_hash, attempts=attempts, check=check
                )
                break
            except BadReply as error:
                last_error = str(error)
                parsed = None
                # every attempt reaches the ledger , the failed ones as bad_reply
                self._ledger_raw(stage, prompt, model, raw, tag, "bad_reply")
                if attempts >= 2:
                    raise BadReply(
                        f"{stage} for {tag}: {last_error}", attempts, raw.text
                    ) from error
        assert raw is not None
        write_atomic(
            cache_path,
            json.dumps(
                {
                    "model": model,
                    "temperature": temperature,
                    "messages": messages,
                    "reply": raw.text,
                    "tokens_in": raw.tokens_in,
                    "tokens_out": raw.tokens_out,
                    "cost_usd": raw.cost_usd,
                    "requested_at": requested_at,
                },
                ensure_ascii=False,
                indent=1,
            ),
        )
        reply = Reply(
            parsed=parsed,
            raw=raw.text,
            prompt=prompt,
            model=model,
            profile=profile.name,
            payload_sha256=payload_hash,
            requested_at=requested_at,
            tokens_in=raw.tokens_in,
            tokens_out=raw.tokens_out,
            cost_usd=raw.cost_usd,
            cached=False,
            attempts=attempts,
        )
        self._ledger(stage, prompt, reply, tag)
        return reply

    def _messages_key(
        self,
        stage: str,
        messages: list[dict],
        model: str | None,
        temperature: Any,
        max_tokens: int | None,
        reasoning_effort: str | None,
        thinking_budget: int | None,
    ) -> tuple[str, ModelProfile, float | None, str | None, str]:
        """(model, profile, temperature, reasoning effort, cache key) of a messages call, after the
        model's profile has had its say: what `complete_messages` sends and keys the cache on."""
        section = self.config.get(stage) or {}
        model = model or section.get("model")
        if not model:
            raise NotConfigured(f"[{stage}].model is not set; the {stage} stage is off")
        profile = model_profile(model)
        if temperature == "config":
            temperature = section.get("temperature", 0.2)
        if profile.temperature == VENDOR_DEFAULT_ONLY:
            temperature = None
        if not profile.reasoning_effort:
            reasoning_effort = None
        key = messages_sha256(
            model,
            temperature,
            messages,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            thinking_budget=thinking_budget,
        )
        return model, profile, temperature, reasoning_effort, key

    def messages_hash(
        self,
        stage: str,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: Any = "config",
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        thinking_budget: int | None = None,
    ) -> str:
        """The reply-cache key `complete_messages` would use, before the call is made."""
        return self._messages_key(
            stage, messages, model, temperature, max_tokens, reasoning_effort, thinking_budget
        )[4]

    def complete_messages(
        self,
        stage: str,
        messages: list[dict],
        *,
        tag: str,
        model: str | None = None,
        temperature: Any = "config",
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        thinking_budget: int | None = None,
        timeout: float | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        parse=None,
    ) -> Reply:
        """One call from raw messages (a tool that builds its own, with content blocks and
        `cache_control` breakpoints), through the same cache, caps, price and ledger as `complete`.

        The model is `model`, else `[<stage>].model`. `temperature` is the section's (default 0.2)
        unless given (None sends none); the model's profile wins over both. `reasoning_effort` is
        sent only where the profile allows it; `thinking_budget` becomes `thinking`. The reply is
        text: `parsed` is `parse(text)` when `parse` is given (a ValueError is a bad reply), else
        None, and a reply is never retried on its content. Raises Refused and TruncatedReply
        (never cached), NotConfigured (no key and no recording), SpendCapReached, VendorError."""
        model, profile, temperature, reasoning_effort, payload_hash = self._messages_key(
            stage, messages, model, temperature, max_tokens, reasoning_effort, thinking_budget
        )
        section = self.config.get(stage) or {}
        prompt = Prompt(
            id=stage,
            version=1,
            stage=stage,
            output_schema="",
            config_section=stage,
            body="",
            path=Path(stage),
            sha256=payload_hash,
        )
        requested_at = clock_mod.iso(self.clock())
        cache_path = self.cache_dir / f"{payload_hash}.json"

        def reply_from(cached: dict, *, from_cache: bool, raw: RawReply | None = None) -> Reply:
            text = cached["reply"] if raw is None else raw.text
            try:
                parsed = parse(text) if parse is not None else None
            except ValueError as error:
                raise BadReply(f"reply does not parse: {error}", 1, text) from error
            source = cached if raw is None else raw
            get = source.get if isinstance(source, dict) else lambda n, d=0: getattr(source, n, d)
            return Reply(
                parsed=parsed,
                raw=text,
                prompt=prompt,
                model=model,
                profile=profile.name,
                payload_sha256=payload_hash,
                requested_at=requested_at,
                tokens_in=int(get("tokens_in", 0)),
                tokens_out=int(get("tokens_out", 0)),
                cost_usd=0.0 if from_cache else (raw.cost_usd if raw else None),
                cached=from_cache,
                cache_read=int(get("cache_read", 0)),
                cache_write=int(get("cache_write", 0)),
                finish_reason=str(get("finish_reason", "stop")),
            )

        if cache_path.exists():
            reply = reply_from(json.loads(cache_path.read_text(encoding="utf-8")), from_cache=True)
            self._ledger(stage, prompt, reply, tag)
            return reply
        key_env = vendor_key_env(model)
        if (
            self.transport.name == "litellm"
            and key_env
            and not api_key
            and not os.environ.get(key_env)
        ):
            raise NotConfigured(f"{key_env} is not set; {stage} is off")
        self.check_spend_cap(stage, tag)
        params: dict = {"_stage": stage}
        if temperature is not None:
            params["temperature"] = temperature
        for name, value in (
            ("max_tokens", max_tokens),
            ("reasoning_effort", reasoning_effort),
            ("api_base", api_base),
            ("api_key", api_key),
        ):
            if value:
                params[name] = value
        if thinking_budget is not None:
            params["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        if api_base and reasoning_effort:
            # An OpenAI-compatible endpoint (a proxy, Ollama) decides for itself whether it takes
            # an effort; litellm's own model map does not know the model behind it and would
            # raise UnsupportedParamsError before sending (authorllm test_e2e, 2026-10-22).
            params["allowed_openai_params"] = ["reasoning_effort"]
        if (timeout or section.get("timeout_seconds")) and self.transport.name == "litellm":
            params["timeout"] = float(timeout or section["timeout_seconds"])
        started = time.monotonic()
        raw = self.transport.send(
            model,
            messages,
            {k: v for k, v in params.items() if k != "_stage" or self.transport.name == "scripted"},
            tag=tag,
        )
        elapsed = time.monotonic() - started
        if raw.cost_usd is None:
            priced = self.prices.price_usd(
                model, raw.tokens_in, raw.tokens_out, raw.cache_read, raw.cache_write
            )
            if priced is not None:
                raw = replace(raw, cost_usd=priced)
        self.calls += 1
        self.spent_usd += raw.cost_usd or 0.0
        self.log_block(
            stage,
            f"{tag} payload={payload_hash[:12]} {elapsed:.1f}s in={raw.tokens_in} "
            f"out={raw.tokens_out} cost={raw.cost_usd}",
            flat_text(messages),
            raw.text,
        )
        if self.capped and raw.cost_usd is None:
            self._ledger_raw(stage, prompt, model, raw, tag, "bad_reply")
            cap_name = RUN_CAP if self.max_usd is not None else DAY_CAP
            raise SpendCapReached(
                f"{model} has no price in the table, so the spend caps cannot "
                f"be enforced; add a row to llmlm/prices.py OVERRIDES or remove {cap_name}",
                cap_name,
            )
        try:
            reply = reply_from({}, from_cache=False, raw=raw)
        except BadReply:
            self._ledger_raw(stage, prompt, model, raw, tag, "bad_reply")
            raise
        write_atomic(
            cache_path,
            json.dumps(
                {
                    "model": model,
                    "temperature": temperature,
                    "messages": without_cache_control(messages),
                    "reply": raw.text,
                    "tokens_in": raw.tokens_in,
                    "tokens_out": raw.tokens_out,
                    "cache_read": raw.cache_read,
                    "cache_write": raw.cache_write,
                    "finish_reason": raw.finish_reason,
                    "cost_usd": raw.cost_usd,
                    "requested_at": requested_at,
                },
                ensure_ascii=False,
                indent=1,
            ),
        )
        self._ledger(stage, prompt, reply, tag)
        return reply

    def estimate(
        self,
        prompt: Prompt,
        variables: Iterable[dict[str, str]],
        *,
        model: str | None = None,
        expected_output_tokens: int | None = None,
    ) -> Estimate:
        """What completing `prompt` once per item of `variables` would cost at the configured
        model, before any call: payloads already cached are free. The output size is
        `expected_output_tokens`, else the prompt's front matter, else 0."""
        payloads = []
        resolved = model
        for item in variables:
            resolved, _profile, _temperature, messages, payload_hash = self._payload(
                prompt, item, model
            )
            payloads.append((messages, payload_hash))
        if resolved is None:
            resolved = self.stage_config(prompt)["model"]
        return estimate(
            resolved,
            payloads,
            expected_output_tokens=(
                expected_output_tokens
                if expected_output_tokens is not None
                else prompt.expected_output_tokens or 0
            ),
            prices=self.prices,
            cache_dir=self.cache_dir,
        )

    def replay(
        self, stage: str, prompt: Prompt, payload_hash: str, *, tag: str, check=None
    ) -> Reply:
        """Serve the reply behind a recorded payload hash from the cache without
        rebuilding the payload: the road a held (stale-on-config) AI cleanup takes
        when a verdict or a memory rule lands on it (a held pass). The model is
        never asked; `check` runs as it does on any cached reply. CacheMiss
        when the entry is gone."""
        cache_path = self.cache_dir / f"{payload_hash}.json" if payload_hash else None
        if cache_path is None or not cache_path.exists():
            raise CacheMiss(
                f"{stage} for {tag}: no cached reply for payload {payload_hash[:12] or '?'}"
            )
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        parsed = self._parse(prompt, cached["reply"], stage, payload_hash, attempts=1, check=check)
        reply = Reply(
            parsed=parsed,
            raw=cached["reply"],
            prompt=prompt,
            model=str(cached.get("model", "")),
            profile="replay",
            payload_sha256=payload_hash,
            requested_at=clock_mod.iso(self.clock()),
            tokens_in=int(cached.get("tokens_in", 0)),
            tokens_out=int(cached.get("tokens_out", 0)),
            cost_usd=0.0,
            cached=True,
        )
        self._ledger(stage, prompt, reply, tag)
        return reply

    def _parse(
        self, prompt: Prompt, text: str, stage: str, payload_hash: str, *, attempts: int, check=None
    ):
        try:
            parsed = parse_json_object(text)
        except ValueError as error:
            raise BadReply(f"reply is not a JSON object ({error})", attempts, text) from error
        if self.validator is not None:
            try:
                self.validator(prompt.output_schema, parsed)
            except ValueError as error:
                raise BadReply(
                    f"reply does not match {prompt.output_schema}: {error}", attempts, text
                ) from error
        if check is not None:
            try:
                check(parsed)
            except ValueError as error:
                raise BadReply(str(error), attempts, text) from error
        return parsed

    def _ledger_raw(
        self, stage: str, prompt: Prompt, model: str, raw: RawReply, tag: str, outcome: str
    ) -> None:
        self.ledger.usage(
            {
                "stage": stage,
                "prompt_id": prompt.id,
                "model": model,
                "tokens_in": raw.tokens_in,
                "tokens_out": raw.tokens_out,
                "cost_usd": raw.cost_usd,
                "kind": "api",
                "cached": False,
                "tag": tag,
                "outcome": outcome,
                **_cache_fields(raw.cache_read, raw.cache_write),
            }
        )

    def _ledger(self, stage: str, prompt: Prompt, reply: Reply, tag: str) -> None:
        self.ledger.usage(
            {
                "stage": stage,
                "prompt_id": prompt.id,
                "model": reply.model,
                "tokens_in": reply.tokens_in,
                "tokens_out": reply.tokens_out,
                "cost_usd": reply.cost_usd,
                "kind": "api",
                "cached": reply.cached,
                "tag": tag,
                "outcome": "ok",
                **_cache_fields(reply.cache_read, reply.cache_write),
            }
        )

    def log_block(self, stage: str, header: str, payload: str, reply: str) -> None:
        """One payload and reply, tagged with its stage, into the invocation's
        log (there are no per-stage logs)."""
        path = self.log_path or self._own_log()
        with path.open("a", encoding="utf-8") as file:
            file.write(
                f"\n===== [{stage}] {clock_mod.iso(self.clock())} {header}\n"
                f"----- [{stage}] payload\n{payload.rstrip(chr(10))}\n"
                f"----- [{stage}] reply\n{reply.rstrip(chr(10))}\n"
                f"===== [{stage}] end\n"
            )

    def _own_log(self) -> Path:
        """`logs/runs/<UTC stamp>-<verb>-<short id>.log`, made once (never an
        existing file) for a verb that calls a model outside a run."""
        import secrets

        directory = self.root / LOGS_DIR / RUN_LOGS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        stamp = self.clock().astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        while True:
            path = directory / f"{stamp}-{self.verb}-{secrets.token_hex(3)}.log"
            try:
                path.open("x", encoding="utf-8").close()
                break
            except FileExistsError:
                continue
        self.log_path = path
        return path

    def _log(
        self,
        stage: str,
        tag: str,
        payload_hash: str,
        messages: list[dict],
        raw: RawReply,
        elapsed: float,
        attempt: int,
    ) -> None:
        self.log_block(
            stage,
            f"{tag} attempt={attempt} payload={payload_hash[:12]} {elapsed:.1f}s "
            f"in={raw.tokens_in} out={raw.tokens_out} cost={raw.cost_usd}",
            "\n".join(message["content"] for message in messages),
            raw.text,
        )


__all__ = [
    "Refused",
    "messages_sha256",
    "Client",
    "Reply",
    "BadReply",
    "SpendCapReached",
    "CacheMiss",
    "TruncatedReply",
    "VendorError",
    "NotConfigured",
    "RawReply",
    "Transport",
    "ScriptedTransport",
    "LiteLLMTransport",
    "default_transport",
    "model_profile",
    "MODEL_PROFILES",
    "price_usd",
    "payload_sha256",
    "Estimate",
]
