"""Prices: what a token costs on each model, cached on this machine and refreshed weekly.

A price is state, not configuration: it is a fetched fact with a date, so it lives in one
machine-wide state file (`CACHE_PATH`, `~/.llmlm/prices.json`, shared by every repo; a repo's
config names only its models and caps). The file holds `checked_at`, the `source` URL and one
row per model: dollars per million tokens for input, output, cache reads and cache writes.

The source is litellm's published model map (`SOURCE_URL`), fetched whole. `Prices.rates` refreshes
the file when it is missing or older than `MAX_AGE` (seven days) before answering, so a price a
tool uses is never more than a week old; a fetch that fails keeps the old file and names the
problem (`Prices.problems`, for a tool's `doctor`). A changed price is applied and announced
through `superlm-notify` when that notifier is installed.

`OVERRIDES` are the rows the map lacks or gets wrong (TypeSafe's `jev-latest`: litellm does not
know it). Each carries the page it was verified at and the date; an override older than
`MAX_AGE` is a `doctor` warning naming the page to check, never silently refreshed. An override
wins over the map.

A model in neither place prices to None: an unknown cost is null, never an estimate.

Never imports: llm, ledger, usage.

Terms:
- **price cache**: `~/.llmlm/prices.json`, the fetched model map with its `checked_at`.
- **override**: a hand-verified price row with its source page and date, for a model the map
  lacks or gets wrong.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from . import clock as clock_mod
from .files import write_atomic

CACHE_PATH = Path.home() / ".llmlm" / "prices.json"
SOURCE_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)
MAX_AGE = timedelta(days=7)
FETCH_TIMEOUT_S = 30


@dataclass(frozen=True, slots=True)
class Rates:
    """Dollars per million tokens. `source` is the page or map the row came from, `verified` the
    date (YYYY-MM-DD) it was checked there."""

    input_per_m: float
    output_per_m: float
    cache_read_per_m: float = 0.0
    cache_write_per_m: float = 0.0
    source: str = ""
    verified: str = ""

    def usd(
        self, tokens_in: int, tokens_out: int, cache_read: int = 0, cache_write: int = 0
    ) -> float:
        return round(
            tokens_in / 1e6 * self.input_per_m
            + tokens_out / 1e6 * self.output_per_m
            + cache_read / 1e6 * self.cache_read_per_m
            + cache_write / 1e6 * self.cache_write_per_m,
            6,
        )


# Hand-verified rows, each with the page it was read from and the date. Longest exact key wins;
# no wildcards. Add a row only for a model the map lacks or gets wrong.
OVERRIDES: tuple[tuple[str, Rates], ...] = (
    # TypeSafe bills input tokens only. Keyed on the alias the owner chose to follow.
    (
        "jev-latest",
        Rates(0.042, 0.0, source="https://docs.typesafe.ai/models", verified="2026-09-29"),
    ),
)


def fetch_litellm_map(url: str = SOURCE_URL) -> dict:
    """The published map, as litellm ships it: one object per model with per-token costs."""
    with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT_S) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def rows_from_map(raw: dict) -> dict[str, dict]:
    """The map's rows that carry a price, as dollars per million tokens."""
    rows: dict[str, dict] = {}
    for name, info in raw.items():
        if not isinstance(info, dict) or info.get("input_cost_per_token") is None:
            continue
        rows[name] = {
            "input_per_m": round(float(info["input_cost_per_token"]) * 1e6, 6),
            "output_per_m": round(float(info.get("output_cost_per_token") or 0.0) * 1e6, 6),
            "cache_read_per_m": round(
                float(info.get("cache_read_input_token_cost") or 0.0) * 1e6, 6
            ),
            "cache_write_per_m": round(
                float(info.get("cache_creation_input_token_cost") or 0.0) * 1e6, 6
            ),
        }
    return rows


def notify_change(changed: list[str]) -> None:
    """Announce changed prices through the SuperLM notifier when it is installed."""
    notifier = shutil.which("superlm-notify")
    if notifier is None:
        return
    body = "; ".join(changed[:10]) + (" …" if len(changed) > 10 else "")
    try:
        subprocess.run(
            [notifier, "--repo", "llmlm", "--title", "Model prices changed", "--body", body],
            check=False,
            timeout=10,
            capture_output=True,
        )
    except (OSError, subprocess.SubprocessError):
        return


class Prices:
    """The price cache: reads the file, refreshes it when stale, answers rates."""

    def __init__(
        self,
        path: Path = CACHE_PATH,
        *,
        fetch: Callable[[], dict] | None = None,
        clock: clock_mod.Clock = clock_mod.utc_now,
        notify: Callable[[list[str]], None] = notify_change,
        max_age: timedelta = MAX_AGE,
        overrides: Iterable[tuple[str, Rates]] = OVERRIDES,
    ):
        self.path = Path(path)
        self.fetch = fetch or fetch_litellm_map
        self.clock = clock
        self.notify = notify
        self.max_age = max_age
        self.overrides = dict(overrides)
        self.problem: str | None = None
        self._cache: dict | None = None

    # -- the file --
    def _load(self) -> dict:
        if self._cache is None:
            try:
                self._cache = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._cache = {"checked_at": None, "source": SOURCE_URL, "models": {}}
        return self._cache

    def checked_at(self) -> datetime | None:
        stamp = self._load().get("checked_at")
        if not stamp:
            return None
        return datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))

    def stale(self) -> bool:
        checked = self.checked_at()
        return checked is None or self.clock() - checked > self.max_age

    def refresh(self, *, force: bool = False) -> dict:
        """Fetch the map when the file is stale (or `force`). Returns a report: `refreshed`,
        `changed` (model: old → new), `problem`. A failed fetch keeps the old file."""
        if not force and not self.stale():
            return {"refreshed": False, "changed": [], "problem": None}
        before = self._load().get("models", {})
        try:
            rows = rows_from_map(self.fetch())
        except Exception as error:  # noqa: BLE001 - any fetch failure keeps the old prices
            self.problem = (
                f"price cache {self.path}: fetch of {SOURCE_URL} failed "
                f"({type(error).__name__}: {str(error)[:120]}); prices kept from "
                f"{self._load().get('checked_at') or 'never'}"
            )
            return {"refreshed": False, "changed": [], "problem": self.problem}
        changed = [
            f"{name}: {before[name]} → {row}"
            for name, row in rows.items()
            if name in before and before[name] != row
        ]
        self._cache = {
            "checked_at": clock_mod.iso(self.clock()),
            "source": SOURCE_URL,
            "models": rows,
        }
        write_atomic(self.path, json.dumps(self._cache, indent=1, sort_keys=True))
        self.problem = None
        if changed:
            self.notify(changed)
        return {"refreshed": True, "changed": changed, "problem": None}

    # -- answers --
    def rates(self, model: str) -> Rates | None:
        """The rates for `model`: an override, else the map (refreshed first when stale), else
        None. The map is tried under the exact name, then without its vendor prefix."""
        if model in self.overrides:
            return self.overrides[model]
        if self.stale():
            self.refresh()
        rows = self._load().get("models", {})
        for key in (model, model.split("/", 1)[-1]):
            row = rows.get(key)
            if row:
                return Rates(
                    **row, source=SOURCE_URL, verified=str(self._load()["checked_at"])[:10]
                )
        return None

    def price_usd(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int,
        cache_read: int = 0,
        cache_write: int = 0,
    ) -> float | None:
        rates = self.rates(model)
        if rates is None:
            return None
        return rates.usd(tokens_in, tokens_out, cache_read, cache_write)

    def problems(self, models: Iterable[str] = ()) -> list[str]:
        """What a tool's `doctor` says: a stale cache that could not be refreshed, an override
        past its verification age, and every one of `models` with no price."""
        out: list[str] = []
        if self.problem:
            out.append(self.problem)
        elif self.stale():
            out.append(f"price cache {self.path} is older than {self.max_age.days} days")
        for name, rates in self.overrides.items():
            verified = datetime.fromisoformat(rates.verified + "T00:00:00+00:00")
            if self.clock() - verified > self.max_age:
                out.append(
                    f"price override for {name} was verified {rates.verified}; check "
                    f"{rates.source} and bump the date in llmlm/prices.py OVERRIDES"
                )
        for model in models:
            if self.rates(model) is None:
                out.append(f"{model} has no price (not in the map or OVERRIDES); its cost is null")
        return out

    def table(self, models: Iterable[str]) -> list[dict]:
        """One row per model: its rates and where they came from, for a tool's board."""
        out = []
        for model in models:
            rates = self.rates(model)
            out.append({"model": model, **(asdict(rates) if rates else {"source": None})})
        return out


_default: Prices | None = None


def default() -> Prices:
    """The one machine-wide cache, made on first use."""
    global _default
    if _default is None:
        _default = Prices()
    return _default
