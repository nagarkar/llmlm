"""Usage: the board of model spend, built from the ledgers and numbers only.

`model_spend(ledgers, month)` reads each tool's `usage.jsonl` (with its rotated generations)
and returns, per tool, the month's totals, by stage and by day, beside the per-run and per-day
cap the tool's own settings file sets. A tool's `usage` verb calls `tool_spend` for itself; the
one served dashboard (corpuslm's usage page) calls `model_spend` for every tool. Reading the
board writes nothing.

Counts and money only: a ledger line's prompt, model and tag are never copied into the board. A
call with no known price is counted as `cost_unknown`, and a chat-road call is counted but never
summed with billed ones. Dollar figures are the ledger's own figures from the price table; the
vendor's meter is the authority (`NOTE`).

Never imports: llm, prices.

Terms:
- **model spend**: the dollars, calls, cached calls and tokens in a tool's `usage.jsonl`.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from . import ledger

NOTE = (
    "Dollar figures are each tool's own ledger figures, worked out from its price table. "
    "The vendor's own meter is the authority."
)
SPEND_FIELDS = ("calls", "cached", "tokens_in", "tokens_out", "cost_usd", "cost_unknown", "chat")


def _empty() -> dict:
    return {"calls": 0, "cached": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
            "cost_unknown": 0, "chat": 0}  # fmt: skip


def _add(row: dict, line: dict) -> None:
    if line.get("kind") == "chat":
        row["chat"] += 1
        return
    row["calls"] += 1
    row["cached"] += 1 if line.get("cached") else 0
    row["tokens_in"] += int(line.get("tokens_in") or 0)
    row["tokens_out"] += int(line.get("tokens_out") or 0)
    cost = line.get("cost_usd")
    if cost is None and not line.get("cached"):
        row["cost_unknown"] += 1
    else:
        row["cost_usd"] = round(row["cost_usd"] + float(cost or 0.0), 6)


def caps(settings_file: Path | None) -> dict:
    """The two spend caps (`[llm].max_usd_per_run`, `[llm].max_usd_per_day`) from a tool's own
    TOML settings file; None where it sets none."""
    spend_caps: dict = {"per_run_usd": None, "per_day_usd": None, "problem": ""}
    if settings_file is None:
        return spend_caps
    try:
        table = tomllib.loads(Path(settings_file).read_text("utf-8")).get("llm", {})
    except (OSError, tomllib.TOMLDecodeError) as error:
        return spend_caps | {"problem": f"cannot read {settings_file}: {type(error).__name__}"}
    for key, name in (("per_run_usd", "max_usd_per_run"), ("per_day_usd", "max_usd_per_day")):
        value = table.get(name) if isinstance(table, dict) else None
        if isinstance(value, int | float) and not isinstance(value, bool):
            spend_caps[key] = float(value)
    return spend_caps


def tool_spend(tool: str, folder: Path, settings_file: Path | None, month: str) -> dict:
    """One tool's model spend for `month` (YYYY-MM): totals, by stage, by day, and its caps."""
    path = Path(folder) / "usage.jsonl"
    found = path.exists() or path.with_name("usage.jsonl.1").exists()
    total, by_stage, by_day = _empty(), {}, {}
    for line in ledger.read(path, with_rotated=True):
        timestamp = str(line.get("at", ""))
        if not timestamp.startswith(month):
            continue
        _add(total, line)
        _add(by_stage.setdefault(str(line.get("stage", "?")), _empty()), line)
        _add(by_day.setdefault(timestamp[:10], _empty()), line)
    spend_caps = caps(settings_file)
    problems = [spend_caps.pop("problem")] if spend_caps["problem"] else []
    spend_caps.pop("problem", None)
    if not found:
        problems.append(f"{tool}: no usage.jsonl in {folder}")
    return {
        "tool": tool,
        "ledger": str(path),
        "ledger_found": found,
        **spend_caps,
        "total": total,
        "biggest_day_usd": max((day["cost_usd"] for day in by_day.values()), default=0.0),
        "by_stage": dict(sorted(by_stage.items())),
        "by_day": dict(sorted(by_day.items())),
        "problems": problems,
    }


def model_spend(
    ledgers: dict[str, Path], month: str, tool_settings: dict[str, Path] | None = None
) -> dict:
    """Every tool's spend for `month`: `tools` by name, the summed `cost_usd`, `problems`."""
    tools = [
        tool_spend(name, folder, (tool_settings or {}).get(name), month)
        for name, folder in ledgers.items()
    ]
    return {
        "tools": {tool["tool"]: tool for tool in tools},
        "cost_usd": round(sum(tool["total"]["cost_usd"] for tool in tools), 6),
        "note": NOTE,
        "problems": [problem for tool in tools for problem in tool["problems"]],
    }
