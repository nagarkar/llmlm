"""The price cache (`llmlm.prices`): a state file refreshed from litellm's map when over a week
old, overrides with a source and date for what the map lacks, a failed fetch keeps the old
prices, an unknown model prices to null, and a changed price is announced."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from llmlm import prices
from llmlm.prices import Prices, Rates

NOW = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
FLASH = "gemini/gemini-2.5-flash"


def litellm_map(input_cost=3e-7, output_cost=2.5e-6):
    return {
        FLASH: {
            "input_cost_per_token": input_cost,
            "output_cost_per_token": output_cost,
            "cache_read_input_token_cost": 7.5e-8,
            "litellm_provider": "gemini",
        },
        "gemini-2.5-flash": {
            "input_cost_per_token": input_cost,
            "output_cost_per_token": output_cost,
        },
        "some/free-model": {"litellm_provider": "x"},  # no price: not a row
        "sample_spec": {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0},
    }


def cache(tmp_path, *, fetch=None, now=NOW, notes=None, overrides=prices.OVERRIDES):
    fetches = []

    def fetching():
        fetches.append(1)
        return (fetch or litellm_map)()

    table = Prices(
        tmp_path / "prices.json",
        fetch=fetching,
        clock=lambda: now,
        notify=(notes if notes is not None else []).append,
        overrides=overrides,
    )
    return table, fetches


def test_a_missing_cache_is_fetched_once_and_written_with_its_checked_at(tmp_path):
    table, fetches = cache(tmp_path)
    assert table.rates(FLASH) == Rates(0.3, 2.5, 0.075, 0.0, prices.SOURCE_URL, "2026-10-21")
    assert table.rates("gemini-2.5-flash").input_per_m == 0.3  # the bare name too
    assert table.price_usd(FLASH, 1_000_000, 1_000_000) == 2.8
    assert table.rates("vendor/unknown") is None
    assert len(fetches) == 1
    stored = json.loads((tmp_path / "prices.json").read_text())
    assert stored["checked_at"] == "2026-10-21T12:00:00Z" and stored["source"] == prices.SOURCE_URL
    assert "some/free-model" not in stored["models"] and "sample_spec" in stored["models"]


def test_a_fresh_cache_is_read_not_fetched_and_a_week_old_one_is_refreshed(tmp_path):
    first, fetches = cache(tmp_path)
    first.rates(FLASH)
    later, fetches_later = cache(tmp_path, now=NOW + timedelta(days=6))
    assert later.rates(FLASH).input_per_m == 0.3 and fetches_later == []
    stale, fetches_stale = cache(
        tmp_path, now=NOW + timedelta(days=8), fetch=lambda: litellm_map(input_cost=4e-7)
    )
    assert stale.rates(FLASH).input_per_m == 0.4 and len(fetches_stale) == 1


def test_a_changed_price_is_announced(tmp_path):
    notes = []
    first, _ = cache(tmp_path, notes=notes)
    first.refresh(force=True)
    assert notes == []
    second, _ = cache(
        tmp_path,
        now=NOW + timedelta(days=8),
        fetch=lambda: litellm_map(output_cost=3e-6),
        notes=notes,
    )
    report = second.refresh()
    assert report["refreshed"] is True and len(report["changed"]) == 2
    assert len(notes) == 1 and FLASH in notes[0][0]


def test_a_failed_fetch_keeps_the_old_prices_and_names_the_problem(tmp_path):
    first, _ = cache(tmp_path)
    first.rates(FLASH)

    def offline():
        raise OSError("no route to host")

    later, _ = cache(tmp_path, now=NOW + timedelta(days=9), fetch=offline)
    assert later.rates(FLASH).input_per_m == 0.3  # kept
    problems = later.problems([FLASH, "vendor/unknown"])
    assert any("fetch" in problem and "no route to host" in problem for problem in problems)
    assert any("vendor/unknown has no price" in problem for problem in problems)
    assert (tmp_path / "prices.json").exists()


def test_an_override_wins_and_names_its_source_and_a_stale_one_is_a_doctor_warning(tmp_path):
    row = Rates(0.042, 0.0, source="https://docs.typesafe.ai/models", verified="2026-10-20")
    table, fetches = cache(
        tmp_path, overrides=[("jev-latest", row), (FLASH, Rates(9.0, 9.0, verified="2026-10-20"))]
    )
    assert (
        table.rates("jev-latest") is row and table.price_usd("jev-latest", 1_000_000, 50) == 0.042
    )
    assert table.rates(FLASH).input_per_m == 9.0 and fetches == []  # an override needs no fetch
    table.refresh(force=True)
    assert table.problems() == []
    old, _ = cache(tmp_path, now=NOW + timedelta(days=30), overrides=[("jev-latest", row)])
    old.refresh(force=True)
    (warning,) = old.problems()
    assert "jev-latest" in warning and "2026-10-20" in warning and row.source in warning


def test_the_shipped_overrides_carry_a_source_and_a_date():
    for name, row in prices.OVERRIDES:
        assert row.source.startswith("https://") and len(row.verified) == 10, name
    assert dict(prices.OVERRIDES)["jev-latest"].output_per_m == 0.0


def test_the_table_shows_where_each_price_came_from(tmp_path):
    table, _ = cache(tmp_path)
    rows = {row["model"]: row for row in table.table([FLASH, "jev-latest", "vendor/unknown"])}
    assert rows[FLASH]["source"] == prices.SOURCE_URL
    assert rows["jev-latest"]["source"] == "https://docs.typesafe.ai/models"
    assert rows["vendor/unknown"]["source"] is None


def test_the_cache_is_refreshed_after_a_week():
    assert prices.MAX_AGE == timedelta(days=7)


def test_rows_from_the_map_drop_unpriced_entries():
    rows = prices.rows_from_map({"a": {"input_cost_per_token": 1e-6}, "b": {}, "c": "not a row"})
    assert rows == {
        "a": {
            "input_per_m": 1.0,
            "output_per_m": 0.0,
            "cache_read_per_m": 0.0,
            "cache_write_per_m": 0.0,
        }
    }


@pytest.mark.parametrize("tokens", [(1_000_000, 0, 0, 0), (0, 0, 1_000_000, 1_000_000)])
def test_rates_price_cache_reads_and_writes(tokens):
    rates = Rates(1.0, 2.0, 0.1, 1.25)
    assert rates.usd(*tokens) in (1.0, 1.35)
