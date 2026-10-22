"""Tests never read the machine-wide price cache, the network, or a git hook's environment: the
default price cache is replaced by one whose fetch fails, so a test that prices anything must
build its own `Prices` (see tests/test_llm.py)."""

import os

import pytest

from llmlm import prices

for key in [name for name in os.environ if name.startswith("GIT_")]:
    os.environ.pop(key, None)


def _no_network():
    raise AssertionError("a test reached for the network price map; build a Prices of its own")


@pytest.fixture(autouse=True)
def _no_default_price_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "CACHE_PATH", tmp_path / "never-prices.json")
    monkeypatch.setattr(
        prices, "_default", prices.Prices(tmp_path / "never-prices.json", fetch=_no_network)
    )
