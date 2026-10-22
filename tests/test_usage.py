"""The spend board (`llmlm.usage`): a tool's month from its ledger beside its caps, numbers only,
reading nothing but the ledger and the settings file, writing nothing."""

import json
from datetime import datetime

from llmlm import ledger, usage


def event(**over):
    return {
        "stage": "cleanup",
        "prompt_id": "cleanup",
        "model": "some/model",
        "tokens_in": 1000,
        "tokens_out": 200,
        "cost_usd": 0.25,
        "kind": "api",
        "cached": False,
        "tag": "SECRET-TAG-TEXT",
        "outcome": "ok",
    } | over


def spend(folder, day: str, **over):
    moment = datetime.fromisoformat(f"{day}T08:00:00+00:00")
    ledger.Ledger(folder, clock=lambda: moment).usage(event(**over))


def ledgers(tmp_path):
    spend(tmp_path / "ytlm-logs", "2026-10-05", stage="pass3")
    spend(tmp_path / "ytlm-logs", "2026-10-05", stage="pass3", cached=True, cost_usd=0.0)
    spend(tmp_path / "ytlm-logs", "2026-10-06", stage="digest", cost_usd=None)
    spend(tmp_path / "ytlm-logs", "2026-10-06", stage="digest", kind="chat", cost_usd=None)
    spend(tmp_path / "ytlm-logs", "2026-09-30", stage="pass3", cost_usd=9.0)
    spend(tmp_path / "beelm-logs", "2026-10-09", stage="model_cleanup", cost_usd=0.5)
    for tool, per_run in (("ytlm", 2.0), ("beelm", 0.5)):
        (tmp_path / f"{tool}.toml").write_text(f"[llm]\nmax_usd_per_run = {per_run}\n")
    return (
        {"ytlm": tmp_path / "ytlm-logs", "beelm": tmp_path / "beelm-logs"},
        {"ytlm": tmp_path / "ytlm.toml", "beelm": tmp_path / "beelm.toml"},
    )


def test_the_board_sums_each_tools_month_by_stage_and_day_beside_its_caps(tmp_path):
    folders, settings = ledgers(tmp_path)
    board = usage.model_spend(folders, "2026-10", settings)
    ytlm = board["tools"]["ytlm"]
    assert ytlm["total"]["calls"] == 3 and ytlm["total"]["cached"] == 1
    assert ytlm["total"]["cost_usd"] == 0.25  # September's $9 is another month
    assert ytlm["per_run_usd"] == 2.0 and ytlm["per_day_usd"] is None
    assert ytlm["by_stage"]["pass3"]["calls"] == 2
    assert ytlm["by_day"]["2026-10-05"]["cost_usd"] == 0.25 and ytlm["biggest_day_usd"] == 0.25
    assert board["tools"]["beelm"]["per_run_usd"] == 0.5
    assert board["cost_usd"] == 0.75 and board["problems"] == [] and board["note"] == usage.NOTE


def test_cost_unknown_and_chat_calls_are_counted_never_summed(tmp_path):
    folders, settings = ledgers(tmp_path)
    digest = usage.tool_spend("ytlm", folders["ytlm"], settings["ytlm"], "2026-10")["by_stage"][
        "digest"
    ]
    assert digest["calls"] == 1 and digest["cost_unknown"] == 1
    assert digest["chat"] == 1 and digest["cost_usd"] == 0.0


def test_the_board_carries_numbers_never_content_and_writes_nothing(tmp_path):
    folders, settings = ledgers(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    text = json.dumps(usage.model_spend(folders, "2026-10", settings))
    assert "SECRET-TAG-TEXT" not in text and "some/model" not in text
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_a_missing_ledger_or_unreadable_settings_is_a_problem_and_nothing_is_created(tmp_path):
    board = usage.tool_spend("beelm", tmp_path / "nowhere", tmp_path / "absent.toml", "2026-10")
    assert board["ledger_found"] is False
    assert any("beelm: no usage.jsonl" in problem for problem in board["problems"])
    assert any("cannot read" in problem for problem in board["problems"])
    assert not (tmp_path / "nowhere").exists()
    assert usage.caps(None) == {"per_run_usd": None, "per_day_usd": None, "problem": ""}
