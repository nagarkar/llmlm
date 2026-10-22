"""Tests for the ledgers (module llmlm.ledger); moved from ytlm.

Written from the requirement and the public interface only (metalm
testing.md, "Anti-gaming" (c)); no implementation file was read.

Sources:
- ytlm's design rules, paragraph "Three ledgers, one architecture":
  JSONL under logs/, one line per event, rotated at 5 MB with two
  generations kept, written best-effort so a ledger failure never fails a
  run, each with an off-switch under [logs]; db-perf off by default.
- src/llmlm/schemas/usage_line.json and trace_line.json (the line schemas).
- metalm coding.md, "Failure" (fail loudly on our own bad data) and
  "Idempotence and writes" (validate every artifact at write and at read).
- metalm operations.md, "Scheduled and unattended work".

Expected values are derived by hand from those documents.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from llmlm import ledger, schemas


def usage_event(**over):
    event = {
        "stage": "cleanup",
        "prompt_id": "cleanup",
        "model": "gemini/gemini-2.5-flash",
        "tokens_in": 1200,
        "tokens_out": 300,
        "cost_usd": 0.0012,
        "kind": "api",
        "cached": False,
        "tag": "vidAAAAAAA1 batch 1",
        "outcome": "ok",
    }
    event.update(over)
    return event


def trace_event(**over):
    event = {
        "argv": ["run", "--now"],
        "client": "cli",
        "workspace": "/Volumes/X6/ytlm",
        "seconds": 1.5,
        "exit_code": 0,
    }
    event.update(over)
    return event


def lines(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


# 1. One JSON line per event, with `at`; off-switches --------------------------


def test_the_rotation_size_generations_and_defaults_are_the_documented_ones():
    # "rotated at 5 MB with two generations kept"; db-perf "off by default"
    assert ledger.ROTATE_BYTES == 5 * 1024 * 1024
    assert ledger.GENERATIONS == 2
    assert ledger.DEFAULTS == {"usage": True, "trace": True, "db_perf": False}


def test_usage_and_trace_write_one_line_each_with_at(tmp_path):
    # ytlm's design rules: "JSONL files under logs/ ... one line per event";
    # schemas require `at` (ISO, UTC)
    logs_directory = tmp_path / "logs"
    led = ledger.Ledger(logs_directory)
    led.usage(usage_event())
    led.usage(usage_event(kind="chat", cost_usd=None, cached=True))
    led.trace(trace_event())
    usage_lines = lines(logs_directory / "usage.jsonl")
    trace_lines = lines(logs_directory / "trace.jsonl")
    assert len(usage_lines) == 2 and len(trace_lines) == 1
    for row in usage_lines + trace_lines:
        assert isinstance(row["at"], str)
        datetime.fromisoformat(row["at"].replace("Z", "+00:00"))
    assert {key: value for key, value in usage_lines[0].items() if key != "at"} == usage_event()
    assert usage_lines[1]["cost_usd"] is None and usage_lines[1]["kind"] == "chat"
    assert {key: value for key, value in trace_lines[0].items() if key != "at"} == trace_event()
    # each line is a standalone JSON document
    raw = (logs_directory / "usage.jsonl").read_text(encoding="utf-8")
    assert raw.count("\n") == 2


def test_db_perf_off_by_default_writes_nothing(tmp_path):
    # "db-perf.jsonl ... off by default ([logs].db_perf, currently false)"
    logs_directory = tmp_path / "logs"
    led = ledger.Ledger(logs_directory)
    led.usage(usage_event())
    led.trace(trace_event())
    assert not (logs_directory / "db-perf.jsonl").exists()


@pytest.mark.parametrize("off", ["usage", "trace"])
def test_section_switched_off_writes_nothing(tmp_path, off):
    # "each with an off-switch under a [logs] section in config.toml"
    logs_directory = tmp_path / "logs"
    led = ledger.Ledger(logs_directory, {off: False})
    led.usage(usage_event())
    led.trace(trace_event())
    other_section = "trace" if off == "usage" else "usage"
    assert not (logs_directory / f"{off}.jsonl").exists()
    assert len(lines(logs_directory / f"{other_section}.jsonl")) == 1


# 2. Rotation -------------------------------------------------------------------


def test_rotation_keeps_two_generations_and_never_more(tmp_path):
    # "rotated at 5 MB with two generations kept"
    logs_directory = tmp_path / "logs"
    logs_directory.mkdir()
    led = ledger.Ledger(logs_directory)
    live = logs_directory / "usage.jsonl"
    filler = "x" * ledger.ROTATE_BYTES
    for round_number in range(4):
        # a full live file, marked with its round number on line one
        live.write_text(
            json.dumps({"marker": round_number}) + "\n" + filler + "\n", encoding="utf-8"
        )
        led.usage(usage_event(tag=f"round {round_number}"))
    rotated = sorted(
        rotated_path
        for rotated_path in logs_directory.iterdir()
        if rotated_path.name.startswith("usage") and rotated_path != live
    )
    assert len(rotated) == ledger.GENERATIONS  # .1 and .2, never a third
    markers = set()
    for rotated_path in rotated:
        first = rotated_path.read_text(encoding="utf-8").splitlines()[0]
        markers.add(json.loads(first)["marker"])
    assert markers == {2, 3}  # the two newest generations kept
    names = {rotated_path.name for rotated_path in rotated}
    assert names == {"usage.jsonl.1", "usage.jsonl.2"}
    newest = json.loads(
        (logs_directory / "usage.jsonl.1").read_text(encoding="utf-8").splitlines()[0]
    )
    assert newest["marker"] == 3
    # the live file stays under the limit after rotation
    assert (not live.exists()) or live.stat().st_size < ledger.ROTATE_BYTES


def test_no_rotation_below_limit(tmp_path):
    logs_directory = tmp_path / "logs"
    led = ledger.Ledger(logs_directory)
    for _ in range(5):
        led.usage(usage_event())
    assert sorted(path.name for path in logs_directory.iterdir()) == ["usage.jsonl"]


# 3. Best effort --------------------------------------------------------------


def test_unwritable_directory_does_not_raise(tmp_path):
    # "written best-effort so a ledger failure never fails a run"
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("a regular file", encoding="utf-8")
    led = ledger.Ledger(blocker / "logs")
    led.usage(usage_event())
    led.trace(trace_event())
    assert blocker.read_text(encoding="utf-8") == "a regular file"


# 4. A malformed event is our bug, not best effort ------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        usage_event(kind="bogus"),  # enum api | chat
        usage_event(tokens_in=-1),  # minimum 0
        {key: value for key, value in usage_event().items() if key != "model"},  # required
        usage_event(extra_field=1),  # additionalProperties false
    ],
)
def test_malformed_usage_event_raises(tmp_path, bad):
    # coding.md "Failure"/"validate every artifact at write"; the line schema rejects it
    logs_directory = tmp_path / "logs"
    led = ledger.Ledger(logs_directory)
    with pytest.raises(schemas.SchemaError):
        led.usage(bad)
    assert (
        not (logs_directory / "usage.jsonl").exists() or lines(logs_directory / "usage.jsonl") == []
    )


def test_malformed_trace_event_raises(tmp_path):
    logs_directory = tmp_path / "logs"
    led = ledger.Ledger(logs_directory)
    with pytest.raises(schemas.SchemaError):
        led.trace(trace_event(argv="run --now"))  # argv must be an array
    assert (
        not (logs_directory / "trace.jsonl").exists() or lines(logs_directory / "trace.jsonl") == []
    )


# 5. read() -------------------------------------------------------------------


def test_read_missing_file_is_empty(tmp_path):
    assert ledger.read(tmp_path / "nope.jsonl") == []


def test_read_skips_and_counts_malformed_lines(tmp_path, capsys):
    # read(): "a malformed line is skipped, never raised, so a damaged ledger
    # still reads, and the number skipped is printed as one note on stderr"
    logs_directory = tmp_path / "logs"
    led = ledger.Ledger(logs_directory)
    led.usage(usage_event(tag="first"))
    led.usage(usage_event(tag="second"))
    usage_path = logs_directory / "usage.jsonl"
    good = usage_path.read_text(encoding="utf-8").splitlines()
    damaged = [
        good[0],
        "{this is not json",
        json.dumps({"at": "2026-10-03T00:00:00Z", "stage": "x"}),  # fails the line schema
        good[1],
    ]
    usage_path.write_text("\n".join(damaged) + "\n", encoding="utf-8")
    out = ledger.read(usage_path)
    assert [record["tag"] for record in out] == ["first", "second"]
    error_output = capsys.readouterr().err
    assert "2 malformed" in error_output and "usage.jsonl" in error_output


def test_read_returns_what_was_written(tmp_path, capsys):
    logs_directory = tmp_path / "logs"
    led = ledger.Ledger(logs_directory)
    led.trace(trace_event(exit_code=1))
    out = ledger.read(logs_directory / "trace.jsonl")
    assert len(out) == 1
    assert out[0]["exit_code"] == 1 and out[0]["argv"] == ["run", "--now"]
    assert "malformed" not in capsys.readouterr().err


def test_read_with_rotated_reads_older_generations_first(tmp_path):
    logs_directory = tmp_path / "logs"
    led = ledger.Ledger(logs_directory)
    for tag in ("old", "mid", "new"):
        led.usage(usage_event(tag=tag))
    path = logs_directory / "usage.jsonl"
    older = path.read_text().splitlines()
    path.write_text(older[2] + "\n")
    path.with_name("usage.jsonl.1").write_text(older[1] + "\n")
    path.with_name("usage.jsonl.2").write_text(older[0] + "\n")
    assert [record["tag"] for record in ledger.read(path)] == ["new"]
    assert [record["tag"] for record in ledger.read(path, with_rotated=True)] == [
        "old",
        "mid",
        "new",
    ]


def test_a_tool_may_stamp_its_lines_with_the_verb_the_run_and_the_client(tmp_path):
    folder = tmp_path / "logs"
    ledger.Ledger(folder).usage(
        usage_event(
            invocation="collect --auto",
            run="abc123",
            client={"engine": "claude-code", "session": "s1", "precision": "exact"},
        )
    )
    (line,) = ledger.read(folder / "usage.jsonl")
    assert line["invocation"] == "collect --auto" and line["run"] == "abc123"
    assert line["client"]["session"] == "s1"
    with pytest.raises(schemas.SchemaError):
        ledger.Ledger(folder).usage(usage_event(invocation=3))
