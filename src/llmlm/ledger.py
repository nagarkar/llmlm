"""Ledger: the JSONL logs of model spend and verbs.

Ledgers are JSONL under the directory the tool passes (`<root>/logs/`), one line per event, rotated
at 5 MB with two generations kept, written best-effort so a ledger failure never fails a run, each
with an off switch passed as `logs_config`.

- `usage.jsonl`   every model call: stage, prompt id, model, tokens (and cache reads and
                  writes where the vendor reports them), cost or null, kind = api | chat,
                  cached, tag, outcome.
- `trace.jsonl`   every verb: argv, client id, workspace, duration, exit code.
- `db-perf.jsonl` query shapes and timings; off by default, no writer yet.

A line is validated against its schema (`usage_line`, `trace_line`, in `llmlm.schemas`) where
it is produced, and again where it is read. A line our own code shaped wrong is a bug, so writing
one raises `SchemaError`; reading skips a malformed line and prints how many on stderr.

Never imports: llm, prices, usage.

Terms:
- **ledger**: one of three JSONL logs: `trace.jsonl`, `usage.jsonl`, `db-perf.jsonl`.
- **usage**: model spend: `usage.jsonl`, one line per model call.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from . import clock as clock_mod
from . import schemas

ROTATE_BYTES = 5 * 1024 * 1024
GENERATIONS = 2
DEFAULTS = {"usage": True, "trace": True, "db_perf": False}


class Ledger:
    def __init__(
        self,
        directory: Path,
        logs_config: dict | None = None,
        *,
        clock: clock_mod.Clock = clock_mod.utc_now,
    ):
        self.directory = Path(directory)
        self.clock = clock
        self.enabled = DEFAULTS | {
            log_name: bool(enabled) for log_name, enabled in (logs_config or {}).items()
        }

    def usage(self, event: dict) -> None:
        self._write("usage", event)

    def trace(self, event: dict) -> None:
        self._write("trace", event)

    def _write(self, name: str, event: dict) -> None:
        if not self.enabled.get(name, False):
            return
        record = {"at": clock_mod.iso(self.clock())} | event
        # Every artifact line is validated where it is produced. A line
        # our own code shaped wrong is a bug, so this one is not best-effort:
        # it raises schemas.SchemaError.
        schemas.validate(f"{name}_line", record)
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{name.replace('_', '-')}.jsonl"
            self._rotate(path)
            line = json.dumps(record, ensure_ascii=False, sort_keys=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
        except OSError:
            return  # best effort, never fatal

    @staticmethod
    def _rotate(path: Path) -> None:
        if not path.exists() or path.stat().st_size < ROTATE_BYTES:
            return
        for generation in range(GENERATIONS, 0, -1):
            older = path.with_name(f"{path.name}.{generation}")
            newer = path if generation == 1 else path.with_name(f"{path.name}.{generation - 1}")
            if newer.exists():
                os.replace(newer, older)


def read(path: Path, *, with_rotated: bool = False) -> list[dict]:
    """Lines validated at read; a malformed line is skipped, never
    raised, so a damaged ledger still reads, and the number skipped is
    printed as one note on stderr. `with_rotated` reads the older
    generations (`<name>.2`, `<name>.1`) first, oldest line first."""
    name = path.stem.replace("-", "_") + "_line"
    files = [path]
    if with_rotated:
        files = [
            path.with_name(f"{path.name}.{generation}") for generation in range(GENERATIONS, 0, -1)
        ] + files
    out: list[dict] = []
    for file in files:
        out.extend(_read_file(file, name))
    return out


def _read_file(path: Path, name: str) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            schemas.validate(name, record)
        except (json.JSONDecodeError, schemas.SchemaError):
            skipped += 1
            continue
        out.append(record)
    if skipped:
        print(f"note: {skipped} malformed line(s) skipped in {path.name}", file=sys.stderr)
    return out
