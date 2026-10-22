"""The one module that reads the wall clock (metalm coding.md, "Module
shape": inject `now` wherever time matters). Everything else takes a
`clock` (a no-argument callable returning an aware UTC datetime) or a
`now`, defaulting to `utc_now`, so a test pins time instead of waiting."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso(when: datetime) -> str:
    """`2026-01-02T03:04:05Z`: the stamp every record carries."""
    return when.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def epoch_seconds() -> float:
    """Seconds since the epoch, for pacing across processes."""
    return time.time()
