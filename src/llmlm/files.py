"""Files: writing a file whole or not at all.

Every file a tool writes whole goes through `write_atomic` (a temp file in the same directory,
then a rename), so a crash leaves the old file or the new one, never a torn one. Appends are the
one other way a file grows.

Never imports: llm.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_atomic(path: Path, data: str | bytes, *, mode: int | None = None) -> Path:
    """Write `data` to `path` whole or not at all: a temp file in the same
    directory, flushed and fsynced, then os.replace (metalm coding.md,
    "Idempotence and writes"). `mode` sets the file's permissions (the temp
    file is created with them, so a secret is never briefly world-readable);
    without it an existing file keeps its mode. Text is UTF-8. On any
    failure the temp file is removed and the old file is left as it was."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode is None and path.exists():
        mode = path.stat().st_mode & 0o777
    raw = data.encode("utf-8") if isinstance(data, str) else data
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "wb") as file:
            file.write(raw)
            file.flush()
            os.fsync(file.fileno())
        os.chmod(temporary_path, 0o644 if mode is None else mode)
        os.replace(temporary_path, path)
    except BaseException:
        Path(temporary_path).unlink(missing_ok=True)
        raise
    return path
