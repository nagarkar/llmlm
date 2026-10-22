"""docs/generated/ matches the code (metalm guidelines/design.md#generated-docs, layer 3;
the hooks in docs/design/testing.md).

The pre-commit hook and the docs workflow regenerate it; this test catches a
commit that skipped both. Fix: `.venv/bin/metalm-gendocs`, then commit
docs/generated/.
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENDOCS = ROOT / ".venv" / "bin" / "metalm-gendocs"


def test_generated_docs_are_current():
    assert (
        GENDOCS.exists()
    ), f"{GENDOCS} missing: .venv/bin/pip install -e '.[dev]' (metalm-gendocs is a dev dependency)"
    out = subprocess.run(
        [str(GENDOCS), "--check", "--root", str(ROOT)], capture_output=True, text=True
    )
    assert out.returncode == 0, (
        f"docs/generated/ is stale; run .venv/bin/metalm-gendocs and commit it:\n"
        f"{out.stdout}{out.stderr}"
    )
