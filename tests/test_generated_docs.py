"""docs/generated/ matches the code (metalm guidelines/design.md#generated-docs, layer 3;
the hooks in docs/design/testing.md).

The pre-commit hook and the docs workflow regenerate it; this test catches a
commit that skipped both. Fix: `.venv/bin/metalm-gendocs`, then commit
docs/generated/.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The launcher sits beside whichever interpreter runs the tests: .venv/bin here, the runner's
# python in CI (pip install -e ".[dev]" puts it there).
GENDOCS = Path(sys.executable).parent / "metalm-gendocs"


def test_generated_docs_are_current():
    assert (
        GENDOCS.exists()
    ), f"{GENDOCS} missing: pip install -e '.[dev]' (metalm-gendocs is a dev dependency)"
    out = subprocess.run(
        [str(GENDOCS), "--dry-run", "--root", str(ROOT)], capture_output=True, text=True
    )
    assert out.returncode == 0, (
        f"docs/generated/ is stale; run .venv/bin/metalm-gendocs and commit it:\n"
        f"{out.stdout}{out.stderr}"
    )
