"""Schemas: the JSON schemas of the files llmlm owns, the ledger lines.

A schema lives in `src/llmlm/schemas/<name>.json`, every property described. `validate` checks
a document against it and names the first failing property. A tool's own reply schemas are the
tool's: it passes its validator to `llmlm.llm.Client`.

Never imports: llm, ledger.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

SCHEMAS_DIR = Path(__file__).parent / "schemas"


class SchemaError(ValueError):
    pass


def names() -> list[str]:
    return sorted(path.stem for path in SCHEMAS_DIR.glob("*.json"))


@cache
def schema(name: str) -> dict:
    path = SCHEMAS_DIR / f"{name}.json"
    if not path.exists():
        raise SchemaError(f"no schema named {name!r}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_against(name: str, schema_document: dict, document) -> None:
    """The first schema error, by property path, named in the message."""
    import jsonschema

    validator = jsonschema.Draft202012Validator(schema_document)
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.absolute_path))
    if errors:
        first_error = errors[0]
        where = ".".join(str(part) for part in first_error.absolute_path) or "(root)"
        raise SchemaError(f"{name}: {where}: {first_error.message}")


def validate(name: str, document) -> None:
    validate_against(name, schema(name), document)
