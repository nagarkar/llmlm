"""Prompts: versioned prompt files, one file per version, loaded by id.

A prompt is a file, `<directory>/<id>_v<N>.md`: TOML front matter above a markdown body with
`{placeholders}`. A change is a copy to the next version, never an edit in place;
`load(id)` takes the newest version, `load(id, version=N)` an exact one, so every prompt id and
version a ledger recorded still resolves. A prompt's sha256 travels in every artifact it
produced.

Prompts are searched in the directories a tool registered with `register_directory`, or in
the `directories=` one call passes; llmlm ships none. The same id and version in two directories
is an error naming both files, so a released version is never shadowed.

Front matter: `id`, `version`, `stage`, `output_schema`, `config_section`, and optionally
`expected_output_tokens`, the reply size `llmlm.estimate` assumes per call.

Never imports: llm, ledger.

Terms:
- **prompt**: one versioned file with front matter (id, version, stage, output schema,
  configuration section, expected output tokens) and a body.
"""

from __future__ import annotations

import hashlib
import re
import string
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_REQUIRED = ("id", "version", "stage", "output_schema", "config_section")
_OPTIONAL = ("expected_output_tokens",)
_registered: list[Path] = []


class PromptError(ValueError):
    pass


def register_directory(directory: Path) -> None:
    """Add `directory` to the prompts every call searches (once; a repeat is ignored)."""
    directory = Path(directory)
    if directory not in _registered:
        _registered.append(directory)


def search_directories() -> list[Path]:
    """Each registered directory, in the order they are searched."""
    return list(_registered)


@dataclass(frozen=True, slots=True)
class Prompt:
    id: str
    version: int
    stage: str
    output_schema: str
    config_section: str
    body: str
    path: Path
    sha256: str
    expected_output_tokens: int | None = None

    def placeholders(self) -> list[str]:
        return [
            field_name for _, field_name, _, _ in string.Formatter().parse(self.body) if field_name
        ]

    def render(self, variables: dict[str, str]) -> str:
        missing = [
            placeholder for placeholder in self.placeholders() if placeholder not in variables
        ]
        if missing:
            raise PromptError(f"prompt {self.id}: missing variables {missing}")
        return self.body.format(**{name: str(value) for name, value in variables.items()})


def _parse(path: Path) -> Prompt:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        raise PromptError(f"{path}: no TOML front matter")
    _, front, body = raw.split("---\n", 2)
    metadata = tomllib.loads(front)
    missing = [key for key in _REQUIRED if key not in metadata]
    if missing:
        raise PromptError(f"{path}: front matter missing {missing}")
    unknown = sorted(set(metadata) - set(_REQUIRED) - set(_OPTIONAL))
    if unknown:
        raise PromptError(f"{path}: unknown front matter keys {unknown}")
    return Prompt(
        id=str(metadata["id"]),
        version=int(metadata["version"]),
        stage=str(metadata["stage"]),
        output_schema=str(metadata["output_schema"]),
        config_section=str(metadata["config_section"]),
        body=body.lstrip("\n"),
        path=path,
        sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        expected_output_tokens=(
            int(metadata["expected_output_tokens"])
            if "expected_output_tokens" in metadata
            else None
        ),
    )


_FILENAME = re.compile(r"^(?P<id>.+)_v(?P<version>\d+)\.md$")


def _checked(path: Path) -> Prompt:
    """Parse `path` and decline it unless its name is `<id>_v<version>.md`
    as its front matter says."""
    match = _FILENAME.match(path.name)
    prompt = _parse(path)
    if match is None or match["id"] != prompt.id or int(match["version"]) != prompt.version:
        raise PromptError(
            f"{path.name}: the file name must be {prompt.id}_v{prompt.version}.md (id and version "
            "from its front matter); copy to a new version, never rename or edit one"
        )
    return prompt


def _searched(directories: Iterable[Path] | None) -> list[Path]:
    return (
        search_directories()
        if directories is None
        else [Path(directory) for directory in directories]
    )


def versions(prompt_id: str, directories: Iterable[Path] | None = None) -> dict[int, Path]:
    """Every version file of `prompt_id`, by version number."""
    out: dict[int, Path] = {}
    for directory in _searched(directories):
        for path in directory.glob(f"{prompt_id}_v*.md"):
            match = _FILENAME.match(path.name)
            if not (match and match["id"] == prompt_id):
                continue
            number = int(match["version"])
            if number in out and out[number] != path:
                raise PromptError(
                    f"prompt {prompt_id!r} version {number} is in two places: "
                    f"{out[number]} and {path}"
                )
            out[number] = path
    return out


def load(
    prompt_id: str, version: int | None = None, *, directories: Iterable[Path] | None = None
) -> Prompt:
    """The newest version of `prompt_id`, or exactly `version`. A file stem
    (`cleanup_v2`) is accepted as the id with its version."""
    searched = _searched(directories)
    match = _FILENAME.match(f"{prompt_id}.md")
    if match and version is None and not versions(prompt_id, searched):
        prompt_id, version = match["id"], int(match["version"])
    found = versions(prompt_id, searched)
    if not found:
        raise PromptError(
            f"no prompt named {prompt_id!r} under "
            f"{', '.join(str(directory) for directory in searched)}"
        )
    if version is None:
        version = max(found)
    if version not in found:
        raise PromptError(f"prompt {prompt_id!r} has no version {version} (has {sorted(found)})")
    return _checked(found[version])


def list_prompts(directories: Iterable[Path] | None = None) -> list[Prompt]:
    paths = [path for directory in _searched(directories) for path in directory.glob("*.md")]
    return [_checked(path) for path in sorted(paths, key=lambda path: (path.name, str(path)))]
