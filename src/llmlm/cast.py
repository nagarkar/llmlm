"""Cast: one look for a character across many pictures, on a folder of images and plain data.

A character is rendered once as a plate, the caller picks one plate, and every later render that
names the character gets that plate attached as a reference. The caller hands over plain data: its
`Character` list (each with its picked plate file, or none), a `Request` (description, named
characters, an optional prior plate, the look text, a count, an optional input image) and a `Pin`
(model and size). This module resolves the references and reports each one it cannot attach as a
sentence, writes the reference text, composes the prompt, renders the candidates through
`llmlm.images.generate_image`, names each file `<slug>-<deschash>-<stylehash>-<NN>.<ext>`, stores
the prompt, look, model, cast and date inside the image file and reads them back, and reads and
writes the folder's pin file.

It owns, inside the caller's image folder: the candidate files and `settings.toml` (the pin).
The image fields are stored under the caller's prefix (`<prefix>:prompt`, `:caption`, `:style`,
`:model`, `:size`, `:off-model`, `:cast`, `:prior`, `:date`, then the request's `extra`); the
look text is stored under `style`, the name the first images were written with.

The ledger line of a render is the caller's: pass a `generator` that calls `generate_image` and
records `llmlm.images.usage_line`, or record one line per name in `Rendered.written`.

Never imports: llm, usage, estimate, prices, ledger, prompts (enforced by the import contract
"the cast calls only the image road" in pyproject.toml).

Decisions:
- The cast code lives here and authorlm and musiclm both call it. Here: looking up a character's
  picked plate and reporting one that is missing or unpicked; the reference text sent with the
  plates; rendering N candidates with the plates attached; candidate file names and the hashes
  behind them; the fields stored inside each image file and reading them back; the pin file and
  the off-model mark. In the caller: finding its characters, its look rules, its description
  files, and its picks (authorlm: the embed line; musiclm: the request file). Why: musiclm's
  stills need the same likeness machinery authorlm's illustrations have, and two copies would
  drift. It never reads a manuscript: it works on a folder of images and the plain data it is
  handed. Enforced by:
  tests/test_cast.py::test_references_come_back_in_cast_order_then_the_prior_plate. (owner)
- A render that names a character with no picked plate does not run text-only; it stops with
  every reason as a sentence. Enforced by:
  tests/test_cast.py::test_render_declines_text_only_when_a_named_character_has_no_plate. (owner)
- "of this book" in the reference text and the `authorlm:` prefix on stored fields are settings
  (`Settings.series`, `Settings.prefix`), so authorlm's existing images keep reading correctly.
  Enforced by: tests/test_cast.py::test_an_image_authorlm_wrote_before_the_move_still_reads. (owner)
- Candidate names and their hashes are byte for byte what authorlm wrote before the move.
  Enforced by: tests/test_cast.py::test_the_hashes_are_what_authorlm_always_wrote.

Terms:
- **character**: a figure or a stage that must look the same in every picture; a slug and, once
  picked, a plate.
- **plate**: the one candidate image the caller picked for a character; the file attached as a
  reference.
- **reference**: a plate attached to a render, in the order the reference text names it: the
  request's cast, then its prior plate.
- **prior plate**: one plate a render follows for stage and composition, not for likeness.
- **candidate**: one rendered image file of a request; a request's candidates are numbered on
  from 01 and the number never resets.
- **look text**: the rendering rules sent after the description; the caller writes it and its
  hash is the `stylehash` part of a candidate name.
- **pin**: the model and size a folder's renders run on, in its `settings.toml`; stamped once
  from the caller's default and owned by the folder from then on.
- **off-model**: a render made on a model or size other than the pin's; marked in the image.
"""

from __future__ import annotations

import hashlib
import re
import struct
import tomllib
import zlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from . import clock as clock_module
from . import files, images

MAX_CAST = 3
PIN_FILE = "settings.toml"
PIN_HEADER = (
    "# settings.toml: the image model and size this folder's renders run on (the pin).\n"
    "# Stamped once from the tool's default and owned here from then on. A render on any\n"
    "# other model or size is marked off-model.\n"
)

_CANDIDATE = re.compile(
    r"^(?P<key>.+)-(?P<desc>[0-9a-f]{8})-(?P<style>[0-9a-f]{4})"
    r"-(?P<src>i?)(?P<n>\d{2})\.(?P<ext>png|jpe?g)$"
)
_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_JPEG_SIG = b"\xff\xd8"
_SMALL_WORDS = {"of", "the", "and", "in", "a", "on"}

# (prompt, input image or None, reference plates in order) -> image bytes
Generator = Callable[[str, bytes | None, list[bytes]], bytes]


class CastError(LookupError):
    """A render was declined: a named character has no plate to attach. The message carries
    every reason."""


@dataclass(frozen=True)
class Settings:
    """The caller's words. `series` completes "the Dragon of ..." in the reference text;
    `prefix` goes before every stored field name; `undefined_hint` (with `{slug}`) ends the
    sentence for a name no character carries, saying where the caller defines one."""

    series: str = "this series"
    prefix: str = "llmlm"
    undefined_hint: str = ""


@dataclass(frozen=True)
class Character:
    """`plate` is the picked plate file, None while unpicked. `where` is the caller's pointer
    to the place the character is defined, quoted in the unpicked sentence."""

    slug: str
    plate: Path | None = None
    where: str = ""


@dataclass(frozen=True)
class Request:
    """One picture to render. `cast` names up to `MAX_CAST` characters, `prior` one plate to
    follow. `where` starts every sentence about this request. `caption` and `extra` are stored
    in the image and sent to no model."""

    description: str
    cast: tuple[str, ...] = ()
    prior: str | None = None
    look: str = ""
    count: int = 1
    input_image: bytes | None = None
    where: str = ""
    caption: str = ""
    extra: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Reference:
    slug: str
    role: str  # "cast" or "prior"
    path: Path

    @property
    def name(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class Resolved:
    references: list[Reference]
    errors: list[str]


@dataclass(frozen=True)
class Pin:
    """What one render runs on, and what the folder is pinned to."""

    model: str
    size: str
    off_model: bool
    pinned_model: str
    pinned_size: str


@dataclass(frozen=True)
class Rendered:
    written: list[str]
    desc_hash: str
    style_hash: str
    composed: str
    references: list[Reference]
    model: str
    off_model: bool


# ------------------------------------------------------- names and their hashes


def desc_hash(description: str) -> str:
    """8 hex of the whitespace-collapsed description: which text a candidate was made for."""
    canonical = " ".join(description.split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def style_hash(look: str) -> str:
    """4 hex of the look text exactly as the model sees it; '0000' when there is none."""
    if not look:
        return "0000"
    return hashlib.sha256(look.encode("utf-8")).hexdigest()[:4]


def slug(text: str) -> str:
    """A file-name slug from the first four words of a description."""
    words = re.sub(r"[^a-z0-9 ]", "", text.lower()).split()
    return "-".join(words[:4]) or "illustration"


def display_name(character_slug: str) -> str:
    """`tower-of-light` -> `Tower of Light`: how the reference text names a character."""
    return " ".join(
        word if (word in _SMALL_WORDS and position) else word.capitalize()
        for position, word in enumerate(character_slug.split("-"))
    )


def image_ext(image: bytes) -> str:
    """The extension the bytes call for: the model decides the format, the name must match."""
    if image.startswith(_PNG_SIG):
        return "png"
    if image.startswith(_JPEG_SIG):
        return "jpg"
    return "png"  # unknown: stored as it came under the default name


def candidate_name(
    slug: str, desc: str, style: str, number: int, ext: str, *, imported: bool = False
) -> str:
    """`<slug>-<deschash>-<stylehash>-<NN>.<ext>`; `i` before the number marks an image that
    was brought in, not rendered."""
    return f"{slug}-{desc}-{style}-{'i' if imported else ''}{number:02d}.{ext}"


def candidates(folder: Path, slug: str | None = "") -> list[dict]:
    """The candidate files in `folder` as {name, key, desc, style, src, n, ext}. With a `slug`,
    that slug's only, in number order; '' lists every candidate by name; None (a request with
    no slug yet) has none."""
    folder = Path(folder)
    if slug is None or not folder.is_dir():
        return []
    found = []
    for path in sorted(folder.iterdir()):
        match = _CANDIDATE.match(path.name)
        if match:
            found.append({"name": path.name, **match.groupdict()})
    if not slug:
        return found
    return sorted((c for c in found if c["key"] == slug), key=lambda c: int(c["n"]))


def next_number(folder: Path, slug: str) -> int:
    return max((int(c["n"]) for c in candidates(folder, slug)), default=0) + 1


# ------------------------------------------------------------------ references


def resolve_references(
    characters: Iterable[Character], request: Request, settings: Settings | None = None
) -> Resolved:
    """The plates a render of `request` attaches, in order (its cast, then its prior plate), and
    every reason one could not be attached, as a sentence. Errors are returned, never raised:
    `attach` and `render` raise on them."""
    settings = settings or Settings()
    by_slug: dict[str, Character] = {}
    for character in characters:
        by_slug.setdefault(character.slug, character)
    start = f"{request.where}: " if request.where else ""
    references: list[Reference] = []
    errors: list[str] = []
    names = list(request.cast)
    if len(names) > MAX_CAST:
        errors.append(f"{start}cast names {len(names)} ids; at most {MAX_CAST}")

    def attach_one(name: str, role: str) -> None:
        character = by_slug.get(name)
        if character is None:
            hint = settings.undefined_hint.format(slug=name)
            errors.append(f"{start}{role} '{name}' is not defined" + (f" — {hint}" if hint else ""))
            return
        if character.plate is None:
            errors.append(
                f"{start}{role} plate '{name}' has no pick — render and pick it first"
                + (f" ({character.where})" if character.where else "")
            )
            return
        plate = Path(character.plate)
        if not plate.exists():
            errors.append(
                f"{start}{role} plate '{name}' is picked but its file {plate.name} is missing"
            )
            return
        references.append(Reference(name, role, plate))

    seen: set[str] = set()
    for name in names[:MAX_CAST]:
        if name not in seen:
            seen.add(name)
            attach_one(name, "cast")
    if request.prior:
        attach_one(request.prior, "prior")
    return Resolved(references, errors)


def attach(
    characters: Iterable[Character], request: Request, settings: Settings | None = None
) -> list[Reference]:
    """The request's references, or `CastError` naming every one that could not be attached:
    a request that names a character is never rendered text-only."""
    resolved = resolve_references(characters, request, settings)
    if resolved.errors:
        raise CastError("cast unresolved — " + "; ".join(resolved.errors))
    declared = list(request.cast) + ([request.prior] if request.prior else [])
    attached = {reference.slug for reference in resolved.references}
    missing = [name for name in declared if name not in attached]
    if missing:
        start = f"{request.where} " if request.where else ""
        raise CastError(
            f"{start}declares {', '.join(missing)} but no reference resolved for "
            f"{'it' if len(missing) == 1 else 'them'}; not rendering text-only"
        )
    return resolved.references


def reference_block(references: list[Reference], settings: Settings | None = None) -> str:
    """The reference text: one line per attached image, in attachment order. It identifies and
    nothing more; what a character means is never sent, because likeness is the model's
    business and meaning is the caller's."""
    if not references:
        return ""
    settings = settings or Settings()
    lines = ["REFERENCES (the attached images, in this order — likeness only, never subject):"]
    for number, reference in enumerate(references, 1):
        if reference.role == "prior":
            lines.append(
                f"Image {number}: the previous plate in this series; "
                "keep its stage and composition."
            )
        else:
            lines.append(
                f"Image {number}: the {display_name(reference.slug)} of {settings.series}. "
                "Draw the same figure again."
            )
    return "\n".join(lines)


def compose(description: str, reference_text: str = "", look: str = "") -> str:
    """The prompt a render sends: the description first, the reference text, the look last.
    Image models weight early tokens and settle a conflict toward whatever claims authority, so
    the description is given precedence over the look in words. A bare description goes as it
    is."""
    if not (reference_text or look):
        return description
    parts = [
        "DEPICT (content commands — costume, era, objects, and composition here override "
        f"anything the style implies):\n{description}"
    ]
    if reference_text:
        parts.append(reference_text)
    if look:
        parts.append("STYLE (technique and rendering only):\n" + look)
    return "\n\n".join(parts)


# -------------------------------------------------- fields stored in the image


def _itxt_chunk(keyword: str, text: str) -> bytes:
    data = keyword.encode("latin-1") + b"\x00\x00\x00\x00\x00" + text.encode("utf-8")
    return (
        struct.pack(">I", len(data))
        + b"iTXt"
        + data
        + struct.pack(">I", zlib.crc32(b"iTXt" + data) & 0xFFFFFFFF)
    )


def image_with_metadata(image: bytes, fields: Mapping[str, str]) -> bytes:
    """`image` with `fields` stored inside it (full names, prefix included): iTXt chunks in a
    PNG, COM segments in a JPEG. An empty value is left out. A copy for people, so a file
    explains itself away from its folder; an unknown format is returned untouched."""
    if image.startswith(_PNG_SIG):
        ihdr_end = 8 + 8 + 13 + 4  # signature, then the IHDR chunk's header, data and crc
        chunks = b"".join(_itxt_chunk(key, value) for key, value in fields.items() if value)
        return image[:ihdr_end] + chunks + image[ihdr_end:]
    if image.startswith(_JPEG_SIG):
        segments = b""
        for key, value in fields.items():
            if not value:
                continue
            body = f"{key}: {value}".encode()[:65531]
            segments += b"\xff\xfe" + struct.pack(">H", len(body) + 2) + body
        return image[:2] + segments + image[2:]
    return image


def read_metadata(path: Path, prefix: str | None = None) -> dict[str, str]:
    """The fields `image_with_metadata` stored, read back; empty for an image with none. With a
    `prefix`, only that prefix's fields, named without it."""
    data = Path(path).read_bytes()
    out: dict[str, str] = {}
    if data.startswith(_PNG_SIG):
        position = 8
        while position + 8 <= len(data):
            length = int.from_bytes(data[position : position + 4], "big")
            kind = data[position + 4 : position + 8]
            body = data[position + 8 : position + 8 + length]
            if kind == b"iTXt":
                try:
                    keyword, rest = body.split(b"\x00", 1)
                    rest = rest[2:]  # compression flag and method
                    _language, rest = rest.split(b"\x00", 1)
                    _translated, text = rest.split(b"\x00", 1)
                    out[keyword.decode("latin-1")] = text.decode("utf-8", "replace")
                except ValueError:
                    pass
            if kind == b"IEND":
                break
            position += 12 + length
    elif data.startswith(_JPEG_SIG):
        position = 2
        while position + 4 <= len(data) and data[position] == 0xFF:
            marker = data[position + 1]
            if marker in (0xD8, 0xD9):
                position += 2
                continue
            length = int.from_bytes(data[position + 2 : position + 4], "big")
            if marker == 0xFE:
                body = data[position + 4 : position + 2 + length].decode("utf-8", "replace")
                if ": " in body:
                    key, value = body.split(": ", 1)
                    out[key] = value
            if marker == 0xDA:
                break
            position += 2 + length
    if prefix is None:
        return out
    start = f"{prefix}:"
    return {key[len(start) :]: value for key, value in out.items() if key.startswith(start)}


def _cast_record(references: list[Reference], role: str) -> str:
    return ";".join(f"{r.slug}={r.name}" for r in references if r.role == role)


def recorded_references(fields: Mapping[str, str], prefix: str) -> set[tuple[str, str, str]]:
    """The (role, slug, plate file name) set an image was rendered against, from its stored
    fields (`read_metadata(path)`). Compare with the current references to tell a picture whose
    plates have since been re-picked."""
    out: set[tuple[str, str, str]] = set()
    for role in ("cast", "prior"):
        for pair in (fields.get(f"{prefix}:{role}") or "").split(";"):
            if "=" in pair:
                name, plate = pair.split("=", 1)
                out.add((role, name, plate))
    return out


# ------------------------------------------------------------------------ pin


def pin_path(folder: Path) -> Path:
    return Path(folder) / PIN_FILE


def load_pin(folder: Path) -> dict[str, str] | None:
    """The folder's pin as {model, image_size}; None when it has none yet."""
    path = pin_path(folder)
    if not path.exists():
        return None
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"{path} is not valid TOML: {error}") from error
    return {"model": str(data.get("model", "")), "image_size": str(data.get("image_size", ""))}


def write_pin(folder: Path, model: str, image_size: str, *, header: str = PIN_HEADER) -> Path:
    """Write the pin whole. `header` is the comment block a tool puts above it (how to change
    the pin with that tool's own verb)."""
    return files.write_atomic(
        pin_path(folder), f'{header}\nmodel = "{model}"\nimage_size = "{image_size}"\n'
    )


def ensure_pin(
    folder: Path, model: str, image_size: str, *, header: str = PIN_HEADER
) -> dict[str, str]:
    """The folder's pin, stamped from the caller's default (`model`, `image_size`) when it has
    none. Copy, then own: once stamped, the default no longer reaches this folder."""
    pin = load_pin(folder)
    if pin is not None:
        return pin
    write_pin(folder, model, image_size, header=header)
    return {"model": model, "image_size": image_size}


def resolve_pin(
    pinned: Mapping[str, str], model: str | None = None, size: str | None = None
) -> Pin:
    """What a render runs on: the pin (`load_pin` or `ensure_pin`), unless another model or
    size was named for this render, which is allowed and marked off-model, never silent."""
    runs_on = model or pinned["model"]
    at_size = size if size is not None else pinned["image_size"]
    return Pin(
        model=runs_on,
        size=at_size,
        off_model=runs_on != pinned["model"] or at_size != pinned["image_size"],
        pinned_model=pinned["model"],
        pinned_size=pinned["image_size"],
    )


# --------------------------------------------------------------------- render


def render(
    folder: Path,
    slug: str,
    request: Request,
    characters: Iterable[Character],
    pin: Pin,
    settings: Settings | None = None,
    *,
    generator: Generator | None = None,
    now: clock_module.Clock = clock_module.utc_now,
) -> Rendered:
    """Render `request.count` new candidates into `folder` under `slug`, each with the request's
    plates attached and the record of what it was made from stored inside it. `CastError`, with
    nothing written and no model called, when a named character has no plate. Picking a
    candidate is the caller's act."""
    settings = settings or Settings()
    folder = Path(folder)
    references = attach(characters, request, settings)
    composed = compose(request.description, reference_block(references, settings), request.look)
    plates = [reference.path.read_bytes() for reference in references]
    if generator is None:

        def generator(prompt: str, input_image: bytes | None, plates: list[bytes]) -> bytes:
            inputs = ([input_image] if input_image is not None else []) + plates
            return images.generate_image(pin.model, prompt, images=inputs, size=pin.size)

    deschash = desc_hash(request.description)
    stylehash = style_hash(request.look)
    first = next_number(folder, slug)
    prefix = settings.prefix
    written: list[str] = []
    for offset in range(request.count):
        image = generator(composed, request.input_image, plates)
        name = candidate_name(slug, deschash, stylehash, first + offset, image_ext(image))
        stored = {
            f"{prefix}:prompt": request.description,
            f"{prefix}:caption": request.caption,
            f"{prefix}:style": request.look,
            f"{prefix}:model": pin.model,
            f"{prefix}:size": pin.size,
            f"{prefix}:off-model": "true" if pin.off_model else "",
            f"{prefix}:cast": _cast_record(references, "cast"),
            f"{prefix}:prior": _cast_record(references, "prior"),
            f"{prefix}:date": now().strftime("%Y-%m-%d"),
            **{f"{prefix}:{key}": value for key, value in request.extra.items()},
        }
        files.write_atomic(folder / name, image_with_metadata(image, stored))
        written.append(name)
    return Rendered(
        written=written,
        desc_hash=deschash,
        style_hash=stylehash,
        composed=composed,
        references=references,
        model=pin.model,
        off_model=pin.off_model,
    )
