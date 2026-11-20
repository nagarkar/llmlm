"""Cast (`src/llmlm/cast.py`): picked plates attached to every render that names a character.

Offline: every render goes through a fake generator or a fake `generate_image`.
"""

import struct
import zlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from llmlm import cast, images


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


PNG = (
    b"\x89PNG\r\n\x1a\n"
    + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    + _chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
    + _chunk(b"IEND", b"")
)
JPG = b"\xff\xd8\xff\xdb\x00\x04\x00\x00\xff\xda\x00\x02rest\xff\xd9"
NOW = datetime(2026, 11, 20, 23, 30, tzinfo=UTC)
AUTHORLM = cast.Settings(
    series="this book", prefix="authorlm", undefined_hint="no tag carries 'cast-id: {slug}'"
)


def _plate(folder: Path, name: str, blob: bytes = PNG) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(blob + name.encode())
    return path


def _pin(model="gemini/test-image", size="1024x1024"):
    return cast.resolve_pin({"model": model, "image_size": size})


class Generator:
    def __init__(self, blob=PNG):
        self.calls = []
        self.blob = blob

    def __call__(self, prompt, input_image, references):
        self.calls.append((prompt, input_image, list(references)))
        return self.blob


# ------------------------------------------------------------------ references


def test_references_come_back_in_cast_order_then_the_prior_plate(tmp_path):
    dragon = _plate(tmp_path, "dragon-aaaaaaaa-0000-01.png")
    seeker = _plate(tmp_path, "seeker-bbbbbbbb-0000-02.png")
    stage = _plate(tmp_path, "stage-cccccccc-0000-01.png")
    characters = [
        cast.Character("seeker", seeker),
        cast.Character("dragon", dragon),
        cast.Character("stage", stage),
    ]
    request = cast.Request("the rider meets the serpent", cast=("dragon", "seeker"), prior="stage")
    resolved = cast.resolve_references(characters, request)
    assert resolved.errors == []
    assert [(r.slug, r.role, r.name) for r in resolved.references] == [
        ("dragon", "cast", dragon.name),
        ("seeker", "cast", seeker.name),
        ("stage", "prior", stage.name),
    ]


def test_a_missing_or_unpicked_character_is_a_sentence_never_a_silent_omission(tmp_path):
    characters = [cast.Character("dragon", None, where="cast.md:5")]
    request = cast.Request("x", cast=("dragon", "ghost"), where="01-road.md:12")
    resolved = cast.resolve_references(characters, request, AUTHORLM)
    assert resolved.references == []
    assert resolved.errors == [
        "01-road.md:12: cast plate 'dragon' has no pick — render and pick it first (cast.md:5)",
        "01-road.md:12: cast 'ghost' is not defined — no tag carries 'cast-id: ghost'",
    ]
    plain = cast.resolve_references([cast.Character("dragon")], cast.Request("x", prior="dragon"))
    assert plain.errors == ["prior plate 'dragon' has no pick — render and pick it first"]
    assert cast.resolve_references([], cast.Request("x", cast=("ghost",))).errors == [
        "cast 'ghost' is not defined"
    ]


def test_a_picked_plate_whose_file_is_gone_is_reported(tmp_path):
    characters = [cast.Character("dragon", tmp_path / "gone.png")]
    resolved = cast.resolve_references(characters, cast.Request("x", cast=("dragon",)))
    assert resolved.references == []
    assert resolved.errors == ["cast plate 'dragon' is picked but its file gone.png is missing"]


def test_more_than_three_cast_names_is_an_error_and_a_repeat_attaches_once(tmp_path):
    plate = _plate(tmp_path, "a-aaaaaaaa-0000-01.png")
    characters = [cast.Character(name, plate) for name in "abcd"]
    crowded = cast.resolve_references(
        characters, cast.Request("x", cast=("a", "b", "a", "c", "d"), where="f:1")
    )
    assert crowded.errors == ["f:1: cast names 5 ids; at most 3"]
    assert [r.slug for r in crowded.references] == ["a", "b"]


def test_attach_declines_when_a_named_character_has_no_plate():
    with pytest.raises(cast.CastError, match="cast unresolved — cast 'ghost' is not defined"):
        cast.attach([], cast.Request("x", cast=("ghost",)))
    assert issubclass(cast.CastError, LookupError)


# --------------------------------------------------------------- reference text


def test_the_reference_text_identifies_each_image_and_the_series_is_a_setting(tmp_path):
    refs = [
        cast.Reference("tower-of-light", "cast", tmp_path / "t.png"),
        cast.Reference("stage", "prior", tmp_path / "s.png"),
    ]
    assert cast.reference_block(refs, AUTHORLM) == (
        "REFERENCES (the attached images, in this order — likeness only, never subject):\n"
        "Image 1: the Tower of Light of this book. Draw the same figure again.\n"
        "Image 2: the previous plate in this series; keep its stage and composition."
    )
    video = cast.Settings(series="this music video")
    assert "Image 1: the Tower of Light of this music video. Draw" in cast.reference_block(
        refs, video
    )
    assert cast.reference_block([], AUTHORLM) == ""


def test_the_composed_prompt_puts_the_description_first_then_references_then_the_look():
    assert cast.compose("a lone pilgrim") == "a lone pilgrim"
    assert cast.compose("a lone pilgrim", "REFS", "ink on paper") == (
        "DEPICT (content commands — costume, era, objects, and composition here override "
        "anything the style implies):\na lone pilgrim\n\nREFS\n\n"
        "STYLE (technique and rendering only):\nink on paper"
    )
    assert cast.compose("a", look="ink").endswith("a\n\nSTYLE (technique and rendering only):\nink")
    assert cast.compose("a", "REFS").endswith(":\na\n\nREFS")


# ------------------------------------------------------- names and their hashes


def test_the_hashes_are_what_authorlm_always_wrote():
    assert cast.desc_hash("a  lone\n pilgrim") == cast.desc_hash("a lone pilgrim") == "8d1ff657"
    assert cast.style_hash("") == "0000"
    assert cast.style_hash("ink on paper") == "55fe"
    assert cast.slug("A lone Pilgrim, with a wide hat!") == "a-lone-pilgrim-with"
    assert cast.slug("!!") == "illustration"
    assert cast.display_name("tower-of-light") == "Tower of Light"
    assert cast.display_name("the-seeker") == "The Seeker"


def test_candidate_names_parse_back_and_number_on_across_generations(tmp_path):
    assert cast.candidate_name("pilgrim", "8d1ff657", "0000", 3, "png") == (
        "pilgrim-8d1ff657-0000-03.png"
    )
    assert cast.candidate_name("pilgrim", "8d1ff657", "55fe", 4, "jpg", imported=True) == (
        "pilgrim-8d1ff657-55fe-i04.jpg"
    )
    for name in (
        "pilgrim-8d1ff657-0000-02.png",
        "pilgrim-8d1ff657-0000-01.png",
        "pilgrim-22222222-55fe-i03.jpg",
        "pilgrim-road-8d1ff657-0000-01.png",
        "notes.txt",
        "settings.toml",
    ):
        (tmp_path / name).write_bytes(b"")
    mine = cast.candidates(tmp_path, "pilgrim")
    assert [c["name"] for c in mine] == [
        "pilgrim-8d1ff657-0000-01.png",
        "pilgrim-8d1ff657-0000-02.png",
        "pilgrim-22222222-55fe-i03.jpg",
    ]
    assert mine[2] == {
        "name": "pilgrim-22222222-55fe-i03.jpg",
        "key": "pilgrim",
        "desc": "22222222",
        "style": "55fe",
        "src": "i",
        "n": "03",
        "ext": "jpg",
    }
    assert len(cast.candidates(tmp_path)) == 4
    assert cast.next_number(tmp_path, "pilgrim") == 4
    assert cast.next_number(tmp_path, "nobody") == 1
    assert cast.candidates(tmp_path / "absent") == [] and cast.candidates(tmp_path, None) == []


# -------------------------------------------------- fields stored in the image


@pytest.mark.parametrize("blob", [PNG, JPG])
def test_fields_stored_in_an_image_read_back_and_empty_ones_are_left_out(tmp_path, blob):
    fields = {"authorlm:prompt": "a lone pilgrim — très bien", "authorlm:caption": "", "x:n": "1"}
    stamped = cast.image_with_metadata(blob, fields)
    path = tmp_path / "image"
    path.write_bytes(stamped)
    assert cast.read_metadata(path) == {"authorlm:prompt": "a lone pilgrim — très bien", "x:n": "1"}
    assert cast.read_metadata(path, prefix="authorlm") == {"prompt": "a lone pilgrim — très bien"}
    assert cast.image_ext(stamped) == ("png" if blob is PNG else "jpg")


def test_an_image_authorlm_wrote_before_the_move_still_reads(tmp_path):
    """Bytes built the way authorlm's `_itxt_chunk` built them, not by this module."""
    data = b"authorlm:cast" + b"\x00\x00\x00\x00\x00" + b"dragon=d-01.png;seeker=s-02.png"
    old = PNG[:33] + _chunk(b"iTXt", data) + PNG[33:]
    path = tmp_path / "old.png"
    path.write_bytes(old)
    fields = cast.read_metadata(path)
    assert fields == {"authorlm:cast": "dragon=d-01.png;seeker=s-02.png"}
    assert cast.recorded_references(fields, "authorlm") == {
        ("cast", "dragon", "d-01.png"),
        ("cast", "seeker", "s-02.png"),
    }
    assert cast.recorded_references({}, "authorlm") == set()


def test_an_unknown_format_is_stored_untouched_and_carries_nothing(tmp_path):
    assert cast.image_with_metadata(b"GIF89a", {"a": "b"}) == b"GIF89a"
    (tmp_path / "x").write_bytes(b"GIF89a")
    assert cast.read_metadata(tmp_path / "x") == {}


# ------------------------------------------------------------------------ pin


def test_the_pin_is_stamped_once_and_owned_by_the_folder_from_then_on(tmp_path):
    assert cast.load_pin(tmp_path) is None
    pin = cast.ensure_pin(tmp_path, "openai/test-image", "1024x1536")
    assert pin == {"model": "openai/test-image", "image_size": "1024x1536"}
    assert cast.pin_path(tmp_path) == tmp_path / "settings.toml"
    assert cast.ensure_pin(tmp_path, "gemini/other", "1024x1024") == pin
    cast.write_pin(tmp_path, "openai/test-image", "1024x1024", header="# mine\n")
    assert cast.pin_path(tmp_path).read_text() == (
        '# mine\n\nmodel = "openai/test-image"\nimage_size = "1024x1024"\n'
    )
    assert cast.load_pin(tmp_path)["image_size"] == "1024x1024"


def test_a_render_on_another_model_or_size_is_marked_off_model(tmp_path):
    pinned = {"model": "openai/test-image", "image_size": "1024x1536"}
    held = cast.resolve_pin(pinned)
    assert (held.model, held.size, held.off_model) == ("openai/test-image", "1024x1536", False)
    off = cast.resolve_pin(pinned, model="gemini/other")
    assert off.model == "gemini/other" and off.off_model and off.pinned_model == pinned["model"]
    assert cast.resolve_pin(pinned, size="").off_model
    assert not cast.resolve_pin(pinned, size="1024x1536").off_model


def test_a_broken_pin_file_is_a_sentence_naming_the_file(tmp_path):
    cast.pin_path(tmp_path).write_text("model = ")
    with pytest.raises(ValueError, match="settings.toml is not valid TOML"):
        cast.load_pin(tmp_path)


# --------------------------------------------------------------------- render


def test_render_writes_n_candidates_with_the_plates_attached_and_the_record_inside(tmp_path):
    folder = tmp_path / "stills"
    dragon = _plate(folder, "dragon-aaaaaaaa-0000-01.png")
    seeker = _plate(folder, "seeker-bbbbbbbb-0000-01.png")
    characters = [cast.Character("dragon", dragon), cast.Character("seeker", seeker)]
    request = cast.Request(
        "the rider  meets the serpent",
        cast=("dragon", "seeker"),
        look="ink on paper",
        count=2,
        caption="The meeting",
        extra={"shot": "12"},
    )
    generator = Generator()
    out = cast.render(
        folder,
        "meeting",
        request,
        characters,
        _pin(),
        AUTHORLM,
        generator=generator,
        now=lambda: NOW,
    )
    deschash = cast.desc_hash("the rider meets the serpent")
    assert out.written == [f"meeting-{deschash}-55fe-01.png", f"meeting-{deschash}-55fe-02.png"]
    assert (out.desc_hash, out.style_hash, out.model, out.off_model) == (
        deschash,
        "55fe",
        "gemini/test-image",
        False,
    )
    assert [(r.role, r.slug) for r in out.references] == [("cast", "dragon"), ("cast", "seeker")]
    assert len(generator.calls) == 2
    prompt, input_image, references = generator.calls[0]
    assert (
        prompt
        == out.composed
        == cast.compose(
            request.description, cast.reference_block(out.references, AUTHORLM), "ink on paper"
        )
    )
    assert input_image is None
    assert references == [dragon.read_bytes(), seeker.read_bytes()]
    assert cast.read_metadata(folder / out.written[0], prefix="authorlm") == {
        "prompt": "the rider  meets the serpent",
        "caption": "The meeting",
        "style": "ink on paper",
        "model": "gemini/test-image",
        "size": "1024x1024",
        "cast": f"dragon={dragon.name};seeker={seeker.name}",
        "date": "2026-11-20",
        "shot": "12",
    }
    again = cast.render(
        folder, "meeting", request, characters, _pin(), AUTHORLM, generator=generator
    )
    assert again.written[0].endswith("-03.png")


def test_render_stores_under_the_callers_prefix_and_marks_off_model(tmp_path):
    stage = _plate(tmp_path, "stage-aaaaaaaa-0000-01.png")
    pin = cast.resolve_pin({"model": "gemini/test-image", "image_size": ""}, model="gemini/other")
    out = cast.render(
        tmp_path,
        "verse-one",
        cast.Request("a river at dusk", prior="stage", input_image=b"seed"),
        [cast.Character("stage", stage)],
        pin,
        cast.Settings(prefix="musiclm"),
        generator=Generator(JPG),
        now=lambda: NOW,
    )
    assert out.written == [f"verse-one-{cast.desc_hash('a river at dusk')}-0000-01.jpg"]
    assert out.off_model
    assert cast.read_metadata(tmp_path / out.written[0], prefix="musiclm") == {
        "prompt": "a river at dusk",
        "model": "gemini/other",
        "off-model": "true",
        "prior": f"stage={stage.name}",
        "date": "2026-11-20",
    }


def test_render_declines_text_only_when_a_named_character_has_no_plate(tmp_path):
    generator = Generator()
    with pytest.raises(cast.CastError, match="has no pick"):
        cast.render(
            tmp_path / "stills",
            "meeting",
            cast.Request("the rider meets the serpent", cast=("dragon",)),
            [cast.Character("dragon")],
            _pin(),
            generator=generator,
        )
    assert generator.calls == [] and not (tmp_path / "stills").exists()


def test_render_goes_through_generate_image_with_the_input_image_first(tmp_path, monkeypatch):
    plate = _plate(tmp_path, "dragon-aaaaaaaa-0000-01.png")
    seen = []

    def fake(model, prompt, *, images, size):
        seen.append((model, prompt, images, size))
        return PNG

    monkeypatch.setattr(images, "generate_image", fake)
    out = cast.render(
        tmp_path,
        "meeting",
        cast.Request("the serpent", cast=("dragon",), input_image=b"seed"),
        [cast.Character("dragon", plate)],
        _pin(size="1024x1536"),
    )
    assert seen == [("gemini/test-image", out.composed, [b"seed", plate.read_bytes()], "1024x1536")]
    assert len(out.written) == 1
