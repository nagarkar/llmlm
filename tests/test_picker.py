"""Picker (`src/llmlm/picker/`): the picking page as HTML text, its two static files, and the
posted reply read back into plain data.

Offline: no model, no network beyond a loopback server on a spare port (the example host in
`tests/picker_example_host.py`), which is stopped before each test ends.

Sources of the expected values: nagarkar/llmlm#22 (the owner's ruling and the screen it asks for)
and the house rule that a served page runs under `Content-Security-Policy: default-src 'self'`
(no inline script, no style attribute, no outside request).
"""

import json
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from html.parser import HTMLParser
from importlib import resources
from pathlib import Path

import pytest

from llmlm import picker

ROOT = Path(__file__).resolve().parent.parent
HOSTILE = '<script>alert("x")</script> & "quoted" \'single\''
HOSTILE_FILE = 'dragon-"><img src=x onerror=alert(1)>-01.png'


def _page(**changes) -> picker.Page:
    """Two choices: a single-pick plate with one pick and an earlier-brief candidate, and a
    several-picks still with two picks in the order 03 then 01."""
    values = {
        "title": "Plates and stills",
        "back_text": "Back to the song",
        "back_url": "/song/demo",
        "post_url": "/song/demo/picks",
        "static_prefix": "/static/picker",
        "choices": (
            picker.Choice(
                id="dragon",
                title="Dragon",
                description="A green dragon with a broken horn.",
                picks=("dragon-aaaaaaaa-bbbb-01.png",),
                note="the horn is on the wrong side",
                candidates=(
                    picker.Candidate(
                        "dragon-11111111-bbbb-01.png",
                        "/img/dragon-11111111-bbbb-01.png",
                        marks=("from an earlier brief",),
                        group="Earlier brief",
                    ),
                    picker.Candidate("dragon-aaaaaaaa-bbbb-01.png", "/img/dragon-a-01.png"),
                    picker.Candidate("dragon-aaaaaaaa-bbbb-02.png", "/img/dragon-a-02.png"),
                ),
            ),
            picker.Choice(
                id="bridge",
                title="Bridge at dawn",
                several=True,
                picks=("bridge-03.png", "bridge-01.png"),
                candidates=(
                    picker.Candidate("bridge-01.png", "/img/bridge-01.png"),
                    picker.Candidate("bridge-02.png", "/img/bridge-02.png"),
                    picker.Candidate("bridge-03.png", "/img/bridge-03.png"),
                ),
            ),
            picker.Choice(
                id="cave",
                title="Cave",
                held_back="The cave waits for the dragon's plate to be picked.",
                candidates=(),
            ),
        ),
    }
    values.update(changes)
    return picker.Page(**values)


class _Tree(HTMLParser):
    """Every start tag with its attributes, and the text inside each element by id."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)

    def named(self, tag: str) -> list[dict[str, str | None]]:
        return [attrs for name, attrs in self.tags if name == tag]

    def classed(self, name: str) -> list[dict[str, str | None]]:
        return [a for _, a in self.tags if name in (a.get("class") or "").split()]


def _parse(text: str) -> _Tree:
    tree = _Tree()
    tree.feed(text)
    return tree


def _reply(*rows, token: str = "") -> bytes:
    return json.dumps(
        {"token": token, "choices": [{"id": i, "picks": p, "note": n} for i, p, n in rows]}
    ).encode()


GOOD = (
    ("dragon", ["dragon-aaaaaaaa-bbbb-02.png"], "better horn"),
    ("bridge", ["bridge-02.png", "bridge-01.png"], ""),
    ("cave", [], ""),
)


@pytest.mark.parametrize(
    ("row", "problems"),
    [
        (
            ("dragon", ["missing.png", "missing.png"], 7),
            (
                "'missing.png' is not one of the candidates offered for 'dragon'.",
                "'dragon' takes one pick and the reply has 2.",
                "The note for 'dragon' is not text.",
            ),
        ),
        (
            ("dragon", "bad", ""),
            ("The picks for 'dragon' are not a list of file names.",),
        ),
        (
            ("bridge", ["bridge-01.png", "bridge-01.png"], "x" * (picker.MAX_NOTE_CHARS + 1)),
            (
                "'bridge-01.png' is picked twice for 'bridge'.",
                f"The note for 'bridge' is longer than {picker.MAX_NOTE_CHARS} characters.",
            ),
        ),
    ],
)
def test_reply_accumulates_choice_problems_without_returning_partial_picks(row, problems):
    rows = [row if good[0] == row[0] else good for good in GOOD]
    reply = picker.read_reply(_reply(*rows), _page())
    assert reply.problems == problems
    assert reply.picked == ()


def test_reply_preserves_pick_order_and_notes_when_entries_arrive_in_reverse_order():
    reply = picker.read_reply(_reply(*reversed(GOOD), token="save-1"), _page())
    assert reply.problems == ()
    assert reply.token == "save-1"
    assert tuple((picked.id, picked.picks, picked.note) for picked in reply.picked) == tuple(
        (name, tuple(picks), note) for name, picks, note in GOOD
    )


# ---- the page -------------------------------------------------------------------------------


def test_the_page_references_one_stylesheet_and_one_script_under_the_hosts_prefix():
    tree = _parse(picker.render(_page(static_prefix="/assets/pick")))
    sheets = [a["href"] for a in tree.named("link") if a.get("rel") == "stylesheet"]
    scripts = [a.get("src") for a in tree.named("script")]
    assert sheets == ["/assets/pick/picker.css"]
    assert scripts == ["/assets/pick/picker.js"]


def test_the_page_carries_no_inline_script_no_style_and_no_handler_attribute():
    """`Content-Security-Policy: default-src 'self'` blocks each of these."""
    text = picker.render(_page(title=HOSTILE))
    tree = _parse(text)
    assert tree.named("style") == []
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", text)
    for tag, attrs in tree.tags:
        assert "style" not in attrs, tag
        assert not [name for name in attrs if name.startswith("on")], (tag, attrs)


def test_the_page_makes_no_request_outside_the_host():
    tree = _parse(picker.render(_page()))
    for _, attrs in tree.tags:
        for name in ("href", "src", "action", "data-post-url"):
            value = attrs.get(name)
            assert value is None or (value.startswith("/") and not value.startswith("//")), value


def test_each_choice_shows_its_title_description_rule_and_note_box_with_a_label():
    text = picker.render(_page())
    tree = _parse(text)
    sections = tree.classed("choice")
    assert [s["data-choice-id"] for s in sections] == ["dragon", "bridge", "cave"]
    assert [s["data-several"] for s in sections] == ["false", "true", "false"]
    assert "A green dragon with a broken horn." in tree.text
    assert "The cave waits for the dragon's plate to be picked." in "".join(tree.text)
    boxes = tree.named("textarea")
    labels = {a["for"] for a in tree.named("label")}
    assert len(boxes) == 3 and {b["id"] for b in boxes} <= labels
    assert "the horn is on the wrong side" in tree.text
    # every section is named by its heading for a screen reader
    ids = {a.get("id") for _, a in tree.tags}
    assert all(s["aria-labelledby"] in ids for s in sections)


def test_candidates_come_ungrouped_first_then_each_group_in_the_order_given():
    tree = _parse(picker.render(_page()))
    files = [a["data-file"] for a in tree.classed("candidate")]
    assert files[:3] == [
        "dragon-aaaaaaaa-bbbb-01.png",
        "dragon-aaaaaaaa-bbbb-02.png",
        "dragon-11111111-bbbb-01.png",
    ]
    assert "Earlier brief" in tree.text
    assert "from an earlier brief" in tree.text


def test_each_candidate_is_a_link_to_its_full_size_image_and_a_pick_button():
    tree = _parse(picker.render(_page()))
    views = tree.classed("view")
    images = tree.named("img")
    buttons = tree.classed("pick")
    assert len(views) == len(images) == len(buttons) == 6
    assert views[0]["href"] == images[0]["src"] == "/img/dragon-a-01.png"
    assert all(i.get("alt") for i in images)
    assert all(b.get("type") == "button" and b.get("aria-label") for b in buttons)


def test_current_picks_are_drawn_pressed_with_their_order():
    tree = _parse(picker.render(_page()))
    state = {a["data-file"]: a.get("data-order") for a in tree.classed("candidate")}
    assert state["dragon-aaaaaaaa-bbbb-01.png"] == "1"
    assert state["dragon-aaaaaaaa-bbbb-02.png"] == ""
    assert (state["bridge-03.png"], state["bridge-01.png"], state["bridge-02.png"]) == (
        "1",
        "2",
        "",
    )
    pressed = [b["aria-pressed"] for b in tree.classed("pick")]
    assert pressed == ["true", "false", "false", "true", "false", "true"]


def test_the_page_has_one_save_button_a_status_line_and_the_way_back():
    tree = _parse(picker.render(_page(token="sha-1", post_headers={"X-Musiclm": "1"})))
    main = [a for a in tree.named("main") if a.get("id") == "picker"][0]
    assert main["data-post-url"] == "/song/demo/picks"
    assert main["data-token"] == "sha-1"
    assert json.loads(main["data-post-headers"]) == {"X-Musiclm": "1"}
    assert len([a for a in tree.named("button") if a.get("id") == "save"]) == 1
    status = [a for _, a in tree.tags if a.get("id") == "status"][0]
    assert status["role"] == "status"
    back = tree.classed("back")[0]
    assert back["href"] == "/song/demo" and "Back to the song" in tree.text
    assert tree.named("html")[0]["lang"] == "en"


def test_hostile_titles_notes_and_file_names_are_escaped():
    hostile = picker.Choice(
        id='x"><b>',
        title=HOSTILE,
        description=HOSTILE,
        held_back=HOSTILE,
        note="</textarea>" + HOSTILE,
        picks=(HOSTILE_FILE,),
        candidates=(
            picker.Candidate(HOSTILE_FILE, "/img/a b.png?x=1&y=2", marks=(HOSTILE,), group=HOSTILE),
        ),
    )
    text = picker.render(_page(title=HOSTILE, back_text=HOSTILE, choices=(hostile,)))
    tree = _parse(text)
    assert "<script>alert" not in text
    assert "<img src=x" not in text
    assert tree.named("b") == [] and len(tree.named("img")) == 1
    assert len(tree.named("textarea")) == 1
    # the hostile strings survive as text and attribute values, exactly
    assert tree.classed("candidate")[0]["data-file"] == HOSTILE_FILE
    assert tree.classed("choice")[0]["data-choice-id"] == 'x"><b>'
    assert tree.named("img")[0]["src"] == "/img/a b.png?x=1&y=2"
    assert "</textarea>" + HOSTILE in tree.text
    assert tree.text.count(HOSTILE) >= 4


def test_the_brand_links_the_hosts_icons_and_shows_its_mark():
    brand = picker.Brand(
        name="MusicLM",
        home_url="/",
        logo_url="/static/brand/logo.png",
        favicon_url="/static/brand/favicon.ico",
        favicon_png_url="/static/brand/favicon-32.png",
        touch_icon_url="/static/brand/apple-touch-icon.png",
    )
    tree = _parse(picker.render(_page(brand=brand)))
    rels = {(a.get("rel"), a.get("href")) for a in tree.named("link")}
    assert ("icon", "/static/brand/favicon.ico") in rels
    assert ("icon", "/static/brand/favicon-32.png") in rels
    assert ("apple-touch-icon", "/static/brand/apple-touch-icon.png") in rels
    assert tree.classed("brand")[0]["href"] == "/"
    assert "MusicLM" in tree.text
    without = _parse(picker.render(_page()))
    assert without.classed("brand") == []


@pytest.mark.parametrize(
    ("changes", "sentence"),
    [
        (
            {"back_url": "https://elsewhere.example/x"},
            "The way back 'https://elsewhere.example/x' is not a path on the host: "
            "it must start with one '/'.",
        ),
        (
            {"post_url": "javascript:alert(1)"},
            "The post URL 'javascript:alert(1)' is not a path on the host: "
            "it must start with one '/'.",
        ),
        (
            {"static_prefix": "//cdn.example/picker"},
            "The static prefix '//cdn.example/picker' is not a path on the host: "
            "it must start with one '/'.",
        ),
        (
            {
                "choices": (
                    picker.Choice("a", "A", (picker.Candidate("a-01.png", "/a-01.png"),)),
                    picker.Choice("a", "A again", ()),
                )
            },
            "Two choices carry the id 'a'.",
        ),
        (
            {
                "choices": (
                    picker.Choice(
                        "a",
                        "A",
                        (
                            picker.Candidate("a-01.png", "/a-01.png"),
                            picker.Candidate("a-01.png", "/other/a-01.png"),
                        ),
                    ),
                )
            },
            "Choice 'a' offers the file 'a-01.png' twice.",
        ),
        (
            {
                "choices": (
                    picker.Choice(
                        "a", "A", (picker.Candidate("a-01.png", "/a-01.png"),), picks=("a-09.png",)
                    ),
                )
            },
            "Choice 'a' has the pick 'a-09.png', which is not one of its candidates.",
        ),
        (
            {
                "choices": (
                    picker.Choice(
                        "a",
                        "A",
                        (
                            picker.Candidate("a-01.png", "/a-01.png"),
                            picker.Candidate("a-02.png", "/a-02.png"),
                        ),
                        picks=("a-01.png", "a-02.png"),
                    ),
                )
            },
            "Choice 'a' takes one pick and has 2.",
        ),
        (
            {"choices": (picker.Choice("a", "A", (picker.Candidate("a.png", "http://x/a.png"),)),)},
            "The image URL 'http://x/a.png' of 'a.png' is not a path on the host: "
            "it must start with one '/'.",
        ),
    ],
)
def test_a_page_the_browser_could_not_draw_safely_is_declined_with_each_reason(changes, sentence):
    with pytest.raises(picker.PickerError) as declined:
        picker.render(_page(**changes))
    assert sentence in str(declined.value)


# ---- the static files -----------------------------------------------------------------------


def test_the_static_files_are_served_by_name_with_their_content_type():
    css = picker.static_file("picker.css")
    js = picker.static_file("picker.js")
    assert css is not None and js is not None
    assert css.content_type == "text/css; charset=utf-8"
    assert js.content_type == "text/javascript; charset=utf-8"
    assert b"prefers-color-scheme: dark" in css.body
    assert b"addEventListener" in js.body
    assert picker.STATIC_FILES == ("picker.css", "picker.js")


@pytest.mark.parametrize(
    "name",
    [
        "",
        "picker.html",
        "__init__.py",
        "../__init__.py",
        "static/picker.css",
        "/picker.css",
        "picker.css/",
        "PICKER.CSS",
        "picker.css\x00.png",
        "..",
    ],
)
def test_no_other_name_is_served(name):
    assert picker.static_file(name) is None


def test_the_static_files_ship_inside_the_package():
    """An editable install and a wheel both carry `llmlm/picker/static/` (the wheel target is the
    whole `src/llmlm` folder)."""
    folder = resources.files("llmlm.picker") / "static"
    assert sorted(item.name for item in folder.iterdir()) == ["picker.css", "picker.js"]


def test_the_stylesheet_defines_every_token_it_uses_in_both_themes():
    css = picker.static_file("picker.css").body.decode()
    used = set(re.findall(r"var\((--[a-z-]+)\)", css))
    light = re.search(r":root\s*\{([^}]*)\}", css).group(1)
    dark = re.search(r"prefers-color-scheme: dark\)\s*\{\s*:root\s*\{([^}]*)\}", css).group(1)
    assert used and used <= set(re.findall(r"(--[a-z-]+):", light))
    assert set(re.findall(r"(--[a-z-]+):", light)) == set(re.findall(r"(--[a-z-]+):", dark))


def test_the_static_files_reach_for_nothing_outside_the_host():
    for name in picker.STATIC_FILES:
        body = picker.static_file(name).body.decode()
        assert "http://" not in body and "https://" not in body and "@import" not in body, name
    script = picker.static_file("picker.js").body.decode()
    assert "innerHTML" not in script and "eval(" not in script and ".style." not in script


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_script_parses():
    with resources.as_file(resources.files("llmlm.picker") / "static" / "picker.js") as script:
        out = subprocess.run(["node", "--check", str(script)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


# ---- the reply ------------------------------------------------------------------------------


def test_a_valid_reply_reads_back_as_picks_in_order_and_the_note():
    reply = picker.read_reply(_reply(*GOOD, token="sha-1"), _page())
    assert reply.problems == ()
    assert reply.token == "sha-1"
    assert reply.picked == (
        picker.Picked("dragon", ("dragon-aaaaaaaa-bbbb-02.png",), "better horn"),
        picker.Picked("bridge", ("bridge-02.png", "bridge-01.png"), ""),
        picker.Picked("cave", (), ""),
    )


def test_a_reply_as_text_and_a_single_choice_with_no_pick_are_accepted():
    rows = (("dragon", [], ""), GOOD[1], GOOD[2])
    reply = picker.read_reply(_reply(*rows).decode(), _page())
    assert reply.problems == ()
    assert reply.picked[0] == picker.Picked("dragon", (), "")


@pytest.mark.parametrize(
    ("body", "sentence"),
    [
        (b"not json", "The reply is not JSON."),
        (b"\xff\xfe", "The reply is not JSON."),
        (b"[1, 2]", "The reply is not a JSON object with a 'choices' list."),
        (b'{"choices": {}}', "The reply is not a JSON object with a 'choices' list."),
        (b'{"choices": [], "token": 7}', "The reply's token is not text."),
        (
            _reply(("dragon", ["bridge-01.png"], ""), GOOD[1], GOOD[2]),
            "'bridge-01.png' is not one of the candidates offered for 'dragon'.",
        ),
        (
            _reply(
                ("dragon", ["dragon-aaaaaaaa-bbbb-01.png", "dragon-aaaaaaaa-bbbb-02.png"], ""),
                GOOD[1],
                GOOD[2],
            ),
            "'dragon' takes one pick and the reply has 2.",
        ),
        (
            _reply(GOOD[0], ("bridge", ["bridge-01.png", "bridge-01.png"], ""), GOOD[2]),
            "'bridge-01.png' is picked twice for 'bridge'.",
        ),
        (
            _reply(*GOOD, ("castle", [], "")),
            "The reply names 'castle', which is not on this page.",
        ),
        (_reply(*GOOD, GOOD[0]), "The reply names 'dragon' twice."),
        (_reply(GOOD[0], GOOD[1]), "The reply leaves out 'cave'."),
        (
            _reply(("dragon", "dragon-aaaaaaaa-bbbb-02.png", ""), GOOD[1], GOOD[2]),
            "The picks for 'dragon' are not a list of file names.",
        ),
        (
            _reply(("dragon", [3], ""), GOOD[1], GOOD[2]),
            "The picks for 'dragon' are not a list of file names.",
        ),
        (_reply(("dragon", [], 5), GOOD[1], GOOD[2]), "The note for 'dragon' is not text."),
        (
            _reply(("dragon", [], "x" * (picker.MAX_NOTE_CHARS + 1)), GOOD[1], GOOD[2]),
            f"The note for 'dragon' is longer than {picker.MAX_NOTE_CHARS} characters.",
        ),
        (
            json.dumps({"choices": ["dragon"]}).encode(),
            "Entry 1 of the reply is not an object with an 'id'.",
        ),
        (
            b" " * (picker.MAX_REPLY_BYTES + 1),
            f"The reply is larger than {picker.MAX_REPLY_BYTES} bytes.",
        ),
    ],
)
def test_a_reply_with_a_problem_is_declined_with_the_sentence_and_no_picks(body, sentence):
    reply = picker.read_reply(body, _page())
    assert sentence in reply.problems
    assert reply.picked == ()


def test_every_problem_of_a_reply_is_reported_not_only_the_first():
    body = _reply(("dragon", ["nope.png"], 5), ("bridge", ["bridge-01.png"], ""))
    reply = picker.read_reply(body, _page())
    assert reply.problems == (
        "'nope.png' is not one of the candidates offered for 'dragon'.",
        "The note for 'dragon' is not text.",
        "The reply leaves out 'cave'.",
    )


def test_the_module_opens_no_file_but_its_own_static_assets(tmp_path, monkeypatch):
    """Tripwire for the scope rule: rendering, serving and reading a reply open nothing outside
    `llmlm/picker/static/`. Editing this test is the act that changes that policy."""
    import builtins
    import io

    opened: list[str] = []
    real_open, real_io_open = builtins.open, io.open

    def watching(file, *args, **kwargs):
        opened.append(str(file))
        return real_io_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", watching)
    monkeypatch.setattr(io, "open", watching)
    monkeypatch.chdir(tmp_path)
    picker.render(_page())
    picker.read_reply(_reply(*GOOD), _page())
    for name in picker.STATIC_FILES:
        picker.static_file(name)
    picker.static_file("../cast.py")
    monkeypatch.setattr(builtins, "open", real_open)
    static = str(resources.files("llmlm.picker") / "static")
    assert opened and all(path.startswith(static) for path in opened), opened
    assert list(tmp_path.iterdir()) == []


# ---- a host, end to end ---------------------------------------------------------------------


@pytest.fixture
def host(tmp_path):
    import picker_example_host

    server = picker_example_host.build(tmp_path / "images", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _post(url: str, body: bytes) -> tuple[int, dict]:
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json", "X-Example": "1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as answer:
            return answer.status, json.loads(answer.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


@pytest.mark.cuj(
    "The owner opens a host app's picking page on a phone, picks among the candidate images of "
    "each choice and taps Save via the host's own server, and sees the host record the picked "
    "file names in order with the note. If a picked name is not one of the candidates offered, "
    "they see the sentence naming it and nothing is recorded."
)
def test_a_host_serves_the_page_and_records_a_valid_reply_its_own_way(host):
    server, base = host
    with urllib.request.urlopen(base + "/", timeout=5) as answer:
        text = answer.read().decode()
        assert answer.headers["Content-Security-Policy"] == "default-src 'self'"
    tree = _parse(text)
    main = [a for a in tree.named("main") if a.get("id") == "picker"][0]
    for attrs in tree.named("link") + tree.named("script") + tree.named("img"):
        with urllib.request.urlopen(base + (attrs.get("href") or attrs["src"]), timeout=5) as got:
            assert got.status == 200 and got.read()
    with pytest.raises(urllib.error.HTTPError) as missing:
        urllib.request.urlopen(base + "/static/picker/__init__.py", timeout=5)
    assert missing.value.code == 404

    files = [a["data-file"] for a in tree.classed("candidate")]
    plate, stills = files[1], [files[6], files[4]]
    body = json.dumps(
        {
            "token": main["data-token"],
            "choices": [
                {"id": "dragon", "picks": [plate], "note": "horn"},
                {"id": "bridge", "picks": stills, "note": ""},
                {"id": "cave", "picks": [], "note": ""},
            ],
        }
    ).encode()
    status, answer = _post(base + main["data-post-url"], body)
    assert status == 200, answer
    assert server.picks["dragon"] == {"picks": [plate], "note": "horn"}
    assert server.picks["bridge"] == {"picks": stills, "note": ""}

    bad = json.dumps({"token": "", "choices": [{"id": "dragon", "picks": ["x.png"], "note": ""}]})
    status, answer = _post(base + main["data-post-url"], bad.encode())
    assert status == 400
    assert "'x.png' is not one of the candidates offered for 'dragon'." in answer["problems"]
    assert server.picks["dragon"] == {"picks": [plate], "note": "horn"}
