"""Picker: the page for choosing among candidate images, as HTML text a host app serves itself.

A host hands over plain data (a `Page`: its title, the way back, the URL to post to, and an
ordered list of `Choice`s, each with its `Candidate`s, its current picks and its pick note) and
gets back three things: `render(page)`, the page as HTML text; `static_file(name)`, the bytes and
content type of the one stylesheet and the one script the page references; and
`read_reply(body, page)`, the posted picking reply as plain data, or every problem with it as a
sentence. The host serves all three on its own server and decides what a valid reply changes.

It owns only `static/picker.css` and `static/picker.js` inside this package. It opens no other
file: not an image, not a pick, not a request file, not a manuscript.

Never imports: any other llmlm module (enforced by the import contract "the picker stands alone"
in pyproject.toml).

Decisions:
- The shared picking page lives here, beside `llmlm.cast`. Each app shows it on its own server
  and records the pick in its own way (musiclm: the request file; authorlm: the embed line).
  llmlm stays a library: no server, no command line, no new port. Enforced by:
  tests/test_picker.py::test_the_module_opens_no_file_but_its_own_static_assets. (owner)
- The page carries no inline script, no `style` element or attribute and no handler attribute,
  and every URL in it is a path on the host (one leading `/`), so it runs under
  `Content-Security-Policy: default-src 'self'` and makes no outside request. A page that would
  break this is declined (`PickerError`) with each reason. Enforced by:
  tests/test_picker.py::test_the_page_carries_no_inline_script_no_style_and_no_handler_attribute.
- Save posts JSON, not a form: `{"token": text, "choices": [{"id", "picks": [file, ...],
  "note"}]}` with `Content-Type: application/json` plus the host's `post_headers`. Why: the
  order of several picks is a list, which a form flattens; the hosts' write check already wants
  JSON with an `X-<App>: 1` header; and a save that fails keeps every tap and note on the screen
  instead of reloading it. Without the script the page still shows the current picks, read-only.
- The host answers a save with any 2xx when it recorded the reply, optionally a JSON object with
  `message` (shown after "Saved") and `token` (the next save sends it back); otherwise a non-2xx
  with `{"problems": [sentence, ...]}` or `{"error": sentence}`, shown after "Not saved".
- `token` is the host's own text (a file hash, a version), sent back unchanged with every save,
  so the host can decline a save made on a stale page. The picker never reads it.
- A reply must name every choice of the page exactly once; one it leaves out is a problem, never
  "no picks". Why: a missing choice read as empty would erase a pick. A reply with any problem
  reads back with no picks at all.
- A single-pick choice may have no pick: tapping the picked candidate again removes it.
- A held-back choice shows its sentence and stays pickable when it has candidates; the host
  offers none when nothing may be picked.
- Candidates are drawn ungrouped first, then each group in the order its label first appears,
  under the label as a heading. Why: current candidates come before those of an earlier brief.
- Candidates are one column below about 600 px of width and side by side above it; tapping an
  image opens it full size in a dialog with Previous, Next and the same Pick button.
- The texts the script writes (the status line, the picked line) are in `static/picker.js`; the
  page is in English.

Terms:
- **host**: the app that serves the picking page on its own server and records the picks
  (musiclm, authorlm).
- **choice**: one thing the owner picks for (a character's plate, a scene's stills): an id, a
  title, its candidates, its picks and its pick note. It takes one pick, or several in order.
- **pick**: the file name of a candidate chosen for a choice; the picks of a choice are ordered.
- **pick note**: the owner's free text on one choice, saying what is off.
- **mark**: a short phrase the host puts on a candidate ("from an earlier brief").
- **held back**: a choice the host cannot fill yet, with the sentence saying why.
- **picking reply**: what Save posts: per choice, its picks in order and its pick note.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources

STATIC_FILES = ("picker.css", "picker.js")
MAX_NOTE_CHARS = 2000
MAX_REPLY_BYTES = 64 * 1024

_CONTENT_TYPES = {
    "picker.css": "text/css; charset=utf-8",
    "picker.js": "text/javascript; charset=utf-8",
}


class PickerError(ValueError):
    """A page was declined before it was drawn. The message carries every reason."""


@dataclass(frozen=True)
class Candidate:
    """One image offered for a choice. `file` is the name a pick records; `url` is the path the
    host serves the image at. `marks` are short phrases shown on it; `group` is the heading it
    is drawn under ("" for the current candidates, drawn first)."""

    file: str
    url: str
    marks: Sequence[str] = ()
    group: str = ""


@dataclass(frozen=True)
class Choice:
    """One thing to pick for. `picks` are the current picks in order (file names of its
    candidates); `several` allows more than one; `held_back` is the sentence saying why the host
    cannot fill it yet; `note` is the saved pick note."""

    id: str
    title: str
    candidates: Sequence[Candidate] = ()
    description: str = ""
    held_back: str = ""
    several: bool = False
    picks: Sequence[str] = ()
    note: str = ""


@dataclass(frozen=True)
class Brand:
    """The host's own mark, shown in the header and linked as the page's icons. Every URL is a
    path on the host."""

    name: str
    home_url: str = "/"
    logo_url: str = ""
    favicon_url: str = ""
    favicon_png_url: str = ""
    touch_icon_url: str = ""


@dataclass(frozen=True)
class Page:
    """The whole picking page. `static_prefix` is the path the host serves `static_file` names
    under; `post_headers` are sent with the save (the host's write-check header); `token` is the
    host's own text, sent back unchanged with the save."""

    title: str
    back_text: str
    back_url: str
    post_url: str
    choices: Sequence[Choice]
    static_prefix: str = "/static/picker"
    post_headers: Mapping[str, str] = field(default_factory=dict)
    token: str = ""
    brand: Brand | None = None


@dataclass(frozen=True)
class StaticFile:
    body: bytes
    content_type: str


@dataclass(frozen=True)
class Picked:
    """One choice of a valid picking reply: its picks in order and its pick note."""

    id: str
    picks: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class Reply:
    """A picking reply read back. `problems` is empty when it is valid; otherwise `picked` is
    empty and each problem is a sentence."""

    picked: tuple[Picked, ...] = ()
    problems: tuple[str, ...] = ()
    token: str = ""


def static_file(name: str) -> StaticFile | None:
    """The stylesheet or the script by its name in `STATIC_FILES`; None for any other name."""
    if name not in STATIC_FILES:
        return None
    body = (resources.files(__name__) / "static" / name).read_bytes()
    return StaticFile(body, _CONTENT_TYPES[name])


def _on_host(url: str) -> bool:
    return url.startswith("/") and not url.startswith("//") and "\\" not in url


def _page_problems(page: Page) -> list[str]:
    problems: list[str] = []

    def path(what: str, url: str, of: str = "") -> None:
        if not _on_host(url):
            problems.append(
                f"{what} '{url}'{of} is not a path on the host: it must start with one '/'."
            )

    path("The way back", page.back_url)
    path("The post URL", page.post_url)
    path("The static prefix", page.static_prefix)
    if page.brand:
        for url in (
            page.brand.home_url,
            page.brand.logo_url,
            page.brand.favicon_url,
            page.brand.favicon_png_url,
            page.brand.touch_icon_url,
        ):
            if url:
                path("The brand URL", url)
    seen: set[str] = set()
    for choice in page.choices:
        if choice.id in seen:
            problems.append(f"Two choices carry the id '{choice.id}'.")
        seen.add(choice.id)
        files: set[str] = set()
        for candidate in choice.candidates:
            if candidate.file in files:
                problems.append(f"Choice '{choice.id}' offers the file '{candidate.file}' twice.")
            files.add(candidate.file)
            path("The image URL", candidate.url, f" of '{candidate.file}'")
        for pick in choice.picks:
            if pick not in files:
                problems.append(
                    f"Choice '{choice.id}' has the pick '{pick}', "
                    "which is not one of its candidates."
                )
        if not choice.several and len(choice.picks) > 1:
            problems.append(f"Choice '{choice.id}' takes one pick and has {len(choice.picks)}.")
    return problems


def _e(text: str) -> str:
    return html.escape(text, quote=True)


def _picked_line(choice: Choice) -> str:
    if not choice.picks:
        return "Picked: none"
    if not choice.several:
        return f"Picked: {choice.picks[0]}"
    return "Picked: " + ", ".join(f"{n}. {name}" for n, name in enumerate(choice.picks, 1))


def _ordered(candidates: Sequence[Candidate]) -> list[tuple[str, list[Candidate]]]:
    groups: dict[str, list[Candidate]] = {"": []}
    for candidate in candidates:
        groups.setdefault(candidate.group, []).append(candidate)
    return [(label, members) for label, members in groups.items() if members]


def _candidate(choice: Choice, candidate: Candidate, position: int, total: int) -> str:
    order = choice.picks.index(candidate.file) + 1 if candidate.file in choice.picks else 0
    badge = (str(order) if choice.several else "✓") if order else ""
    where = f"candidate {position} of {total} for {choice.title}"
    marks = "".join(f'<span class="mark">{_e(mark)}</span>' for mark in candidate.marks)
    return (
        f'<li class="candidate" data-file="{_e(candidate.file)}" data-order="{order or ""}">'
        f'<a class="view" href="{_e(candidate.url)}" target="_blank" rel="noopener">'
        f'<img src="{_e(candidate.url)}" alt="{_e("C" + where[1:])}" loading="lazy">'
        f'<span class="order" aria-hidden="true">{badge}</span></a>'
        f'<p class="file">{_e(candidate.file)}</p>'
        + (f'<p class="marks">{marks}</p>' if marks else "")
        + f'<button type="button" class="button pick" aria-pressed="{"true" if order else "false"}"'
        f' aria-label="{_e("Pick " + where)}">'
        '<span class="when-off">Pick</span><span class="when-on">Picked · remove</span>'
        "</button></li>"
    )


def _choice(choice: Choice, number: int) -> str:
    key = f"choice-{number}"
    parts = [
        f'<section class="choice" data-choice-id="{_e(choice.id)}"'
        f' data-several="{"true" if choice.several else "false"}" aria-labelledby="{key}-title">',
        f'<h2 id="{key}-title">{_e(choice.title)}</h2>',
    ]
    if choice.description:
        parts.append(f'<p class="description">{_e(choice.description)}</p>')
    if choice.held_back:
        parts.append(f'<p class="held"><strong>Held back.</strong> {_e(choice.held_back)}</p>')
    total = len(choice.candidates)
    if total:
        rule = (
            "Pick any number. The order you tap is the order kept."
            if choice.several
            else "Pick one."
        )
        parts.append(f'<p class="rule">{rule}</p>')
        position = 0
        for label, members in _ordered(choice.candidates):
            if label:
                parts.append(f'<h3 class="group">{_e(label)}</h3>')
            parts.append('<ul class="candidates">')
            for candidate in members:
                position += 1
                parts.append(_candidate(choice, candidate, position, total))
            parts.append("</ul>")
        parts.append(f'<p class="picked-line" aria-live="polite">{_e(_picked_line(choice))}</p>')
    else:
        parts.append('<p class="empty">No candidates to pick from yet.</p>')
    parts.append(
        f'<div class="note"><label for="{key}-note">What is off</label>'
        f'<textarea id="{key}-note" rows="3" maxlength="{MAX_NOTE_CHARS}">'
        f"{_e(choice.note)}</textarea></div>"
    )
    parts.append("</section>")
    return "\n".join(parts)


def render(page: Page) -> str:
    """The picking page as HTML text. Raises `PickerError`, with every reason, for a page that
    could not be drawn safely (a URL off the host, two choices with one id, a pick that is not a
    candidate)."""
    problems = _page_problems(page)
    if problems:
        raise PickerError(" ".join(problems))
    prefix = page.static_prefix.rstrip("/")
    head = [
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_e(page.title)}</title>",
    ]
    brand = ""
    if page.brand:
        mark = page.brand
        if mark.favicon_url:
            head.append(f'<link rel="icon" href="{_e(mark.favicon_url)}" sizes="any">')
        if mark.favicon_png_url:
            head.append(
                '<link rel="icon" type="image/png" sizes="32x32"'
                f' href="{_e(mark.favicon_png_url)}">'
            )
        if mark.touch_icon_url:
            head.append(f'<link rel="apple-touch-icon" href="{_e(mark.touch_icon_url)}">')
        logo = (
            f'<img src="{_e(mark.logo_url)}" alt="" width="32" height="32">'
            if mark.logo_url
            else ""
        )
        brand = (
            f'<a class="brand" href="{_e(mark.home_url)}">{logo}<span>{_e(mark.name)}</span></a>'
        )
    head.append(f'<link rel="stylesheet" href="{_e(prefix)}/picker.css">')
    head.append(f'<script src="{_e(prefix)}/picker.js" defer></script>')
    choices = "\n".join(_choice(choice, n) for n, choice in enumerate(page.choices, 1))
    if not page.choices:
        choices = '<p class="empty">Nothing to pick yet.</p>'
    headers = json.dumps(dict(page.post_headers))
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n' + "\n".join(head) + "\n</head>\n<body>\n"
        f'<header class="top">{brand}'
        f'<a class="button back" href="{_e(page.back_url)}">{_e(page.back_text)}</a></header>\n'
        f'<main id="picker" data-post-url="{_e(page.post_url)}"'
        f' data-post-headers="{_e(headers)}" data-token="{_e(page.token)}">\n'
        f"<h1>{_e(page.title)}</h1>\n"
        "<noscript><p>Picking needs the page's script; without it this page only shows the "
        "current picks.</p></noscript>\n"
        f"{choices}\n"
        '<div class="savebar"><button type="button" id="save" class="button main">Save</button>'
        '<p id="status" role="status" data-state="clean">No changes yet.</p></div>\n'
        "</main>\n</body>\n</html>\n"
    )


def _read_picked(entry: dict, choice: Choice, problems: list[str]) -> Picked:
    """Read one offered choice's picks and note, appending validation problems."""
    name = entry["id"]
    picks = entry.get("picks", [])
    if not isinstance(picks, list) or not all(isinstance(pick, str) for pick in picks):
        problems.append(f"The picks for '{name}' are not a list of file names.")
        picks = []
    files = {candidate.file for candidate in choice.candidates}
    for position, pick in enumerate(picks):
        if pick not in files:
            problems.append(f"'{pick}' is not one of the candidates offered for '{name}'.")
        elif pick in picks[:position]:
            problems.append(f"'{pick}' is picked twice for '{name}'.")
    if not choice.several and len(picks) > 1:
        problems.append(f"'{name}' takes one pick and the reply has {len(picks)}.")
    note = entry.get("note", "")
    if not isinstance(note, str):
        problems.append(f"The note for '{name}' is not text.")
        note = ""
    elif len(note) > MAX_NOTE_CHARS:
        problems.append(f"The note for '{name}' is longer than {MAX_NOTE_CHARS} characters.")
    return Picked(name, tuple(picks), note)


def read_reply(body: bytes | str, page: Page) -> Reply:
    """The posted picking reply as plain data, checked against the page it was offered on (the
    host rebuilds the same `Page` it rendered). Every problem is a sentence in `problems`; a
    reply with any problem carries no picks."""
    raw = body.encode() if isinstance(body, str) else body
    if len(raw) > MAX_REPLY_BYTES:
        return Reply(problems=(f"The reply is larger than {MAX_REPLY_BYTES} bytes.",))
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError:
        return Reply(problems=("The reply is not JSON.",))
    if not isinstance(data, dict) or not isinstance(data.get("choices"), list):
        return Reply(problems=("The reply is not a JSON object with a 'choices' list.",))
    problems: list[str] = []
    token = data.get("token", "")
    if not isinstance(token, str):
        problems.append("The reply's token is not text.")
        token = ""
    offered = {choice.id: choice for choice in page.choices}
    read: dict[str, Picked] = {}
    for number, entry in enumerate(data["choices"], 1):
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            problems.append(f"Entry {number} of the reply is not an object with an 'id'.")
            continue
        name = entry["id"]
        if name not in offered:
            problems.append(f"The reply names '{name}', which is not on this page.")
            continue
        if name in read:
            problems.append(f"The reply names '{name}' twice.")
            continue
        read[name] = _read_picked(entry, offered[name], problems)
    for name in offered:
        if name not in read:
            problems.append(f"The reply leaves out '{name}'.")
    if problems:
        return Reply(problems=tuple(dict.fromkeys(problems)))
    return Reply(picked=tuple(read[name] for name in offered), token=token)
