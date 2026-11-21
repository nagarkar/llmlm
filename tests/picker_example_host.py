"""A tiny example host for `llmlm.picker`: what an app does to show the picking page.

    .venv/bin/python tests/picker_example_host.py [port]

It binds loopback only, writes a few generated images into a temporary folder that is removed on
exit, and keeps the picks in memory (a real host records them its own way: musiclm in its request
file, authorlm as an embed line). It calls no model. Stop it with Ctrl-C.

The three calls a host makes are marked (1) render, (2) static files, (3) read the reply.
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from llmlm import picker

STATIC = "/static/picker"
HEADERS = {
    "Content-Security-Policy": "default-src 'self'",
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
}
# file name -> (top colour, bottom colour)
IMAGES = {
    "dragon-0a1b2c3d-9f8e-01.png": ((34, 120, 74), (12, 40, 30)),
    "dragon-0a1b2c3d-9f8e-02.png": ((60, 150, 60), (20, 30, 60)),
    "dragon-77e1d2c3-9f8e-01.png": ((120, 130, 40), (40, 30, 10)),
    "bridge-4d5e6f70-9f8e-01.png": ((250, 180, 120), (60, 70, 120)),
    "bridge-4d5e6f70-9f8e-02.png": ((240, 140, 150), (40, 50, 110)),
    "bridge-4d5e6f70-9f8e-03.png": ((255, 210, 150), (90, 80, 130)),
    "bridge-4d5e6f70-9f8e-04.png": ((200, 160, 200), (30, 40, 90)),
}


def _png(top: tuple[int, int, int], bottom: tuple[int, int, int], width=480, height=600) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    rows = bytearray()
    for y in range(height):
        colour = bytes(a + (b - a) * y // (height - 1) for a, b in zip(top, bottom, strict=True))
        rows += b"\x00" + colour * width
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(rows)))
        + chunk(b"IEND", b"")
    )


class ExampleHost(ThreadingHTTPServer):
    """`picks` is this host's own record: choice id -> {"picks": [file, ...], "note": text}."""

    daemon_threads = True

    def __init__(self, folder: Path, port: int):
        self.folder = folder
        self.saves = 0
        self.picks: dict[str, dict] = {
            "dragon": {"picks": ["dragon-0a1b2c3d-9f8e-01.png"], "note": ""},
            "bridge": {"picks": [], "note": "the second one has two suns"},
            "cave": {"picks": [], "note": ""},
        }
        super().__init__(("127.0.0.1", port), _Handler)

    def page(self) -> picker.Page:
        def candidates(slug: str, earlier: str = "") -> list[picker.Candidate]:
            return [
                picker.Candidate(
                    name,
                    f"/img/{name}",
                    marks=("from an earlier brief",) if earlier and earlier in name else (),
                    group="Earlier brief" if earlier and earlier in name else "",
                )
                for name in IMAGES
                if name.startswith(slug)
            ]

        return picker.Page(
            title="Example: plates and stills",
            back_text="Back to the example",
            back_url="/",
            post_url="/picks",
            static_prefix=STATIC,
            post_headers={"X-Example": "1"},
            token=f"save-{self.saves}",
            choices=[
                picker.Choice(
                    "dragon",
                    "Dragon (plate)",
                    candidates("dragon", earlier="77e1d2c3"),
                    description="A green dragon with a broken left horn, seen from the side.",
                    picks=self.picks["dragon"]["picks"],
                    note=self.picks["dragon"]["note"],
                ),
                picker.Choice(
                    "bridge",
                    "Bridge at dawn (stills)",
                    candidates("bridge"),
                    description="The stone bridge as the sun comes up; the dragon is a far shape.",
                    several=True,
                    picks=self.picks["bridge"]["picks"],
                    note=self.picks["bridge"]["note"],
                ),
                picker.Choice(
                    "cave",
                    "Cave (stills)",
                    held_back="The cave waits for the dragon's plate to be picked.",
                    note=self.picks["cave"]["note"],
                ),
            ],
        )


class _Handler(BaseHTTPRequestHandler):
    server: ExampleHost

    def log_message(self, *args) -> None:  # quiet
        pass

    def _send(self, status: int, kind: str, body: bytes) -> None:
        self.send_response(status)
        for name, value in {**HEADERS, "Content-Type": kind}.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: dict) -> None:
        self._send(status, "application/json", json.dumps(data).encode())

    def do_GET(self) -> None:  # noqa: N802 (the base class names these)
        path = self.path.split("?", 1)[0]
        if path == "/":
            # (1) render: plain data in, HTML text out
            text = picker.render(self.server.page())
            return self._send(200, "text/html; charset=utf-8", text.encode())
        if path == "/health":
            return self._json(200, {"ok": True})
        if path.startswith(STATIC + "/"):
            # (2) static files: by name, None for anything else
            asset = picker.static_file(path[len(STATIC) + 1 :])
            if asset:
                return self._send(200, asset.content_type, asset.body)
        if path.startswith("/img/") and path[5:] in IMAGES:
            return self._send(200, "image/png", (self.server.folder / path[5:]).read_bytes())
        self._send(404, "text/plain; charset=utf-8", b"Not found.")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/picks":
            return self._json(404, {"error": "Not found."})
        if self.headers.get("X-Example") != "1" or not (
            self.headers.get("Content-Type") or ""
        ).startswith("application/json"):
            return self._json(403, {"error": "This save did not come from the picking page."})
        length = int(self.headers.get("Content-Length") or 0)
        if length > picker.MAX_REPLY_BYTES:
            return self._json(413, {"error": "The reply is too large."})
        # (3) read the reply against the same page that was offered
        reply = picker.read_reply(self.rfile.read(length), self.server.page())
        if reply.problems:
            return self._json(400, {"problems": list(reply.problems)})
        if reply.token and reply.token != f"save-{self.server.saves}":
            return self._json(409, {"error": "The picks changed elsewhere; reload this page."})
        # the host records a valid reply its own way; this one keeps it in memory
        for picked in reply.picked:
            self.server.picks[picked.id] = {"picks": list(picked.picks), "note": picked.note}
        self.server.saves += 1
        self._json(200, {"token": f"save-{self.server.saves}"})


def build(folder: Path, port: int = 0) -> ExampleHost:
    """Write the example images into `folder` and return the server, not yet serving."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, (top, bottom) in IMAGES.items():
        (folder / name).write_bytes(_png(top, bottom))
    return ExampleHost(folder, port)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="llmlm-picker-example-") as temp:
        host = build(Path(temp), int(sys.argv[1]) if len(sys.argv) > 1 else 0)
        print(f"http://127.0.0.1:{host.server_address[1]}/  (Ctrl-C stops it)", flush=True)
        try:
            host.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            host.server_close()
            print(json.dumps(host.picks, indent=2))
