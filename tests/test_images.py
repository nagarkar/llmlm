"""Images (`src/llmlm/images.py`): one render per call; every failure is one readable sentence."""

import base64
import io
import json
import urllib.error
from types import SimpleNamespace

import pytest

from llmlm import images

PNG = b"\x89PNG\r\n\x1a\nrest"
JPG = b"\xff\xd8\xffrest"


def _gemini_reply(data=PNG):
    body = {
        "candidates": [
            {"content": {"parts": [{"inlineData": {"data": base64.b64encode(data).decode()}}]}}
        ]
    }
    return io.BytesIO(json.dumps(body).encode())


def test_gemini_takes_the_prompt_and_the_input_images_in_order():
    seen = []

    def urlopen(request, timeout):
        seen.append(json.loads(request.data))
        return _gemini_reply()

    out = images.generate_image(
        "gemini/gemini-3.1-flash-lite-image", "draw", images=[PNG, JPG], key="k", urlopen=urlopen
    )
    assert out == PNG
    parts = seen[0]["contents"][0]["parts"]
    assert parts[0] == {"text": "draw"}
    assert [part["inline_data"]["mime_type"] for part in parts[1:]] == ["image/png", "image/jpeg"]


def test_a_gemini_timeout_is_tried_once_more_and_a_refusal_is_named():
    calls = []

    def urlopen(request, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError
        return io.BytesIO(json.dumps({"candidates": [{"finishReason": "SAFETY"}]}).encode())

    with pytest.raises(images.ImageError, match="SAFETY"):
        images.generate_image("gemini/x", "draw", key="k", urlopen=urlopen)
    assert len(calls) == 2


def test_an_http_error_is_a_sentence_with_its_code_and_no_key_is_named():
    def urlopen(request, timeout):
        raise urllib.error.HTTPError("u", 429, "busy", {}, io.BytesIO(b"slow down"))

    with pytest.raises(images.ImageError, match=r"429.*slow down"):
        images.generate_image("gemini/x", "draw", key="k", urlopen=urlopen)
    with pytest.raises(images.ImageError, match="GEMINI_API_KEY"):
        images.generate_image("gemini/x", "draw", key="")


def test_openai_with_input_images_uses_the_edits_endpoint():
    sent = {}

    class Images:
        def edit(self, **kwargs):
            sent.update(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(PNG).decode())])

    client = SimpleNamespace(images=Images())
    out = images.generate_image(
        "openai/gpt-image-1",
        "edit",
        images=[PNG, JPG],
        size="1024x1024",
        key="k",
        openai_client=client,
    )
    assert out == PNG and sent["model"] == "gpt-image-1" and len(sent["image"]) == 2
    assert sent["size"] == "1024x1024"


def test_text_to_image_goes_through_litellm_and_declines_input_images_there():
    def generation(**kwargs):
        assert kwargs["size"] == "1536x1024"
        return {"data": [{"b64_json": base64.b64encode(PNG).decode()}]}

    assert (
        images.generate_image(
            "openai/dall-e-3", "draw", size="1536x1024", image_generation=generation
        )
        == PNG
    )
    with pytest.raises(images.ImageError, match="text-to-image only"):
        images.generate_image("stability/sd", "draw", images=[PNG], key="k")


def test_the_ledger_line_is_a_count_with_no_invented_cost():
    line = images.usage_line("gemini/gemini-3.1-flash-lite-image", "slot-3")
    assert line["tokens_in"] == 0 and line["cost_usd"] is None and line["stage"] == "illustrations"
