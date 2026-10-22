"""Images: one rendered image per call (`llmlm[images]`), the third road beside text and TypeSafe.

`generate_image(model, prompt, ...)` returns the image bytes. A `gemini/*` model takes the prompt
and any input images (an image to edit, cast plates to hold a likeness to) through
`generateContent`; an `openai/*` model with input images goes through the images EDITS endpoint
(the `openai` package, up to 16 inputs); every other model goes through `litellm.image_generation`,
which is text-to-image only and so declines input images. Rendering is an explicit act, so there is
no silent degradation: every failure is an `ImageError` with a readable sentence, never a provider
traceback. The key follows the model's vendor prefix (`llm.vendor_key_env`), never the text
model's.

A render has no token count and the vendors publish no flat per-image price, so its ledger line
(`usage_line`) is a count: zero tokens, a null cost, kind `api`, which the usage board counts as
`cost_unknown` rather than inventing a dollar figure.

Never imports: usage, estimate, prices. Needs the `images` extra (`openai`) only for edits.

Terms:
- **render**: one `generate_image` call, one image back.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request

from . import llm

DEFAULT_MODEL = "gemini/gemini-2.5-flash-image"
STAGE = "illustrations"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class ImageError(RuntimeError):
    """A render failed; the message says why and what to change."""


def image_mime(blob: bytes) -> str:
    return "image/png" if blob.startswith(b"\x89PNG") else "image/jpeg"


def generate_image(
    model: str,
    prompt: str,
    *,
    images: list[bytes] | None = None,
    size: str = "",
    timeout: float = 120,
    key: str | None = None,
    urlopen=None,
    image_generation=None,
    openai_client=None,
) -> bytes:
    """One image for `prompt` on `model`. `images` are the inputs (the image to edit first, then
    reference plates, in the order the prompt names them). `size` is the vendor's size string, ''
    for its default."""
    images = list(images or [])
    urlopen = urlopen or urllib.request.urlopen  # looked up now, so a test can patch it
    key = key if key is not None else os.environ.get(llm.vendor_key_env(model), "")
    if model.startswith("gemini/"):
        if not key:
            raise ImageError("no API key for image generation: set GEMINI_API_KEY in .env")
        return _gemini(model.split("/", 1)[1], prompt, images, key, timeout, urlopen)
    if images and model.startswith("openai/"):
        if not key:
            raise ImageError("no API key for image generation: set OPENAI_API_KEY in .env")
        return _openai_edit(
            model.split("/", 1)[1], prompt, images, size, key, timeout, openai_client
        )
    if images:
        raise ImageError(
            f"an input image or reference plates need a gemini/* or openai/* image model; "
            f"'{model}' goes through litellm.image_generation, which is text-to-image only"
        )
    return _text_to_image(model, prompt, size, timeout, image_generation)


def usage_line(model: str, tag: str, *, stage: str = STAGE, outcome: str = "ok") -> dict:
    """The `usage.jsonl` event of one render (`llmlm.ledger.Ledger.usage`): a count, no cost."""
    return {
        "stage": stage,
        "prompt_id": "image",
        "model": model,
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": None,
        "kind": "api",
        "cached": False,
        "tag": tag,
        "outcome": outcome,
    }


def _text_to_image(model, prompt, size, timeout, image_generation) -> bytes:
    if image_generation is None:
        import litellm

        litellm.suppress_debug_info = True
        image_generation = litellm.image_generation
    try:
        response = image_generation(
            model=model, prompt=prompt, timeout=timeout, **({"size": size} if size else {})
        )
    except Exception as error:  # noqa: BLE001 - a vendor's own types; one readable sentence
        raise ImageError(f"image model '{model}' failed: {str(error).strip()[:300]}") from error
    data = (response["data"] if isinstance(response, dict) else response.data)[0]
    b64 = data["b64_json"] if isinstance(data, dict) else data.b64_json
    if not b64:
        raise ImageError(f"image model '{model}' returned no image data")
    return base64.b64decode(b64)


def _openai_edit(model, prompt, images, size, key, timeout, client) -> bytes:
    """The images EDITS endpoint: the inputs plus the prompt, one image back."""
    files = []
    for number, blob in enumerate(images, 1):
        handle = io.BytesIO(blob)
        handle.name = (
            f"image-{number}.png" if image_mime(blob) == "image/png" else f"image-{number}.jpg"
        )
        files.append(handle)
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ImageError(
                "the 'openai' package is needed for reference-image renders on openai/* "
                "image models (pip install 'llmlm[images]')"
            ) from error
        client = OpenAI(api_key=key, timeout=timeout)
    try:
        response = client.images.edit(
            model=model,
            image=files if len(files) > 1 else files[0],
            prompt=prompt,
            n=1,
            **({"size": size} if size else {}),
        )
    except Exception as error:  # noqa: BLE001
        raise ImageError(
            f"image model 'openai/{model}' failed: {str(error).strip()[:300]}"
        ) from error
    data = response.data[0] if response.data else None
    b64 = getattr(data, "b64_json", None) if data is not None else None
    if not b64:
        raise ImageError(f"image model 'openai/{model}' returned no image data")
    return base64.b64decode(b64)


def _gemini(model, prompt, images, key, timeout, urlopen) -> bytes:
    parts: list[dict] = [{"text": prompt}]
    for blob in images:
        parts.append(
            {
                "inline_data": {
                    "mime_type": image_mime(blob),
                    "data": base64.b64encode(blob).decode("ascii"),
                }
            }
        )
    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
    }
    request = urllib.request.Request(
        GEMINI_URL.format(model=model),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    try:
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode())
        except (TimeoutError, urllib.error.URLError) as error:
            if isinstance(error, urllib.error.HTTPError):
                raise
            # Renders run long and stall transiently; one fresh attempt recovers most timeouts.
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:400]
        raise ImageError(f"Gemini image call failed ({error.code}): {detail}") from error
    except urllib.error.URLError as error:
        raise ImageError(f"Gemini image call failed: {error}") from error
    for candidate in payload.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            blob = part.get("inlineData") or part.get("inline_data")
            if blob and blob.get("data"):
                return base64.b64decode(blob["data"])
    finish = (payload.get("candidates") or [{}])[0].get("finishReason")
    raise ImageError(
        f"Gemini returned no image (finishReason={finish}): the prompt may have been refused; "
        "reword and re-render"
    )
