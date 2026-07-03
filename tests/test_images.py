"""Tests for image-token estimation, planning and reduction (issue #61)."""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from tokenslim import (
    estimate_image_tokens,
    plan_image_reduction,
    reduce_image_tokens,
)
from tokenslim.images import _image_dimensions

# --- helpers ----------------------------------------------------------------


def _png_bytes(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (30, 30, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def _openai_message(raw: bytes, mime: str = "image/png", detail: str | None = None) -> dict:
    image_url: dict = {"url": f"data:{mime};base64,{base64.b64encode(raw).decode()}"}
    if detail is not None:
        image_url["detail"] = detail
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is in this image?"},
            {"type": "image_url", "image_url": image_url},
        ],
    }


def _anthropic_message(raw: bytes, mime: str = "image/png") -> dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": base64.b64encode(raw).decode(),
                },
            },
            {"type": "text", "text": "describe"},
        ],
    }


def _decode_openai_image(message: dict) -> Image.Image:
    url = message["content"][1]["image_url"]["url"]
    payload = url.partition(",")[2]
    return Image.open(io.BytesIO(base64.b64decode(payload)))


# --- estimate_image_tokens: published formula values -------------------------


def test_openai_high_1024_square_is_765():
    # 1024x1024 -> short side scaled to 768 -> 4 tiles -> 85 + 4*170 = 765.
    assert estimate_image_tokens(1024, 1024, "openai", detail="high") == 765


def test_openai_high_docs_example_2048x4096_is_1105():
    # OpenAI docs example: fit 2048^2 -> 1024x2048, short side 768 -> 768x1536,
    # 2x3 tiles -> 85 + 6*170 = 1105.
    assert estimate_image_tokens(2048, 4096, "openai", detail="high") == 1105


def test_openai_low_detail_is_flat_85():
    assert estimate_image_tokens(4096, 4096, "openai", detail="low") == 85
    assert estimate_image_tokens(100, 100, "openai", detail="low") == 85


def test_openai_single_tile_minimum():
    assert estimate_image_tokens(512, 512, "openai", detail="high") == 85 + 170


def test_openai_auto_picks_low_for_small_high_for_large():
    assert estimate_image_tokens(400, 400, "openai", detail="auto") == 85
    assert estimate_image_tokens(1024, 1024, "openai", detail="auto") == 765


def test_anthropic_pixels_over_750():
    # Anthropic docs example size: 1092x1092 ~ 1590 tokens.
    assert estimate_image_tokens(1092, 1092, "anthropic") == 1590
    assert estimate_image_tokens(200, 200, "anthropic") == 54  # ceil(40000/750)


def test_anthropic_scales_long_edge_then_caps():
    # 4000x4000 is scaled to 1568x1568 first, then capped at ~1600 tokens.
    assert estimate_image_tokens(4000, 4000, "anthropic") == 1600


def test_google_tile_formula():
    assert estimate_image_tokens(512, 512, "google") == 258  # one tile
    assert estimate_image_tokens(768, 768, "google") == 258
    assert estimate_image_tokens(1024, 1024, "google") == 258 * 4
    assert estimate_image_tokens(1536, 768, "google") == 258 * 2


def test_provider_aliases():
    assert estimate_image_tokens(1024, 1024, "gemini") == estimate_image_tokens(
        1024, 1024, "google"
    )
    assert estimate_image_tokens(1024, 1024, "claude") == estimate_image_tokens(
        1024, 1024, "anthropic"
    )


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        estimate_image_tokens(1024, 1024, "mistral")
    with pytest.raises(ValueError):
        estimate_image_tokens(0, 100, "openai")
    with pytest.raises(ValueError):
        estimate_image_tokens(100, 100, "openai", detail="medium")


# --- plan_image_reduction -----------------------------------------------------


def test_plan_passthrough_when_already_small():
    plan = plan_image_reduction(512, 512, "openai")
    assert plan.strategy == "passthrough"
    assert (plan.new_width, plan.new_height) == (512, 512)
    assert plan.est_tokens_after == plan.est_tokens_before


def test_plan_downscales_to_openai_sweet_spot():
    plan = plan_image_reduction(4096, 4096, "openai")
    assert plan.strategy == "downscale"
    assert (plan.new_width, plan.new_height) == (768, 768)
    assert plan.est_tokens_after <= plan.est_tokens_before


def test_plan_downscale_meets_target_budget():
    # 512x512 costs one tile (255) <= 300, and it is the largest such downscale.
    plan = plan_image_reduction(1024, 1024, "openai", target_tokens=300)
    assert plan.strategy == "downscale"
    assert (plan.new_width, plan.new_height) == (512, 512)
    assert plan.est_tokens_before == 765
    assert plan.est_tokens_after == 255


def test_plan_falls_back_to_detail_low_when_target_below_tile_floor():
    # High-detail minimum is 255 tokens; a 100-token budget needs detail=low.
    plan = plan_image_reduction(1024, 1024, "openai", target_tokens=100)
    assert plan.strategy == "detail-low"
    assert plan.est_tokens_after == 85
    assert (plan.new_width, plan.new_height) == (1024, 1024)


def test_plan_low_detail_input_passes_through():
    plan = plan_image_reduction(4096, 4096, "openai", target_tokens=100, detail="low")
    assert plan.strategy == "passthrough"
    assert plan.est_tokens_before == 85


def test_plan_anthropic_sweet_spot_is_115_megapixels():
    plan = plan_image_reduction(3000, 3000, "anthropic")
    assert plan.strategy == "downscale"
    assert max(plan.new_width, plan.new_height) <= 1568
    assert plan.new_width * plan.new_height <= 1_150_000
    assert plan.est_tokens_before == 1600
    assert plan.est_tokens_after < plan.est_tokens_before


def test_plan_anthropic_target_budget():
    plan = plan_image_reduction(2000, 2000, "anthropic", target_tokens=500)
    assert plan.strategy == "downscale"
    assert plan.est_tokens_after <= 500
    assert plan.new_width < 2000


def test_plan_google_target_budget():
    plan = plan_image_reduction(2048, 2048, "google", target_tokens=300)
    assert plan.strategy == "downscale"
    assert plan.est_tokens_after <= 300


# --- reduce_image_tokens ------------------------------------------------------


def test_reduce_openai_downscale_roundtrip():
    messages = [_openai_message(_png_bytes(1024, 1024))]
    out, stats = reduce_image_tokens(messages, "openai", image_max_tokens=300)

    assert stats.images == 1
    assert stats.changed == 1
    assert stats.pillow_available is True
    assert stats.est_tokens_before == 765
    assert stats.est_tokens_after == 255
    assert stats.saved_tokens == 510

    img = _decode_openai_image(out[0])
    assert img.size == (512, 512)
    assert img.format == "PNG"  # original format respected
    # Input is never mutated.
    assert messages[0]["content"][1]["image_url"]["url"] != out[0]["content"][1]["image_url"]["url"]


def test_reduce_anthropic_downscale_roundtrip():
    messages = [_anthropic_message(_png_bytes(2000, 2000))]
    out, stats = reduce_image_tokens(messages, "anthropic")

    assert stats.images == 1
    assert stats.changed == 1
    assert stats.est_tokens_before == 1600
    assert stats.est_tokens_after < 1600

    source = out[0]["content"][0]["source"]
    assert source["type"] == "base64"
    img = Image.open(io.BytesIO(base64.b64decode(source["data"])))
    assert max(img.size) <= 1568
    assert img.size[0] * img.size[1] <= 1_150_000
    assert img.format == "PNG"


def test_reduce_jpeg_keeps_format():
    messages = [_openai_message(_jpeg_bytes(1024, 1024), mime="image/jpeg")]
    out, stats = reduce_image_tokens(messages, "openai", image_max_tokens=300)

    assert stats.changed == 1
    url = out[0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    img = _decode_openai_image(out[0])
    assert img.format == "JPEG"
    assert img.size == (512, 512)


def test_reduce_sets_detail_low_without_touching_data():
    messages = [_openai_message(_png_bytes(1024, 1024))]
    out, stats = reduce_image_tokens(messages, "openai", image_max_tokens=90)

    block = out[0]["content"][1]
    assert block["image_url"]["detail"] == "low"
    assert block["image_url"]["url"] == messages[0]["content"][1]["image_url"]["url"]
    assert stats.changed == 1
    assert stats.est_tokens_after == 85


def test_reduce_respects_block_detail_over_config():
    # The block already asks for low detail -> nothing to reduce.
    messages = [_openai_message(_png_bytes(1024, 1024), detail="low")]
    out, stats = reduce_image_tokens(messages, "openai", image_max_tokens=90)
    assert stats.changed == 0
    assert stats.est_tokens_before == stats.est_tokens_after == 85
    assert out[0] == messages[0]


def test_reduce_ignores_remote_urls_and_text():
    messages = [
        {"role": "user", "content": "plain string content"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            ],
        },
    ]
    out, stats = reduce_image_tokens(messages, "openai", image_max_tokens=100)
    assert out == messages
    assert stats.images == 0
    assert stats.changed == 0


def test_reduce_never_raises_on_garbage_payload():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!notb64"}},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
                },
            ],
        }
    ]
    out, stats = reduce_image_tokens(messages, "anthropic")
    assert out == messages
    assert stats.changed == 0


def test_reduce_plan_only_without_pillow(monkeypatch):
    import tokenslim.images as images_mod

    monkeypatch.setattr(images_mod, "_load_pillow", lambda: None)
    messages = [_openai_message(_png_bytes(1024, 1024))]
    out, stats = reduce_image_tokens(messages, "openai", image_max_tokens=300)

    assert stats.pillow_available is False
    assert out == messages  # passthrough: downscale needs Pillow
    assert stats.images == 1
    assert stats.changed == 0
    assert stats.est_tokens_after == stats.est_tokens_before == 765
    # The intended plan is still reported.
    assert len(stats.plans) == 1
    assert stats.plans[0].strategy == "downscale"
    assert stats.plans[0].est_tokens_after == 255


def test_reduce_passthrough_keeps_payload_identical():
    messages = [_openai_message(_png_bytes(400, 300))]
    out, stats = reduce_image_tokens(messages, "openai")
    assert out == messages
    assert stats.images == 1
    assert stats.changed == 0
    assert stats.plans[0].strategy == "passthrough"


# --- stdlib header parsing ----------------------------------------------------


def test_dimensions_from_headers_without_pillow():
    assert _image_dimensions(_png_bytes(321, 123), pil=None) == (321, 123)
    assert _image_dimensions(_jpeg_bytes(640, 480), pil=None) == (640, 480)

    buf = io.BytesIO()
    Image.new("P", (77, 55)).save(buf, format="GIF")
    assert _image_dimensions(buf.getvalue(), pil=None) == (77, 55)

    assert _image_dimensions(b"not an image", pil=None) is None
