"""Image-token estimation and reduction.

Vision inputs are billed by image dimensions, not by the base64 payload, so
the cheapest wins come from resizing images to each provider's sweet spot (or
flipping OpenAI's ``detail`` flag) before the request is sent.

Three layers, cheapest first:

- :func:`estimate_image_tokens` — pure calculators for the published formulas.
- :func:`plan_image_reduction` — picks the cheapest strategy that stays useful
  (``passthrough`` / ``downscale`` / ``detail-low``) for one image.
- :func:`reduce_image_tokens` — walks a message array, decodes embedded
  base64 images and applies the plans. Pillow is optional (``pip install
  tokenslim[images]``): with it, images are actually resized and re-encoded;
  without it, dimensions are read from PNG/JPEG/GIF headers and downscale
  plans are reported in the stats but the messages pass through unchanged.
"""

from __future__ import annotations

import base64
import copy
import io
import math
from dataclasses import dataclass, field
from typing import Any

from .config import Config, load_config

__all__ = [
    "ImagePlan",
    "ImageStats",
    "estimate_image_tokens",
    "plan_image_reduction",
    "reduce_image_tokens",
]

Message = dict[str, Any]

# --- provider constants (see estimate_image_tokens docstring for sources) ---
_OPENAI_BASE_TOKENS = 85
_OPENAI_TILE_TOKENS = 170
_OPENAI_TILE_PX = 512
_OPENAI_MAX_EDGE = 2048
_OPENAI_SHORT_SIDE = 768
_ANTHROPIC_PX_PER_TOKEN = 750
_ANTHROPIC_MAX_EDGE = 1568
_ANTHROPIC_TOKEN_CAP = 1600
_ANTHROPIC_OPTIMAL_PIXELS = 1_150_000  # ~1.15 megapixels, Anthropic's stated optimum
_GOOGLE_TILE_TOKENS = 258
_GOOGLE_TILE_PX = 768
_GOOGLE_SWEET_EDGE = 1536  # keep at most a 2x2 tile grid when downscaling

# Downscaling below this short side rarely stays useful to a vision model.
_MIN_USEFUL_SIDE = 16

_PROVIDER_ALIASES = {
    "openai": "openai",
    "gpt": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "google": "google",
    "gemini": "google",
}

_DETAILS = frozenset({"auto", "low", "high"})

_FORMAT_BY_MIME = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG",
    "image/gif": "GIF",
    "image/webp": "WEBP",
}

_JPEG_QUALITY = 85


@dataclass(frozen=True)
class ImagePlan:
    """Reduction decision for a single image."""

    # "passthrough" | "downscale" | "detail-low"
    strategy: str
    new_width: int
    new_height: int
    est_tokens_before: int
    est_tokens_after: int


@dataclass
class ImageStats:
    """Aggregate result of :func:`reduce_image_tokens`."""

    # Base64 image blocks found in the messages.
    images: int = 0
    # Blocks actually rewritten (resized or detail-lowered).
    changed: int = 0
    # Token estimates for the returned messages (after == before for any plan
    # that could not be executed, e.g. a downscale without Pillow installed).
    est_tokens_before: int = 0
    est_tokens_after: int = 0
    # One plan per decodable image, in walk order (kept even when unexecuted).
    plans: list[ImagePlan] = field(default_factory=list)
    # Whether Pillow was importable (downscale plans need it to execute).
    pillow_available: bool = False

    @property
    def saved_tokens(self) -> int:
        return self.est_tokens_before - self.est_tokens_after


def _normalize_provider(provider: str) -> str:
    key = provider.strip().lower()
    if key not in _PROVIDER_ALIASES:
        raise ValueError(
            f"unknown provider: {provider!r} (expected 'openai', 'anthropic', or 'google')"
        )
    return _PROVIDER_ALIASES[key]


def _scaled(width: int, height: int, scale: float) -> tuple[int, int]:
    return max(1, round(width * scale)), max(1, round(height * scale))


def _fit(width: int, height: int, max_edge: int) -> tuple[int, int]:
    """Scale ``(width, height)`` down (never up) to fit a square of ``max_edge``."""
    scale = min(1.0, max_edge / max(width, height))
    return (width, height) if scale >= 1.0 else _scaled(width, height, scale)


def _openai_scaled_dims(width: int, height: int) -> tuple[int, int]:
    """Dimensions OpenAI processes in high detail: fit 2048², then short side 768."""
    width, height = _fit(width, height, _OPENAI_MAX_EDGE)
    short = min(width, height)
    if short > _OPENAI_SHORT_SIDE:
        width, height = _scaled(width, height, _OPENAI_SHORT_SIDE / short)
    return width, height


def _estimate_openai(width: int, height: int, detail: str) -> int:
    if detail == "auto":
        detail = "low" if max(width, height) <= _OPENAI_TILE_PX else "high"
    if detail == "low":
        return _OPENAI_BASE_TOKENS
    width, height = _openai_scaled_dims(width, height)
    tiles = math.ceil(width / _OPENAI_TILE_PX) * math.ceil(height / _OPENAI_TILE_PX)
    return _OPENAI_BASE_TOKENS + _OPENAI_TILE_TOKENS * tiles


def _estimate_anthropic(width: int, height: int) -> int:
    width, height = _fit(width, height, _ANTHROPIC_MAX_EDGE)
    return min(math.ceil(width * height / _ANTHROPIC_PX_PER_TOKEN), _ANTHROPIC_TOKEN_CAP)


def _estimate_google(width: int, height: int) -> int:
    tiles = math.ceil(width / _GOOGLE_TILE_PX) * math.ceil(height / _GOOGLE_TILE_PX)
    return _GOOGLE_TILE_TOKENS * tiles


def estimate_image_tokens(
    width: int,
    height: int,
    provider: str,
    detail: str = "auto",
) -> int:
    """Estimate the input tokens a ``width`` x ``height`` image costs at ``provider``.

    Published formulas:

    - **OpenAI** (GPT-4o family; https://platform.openai.com/docs/guides/vision):
      ``detail="low"`` is a flat 85 tokens. ``detail="high"`` first scales the
      image to fit within 2048x2048, then scales the shortest side down to
      768 px, and costs ``85 + 170 * tiles`` where ``tiles`` is the count of
      512x512 tiles covering the scaled image (1024x1024 -> 768x768 -> 4 tiles
      -> 765 tokens; 2048x4096 -> 768x1536 -> 6 tiles -> 1105 tokens).
      ``detail="auto"`` is treated as low when the image fits in 512x512,
      otherwise high.
    - **Anthropic** (Claude 3+;
      https://docs.anthropic.com/en/docs/build-with-claude/vision):
      ``tokens = ceil(width * height / 750)``. Images whose long edge exceeds
      1568 px are scaled down to that edge first, and the result is capped at
      ~1600 tokens.
    - **Google Gemini** (https://ai.google.dev/gemini-api/docs/tokens):
      258 tokens per 768x768 tile after fitting, i.e.
      ``258 * ceil(width / 768) * ceil(height / 768)``; small images (both
      dimensions <= 384 px) cost a single tile of 258, which the tile formula
      already yields. ``detail`` is ignored.

    Provider aliases ``gpt``, ``claude`` and ``gemini`` are accepted. Raises
    :class:`ValueError` for an unknown provider, non-positive dimensions, or a
    ``detail`` outside ``{"auto", "low", "high"}``.
    """
    prov = _normalize_provider(provider)
    if width < 1 or height < 1:
        raise ValueError(f"image dimensions must be positive, got {width}x{height}")
    if detail not in _DETAILS:
        raise ValueError(f"invalid detail: {detail!r} (expected 'auto', 'low', or 'high')")
    if prov == "openai":
        return _estimate_openai(width, height, detail)
    if prov == "anthropic":
        return _estimate_anthropic(width, height)
    return _estimate_google(width, height)


def _sweet_spot_dims(width: int, height: int, provider: str) -> tuple[int, int]:
    """Largest dimensions ``provider`` makes full use of — extra pixels are wasted."""
    if provider == "openai":
        return _openai_scaled_dims(width, height)
    if provider == "anthropic":
        scale = min(
            1.0,
            _ANTHROPIC_MAX_EDGE / max(width, height),
            math.sqrt(_ANTHROPIC_OPTIMAL_PIXELS / (width * height)),
        )
        return (width, height) if scale >= 1.0 else _scaled(width, height, scale)
    return _fit(width, height, _GOOGLE_SWEET_EDGE)


def plan_image_reduction(
    width: int,
    height: int,
    provider: str,
    target_tokens: int | None = None,
    detail: str = "auto",
) -> ImagePlan:
    """Pick the cheapest reduction strategy for one image that stays useful.

    Without ``target_tokens``, images larger than the provider sweet spot
    (OpenAI: 768 short side within 2048x2048; Anthropic: 1568 long edge and
    ~1.15 megapixels; Google: 1536 long edge) get a ``downscale`` plan to that
    spot — the provider would discard the extra pixels anyway. With a
    ``target_tokens`` budget, the plan is the largest downscale (searched in
    5% steps) whose estimate fits the budget; when no useful downscale can
    reach it, OpenAI images fall back to ``detail-low`` (flat 85 tokens) and
    other providers keep the best-effort downscale. Images already within
    budget (or already at the sweet spot) pass through.
    """
    prov = _normalize_provider(provider)
    before = estimate_image_tokens(width, height, prov, detail)

    if prov == "openai" and detail == "auto":
        # Resolve "auto" against the *original* size so a downscale plan means
        # "high detail with fewer tiles" rather than silently flipping to low.
        detail = "low" if max(width, height) <= _OPENAI_TILE_PX else "high"

    if prov == "openai" and detail == "low":
        # Already at the 85-token floor; nothing to reduce.
        return ImagePlan("passthrough", width, height, before, before)

    if target_tokens is None:
        new_w, new_h = _sweet_spot_dims(width, height, prov)
        if (new_w, new_h) == (width, height):
            return ImagePlan("passthrough", width, height, before, before)
        after = estimate_image_tokens(new_w, new_h, prov, detail)
        return ImagePlan("downscale", new_w, new_h, before, after)

    if before <= target_tokens:
        return ImagePlan("passthrough", width, height, before, before)

    # Largest scale (in 5% steps) whose estimate fits the budget.
    smallest: tuple[int, int, int] | None = None
    for step in range(19, 0, -1):
        new_w, new_h = _scaled(width, height, step / 20)
        if min(new_w, new_h) < _MIN_USEFUL_SIDE:
            break
        after = estimate_image_tokens(new_w, new_h, prov, detail)
        if after <= target_tokens:
            return ImagePlan("downscale", new_w, new_h, before, after)
        smallest = (new_w, new_h, after)

    if prov == "openai":
        return ImagePlan("detail-low", width, height, before, _OPENAI_BASE_TOKENS)
    if smallest is not None and smallest[2] < before:
        return ImagePlan("downscale", smallest[0], smallest[1], before, smallest[2])
    return ImagePlan("passthrough", width, height, before, before)


# --- message walking -------------------------------------------------------


def _load_pillow() -> Any:
    """Return ``PIL.Image`` when Pillow is installed, else ``None`` (optional dep)."""
    try:
        from PIL import Image
    except ImportError:
        return None
    return Image


def _parse_data_url(url: str) -> tuple[str, bytes] | None:
    """Split a ``data:<mime>;base64,<payload>`` URL into (media_type, raw bytes)."""
    if not url.startswith("data:"):
        return None
    header, sep, payload = url.partition(",")
    if not sep or not header.endswith(";base64"):
        return None
    media_type = header[5:].split(";", 1)[0] or "application/octet-stream"
    try:
        return media_type, base64.b64decode(payload)
    except ValueError:
        return None


def _extract_image(block: dict[str, Any]) -> tuple[bytes, str, str] | None:
    """Return ``(raw_bytes, media_type, kind)`` for a base64 image block."""
    block_type = block.get("type")
    if block_type == "image_url":
        image_url = block.get("image_url")
        if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
            parsed = _parse_data_url(image_url["url"])
            if parsed is not None:
                return parsed[1], parsed[0], "openai"
    elif block_type == "image":
        source = block.get("source")
        if isinstance(source, dict) and source.get("type") == "base64":
            data = source.get("data")
            if isinstance(data, str):
                try:
                    raw = base64.b64decode(data)
                except ValueError:
                    return None
                return raw, str(source.get("media_type") or "image/png"), "anthropic"
    return None


# SOF0-SOF15 carry frame dimensions, except DHT (C4), JPG (C8) and DAC (CC).
_JPEG_SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}


def _jpeg_dims(raw: bytes) -> tuple[int, int] | None:
    pos = 2
    size = len(raw)
    while pos + 9 < size:
        if raw[pos] != 0xFF:
            return None
        marker = raw[pos + 1]
        if marker == 0xFF:  # fill byte
            pos += 1
            continue
        if marker in _JPEG_SOF_MARKERS:
            height = int.from_bytes(raw[pos + 5 : pos + 7], "big")
            width = int.from_bytes(raw[pos + 7 : pos + 9], "big")
            return width, height
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:  # standalone markers
            pos += 2
            continue
        length = int.from_bytes(raw[pos + 2 : pos + 4], "big")
        if length < 2:
            return None
        pos += 2 + length
    return None


def _dims_from_headers(raw: bytes) -> tuple[int, int] | None:
    """Read dimensions from PNG/GIF/JPEG headers with the stdlib only."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24:
        return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
    if raw[:6] in (b"GIF87a", b"GIF89a") and len(raw) >= 10:
        return int.from_bytes(raw[6:8], "little"), int.from_bytes(raw[8:10], "little")
    if raw.startswith(b"\xff\xd8"):
        return _jpeg_dims(raw)
    return None


def _image_dimensions(raw: bytes, pil: Any = None) -> tuple[int, int] | None:
    dims = _dims_from_headers(raw)
    if dims is not None:
        return dims
    if pil is not None:
        try:
            with pil.open(io.BytesIO(raw)) as img:
                return int(img.width), int(img.height)
        except Exception:
            return None
    return None


def _resize_bytes(
    raw: bytes,
    media_type: str,
    new_width: int,
    new_height: int,
    pil: Any,
) -> bytes | None:
    """Resize with Pillow, re-encoding in the original format (JPEG quality 85)."""
    try:
        img = pil.open(io.BytesIO(raw))
        fmt = (img.format or _FORMAT_BY_MIME.get(media_type, "PNG")).upper()
        resized = img.resize((new_width, new_height), pil.LANCZOS)
        buf = io.BytesIO()
        if fmt in {"JPEG", "JPG"}:
            if resized.mode not in {"RGB", "L"}:
                resized = resized.convert("RGB")
            resized.save(buf, format="JPEG", quality=_JPEG_QUALITY)
        else:
            resized.save(buf, format=fmt)
        return buf.getvalue()
    except Exception:
        return None


def _reduce_block(
    block: dict[str, Any],
    provider: str,
    config: Config,
    pil: Any,
    stats: ImageStats,
) -> None:
    """Plan and (when possible) apply a reduction to one block, in place."""
    extracted = _extract_image(block)
    if extracted is None:
        return
    raw, media_type, kind = extracted
    stats.images += 1

    dims = _image_dimensions(raw, pil)
    if dims is None or min(dims) < 1:
        return
    width, height = dims

    detail = config.image_detail
    if kind == "openai":
        block_detail = block["image_url"].get("detail")
        if isinstance(block_detail, str):
            detail = block_detail
    if detail not in _DETAILS:
        detail = "auto"

    plan = plan_image_reduction(
        width, height, provider, target_tokens=config.image_max_tokens, detail=detail
    )
    stats.plans.append(plan)
    stats.est_tokens_before += plan.est_tokens_before

    if plan.strategy == "detail-low" and kind == "openai":
        block["image_url"]["detail"] = "low"
        stats.changed += 1
        stats.est_tokens_after += plan.est_tokens_after
        return

    if plan.strategy == "downscale" and pil is not None:
        resized = _resize_bytes(raw, media_type, plan.new_width, plan.new_height, pil)
        if resized is not None:
            encoded = base64.b64encode(resized).decode("ascii")
            if kind == "openai":
                block["image_url"]["url"] = f"data:{media_type};base64,{encoded}"
            else:
                block["source"]["data"] = encoded
            stats.changed += 1
            stats.est_tokens_after += plan.est_tokens_after
            return

    # Plan not executable (passthrough plan, no Pillow, or re-encode failure).
    stats.est_tokens_after += plan.est_tokens_before


def reduce_image_tokens(
    messages: list[Message],
    provider: str,
    options: Config | None = None,
    **overrides: Any,
) -> tuple[list[Message], ImageStats]:
    """Reduce the token cost of base64 images embedded in a message array.

    Walks OpenAI ``{"type": "image_url"}`` blocks carrying ``data:`` URLs and
    Anthropic ``{"type": "image", "source": {"type": "base64", ...}}`` blocks,
    plans a reduction per image (see :func:`plan_image_reduction`, driven by
    ``Config.image_max_tokens`` / ``Config.image_detail``), and applies it:
    downscales resize and re-encode via Pillow when it is installed (original
    format kept, JPEG quality 85); ``detail-low`` plans just set the block's
    ``detail`` field. Without Pillow, downscale plans are recorded in
    ``stats.plans`` but the blocks pass through unchanged.

    Remote (``http``/``https``) image URLs are left untouched, and malformed
    image data never raises — the offending block passes through.

    Returns ``(rewritten_messages, stats)``; the input is never mutated.
    """
    config = options.merged(**overrides) if options is not None else load_config(**overrides)
    prov = _normalize_provider(provider)
    pil = _load_pillow()

    stats = ImageStats(pillow_available=pil is not None)
    out = copy.deepcopy(messages)
    for msg in out:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            try:
                _reduce_block(block, prov, config, pil, stats)
            except Exception:
                continue  # never raise mid-walk; leave the block as-is
    return out, stats
