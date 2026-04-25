"""Visual reward: CLIP image-embedding cosine similarity after rendering HTML.

Uses openai/clip-vit-base-patch32 on CPU — no GPU required, stays well
within the 16 GB HF Spaces memory limit (~1.5 GB total for model + inference).

Falls back to PIL pixel-diff if CLIP fails to load.
"""
from __future__ import annotations

import io
import logging
from typing import Optional

from PIL import Image

from openenv.server.rewards import extract_html

logger = logging.getLogger(__name__)

import os as _os
_CLIP_MODEL_NAME = _os.path.expanduser("~/models/clip-vit-base-patch32") if _os.path.isdir(_os.path.expanduser("~/models/clip-vit-base-patch32")) else "openai/clip-vit-base-patch32"
_clip_model = None
_clip_processor = None


def _get_clip():
    """Lazy singleton — loads CLIP once, reuses across calls."""
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        logger.info("Loading CLIP model %s …", _CLIP_MODEL_NAME)
        _clip_model = CLIPModel.from_pretrained(_CLIP_MODEL_NAME)
        _clip_model.eval()
        _clip_processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_NAME)
        logger.info("CLIP model loaded.")
    return _clip_model, _clip_processor


def _render_html(html: str, width: int = 640, height: int = 480) -> Optional[Image.Image]:
    """Render HTML to a PIL Image using Playwright headless Chromium.

    Uses full_page=True so the complete page is captured (no viewport cropping).
    The viewport width is fixed; height auto-expands to fit content.
    """
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = browser.new_page(viewport={"width": width, "height": height})
            page.set_content(html, wait_until="networkidle")
            png_bytes = page.screenshot(full_page=True)
            browser.close()
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")
    except Exception as exc:
        logger.warning("HTML rendering failed: %s", exc)
        return None


_CLIP_RENORM_THRESHOLD = 0.65  # raw cosine similarity ≤ this → score 0; 1.0 → 1.0
# Renormalisation makes the metric stricter: only pages visually similar to reference
# score meaningfully. Blank pages (raw ~0.45) and unstyled pages (raw ~0.75) get pushed
# toward 0, while near-perfect matches (raw ~1.0) remain high.


def _clip_similarity(img_a: Image.Image, img_b: Image.Image) -> float:
    """Compute CLIP image-embedding cosine similarity, renormalised to [0, 1]."""
    import torch
    model, processor = _get_clip()
    inputs = processor(images=[img_a, img_b], return_tensors="pt")
    with torch.no_grad():
        out = model.get_image_features(**inputs)
    # transformers v5 returns a dataclass; v4 returns a plain tensor
    features = out.pooler_output if hasattr(out, "pooler_output") else out
    features = features / features.norm(dim=-1, keepdim=True)
    raw = (features[0] @ features[1]).item()
    # Renormalise: threshold → 0, 1.0 → 1.0
    scale = 1.0 - _CLIP_RENORM_THRESHOLD
    return float(max(0.0, min(1.0, (raw - _CLIP_RENORM_THRESHOLD) / scale)))


def _pil_similarity(img_a: Image.Image, img_b: Image.Image, size: tuple = (128, 128)) -> float:
    """Fallback: pixel-wise similarity in [0, 1]."""
    a = img_a.resize(size).convert("RGB")
    b = img_b.resize(size).convert("RGB")
    pa = list(a.getdata())
    pb = list(b.getdata())
    total_diff = sum(
        abs(int(ra) - int(rb)) + abs(int(ga) - int(gb)) + abs(int(ba) - int(bb))
        for (ra, ga, ba), (rb, gb, bb) in zip(pa, pb)
    )
    max_diff = size[0] * size[1] * 3 * 255
    return 1.0 - total_diff / max_diff


def clip_visual_reward(
    completions: list[list[dict]],
    image: Optional[list[Image.Image]] = None,
    pred_image: Optional[list[Optional[Image.Image]]] = None,
) -> list[float]:
    """Score visual similarity between rendered HTML and reference screenshot.

    Renders each completion's HTML with Playwright (unless pred_image is
    provided), then computes CLIP cosine similarity against the reference.
    Falls back to PIL pixel-diff if CLIP is unavailable.
    Returns 0.5 if rendering fails.

    Args:
        completions: List of completion message lists.
        image:       List of reference PIL Images (one per completion).
        pred_image:  Optional pre-rendered prediction images (skips rendering
                     when provided — avoids duplicate Playwright launches).

    Returns:
        List of float scores in [0.0, 1.0].
    """
    # Determine similarity function — prefer CLIP, fall back to pixel-diff
    try:
        _get_clip()
        sim_fn = _clip_similarity
    except Exception as exc:
        logger.warning("CLIP unavailable, falling back to pixel-diff: %s", exc)
        sim_fn = _pil_similarity

    results = []
    for i, completion in enumerate(completions):
        content = completion[0]["content"]
        html = extract_html(content)
        ref_image = image[i] if image and i < len(image) else None

        # Use pre-rendered image if supplied, otherwise render now
        if pred_image is not None and i < len(pred_image):
            rendered = pred_image[i]
        else:
            rendered = _render_html(html)

        if rendered is None or ref_image is None:
            results.append(0.5)
            continue

        try:
            score = sim_fn(rendered, ref_image.convert("RGB"))
        except Exception as exc:
            logger.warning("Similarity scoring failed: %s", exc)
            score = 0.5

        results.append(score)

    return results
