"""Visual reward: CLIP image-image similarity after rendering HTML."""
from __future__ import annotations

import io
import logging
from typing import Optional

from PIL import Image

from vcoder.rewards import extract_html

logger = logging.getLogger(__name__)

_clip_model = None
_clip_processor = None


def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_processor


def _render_html(html: str, width: int = 1280, height: int = 720) -> Optional[Image.Image]:
    """Render HTML to a PIL Image using Playwright headless Chromium."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = browser.new_page(viewport={"width": width, "height": height})
            page.set_content(html, wait_until="networkidle")
            png_bytes = page.screenshot(full_page=False)
            browser.close()
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")
    except Exception as exc:
        logger.warning("HTML rendering failed: %s", exc)
        return None


def clip_visual_reward(
    completions: list[list[dict]],
    image: Optional[list[Image.Image]] = None,
) -> list[float]:
    """Score visual similarity between rendered HTML and the reference screenshot.

    Renders each completion's HTML with Playwright, then computes cosine
    similarity between the rendered and reference image CLIP embeddings.

    Falls back to 0.0 if rendering or encoding fails.

    Args:
        completions: List of completion message lists.
        image:       List of reference PIL Images (one per completion).

    Returns:
        List of float scores in [0.0, 1.0].
    """
    import torch

    model, processor = _get_clip()
    results = []

    for i, completion in enumerate(completions):
        content = completion[0]["content"]
        html = extract_html(content)
        ref_image = image[i] if image and i < len(image) else None

        rendered = _render_html(html)
        if rendered is None or ref_image is None:
            results.append(0.0)
            continue

        try:
            inputs = processor(
                images=[rendered, ref_image.convert("RGB")],
                return_tensors="pt",
                padding=True,
            )
            with torch.no_grad():
                features = model.get_image_features(**inputs)

            # Cosine similarity then map [-1, 1] → [0, 1]
            sim = torch.nn.functional.cosine_similarity(
                features[0:1], features[1:2]
            ).item()
            score = (sim + 1.0) / 2.0
        except Exception as exc:
            logger.warning("CLIP scoring failed: %s", exc)
            score = 0.0

        results.append(score)

    return results
