"""Visual reward: image similarity after rendering HTML.

Uses PIL pixel-difference similarity — no torch/CLIP required, so the
server stays well within the 16 GB HF Spaces memory limit.
"""
from __future__ import annotations

import io
import logging
from typing import Optional

from PIL import Image

from vcoder.rewards import extract_html

logger = logging.getLogger(__name__)


def _render_html(html: str, width: int = 640, height: int = 480) -> Optional[Image.Image]:
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


def _pil_similarity(img_a: Image.Image, img_b: Image.Image, size: tuple = (128, 128)) -> float:
    """Compute pixel-wise similarity between two images in [0, 1].

    Both images are resized to `size` and compared channel-wise.
    Returns 1.0 for identical images, 0.0 for maximally different.
    """
    a = img_a.resize(size).convert("RGB")
    b = img_b.resize(size).convert("RGB")

    import struct
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
) -> list[float]:
    """Score visual similarity between rendered HTML and reference screenshot.

    Renders each completion's HTML with Playwright, then computes pixel-wise
    similarity against the reference image. Returns 0.5 if rendering fails.

    Args:
        completions: List of completion message lists.
        image:       List of reference PIL Images (one per completion).

    Returns:
        List of float scores in [0.0, 1.0].
    """
    results = []
    for i, completion in enumerate(completions):
        content = completion[0]["content"]
        html = extract_html(content)
        ref_image = image[i] if image and i < len(image) else None

        rendered = _render_html(html)
        if rendered is None or ref_image is None:
            results.append(0.5)
            continue

        try:
            score = _pil_similarity(rendered, ref_image.convert("RGB"))
        except Exception as exc:
            logger.warning("Pixel similarity failed: %s", exc)
            score = 0.5

        results.append(score)

    return results
