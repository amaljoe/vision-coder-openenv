"""Perceptual color reward using CIEDE2000.

Phase 3 of the Design2Code metrics extension.
Renders both the reference and predicted HTML, samples non-white pixels,
and computes mean CIEDE2000 perceptual color distance.

ΔE = 0  → score = 1.0  (perfect color match)
ΔE = 50 → score = 0.0  (completely wrong colors)
"""
from __future__ import annotations

import logging
import random
from typing import Optional

from PIL import Image

from vcoder.rewards import extract_html
from vcoder.rewards.visual_rewards import _render_html

logger = logging.getLogger(__name__)

_N_SAMPLES = 1000          # pixels to sample per image
_WHITE_THRESHOLD = 240     # RGB channel value above which a pixel is "near-white"
_MAX_DELTA_E = 50.0        # ΔE value that maps to score 0.0


def _sample_pixels(
    img: Image.Image,
    n: int = _N_SAMPLES,
    seed: int = 42,
) -> list[tuple[int, int, int]]:
    """Return up to *n* non-white pixels sampled uniformly from *img*."""
    rng = random.Random(seed)
    import numpy as np

    arr = np.array(img.convert("RGB"))
    pixels = [tuple(p) for p in arr.reshape(-1, 3)]
    non_white = [
        (r, g, b)
        for (r, g, b) in pixels
        if not (r > _WHITE_THRESHOLD and g > _WHITE_THRESHOLD and b > _WHITE_THRESHOLD)
    ]
    if not non_white:
        non_white = pixels  # all-white image — fall back to raw pixels
    if len(non_white) <= n:
        return non_white
    return rng.sample(non_white, n)


def color_reward(
    completions: list[list[dict]],
    image: Optional[list[Image.Image]] = None,
) -> list[float]:
    """Score perceptual color accuracy between rendered HTML and reference.

    Samples non-white pixels from both images and computes the mean
    CIEDE2000 distance. Returns 0.5 if rendering fails (neutral penalty).

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

        if ref_image is None:
            results.append(0.5)
            continue

        try:
            import numpy as np
            from skimage.color import deltaE_ciede2000, rgb2lab

            rendered = _render_html(html)
            if rendered is None:
                results.append(0.5)
                continue

            ref_pixels = _sample_pixels(ref_image.convert("RGB"))
            pred_pixels = _sample_pixels(rendered)

            # Align sample counts
            n = min(len(ref_pixels), len(pred_pixels))
            if n == 0:
                results.append(0.5)
                continue

            ref_arr = np.array(ref_pixels[:n], dtype=np.float32) / 255.0
            pred_arr = np.array(pred_pixels[:n], dtype=np.float32) / 255.0

            # rgb2lab expects shape (H, W, 3); use (1, n, 3)
            ref_lab = rgb2lab(ref_arr.reshape(1, n, 3)).reshape(n, 3)
            pred_lab = rgb2lab(pred_arr.reshape(1, n, 3)).reshape(n, 3)

            delta_e = deltaE_ciede2000(ref_lab, pred_lab)
            mean_delta_e = float(np.mean(delta_e))

            score = 1.0 - min(mean_delta_e / _MAX_DELTA_E, 1.0)
            results.append(score)

        except Exception as exc:
            logger.warning("Color reward failed: %s", exc)
            results.append(0.5)

    return results
