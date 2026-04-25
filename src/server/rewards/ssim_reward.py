"""SSIM (Structural Similarity Index) reward.

Pixel-level perceptual similarity — far more sensitive to small visual
changes than CLIP. Fills the variance gap in the near-perfect zone where
CLIP cosine similarity barely moves but pixel layout/color details change.

Complements color_reward (CIEDE2000) by capturing luminance and contrast
patterns rather than just hue. Fast CPU-only via skimage.
"""
from __future__ import annotations

import logging
from typing import Optional

from PIL import Image

from openenv.server.rewards import extract_html
from openenv.server.rewards.visual_rewards import _render_html

logger = logging.getLogger(__name__)

_COMPARE_SIZE = (320, 240)   # downsample for speed; still captures fine detail


def ssim_reward(
    completions: list[list[dict]],
    image: Optional[list[Image.Image]] = None,
    pred_image: Optional[list[Optional[Image.Image]]] = None,
) -> list[float]:
    """Score pixel-level structural similarity between rendered HTML and reference.

    Uses skimage structural_similarity (channel_axis=-1) on RGB images
    downsampled to 320×240. Score is in [0, 1]; returns 0.5 on failure.

    Args:
        completions: List of completion message lists.
        image:       List of reference PIL Images (one per completion).
        pred_image:  Optional pre-rendered prediction images (skips rendering).

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

        if pred_image is not None and i < len(pred_image):
            rendered = pred_image[i]
        else:
            rendered = _render_html(html)

        if rendered is None:
            results.append(0.5)
            continue

        try:
            import numpy as np
            from skimage.metrics import structural_similarity

            ref_arr  = np.array(ref_image.convert("RGB").resize(_COMPARE_SIZE))
            pred_arr = np.array(rendered.convert("RGB").resize(_COMPARE_SIZE))

            score = structural_similarity(
                ref_arr, pred_arr,
                channel_axis=-1,
                data_range=255,
            )
            results.append(float(max(0.0, min(1.0, score))))
        except Exception as exc:
            logger.warning("SSIM reward failed: %s", exc)
            results.append(0.5)

    return results
