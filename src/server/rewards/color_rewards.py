"""Perceptual color reward using CIEDE2000.

Phase 3 of the Design2Code metrics extension.
Resizes both images to 128×128 and computes per-pixel CIEDE2000, averaging
only over pixels where the REFERENCE is non-white. This penalises blank
predictions even for mostly-white UIs (e.g. login forms on #f0f2f5).

ΔE = 0  → score = 1.0  (perfect color match)
ΔE = 50 → score = 0.0  (completely wrong colors)
"""
from __future__ import annotations

import logging
from typing import Optional

from PIL import Image

from openenv.server.rewards import extract_html
from openenv.server.rewards.visual_rewards import _render_html

logger = logging.getLogger(__name__)

_COMPARE_SIZE = (128, 128)   # downsample resolution for ΔE comparison
_WHITE_THRESHOLD = 240       # pixel channel value above which a pixel is "near-white"
_MIN_NONWHITE_FRAC = 0.02    # if reference has fewer non-white pixels than this fraction,
                              # fall back to full spatial comparison
_MAX_DELTA_E = 50.0          # ΔE value that maps to score 0.0


def _nonwhite_mask(arr_rgb: "np.ndarray") -> "np.ndarray":
    """Boolean mask for pixels where any channel is below _WHITE_THRESHOLD."""
    import numpy as np
    return (arr_rgb[:, :, 0] < _WHITE_THRESHOLD) | \
           (arr_rgb[:, :, 1] < _WHITE_THRESHOLD) | \
           (arr_rgb[:, :, 2] < _WHITE_THRESHOLD)


def color_reward(
    completions: list[list[dict]],
    image: Optional[list[Image.Image]] = None,
    pred_image: Optional[list[Optional[Image.Image]]] = None,
) -> list[float]:
    """Score perceptual color accuracy between rendered HTML and reference.

    Resizes both images to 128×128, computes per-pixel CIEDE2000, then
    averages only over pixels where the REFERENCE is non-white. This
    correctly penalises blank predictions even when the reference has a
    mostly-white background (e.g. a login form on #f0f2f5).

    Falls back to full spatial mean when the reference is nearly all white
    (< 2% non-white pixels).  Returns 0.5 if rendering fails.

    Args:
        completions: List of completion message lists.
        image:       List of reference PIL Images (one per completion).
        pred_image:  Optional pre-rendered prediction images (skips rendering
                     when provided — avoids duplicate Playwright launches).

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

            if pred_image is not None and i < len(pred_image):
                rendered = pred_image[i]
            else:
                rendered = _render_html(html)
            if rendered is None:
                results.append(0.5)
                continue

            W, H = _COMPARE_SIZE
            ref_arr  = np.array(ref_image.convert("RGB").resize(_COMPARE_SIZE), dtype=np.float32) / 255.0
            pred_arr = np.array(rendered.convert("RGB").resize(_COMPARE_SIZE), dtype=np.float32) / 255.0

            ref_lab  = rgb2lab(ref_arr.reshape(1, H, W, 3)).reshape(H * W, 3)
            pred_lab = rgb2lab(pred_arr.reshape(1, H, W, 3)).reshape(H * W, 3)
            delta_e  = deltaE_ciede2000(ref_lab, pred_lab)

            # Average ΔE only over pixels where the reference is non-white so
            # that blank predictions are penalised on a mostly-white reference.
            mask = _nonwhite_mask((ref_arr * 255).astype(np.uint8)).ravel()
            if mask.sum() >= _MIN_NONWHITE_FRAC * H * W:
                mean_delta_e = float(np.mean(delta_e[mask]))
            else:
                # Reference is nearly all white — full spatial average
                mean_delta_e = float(np.mean(delta_e))

            score = 1.0 - min(mean_delta_e / _MAX_DELTA_E, 1.0)
            results.append(float(np.clip(score, 0.0, 1.0)))

        except Exception as exc:
            logger.warning("Color reward failed: %s", exc)
            results.append(0.5)

    return results
