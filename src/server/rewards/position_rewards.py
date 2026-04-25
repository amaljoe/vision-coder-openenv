"""Position reward: compare spatial layout of matched text blocks.

Phase 2 of the Design2Code metrics extension.
Reuses text block extraction from Phase 1 (text_block_rewards.py).

For each matched block pair, computes the normalised distance between
bounding-box centres and converts it to a similarity score in [0, 1].
"""
from __future__ import annotations

import logging
import math
from typing import Optional

from openenv.server.rewards import extract_html
from openenv.server.rewards.text_block_rewards import _get_text_blocks

logger = logging.getLogger(__name__)

_VIEWPORT_W = 640
_VIEWPORT_H = 480
# Diagonal of the viewport — used to normalise distances to [0, 1]
_VIEWPORT_DIAG = math.sqrt(_VIEWPORT_W**2 + _VIEWPORT_H**2)


def position_reward(
    completions: list[list[dict]],
    solution: Optional[list[str]] = None,
) -> list[float]:
    """Score positional accuracy of text blocks relative to the reference.

    Matches reference and predicted blocks using the Hungarian algorithm on
    normalised centre-to-centre distance, then averages the position scores
    (1 − normalised_distance) across all matched pairs.

    Args:
        completions: List of completion message lists.
        solution:    List of reference HTML strings (one per completion).

    Returns:
        List of float scores in [0.0, 1.0].
    """
    results = []
    for i, completion in enumerate(completions):
        content = completion[0]["content"]
        html = extract_html(content)
        ref_html = solution[i] if solution and i < len(solution) else None

        if not ref_html:
            results.append(0.0)
            continue

        try:
            import numpy as np
            from scipy.optimize import linear_sum_assignment

            ref_blocks = _get_text_blocks(ref_html)
            pred_blocks = _get_text_blocks(html)

            if not ref_blocks:
                results.append(1.0 if not pred_blocks else 0.5)
                continue

            if not pred_blocks:
                results.append(0.0)
                continue

            n_ref = len(ref_blocks)
            n_pred = len(pred_blocks)

            # Cost = normalised Euclidean distance between block centres
            cost_matrix = np.zeros((n_ref, n_pred), dtype=np.float64)
            for r, ref_block in enumerate(ref_blocks):
                ref_cx = ref_block["x"] + ref_block["width"] / 2
                ref_cy = ref_block["y"] + ref_block["height"] / 2
                for p, pred_block in enumerate(pred_blocks):
                    pred_cx = pred_block["x"] + pred_block["width"] / 2
                    pred_cy = pred_block["y"] + pred_block["height"] / 2
                    dist = math.sqrt((ref_cx - pred_cx) ** 2 + (ref_cy - pred_cy) ** 2)
                    cost_matrix[r, p] = dist / _VIEWPORT_DIAG

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Average positional similarity over ALL reference blocks
            # (unmatched blocks beyond n_pred count as distance=1, i.e. score=0)
            position_scores = [1.0 - cost_matrix[r, p] for r, p in zip(row_ind, col_ind)]
            # Pad zeros for any unmatched reference blocks
            if len(position_scores) < n_ref:
                position_scores += [0.0] * (n_ref - len(position_scores))

            score = max(0.0, sum(position_scores) / n_ref)
            results.append(score)

        except Exception as exc:
            logger.warning("Position reward failed: %s", exc)
            results.append(0.0)

    return results
