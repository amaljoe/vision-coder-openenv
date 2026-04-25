"""Text block reward: match text blocks between reference and predicted HTML.

Phase 1 of the Design2Code metrics extension.
Uses Playwright to extract text element bounding boxes and content,
then matches blocks using the Hungarian algorithm (scipy.optimize.linear_sum_assignment).

Score = 0.5 * block_match_rate + 0.5 * mean_text_similarity
"""
from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Optional

from openenv.server.rewards import extract_html

logger = logging.getLogger(__name__)

_VIEWPORT_W = 640
_VIEWPORT_H = 480
_IOU_MATCH_THRESHOLD = 0.1  # minimum IoU to count a block as "matched"


def _get_text_blocks(html: str, width: int = _VIEWPORT_W, height: int = _VIEWPORT_H) -> list[dict]:
    """Extract text blocks from HTML by rendering with Playwright.

    Returns a list of dicts with keys: 'text', 'x', 'y', 'width', 'height'.
    Only leaf-level DOM elements with direct text content are included.
    """
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = browser.new_page(viewport={"width": width, "height": height})
            page.set_content(html, wait_until="networkidle")

            blocks = page.evaluate("""() => {
                const results = [];
                const walker = document.createTreeWalker(
                    document.body || document.documentElement,
                    NodeFilter.SHOW_ELEMENT,
                    null
                );
                let node;
                while ((node = walker.nextNode())) {
                    // Collect only direct text content (ignore child elements)
                    const directText = Array.from(node.childNodes)
                        .filter(n => n.nodeType === Node.TEXT_NODE)
                        .map(n => n.textContent.trim())
                        .join(' ')
                        .trim();
                    if (!directText) continue;

                    const rect = node.getBoundingClientRect();
                    if (rect.width <= 0 || rect.height <= 0) continue;
                    if (rect.top < 0 || rect.left < 0) continue;

                    results.push({
                        text: directText,
                        x: rect.left,
                        y: rect.top,
                        width: rect.width,
                        height: rect.height,
                    });
                }
                return results;
            }""")

            browser.close()
            return blocks or []
    except Exception as exc:
        logger.warning("Text block extraction failed: %s", exc)
        return []


def _bbox_iou(a: dict, b: dict) -> float:
    """Compute IoU (Intersection over Union) between two bounding boxes."""
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["width"], by1 + b["height"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    a_area = a["width"] * a["height"]
    b_area = b["width"] * b["height"]
    union_area = a_area + b_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _text_similarity(a: str, b: str) -> float:
    """Character-level similarity between two strings using SequenceMatcher."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def text_block_reward(
    completions: list[list[dict]],
    solution: Optional[list[str]] = None,
) -> list[float]:
    """Score text block matching between rendered HTML and reference.

    Uses the Hungarian algorithm to match text blocks optimally by IoU,
    then scores both the block match rate and text content similarity.

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
                # If reference has no text blocks, penalise non-empty predictions
                results.append(1.0 if not pred_blocks else 0.5)
                continue

            if not pred_blocks:
                results.append(0.0)
                continue

            n_ref = len(ref_blocks)
            n_pred = len(pred_blocks)

            # Build IoU cost matrix (cost = 1 - IoU so lower = better match)
            cost_matrix = np.zeros((n_ref, n_pred), dtype=np.float64)
            for r, ref_block in enumerate(ref_blocks):
                for p, pred_block in enumerate(pred_blocks):
                    cost_matrix[r, p] = 1.0 - _bbox_iou(ref_block, pred_block)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched = 0
            text_scores = []
            for r, p in zip(row_ind, col_ind):
                iou = 1.0 - cost_matrix[r, p]
                if iou > _IOU_MATCH_THRESHOLD:
                    matched += 1
                    text_scores.append(
                        _text_similarity(ref_blocks[r]["text"], pred_blocks[p]["text"])
                    )

            block_match_score = matched / n_ref
            text_sim_score = sum(text_scores) / n_ref if text_scores else 0.0
            score = 0.5 * block_match_score + 0.5 * text_sim_score
            results.append(score)

        except Exception as exc:
            logger.warning("Text block reward failed: %s", exc)
            results.append(0.0)

    return results
