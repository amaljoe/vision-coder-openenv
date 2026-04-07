"""Structural reward: DOM tag-sequence similarity + CSS class overlap."""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Optional

from vcoder.rewards import extract_html


def _get_tag_sequence(html: str) -> list[str]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    return [t.name for t in soup.find_all() if t.name]


def _get_css_classes(html: str) -> set[str]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    classes: set[str] = set()
    for tag in soup.find_all(class_=True):
        classes.update(tag.get("class", []))
    return classes


def structural_similarity_reward(
    completions: list[list[dict]],
    solution: Optional[list[str]] = None,
) -> list[float]:
    """Score structural similarity between generated and reference HTML.

    Computes:
      - Tag-sequence similarity via difflib SequenceMatcher  (0–0.5)
      - CSS class overlap (Jaccard over reference classes)    (0–0.5)

    Args:
        completions: List of completion message lists.
        solution:    List of reference HTML strings (one per completion).

    Returns:
        List of float scores in [0.0, 1.0].
    """
    results = []
    for i, completion in enumerate(completions):
        content = completion[0]["content"]
        pred_html = extract_html(content)
        ref_html = solution[i] if solution and i < len(solution) else ""

        try:
            pred_tags = _get_tag_sequence(pred_html)
            ref_tags = _get_tag_sequence(ref_html)

            tag_sim = SequenceMatcher(None, pred_tags, ref_tags).ratio()

            pred_classes = _get_css_classes(pred_html)
            ref_classes = _get_css_classes(ref_html)

            if ref_classes:
                class_overlap = len(pred_classes & ref_classes) / len(ref_classes)
            else:
                class_overlap = 1.0 if not pred_classes else 0.5

            score = 0.5 * tag_sim + 0.5 * class_overlap
        except Exception:
            score = 0.0

        results.append(score)
    return results
