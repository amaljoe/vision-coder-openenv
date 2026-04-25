"""Structural reward: DOM tag-sequence similarity + style coverage."""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Optional

from openenv.server.rewards import extract_html


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


def _get_inline_style_props(html: str) -> set[str]:
    """Return the set of CSS property names used in inline style attributes."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    props: set[str] = set()
    for tag in soup.find_all(style=True):
        for part in tag.get("style", "").split(";"):
            part = part.strip()
            if ":" in part:
                prop = part.split(":", 1)[0].strip().lower()
                if prop:
                    props.add(prop)
    return props


def structural_similarity_reward(
    completions: list[list[dict]],
    solution: Optional[list[str]] = None,
) -> list[float]:
    """Score structural similarity between generated and reference HTML.

    Computes:
      - Tag-sequence similarity via difflib SequenceMatcher      (0–0.5)
      - Style coverage:
          * CSS class overlap when reference uses class-based CSS (0–0.5)
          * Inline style property overlap when ref uses inline CSS (0–0.5)

    Inline style property overlap penalises blank/unstyled predictions
    against styled references without hurting the perfect-match case.

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

            ref_classes = _get_css_classes(ref_html)
            if ref_classes:
                pred_classes = _get_css_classes(pred_html)
                style_score = len(pred_classes & ref_classes) / len(ref_classes)
            else:
                # Reference uses inline styles — compare CSS property coverage
                ref_props = _get_inline_style_props(ref_html)
                if ref_props:
                    pred_props = _get_inline_style_props(pred_html)
                    style_score = len(pred_props & ref_props) / len(ref_props)
                else:
                    style_score = 1.0  # ref has no styling at all → neutral

            score = 0.5 * tag_sim + 0.5 * style_score
        except Exception:
            score = 0.0

        results.append(score)
    return results
