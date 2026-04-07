"""Validity reward: checks HTML parseability and structural completeness."""
from __future__ import annotations

from vcoder.rewards import extract_html

# Minimum unique tags to consider "diverse" content
_MIN_DIVERSE_TAGS = 5


def html_validity_reward(completions: list[list[dict]]) -> list[float]:
    """Score HTML validity on a scale of 0.0–1.0.

    Checks:
      - Parses without crashing                        (required)
      - Has <html>, <head>, <body> structural tags      (0–0.5)
      - Has diverse tag usage (≥8 unique tag types)    (0–0.5)

    Args:
        completions: List of completion message lists.

    Returns:
        List of float scores in [0.0, 1.0].
    """
    from bs4 import BeautifulSoup

    results = []
    for completion in completions:
        content = completion[0]["content"]
        html = extract_html(content)

        try:
            soup = BeautifulSoup(html, "html.parser")

            has_html = soup.find("html") is not None
            has_head = soup.find("head") is not None
            has_body = soup.find("body") is not None
            structure_score = (int(has_html) + int(has_head) + int(has_body)) / 3.0

            all_tags = [t.name for t in soup.find_all() if t.name]
            unique_tags = len(set(all_tags))
            diversity_score = min(unique_tags / _MIN_DIVERSE_TAGS, 1.0)

            score = 0.5 * structure_score + 0.5 * diversity_score
        except Exception:
            score = 0.0

        results.append(score)
    return results
