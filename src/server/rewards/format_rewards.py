"""Format reward: checks markdown fencing and HTML document structure."""
from __future__ import annotations

import re

from openenv.server.rewards import extract_html


def format_reward(completions: list[list[dict]]) -> list[float]:
    """Score whether the completion has correct format.

    Checks:
      - Wrapped in ```html ... ``` markdown fencing  (+0.5)
      - Contains <!DOCTYPE html> or <html> tag        (+0.5)

    Args:
        completions: List of completion message lists.
                     Each inner list contains dicts with a "content" key.

    Returns:
        List of float scores in [0.0, 1.0].
    """
    results = []
    for completion in completions:
        content = completion[0]["content"]

        has_fence = bool(re.search(r"```html.*?```", content, re.DOTALL | re.IGNORECASE))
        html = extract_html(content)
        has_doc = bool(re.search(r"<!doctype\s+html", html, re.IGNORECASE))
        has_html_tag = bool(re.search(r"<html[\s>]", html, re.IGNORECASE))

        score = 0.0
        if has_fence:
            score += 0.5
        if has_doc or has_html_tag:
            score += 0.5

        results.append(score)
    return results
