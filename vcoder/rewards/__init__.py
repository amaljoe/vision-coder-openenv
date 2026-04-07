"""Reward functions for VisionCoder."""

import re


def extract_html(content: str) -> str:
    """Extract raw HTML from a markdown-fenced completion string."""
    match = re.search(r"```html\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return content.strip()
