"""Reward functions for VisionCoder."""

import re


def extract_html(content: str) -> str:
    """Extract raw HTML from a completion string, stripping think blocks and markdown fences."""
    # Strip <think>...</think> blocks (Qwen3 reasoning output)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()
    # Extract from closed markdown code fences
    match = re.search(r"```(?:html)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Handle unclosed fence (truncated output) — strip opening ```html
    content = re.sub(r"^```(?:html)?\s*", "", content, flags=re.IGNORECASE).strip()
    return content
