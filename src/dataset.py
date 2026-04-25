"""Dataset loader for VisionCoder.

Loads HTML files from data/<difficulty>/*.html for instant reset().
Falls back to HF streaming if the directory is missing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_DATASET_CACHE: dict = {}
_DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _load_bundled(difficulty: Optional[str]) -> list[dict] | None:
    """Load HTML files from data/<difficulty>/."""
    key = difficulty if difficulty in ("easy", "medium", "hard") else "easy"
    folder = _DATA_DIR / key
    if not folder.exists():
        return None
    files = sorted(folder.glob("*.html"))
    if not files:
        return None
    return [{"solution": f.read_text()} for f in files]


def load_websight_dataset(
    max_samples: int = 10,
    difficulty: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> list[dict]:
    """Load screenshot-HTML pairs. Returns list of dicts with "solution" (HTML str).
    Images are rendered live by the environment on reset().
    """
    cache_key = (difficulty,)
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    samples = _load_bundled(difficulty)
    if samples:
        _DATASET_CACHE[cache_key] = samples
        return samples

    # Fallback: stream from HF
    token = hf_token or os.environ.get("HF_TOKEN")
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceM4/WebSight", split="train", streaming=True, token=token)
    skip = {"easy": 0, "medium": 5000, "hard": 15000}.get(difficulty or "easy", 0)
    if skip:
        ds = ds.skip(skip)

    samples = []
    for row in ds.take(max_samples):
        samples.append({"solution": row.get("html", row.get("solution", ""))})

    _DATASET_CACHE[cache_key] = samples
    return samples
