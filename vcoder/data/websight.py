"""WebSight dataset loader for VisionCoder.

Loads the HuggingFaceM4/WebSight dataset and optionally filters
samples by DOM complexity for difficulty-based task selection.
"""
from __future__ import annotations

import os
from typing import Optional

_DATASET_CACHE: dict = {}


def load_websight_dataset(
    max_samples: int = 2000,
    difficulty: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> list[dict]:
    """Load WebSight screenshot-HTML pairs from Hugging Face.

    Args:
        max_samples: Maximum number of samples to load.
        difficulty:  One of "easy", "medium", "hard", or None (mixed).
        hf_token:    Hugging Face token (falls back to HF_TOKEN env var).

    Returns:
        List of dicts with keys "image" (PIL.Image) and "solution" (str HTML).
    """
    cache_key = (max_samples, difficulty)
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    token = hf_token or os.environ.get("HF_TOKEN")

    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceM4/WebSight",
        split="train",
        token=token,
    )

    # Map to standard field names
    def _mapper(example):
        return {
            "image": example["image"],
            "solution": example.get("html", example.get("solution", "")),
        }

    ds = ds.map(_mapper, remove_columns=[c for c in ds.column_names if c not in ("image", "html", "solution")])

    # Difficulty-based index slicing (deterministic, no expensive per-row filtering)
    total = len(ds)
    if difficulty == "easy":
        # First third — typically shorter pages in WebSight ordering
        start, end = 0, min(max_samples, total // 3)
    elif difficulty == "medium":
        start = total // 3
        end = min(start + max_samples, 2 * total // 3)
    elif difficulty == "hard":
        start = 2 * total // 3
        end = min(start + max_samples, total)
    else:
        start, end = 0, min(max_samples, total)

    indices = list(range(start, end))
    ds = ds.select(indices)

    samples = [{"image": row["image"], "solution": row["solution"]} for row in ds]
    _DATASET_CACHE[cache_key] = samples
    return samples
