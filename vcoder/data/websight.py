"""WebSight dataset loader for VisionCoder.

Loads from bundled JSON files in data/ for instant reset() response.
Falls back to HF streaming if the bundled files are missing.
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Optional

from PIL import Image

_DATASET_CACHE: dict = {}

# Bundled data lives next to the repo root
_DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _load_bundled(difficulty: Optional[str]) -> list[dict] | None:
    """Load pre-downloaded samples from data/<difficulty>.json."""
    key = difficulty if difficulty in ("easy", "medium", "hard") else "easy"
    path = _DATA_DIR / f"{key}.json"
    if not path.exists():
        return None
    with open(path) as f:
        rows = json.load(f)
    samples = []
    for row in rows:
        img = Image.open(io.BytesIO(base64.b64decode(row["image_b64"]))).convert("RGB")
        samples.append({"image": img, "solution": row["solution"]})
    return samples


def load_websight_dataset(
    max_samples: int = 10,
    difficulty: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> list[dict]:
    """Load WebSight screenshot-HTML pairs.

    Tries bundled data first (instant). Falls back to HF streaming if missing.

    Returns:
        List of dicts with "image" (PIL.Image) and "solution" (str HTML).
    """
    cache_key = (difficulty,)
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    samples = _load_bundled(difficulty)
    if samples:
        _DATASET_CACHE[cache_key] = samples
        return samples

    # Fallback: stream from HF (slow on first call)
    token = hf_token or os.environ.get("HF_TOKEN")
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceM4/WebSight", split="train", streaming=True, token=token)

    skip = {"easy": 0, "medium": 5000, "hard": 15000}.get(difficulty or "easy", 0)
    if skip:
        ds = ds.skip(skip)

    samples = []
    for row in ds.take(max_samples):
        samples.append({
            "image": row["image"],
            "solution": row.get("html", row.get("solution", "")),
        })

    _DATASET_CACHE[cache_key] = samples
    return samples
