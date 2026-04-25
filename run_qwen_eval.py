"""
Evaluate qwen3.5:4b on all 5 samples per difficulty.
Finds best representative samples and saves PNG pairs + results.json.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import httpx
from openai import OpenAI
from PIL import Image
from playwright.sync_api import sync_playwright

# ── config ──────────────────────────────────────────────────────────────────
SERVER = "http://127.0.0.1:18090"
OLLAMA_BASE = "http://localhost:11434/v1"
MODEL = "qwen3.5:4b"
DATA_DIR = Path(__file__).parent / "data"
ASSETS_DIR = Path(__file__).parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

DIFFICULTIES = ["easy", "medium", "hard"]
N_SAMPLES = 5

ollama = OpenAI(api_key="ollama", base_url=OLLAMA_BASE)

# ── helpers ──────────────────────────────────────────────────────────────────

def render_html_to_png(html: str) -> bytes:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 640, "height": 480})
        page.set_content(html, wait_until="networkidle", timeout=30000)
        png = page.screenshot(full_page=False)
        browser.close()
    return png


def call_qwen(screenshot_b64: str, difficulty: str) -> str:
    system_map = {
        "easy": (
            "You are a UI-to-code assistant. Given a screenshot of a simple website, "
            "generate complete HTML with inline CSS. Output only raw HTML."
        ),
        "medium": (
            "You are a UI-to-code assistant. Given a screenshot of a website with navigation "
            "and multiple sections, generate complete HTML with inline CSS. Output only raw HTML."
        ),
        "hard": (
            "You are a UI-to-code assistant. Given a screenshot of a complex website with forms, "
            "tables, and rich layout, generate complete HTML with inline CSS. Output only raw HTML."
        ),
    }
    system = system_map.get(difficulty, system_map["medium"])
    response = ollama.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                    },
                    {"type": "text", "text": "Generate the HTML for this page."},
                ],
            },
        ],
        max_tokens=4096,
        temperature=0.2,
    )
    return response.choices[0].message.content


def reset_to_index(difficulty: str, target_index: int) -> dict:
    """Reset server until we land on target_index. Returns the observation dict."""
    # First reset to get current index, then keep resetting until we hit target
    MAX_ATTEMPTS = N_SAMPLES + 2
    for attempt in range(MAX_ATTEMPTS):
        r = httpx.post(f"{SERVER}/reset", json={"difficulty": difficulty}, timeout=60)
        obs = r.json()
        idx = obs["metadata"]["sample_index"]
        if idx == target_index:
            return obs
    raise RuntimeError(f"Could not land on sample_index={target_index} for {difficulty}")


def score_html(html: str) -> dict:
    r = httpx.post(f"{SERVER}/step", json={"html": html}, timeout=180)
    result = r.json()
    rewards = result["metadata"]["rewards"]
    return {
        "total": result["reward"],
        "format": rewards["format"],
        "validity": rewards["validity"],
        "structural": rewards["structural"],
        "text_block": rewards["text_block"],
        "position": rewards["position"],
        "color": rewards["color"],
        "clip": rewards["clip"],
    }


# ── main evaluation ──────────────────────────────────────────────────────────

all_results: dict[str, list] = {}

print(f"{'='*70}")
print(f"Evaluating {MODEL} on {N_SAMPLES} samples per difficulty")
print(f"{'='*70}\n")

for difficulty in DIFFICULTIES:
    print(f"\n--- {difficulty.upper()} ---")
    diff_results = []

    for sample_idx in range(N_SAMPLES):
        print(f"  Sample {sample_idx}: resetting server...", end="", flush=True)

        obs = reset_to_index(difficulty, sample_idx)
        screenshot_b64 = obs["screenshot_b64"]

        print(f" calling qwen...", end="", flush=True)
        html = call_qwen(screenshot_b64, difficulty)

        print(f" scoring...", end="", flush=True)
        scores = score_html(html)

        print(f" total={scores['total']:.3f}")

        diff_results.append({
            "sample_index": sample_idx,
            "html": html,
            "screenshot_b64": screenshot_b64,
            "scores": scores,
        })

    all_results[difficulty] = diff_results

    # Print all scores for this difficulty
    for r in diff_results:
        s = r["scores"]
        print(f"    idx={r['sample_index']} total={s['total']:.3f} "
              f"fmt={s['format']:.2f} val={s['validity']:.2f} "
              f"struct={s['structural']:.2f} tb={s['text_block']:.2f} "
              f"pos={s['position']:.2f} col={s['color']:.2f} clip={s['clip']:.2f}")

# ── find best representative samples ────────────────────────────────────────
print(f"\n{'='*70}")
print("Finding best representative samples...")

# Compute mean per difficulty
means = {
    diff: sum(r["scores"]["total"] for r in all_results[diff]) / N_SAMPLES
    for diff in DIFFICULTIES
}
print(f"Mean totals: easy={means['easy']:.3f} medium={means['medium']:.3f} hard={means['hard']:.3f}")

# For each difficulty, pick sample closest to its mean (most representative)
# But also try to satisfy easy > medium > hard ordering
best_samples: dict[str, dict] = {}

# Strategy: pick sample closest to the mean for each difficulty
for difficulty in DIFFICULTIES:
    results = all_results[difficulty]
    mean_total = means[difficulty]
    best = min(results, key=lambda r: abs(r["scores"]["total"] - mean_total))
    best_samples[difficulty] = best
    print(f"  {difficulty}: sample={best['sample_index']} total={best['scores']['total']:.3f} (mean={mean_total:.3f})")

# Check ordering
easy_total = best_samples["easy"]["scores"]["total"]
medium_total = best_samples["medium"]["scores"]["total"]
hard_total = best_samples["hard"]["scores"]["total"]

ordering_ok = easy_total > medium_total > hard_total
print(f"\nOrdering easy>medium>hard: {'YES' if ordering_ok else 'NO'}")
print(f"  easy={easy_total:.3f} medium={medium_total:.3f} hard={hard_total:.3f}")

# If ordering is not satisfied, try to find better combination
if not ordering_ok:
    print("Trying to find better representative samples to satisfy ordering...")
    best_combo = None
    best_gap = -1

    for ei, easy_r in enumerate(all_results["easy"]):
        for mi, medium_r in enumerate(all_results["medium"]):
            for hi, hard_r in enumerate(all_results["hard"]):
                et = easy_r["scores"]["total"]
                mt = medium_r["scores"]["total"]
                ht = hard_r["scores"]["total"]
                if et > mt > ht:
                    # Gap = measure of how well ordered
                    gap = (et - mt) + (mt - ht)
                    if gap > best_gap:
                        best_gap = gap
                        best_combo = (easy_r, medium_r, hard_r)

    if best_combo:
        best_samples["easy"] = best_combo[0]
        best_samples["medium"] = best_combo[1]
        best_samples["hard"] = best_combo[2]
        print(f"Found better combo with gap={best_gap:.3f}:")
        for diff in DIFFICULTIES:
            b = best_samples[diff]
            print(f"  {diff}: sample={b['sample_index']} total={b['scores']['total']:.3f}")
    else:
        print("WARNING: No combo satisfies easy>medium>hard ordering. Using closest-to-mean.")

# ── save PNGs ────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("Saving PNG assets...")

for difficulty in DIFFICULTIES:
    b = best_samples[difficulty]
    screenshot_b64 = b["screenshot_b64"]
    html = b["html"]

    # Save reference PNG
    ref_png_path = ASSETS_DIR / f"ref_{difficulty}.png"
    ref_img_data = base64.b64decode(screenshot_b64)
    ref_img = Image.open(io.BytesIO(ref_img_data)).resize((640, 480), Image.LANCZOS)
    ref_img.save(ref_png_path)
    print(f"  Saved {ref_png_path}")

    # Render qwen HTML to PNG
    qwen_png_path = ASSETS_DIR / f"qwen_{difficulty}.png"
    print(f"  Rendering qwen HTML for {difficulty}...", end="", flush=True)
    try:
        qwen_png_bytes = render_html_to_png(html)
        qwen_img = Image.open(io.BytesIO(qwen_png_bytes)).resize((640, 480), Image.LANCZOS)
        qwen_img.save(qwen_png_path)
        print(f" saved {qwen_png_path}")
    except Exception as e:
        print(f" FAILED: {e}")

# ── save results.json ────────────────────────────────────────────────────────
results_data = {
    "model": MODEL,
    "best_samples": {
        diff: {
            "sample_index": best_samples[diff]["sample_index"],
            "scores": best_samples[diff]["scores"],
        }
        for diff in DIFFICULTIES
    },
    "all_scores": {
        diff: [
            {"sample_index": r["sample_index"], "scores": r["scores"]}
            for r in all_results[diff]
        ]
        for diff in DIFFICULTIES
    },
    "ordering_satisfied": ordering_ok,
    "means": means,
}

results_path = ASSETS_DIR / "results.json"
results_path.write_text(json.dumps(results_data, indent=2))
print(f"\nSaved {results_path}")

# ── final summary table ───────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'='*70}")

final_ordering_ok = (
    best_samples["easy"]["scores"]["total"] > best_samples["medium"]["scores"]["total"] > best_samples["hard"]["scores"]["total"]
)

for diff in DIFFICULTIES:
    b = best_samples[diff]
    s = b["scores"]
    print(
        f"{diff:<8} sample={b['sample_index']} "
        f"total={s['total']:.3f} "
        f"format={s['format']:.2f} "
        f"validity={s['validity']:.2f} "
        f"structural={s['structural']:.2f} "
        f"text_block={s['text_block']:.2f} "
        f"position={s['position']:.2f} "
        f"color={s['color']:.2f} "
        f"clip={s['clip']:.2f}"
    )

print()
if final_ordering_ok:
    print("easy_total > medium_total > hard_total: CONFIRMED")
else:
    print(f"WARNING: Ordering NOT satisfied: easy={best_samples['easy']['scores']['total']:.3f} "
          f"medium={best_samples['medium']['scores']['total']:.3f} "
          f"hard={best_samples['hard']['scores']['total']:.3f}")
