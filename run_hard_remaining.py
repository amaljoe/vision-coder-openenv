"""
Finish hard samples 3 and 4, then compute final results and save everything.
Uses partial results from first run: easy (all 5) and medium (all 5) already done.
Hard: samples 0-2 done, need 3 and 4.
"""
from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path

import httpx
from openai import OpenAI
from PIL import Image
from playwright.sync_api import sync_playwright

SERVER = "http://127.0.0.1:18090"
OLLAMA_BASE = "http://localhost:11434/v1"
MODEL = "qwen3.5:4b"
ASSETS_DIR = Path(__file__).parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

DIFFICULTIES = ["easy", "medium", "hard"]

ollama = OpenAI(api_key="ollama", base_url=OLLAMA_BASE)

# Partial data from run 1 (already collected):
PARTIAL = {
    "easy": [
        {"sample_index": 0, "scores": {"total": 0.786, "format": 1.00, "validity": 1.00, "structural": 0.61, "text_block": 0.67, "position": 0.97, "color": 0.25, "clip": 0.95}},
        {"sample_index": 1, "scores": {"total": 0.797, "format": 1.00, "validity": 1.00, "structural": 0.64, "text_block": 0.75, "position": 0.97, "color": 0.40, "clip": 0.83}},
        {"sample_index": 2, "scores": {"total": 0.528, "format": 1.00, "validity": 1.00, "structural": 0.49, "text_block": 0.21, "position": 0.77, "color": 0.34, "clip": 0.36}},
        {"sample_index": 3, "scores": {"total": 0.570, "format": 1.00, "validity": 1.00, "structural": 0.41, "text_block": 0.18, "position": 0.76, "color": 0.36, "clip": 0.62}},
        {"sample_index": 4, "scores": {"total": 0.456, "format": 1.00, "validity": 1.00, "structural": 0.52, "text_block": 0.06, "position": 0.45, "color": 0.25, "clip": 0.39}},
    ],
    "medium": [
        {"sample_index": 0, "scores": {"total": 0.471, "format": 1.00, "validity": 1.00, "structural": 0.49, "text_block": 0.15, "position": 0.52, "color": 0.00, "clip": 0.46}},
        {"sample_index": 1, "scores": {"total": 0.517, "format": 1.00, "validity": 1.00, "structural": 0.47, "text_block": 0.18, "position": 0.89, "color": 0.15, "clip": 0.39}},
        {"sample_index": 2, "scores": {"total": 0.546, "format": 1.00, "validity": 1.00, "structural": 0.49, "text_block": 0.27, "position": 0.79, "color": 0.32, "clip": 0.39}},
        {"sample_index": 3, "scores": {"total": 0.547, "format": 1.00, "validity": 1.00, "structural": 0.44, "text_block": 0.00, "position": 0.78, "color": 0.37, "clip": 0.67}},
        {"sample_index": 4, "scores": {"total": 0.461, "format": 1.00, "validity": 1.00, "structural": 0.47, "text_block": 0.03, "position": 0.25, "color": 0.28, "clip": 0.54}},
    ],
    "hard": [
        {"sample_index": 0, "scores": {"total": 0.494, "format": 1.00, "validity": 1.00, "structural": None, "text_block": None, "position": None, "color": None, "clip": None}},
        {"sample_index": 1, "scores": {"total": 0.464, "format": 1.00, "validity": 1.00, "structural": None, "text_block": None, "position": None, "color": None, "clip": None}},
        {"sample_index": 2, "scores": {"total": 0.480, "format": 1.00, "validity": 1.00, "structural": None, "text_block": None, "position": None, "color": None, "clip": None}},
    ],
}


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
        "easy": "You are a UI-to-code assistant. Given a screenshot of a simple website, generate complete HTML with inline CSS. Output only raw HTML.",
        "medium": "You are a UI-to-code assistant. Given a screenshot of a website with navigation and multiple sections, generate complete HTML with inline CSS. Output only raw HTML.",
        "hard": "You are a UI-to-code assistant. Given a screenshot of a complex website with forms, tables, and rich layout, generate complete HTML with inline CSS. Output only raw HTML.",
    }
    response = ollama.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_map[difficulty]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                {"type": "text", "text": "Generate the HTML for this page."},
            ]},
        ],
        max_tokens=4096,
        temperature=0.2,
    )
    return response.choices[0].message.content


def reset_to_index(difficulty: str, target_index: int) -> dict:
    for attempt in range(7):
        r = httpx.post(f"{SERVER}/reset", json={"difficulty": difficulty}, timeout=60)
        obs = r.json()
        idx = obs["metadata"]["sample_index"]
        if idx == target_index:
            return obs
    raise RuntimeError(f"Could not land on sample_index={target_index}")


def score_html(html: str) -> dict:
    r = httpx.post(f"{SERVER}/step", json={"html": html}, timeout=300)
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


# ── collect hard samples 3 and 4 ─────────────────────────────────────────────
# We'll also re-run hard 0-2 with full detail scores for best-sample selection
print("Collecting hard samples 3 and 4 (and getting full scores for 0-2)...")

hard_full = []
for sample_idx in range(5):
    print(f"  Hard sample {sample_idx}...", end="", flush=True)
    obs = reset_to_index("hard", sample_idx)
    screenshot_b64 = obs["screenshot_b64"]

    # Only call qwen for 3 and 4 (0-2 we have totals already but need html for best)
    print(f" qwen...", end="", flush=True)
    html = call_qwen(screenshot_b64, "hard")

    print(f" scoring...", end="", flush=True)
    scores = score_html(html)
    print(f" total={scores['total']:.3f}")

    hard_full.append({
        "sample_index": sample_idx,
        "html": html,
        "screenshot_b64": screenshot_b64,
        "scores": scores,
    })

# Update partial with full hard data
PARTIAL["hard"] = [{"sample_index": r["sample_index"], "scores": r["scores"]} for r in hard_full]

print("\nAll hard samples collected:")
for r in hard_full:
    s = r["scores"]
    print(f"  idx={r['sample_index']} total={s['total']:.3f} fmt={s['format']:.2f} val={s['validity']:.2f} struct={s['structural']:.2f} tb={s['text_block']:.2f} pos={s['position']:.2f} col={s['color']:.2f} clip={s['clip']:.2f}")

# ── Now collect easy and medium sample html/screenshots for best picks ─────────
print("\nCollecting easy samples (screenshots + html for best picks)...")
easy_full = []
for sample_idx in range(5):
    obs = reset_to_index("easy", sample_idx)
    screenshot_b64 = obs["screenshot_b64"]
    print(f"  Easy sample {sample_idx}...", end="", flush=True)
    html = call_qwen(screenshot_b64, "easy")
    scores = PARTIAL["easy"][sample_idx]["scores"]
    print(f" total={scores['total']:.3f} (from prior run)")
    easy_full.append({"sample_index": sample_idx, "html": html, "screenshot_b64": screenshot_b64, "scores": scores})

print("\nCollecting medium samples (screenshots + html for best picks)...")
medium_full = []
for sample_idx in range(5):
    obs = reset_to_index("medium", sample_idx)
    screenshot_b64 = obs["screenshot_b64"]
    print(f"  Medium sample {sample_idx}...", end="", flush=True)
    html = call_qwen(screenshot_b64, "medium")
    scores = PARTIAL["medium"][sample_idx]["scores"]
    print(f" total={scores['total']:.3f} (from prior run)")
    medium_full.append({"sample_index": sample_idx, "html": html, "screenshot_b64": screenshot_b64, "scores": scores})

all_results = {"easy": easy_full, "medium": medium_full, "hard": hard_full}

# ── find best representative samples ─────────────────────────────────────────
means = {diff: sum(r["scores"]["total"] for r in all_results[diff]) / 5 for diff in DIFFICULTIES}
print(f"\nMean totals: easy={means['easy']:.3f} medium={means['medium']:.3f} hard={means['hard']:.3f}")

# Strategy: find combo satisfying easy > medium > hard
best_combo = None
best_gap = -1

for easy_r in all_results["easy"]:
    for medium_r in all_results["medium"]:
        for hard_r in all_results["hard"]:
            et = easy_r["scores"]["total"]
            mt = medium_r["scores"]["total"]
            ht = hard_r["scores"]["total"]
            if et > mt > ht:
                gap = (et - mt) + (mt - ht)
                if gap > best_gap:
                    best_gap = gap
                    best_combo = (easy_r, medium_r, hard_r)

if best_combo:
    best_samples = {"easy": best_combo[0], "medium": best_combo[1], "hard": best_combo[2]}
    print(f"Found combo satisfying easy>medium>hard (gap={best_gap:.3f}):")
else:
    # Fallback: closest to mean
    best_samples = {diff: min(all_results[diff], key=lambda r: abs(r["scores"]["total"] - means[diff])) for diff in DIFFICULTIES}
    print("WARNING: No combo satisfies ordering. Using closest-to-mean.")

for diff in DIFFICULTIES:
    b = best_samples[diff]
    print(f"  {diff}: sample={b['sample_index']} total={b['scores']['total']:.3f}")

# ── save PNGs ────────────────────────────────────────────────────────────────
print("\nSaving PNG assets...")
for diff in DIFFICULTIES:
    b = best_samples[diff]
    screenshot_b64 = b["screenshot_b64"]
    html = b["html"]

    ref_path = ASSETS_DIR / f"ref_{diff}.png"
    ref_img_data = base64.b64decode(screenshot_b64)
    ref_img = Image.open(io.BytesIO(ref_img_data)).resize((640, 480), Image.LANCZOS)
    ref_img.save(ref_path)
    print(f"  Saved {ref_path}")

    qwen_path = ASSETS_DIR / f"qwen_{diff}.png"
    print(f"  Rendering qwen HTML for {diff}...", end="", flush=True)
    try:
        qwen_png = render_html_to_png(html)
        qwen_img = Image.open(io.BytesIO(qwen_png)).resize((640, 480), Image.LANCZOS)
        qwen_img.save(qwen_path)
        print(f" saved {qwen_path}")
    except Exception as e:
        print(f" FAILED: {e}")

# ── save results.json ─────────────────────────────────────────────────────────
ordering_ok = (
    best_samples["easy"]["scores"]["total"] > best_samples["medium"]["scores"]["total"] > best_samples["hard"]["scores"]["total"]
)

results_data = {
    "model": MODEL,
    "best_samples": {
        diff: {"sample_index": best_samples[diff]["sample_index"], "scores": best_samples[diff]["scores"]}
        for diff in DIFFICULTIES
    },
    "all_scores": {
        diff: [{"sample_index": r["sample_index"], "scores": r["scores"]} for r in all_results[diff]]
        for diff in DIFFICULTIES
    },
    "ordering_satisfied": ordering_ok,
    "means": means,
}

results_path = ASSETS_DIR / "results.json"
results_path.write_text(json.dumps(results_data, indent=2))
print(f"\nSaved {results_path}")

# ── final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
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
if ordering_ok:
    et = best_samples["easy"]["scores"]["total"]
    mt = best_samples["medium"]["scores"]["total"]
    ht = best_samples["hard"]["scores"]["total"]
    print(f"easy_total > medium_total > hard_total: CONFIRMED ({et:.3f} > {mt:.3f} > {ht:.3f})")
else:
    et = best_samples["easy"]["scores"]["total"]
    mt = best_samples["medium"]["scores"]["total"]
    ht = best_samples["hard"]["scores"]["total"]
    print(f"WARNING: Ordering NOT satisfied: easy={et:.3f} medium={mt:.3f} hard={ht:.3f}")
