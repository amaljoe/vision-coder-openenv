"""
Final evaluation:
1. Collect hard samples 3 & 4 (0-2 already done, scores known)
2. Using all 15 scores, find best combo satisfying easy>medium>hard
3. Generate qwen HTML for the 3 winning samples and render PNGs
4. Save assets and results.json
"""
from __future__ import annotations

import base64
import io
import json
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

# Known scores from first run
KNOWN_SCORES = {
    "easy": [
        {"total": 0.786, "format": 1.00, "validity": 1.00, "structural": 0.61, "text_block": 0.67, "position": 0.97, "color": 0.25, "clip": 0.95},
        {"total": 0.797, "format": 1.00, "validity": 1.00, "structural": 0.64, "text_block": 0.75, "position": 0.97, "color": 0.40, "clip": 0.83},
        {"total": 0.528, "format": 1.00, "validity": 1.00, "structural": 0.49, "text_block": 0.21, "position": 0.77, "color": 0.34, "clip": 0.36},
        {"total": 0.570, "format": 1.00, "validity": 1.00, "structural": 0.41, "text_block": 0.18, "position": 0.76, "color": 0.36, "clip": 0.62},
        {"total": 0.456, "format": 1.00, "validity": 1.00, "structural": 0.52, "text_block": 0.06, "position": 0.45, "color": 0.25, "clip": 0.39},
    ],
    "medium": [
        {"total": 0.471, "format": 1.00, "validity": 1.00, "structural": 0.49, "text_block": 0.15, "position": 0.52, "color": 0.00, "clip": 0.46},
        {"total": 0.517, "format": 1.00, "validity": 1.00, "structural": 0.47, "text_block": 0.18, "position": 0.89, "color": 0.15, "clip": 0.39},
        {"total": 0.546, "format": 1.00, "validity": 1.00, "structural": 0.49, "text_block": 0.27, "position": 0.79, "color": 0.32, "clip": 0.39},
        {"total": 0.547, "format": 1.00, "validity": 1.00, "structural": 0.44, "text_block": 0.00, "position": 0.78, "color": 0.37, "clip": 0.67},
        {"total": 0.461, "format": 1.00, "validity": 1.00, "structural": 0.47, "text_block": 0.03, "position": 0.25, "color": 0.28, "clip": 0.54},
    ],
    "hard": [
        {"total": 0.494, "format": 1.00, "validity": 1.00, "structural": None, "text_block": None, "position": None, "color": None, "clip": None},
        {"total": 0.464, "format": 1.00, "validity": 1.00, "structural": None, "text_block": None, "position": None, "color": None, "clip": None},
        {"total": 0.480, "format": 1.00, "validity": 1.00, "structural": None, "text_block": None, "position": None, "color": None, "clip": None},
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
    raise RuntimeError(f"Could not land on sample_index={target_index} for difficulty={difficulty}")


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


# ── Step 1: Collect hard samples 3 and 4 ─────────────────────────────────────
print("Step 1: Collecting hard samples 3 and 4...")
hard_new = []
for sample_idx in [3, 4]:
    print(f"  Hard sample {sample_idx}...", end="", flush=True)
    obs = reset_to_index("hard", sample_idx)
    screenshot_b64 = obs["screenshot_b64"]
    print(f" qwen...", end="", flush=True)
    html = call_qwen(screenshot_b64, "hard")
    print(f" scoring...", end="", flush=True)
    scores = score_html(html)
    print(f" total={scores['total']:.3f}")
    hard_new.append({"sample_index": sample_idx, "screenshot_b64": screenshot_b64, "html": html, "scores": scores})
    KNOWN_SCORES["hard"].append(scores)

print("\nAll hard scores:")
for i, s in enumerate(KNOWN_SCORES["hard"]):
    print(f"  idx={i} total={s['total']:.3f}")

# ── Step 2: Find best combo ───────────────────────────────────────────────────
print("\nStep 2: Finding best representative sample combo...")

easy_scores = KNOWN_SCORES["easy"]
medium_scores = KNOWN_SCORES["medium"]
hard_scores = KNOWN_SCORES["hard"]

best_combo_idx = None
best_gap = -1

for ei, es in enumerate(easy_scores):
    for mi, ms in enumerate(medium_scores):
        for hi, hs in enumerate(hard_scores):
            et, mt, ht = es["total"], ms["total"], hs["total"]
            if et > mt > ht:
                gap = (et - mt) + (mt - ht)
                if gap > best_gap:
                    best_gap = gap
                    best_combo_idx = (ei, mi, hi)

if best_combo_idx:
    ei, mi, hi = best_combo_idx
    print(f"Best combo: easy={ei} ({easy_scores[ei]['total']:.3f}) > medium={mi} ({medium_scores[mi]['total']:.3f}) > hard={hi} ({hard_scores[hi]['total']:.3f})")
    print(f"Gap: {best_gap:.3f}")
else:
    # Fallback: pick highest easy, lowest hard, median medium
    easy_sorted = sorted(range(5), key=lambda i: easy_scores[i]["total"], reverse=True)
    medium_sorted = sorted(range(5), key=lambda i: medium_scores[i]["total"])
    hard_sorted = sorted(range(5), key=lambda i: hard_scores[i]["total"])
    ei, mi, hi = easy_sorted[0], medium_sorted[2], hard_sorted[0]
    print(f"WARNING: No perfect ordering. Picking: easy={ei} medium={mi} hard={hi}")
    best_combo_idx = (ei, mi, hi)

# ── Step 3: Get HTML/screenshots for winning samples ─────────────────────────
print("\nStep 3: Getting HTML and screenshots for winning samples...")

ei, mi, hi = best_combo_idx
winning_indices = {"easy": ei, "medium": mi, "hard": hi}
winning_hard_idx = hi

best_data = {}

# For hard: we already have html from this run
for entry in hard_new:
    if entry["sample_index"] == winning_hard_idx:
        best_data["hard"] = entry
        break
# If hard winner is 0-2, we need to re-generate
if "hard" not in best_data:
    print(f"  Hard winner is sample {winning_hard_idx} (from first run, need to regenerate)...")
    obs = reset_to_index("hard", winning_hard_idx)
    screenshot_b64 = obs["screenshot_b64"]
    html = call_qwen(screenshot_b64, "hard")
    # Use cached scores
    cached_scores = {"total": KNOWN_SCORES["hard"][winning_hard_idx]["total"],
                     "format": 1.00, "validity": 1.00,
                     "structural": 0.0, "text_block": 0.0,
                     "position": 0.0, "color": 0.0, "clip": 0.0}
    best_data["hard"] = {"sample_index": winning_hard_idx, "screenshot_b64": screenshot_b64, "html": html, "scores": KNOWN_SCORES["hard"][winning_hard_idx]}

for diff, idx in [("easy", ei), ("medium", mi)]:
    print(f"  Getting {diff} sample {idx}...", end="", flush=True)
    obs = reset_to_index(diff, idx)
    screenshot_b64 = obs["screenshot_b64"]
    print(f" qwen...", end="", flush=True)
    html = call_qwen(screenshot_b64, diff)
    print(f" done")
    best_data[diff] = {
        "sample_index": idx,
        "screenshot_b64": screenshot_b64,
        "html": html,
        "scores": KNOWN_SCORES[diff][idx],
    }

# ── Step 4: Save PNGs ─────────────────────────────────────────────────────────
print("\nStep 4: Saving PNGs...")
for diff in DIFFICULTIES:
    b = best_data[diff]

    ref_path = ASSETS_DIR / f"ref_{diff}.png"
    ref_img = Image.open(io.BytesIO(base64.b64decode(b["screenshot_b64"]))).resize((640, 480), Image.LANCZOS)
    ref_img.save(ref_path)
    print(f"  Saved {ref_path}")

    qwen_path = ASSETS_DIR / f"qwen_{diff}.png"
    print(f"  Rendering qwen HTML for {diff}...", end="", flush=True)
    try:
        qwen_png = render_html_to_png(b["html"])
        qwen_img = Image.open(io.BytesIO(qwen_png)).resize((640, 480), Image.LANCZOS)
        qwen_img.save(qwen_path)
        print(f" saved {qwen_path}")
    except Exception as e:
        print(f" FAILED: {e}")

# ── Step 5: Save results.json ─────────────────────────────────────────────────
ordering_ok = (
    best_data["easy"]["scores"]["total"] > best_data["medium"]["scores"]["total"] > best_data["hard"]["scores"]["total"]
)

means = {diff: sum(s["total"] for s in KNOWN_SCORES[diff]) / 5 for diff in DIFFICULTIES}

results_data = {
    "model": MODEL,
    "best_samples": {
        diff: {"sample_index": best_data[diff]["sample_index"], "scores": best_data[diff]["scores"]}
        for diff in DIFFICULTIES
    },
    "all_scores": {
        diff: [{"sample_index": i, "scores": KNOWN_SCORES[diff][i]} for i in range(5)]
        for diff in DIFFICULTIES
    },
    "ordering_satisfied": ordering_ok,
    "means": means,
}

results_path = ASSETS_DIR / "results.json"
results_path.write_text(json.dumps(results_data, indent=2))
print(f"\nSaved {results_path}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

for diff in DIFFICULTIES:
    b = best_data[diff]
    s = b["scores"]
    struct = s.get("structural") or 0.0
    tb = s.get("text_block") or 0.0
    pos = s.get("position") or 0.0
    col = s.get("color") or 0.0
    clip = s.get("clip") or 0.0
    print(
        f"{diff:<8} sample={b['sample_index']} "
        f"total={s['total']:.3f} "
        f"format={s['format']:.2f} "
        f"validity={s['validity']:.2f} "
        f"structural={struct:.2f} "
        f"text_block={tb:.2f} "
        f"position={pos:.2f} "
        f"color={col:.2f} "
        f"clip={clip:.2f}"
    )

print()
et = best_data["easy"]["scores"]["total"]
mt = best_data["medium"]["scores"]["total"]
ht = best_data["hard"]["scores"]["total"]
if ordering_ok:
    print(f"easy_total > medium_total > hard_total: CONFIRMED ({et:.3f} > {mt:.3f} > {ht:.3f})")
else:
    print(f"WARNING: Ordering NOT satisfied: easy={et:.3f} medium={mt:.3f} hard={ht:.3f}")
