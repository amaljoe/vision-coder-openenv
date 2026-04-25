#!/usr/bin/env python3
"""Reward function stability test — two-phase design.

Phase 1  --generate
  Runs Playwright once for every (sample × variant) combo and saves:
    outputs/test/{diff}_{idx}/reference.png
    outputs/test/{diff}_{idx}/reference_blocks.json
    outputs/test/{diff}_{idx}/{variant}.png
    outputs/test/{diff}_{idx}/{variant}.html
    outputs/test/{diff}_{idx}/{variant}_blocks.json

Phase 2  (default, --score)
  Loads saved PNGs + block JSONs and scores — no Playwright needed.
  Iterate fast on reward weights / logic without re-rendering.

Variants per sample:
  perfect     — exact reference HTML
  minor_diff  — one colour + two font-sizes changed
  no_style    — inline styles and classes stripped (browser defaults)
  blank       — minimal empty page (no content)

Expected ordering: perfect > minor_diff > no_style > blank

Usage:
  # Generate all 15 × 4 = 60 combos  (needs Playwright / apptainer)
  python scripts/test_rewards.py --generate [--samples easy/0,easy/1,...]

  # Score using saved renders  (fast, no Playwright)
  python scripts/test_rewards.py [--samples easy/0,easy/1,...]

  # All 15 samples
  python scripts/test_rewards.py --all
  python scripts/test_rewards.py --all --generate
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(Path.home() / "playwright-browsers"))

import numpy as np
from PIL import Image

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR  = Path(__file__).parent.parent / "outputs" / "test"

VARIANTS = ["perfect", "minor_diff", "no_style", "blank"]
METRIC_COLS = ["format", "validity", "structural", "text_block", "position", "color", "clip", "total"]

WEIGHTS    = {"format": 1.0, "validity": 1.0, "structural": 0.5,
              "text_block": 3.0, "position": 1.0, "color": 1.0, "clip": 2.0}
WEIGHT_SUM = sum(WEIGHTS.values())  # 9.5

BLANK_HTML = (
    "<!DOCTYPE html><html><head><title>Page</title></head>"
    "<body style=\"background:#fff;\"></body></html>"
)

# ─────────────────────────────────────────────────────────────────────────────
# HTML variant generation
# ─────────────────────────────────────────────────────────────────────────────

def make_variants(ref_html: str) -> dict[str, str]:
    perfect = ref_html

    minor = ref_html
    minor = re.sub(r"(background:\s*)(#[0-9a-fA-F]{6})", r"\g<1>#888888", minor, count=1)
    minor = re.sub(
        r"font-size:(\d+)px",
        lambda m: f"font-size:{max(8, int(m.group(1)) - 4)}px",
        minor, count=2,
    )

    no_style = re.sub(r'\s+style="[^"]*"', "", ref_html)
    no_style = re.sub(r'\s+class="[^"]*"', "", no_style)

    return {"perfect": perfect, "minor_diff": minor, "no_style": no_style, "blank": BLANK_HTML}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — generate (Playwright)
# ─────────────────────────────────────────────────────────────────────────────

def _render_html_pw(html: str, width: int = 640, height: int = 480):
    """Render HTML → PIL Image using Playwright. Returns None on failure."""
    import io
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = browser.new_page(viewport={"width": width, "height": height})
            page.set_content(html, wait_until="networkidle")
            png_bytes = page.screenshot(full_page=True)
            browser.close()
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")
    except Exception as exc:
        print(f"    render failed: {exc}", file=sys.stderr)
        return None


def _extract_blocks_pw(html: str, width: int = 640, height: int = 480) -> list[dict]:
    """Extract leaf text blocks via Playwright DOM walk. Returns [] on failure."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = browser.new_page(viewport={"width": width, "height": height})
            page.set_content(html, wait_until="networkidle")
            blocks = page.evaluate("""() => {
                const results = [];
                const walker = document.createTreeWalker(
                    document.body || document.documentElement,
                    NodeFilter.SHOW_ELEMENT, null
                );
                let node;
                while ((node = walker.nextNode())) {
                    const directText = Array.from(node.childNodes)
                        .filter(n => n.nodeType === Node.TEXT_NODE)
                        .map(n => n.textContent.trim())
                        .join(' ').trim();
                    if (!directText) continue;
                    const rect = node.getBoundingClientRect();
                    if (rect.width <= 0 || rect.height <= 0) continue;
                    if (rect.top < 0 || rect.left < 0) continue;
                    results.push({
                        text: directText,
                        x: rect.left, y: rect.top,
                        width: rect.width, height: rect.height,
                    });
                }
                return results;
            }""")
            browser.close()
            return blocks or []
    except Exception as exc:
        print(f"    block extraction failed: {exc}", file=sys.stderr)
        return []


def generate_sample(difficulty: str, idx: str):
    ref_path = DATA_DIR / difficulty / f"{idx}.html"
    ref_html = ref_path.read_text()
    sample_dir = OUT_DIR / f"{difficulty}_{idx}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Rendering reference …", end=" ", flush=True)
    ref_img = _render_html_pw(ref_html)
    if ref_img is None:
        print("FAILED"); return False
    ref_img.save(sample_dir / "reference.png")

    print("extracting blocks …", end=" ", flush=True)
    ref_blocks = _extract_blocks_pw(ref_html)
    (sample_dir / "reference_blocks.json").write_text(json.dumps(ref_blocks))
    print(f"({len(ref_blocks)} blocks) done")

    variants = make_variants(ref_html)
    for name, html in variants.items():
        print(f"  [{name}] render …", end=" ", flush=True)
        img = _render_html_pw(html)
        if img is None:
            img = Image.new("RGB", (640, 480), (255, 255, 255))
        img.save(sample_dir / f"{name}.png")
        (sample_dir / f"{name}.html").write_text(html)

        print("blocks …", end=" ", flush=True)
        blocks = _extract_blocks_pw(html)
        (sample_dir / f"{name}_blocks.json").write_text(json.dumps(blocks))
        print(f"({len(blocks)} blocks) done")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — scoring (no Playwright)
# ─────────────────────────────────────────────────────────────────────────────

# --- format reward (no Playwright) ---
def _fmt_score(html: str) -> float:
    from vcoder.rewards.format_rewards import format_reward
    return format_reward([[{"content": html}]])[0]


# --- validity reward (no Playwright) ---
def _val_score(html: str) -> float:
    from vcoder.rewards.validity_rewards import html_validity_reward
    return html_validity_reward([[{"content": html}]])[0]


# --- structural reward (no Playwright) ---
def _struct_score(html: str, ref_html: str) -> float:
    from vcoder.rewards.structural_rewards import structural_similarity_reward
    return structural_similarity_reward([[{"content": html}]], solution=[ref_html])[0]


# --- color reward using pre-rendered PIL images (spatial, no Playwright) ---
_COLOR_SIZE = (128, 128)
_COLOR_WHITE_THRESH = 240
_COLOR_MIN_NONWHITE_FRAC = 0.02

def _color_score(pred_img: Image.Image, ref_img: Image.Image) -> float:
    """Spatial CIEDE2000 on reference non-white pixels (ignores white-on-white)."""
    try:
        from skimage.color import deltaE_ciede2000, rgb2lab
        W, H = _COLOR_SIZE
        ref_arr  = np.array(ref_img.convert("RGB").resize(_COLOR_SIZE), dtype=np.float32) / 255.0
        pred_arr = np.array(pred_img.convert("RGB").resize(_COLOR_SIZE), dtype=np.float32) / 255.0

        ref_lab  = rgb2lab(ref_arr.reshape(1, H, W, 3)).reshape(H * W, 3)
        pred_lab = rgb2lab(pred_arr.reshape(1, H, W, 3)).reshape(H * W, 3)
        delta_e  = deltaE_ciede2000(ref_lab, pred_lab)

        ref_u8 = (ref_arr * 255).astype(np.uint8).reshape(H * W, 3)
        mask = (ref_u8[:, 0] < _COLOR_WHITE_THRESH) | \
               (ref_u8[:, 1] < _COLOR_WHITE_THRESH) | \
               (ref_u8[:, 2] < _COLOR_WHITE_THRESH)

        if mask.sum() >= _COLOR_MIN_NONWHITE_FRAC * H * W:
            mean_de = float(np.mean(delta_e[mask]))
        else:
            mean_de = float(np.mean(delta_e))

        score = 1.0 - min(mean_de / 50.0, 1.0)
        return float(np.clip(score, 0.0, 1.0))
    except Exception as exc:
        print(f"    color failed: {exc}", file=sys.stderr)
        return 0.5


# --- CLIP reward using pre-rendered PIL images (no Playwright) ---
def _clip_score(pred_img: Image.Image, ref_img: Image.Image) -> float:
    from vcoder.rewards.visual_rewards import clip_visual_reward
    return clip_visual_reward(
        [[{"content": ""}]],
        image=[ref_img],
        pred_image=[pred_img],
    )[0]


# --- text block reward using saved blocks (no Playwright) ---
def _text_block_score(ref_blocks: list[dict], pred_blocks: list[dict]) -> float:
    from difflib import SequenceMatcher
    from scipy.optimize import linear_sum_assignment

    if not ref_blocks:
        return 1.0 if not pred_blocks else 0.5
    if not pred_blocks:
        return 0.0

    n_ref, n_pred = len(ref_blocks), len(pred_blocks)

    def iou(a, b):
        ax1,ay1,ax2,ay2 = a["x"],a["y"],a["x"]+a["width"],a["y"]+a["height"]
        bx1,by1,bx2,by2 = b["x"],b["y"],b["x"]+b["width"],b["y"]+b["height"]
        ix1,iy1 = max(ax1,bx1),max(ay1,by1)
        ix2,iy2 = min(ax2,bx2),min(ay2,by2)
        if ix2<=ix1 or iy2<=iy1: return 0.0
        inter = (ix2-ix1)*(iy2-iy1)
        union = a["width"]*a["height"] + b["width"]*b["height"] - inter
        return inter/union if union>0 else 0.0

    cost = np.zeros((n_ref, n_pred))
    for r,rb in enumerate(ref_blocks):
        for p,pb in enumerate(pred_blocks):
            cost[r,p] = 1.0 - iou(rb, pb)

    row_ind, col_ind = linear_sum_assignment(cost)
    matched, text_scores = 0, []
    for r,p in zip(row_ind, col_ind):
        if 1.0 - cost[r,p] > 0.1:
            matched += 1
            a,b = ref_blocks[r]["text"], pred_blocks[p]["text"]
            sim = SequenceMatcher(None, a, b).ratio() if a and b else (1.0 if not a and not b else 0.0)
            text_scores.append(sim)

    block_match = matched / n_ref
    text_sim    = sum(text_scores) / n_ref if text_scores else 0.0
    return 0.5 * block_match + 0.5 * text_sim


# --- position reward using saved blocks (no Playwright) ---
def _position_score(ref_blocks: list[dict], pred_blocks: list[dict]) -> float:
    import math
    from scipy.optimize import linear_sum_assignment

    if not ref_blocks:
        return 1.0 if not pred_blocks else 0.5
    if not pred_blocks:
        return 0.0

    DIAG = math.sqrt(640**2 + 480**2)
    n_ref, n_pred = len(ref_blocks), len(pred_blocks)
    cost = np.zeros((n_ref, n_pred))
    for r,rb in enumerate(ref_blocks):
        rcx,rcy = rb["x"]+rb["width"]/2, rb["y"]+rb["height"]/2
        for p,pb in enumerate(pred_blocks):
            pcx,pcy = pb["x"]+pb["width"]/2, pb["y"]+pb["height"]/2
            cost[r,p] = math.sqrt((rcx-pcx)**2+(rcy-pcy)**2) / DIAG

    row_ind, col_ind = linear_sum_assignment(cost)
    pos_scores = [1.0 - cost[r,p] for r,p in zip(row_ind, col_ind)]
    if len(pos_scores) < n_ref:
        pos_scores += [0.0] * (n_ref - len(pos_scores))
    return max(0.0, sum(pos_scores) / n_ref)


def score_variant_from_disk(
    html: str,
    ref_html: str,
    pred_img: Image.Image,
    ref_img: Image.Image,
    pred_blocks: list[dict],
    ref_blocks: list[dict],
) -> dict[str, float]:
    fmt    = _fmt_score(html)
    val    = _val_score(html)
    struct = _struct_score(html, ref_html)
    tb     = _text_block_score(ref_blocks, pred_blocks)
    pos    = _position_score(ref_blocks, pred_blocks)
    col    = _color_score(pred_img, ref_img)
    clip   = _clip_score(pred_img, ref_img)

    scores = {"format": fmt, "validity": val, "structural": struct,
              "text_block": tb, "position": pos, "color": col, "clip": clip}
    scores["total"] = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS) / WEIGHT_SUM
    return scores


def score_sample(difficulty: str, idx: str) -> dict | None:
    sample_dir = OUT_DIR / f"{difficulty}_{idx}"
    ref_html_path = DATA_DIR / difficulty / f"{idx}.html"
    if not (sample_dir / "reference.png").exists():
        return None  # not generated yet

    ref_img    = Image.open(sample_dir / "reference.png").convert("RGB")
    ref_html   = ref_html_path.read_text()
    ref_blocks = json.loads((sample_dir / "reference_blocks.json").read_text())

    results = {}
    for name in VARIANTS:
        png_path  = sample_dir / f"{name}.png"
        html_path = sample_dir / f"{name}.html"
        blk_path  = sample_dir / f"{name}_blocks.json"
        if not png_path.exists():
            continue

        pred_img    = Image.open(png_path).convert("RGB")
        pred_html   = html_path.read_text() if html_path.exists() else ""
        pred_blocks = json.loads(blk_path.read_text()) if blk_path.exists() else []

        t0 = time.time()
        scores = score_variant_from_disk(pred_html, ref_html, pred_img, ref_img, pred_blocks, ref_blocks)
        scores["_elapsed"] = round(time.time() - t0, 2)
        results[name] = scores

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_table(sample_key: str, data: dict):
    print(f"\n{'─'*100}")
    print(f"  {sample_key}")
    print(f"{'─'*100}")
    header = f"  {'variant':<12}" + "".join(f" {c:>10}" for c in METRIC_COLS)
    print(header)
    print(f"  {'─'*12}" + ("  ─────────" * len(METRIC_COLS)))

    for v in VARIANTS:
        if v not in data: continue
        s = data[v]
        row = f"  {v:<12}" + "".join(f" {s.get(c,0):>10.3f}" for c in METRIC_COLS)
        print(row)

    tots = [data[v]["total"] for v in VARIANTS if v in data]
    ordered = all(tots[i] >= tots[i+1] - 0.01 for i in range(len(tots)-1))
    order_strs = [f'{v}={data[v]["total"]:.3f}' for v in VARIANTS if v in data]
    print(f"\n  Ordering: {' > '.join(order_strs)}")
    print(f"  {'✓ PASS' if ordered else '✗ FAIL — ordering violated'}")

    issues = []
    if "blank" in data:
        if data["blank"]["total"] > 0.30:
            issues.append(f"blank total high ({data['blank']['total']:.3f} > 0.30)")
        if data["blank"]["color"] > 0.60:
            issues.append(f"blank color high ({data['blank']['color']:.3f}) — likely blank-vs-white-background false positive")
        if data["blank"]["clip"] > 0.60:
            issues.append(f"blank clip high ({data['blank']['clip']:.3f})")
    if "perfect" in data and data["perfect"]["total"] < 0.80:
        issues.append(f"perfect total low ({data['perfect']['total']:.3f} < 0.80) — check CLIP/render")
    if "no_style" in data and "blank" in data:
        if data["no_style"]["total"] <= data["blank"]["total"]:
            issues.append("no_style ≤ blank — layout stripping not penalised enough")
    for issue in issues:
        print(f"  ⚠  {issue}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_samples(args) -> list[tuple[str, str]]:
    if args.all:
        return [(d, str(i)) for d in ("easy", "medium", "hard") for i in range(5)]
    if args.samples:
        return [tuple(s.split("/", 1)) for s in args.samples.split(",")]
    return [("easy", "0"), ("easy", "1"), ("medium", "0")]


def main():
    parser = argparse.ArgumentParser(description="Reward stability test")
    parser.add_argument("--generate", action="store_true",
                        help="Run Playwright to generate renders + blocks (slow, do once)")
    parser.add_argument("--all", action="store_true", help="All 15 samples")
    parser.add_argument("--samples", metavar="DIFF/IDX,...",
                        help="Comma-separated samples, e.g. easy/0,medium/2")
    args = parser.parse_args()

    samples = parse_samples(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.generate:
        print(f"[GENERATE] {len(samples)} samples × 4 variants = {len(samples)*4} Playwright runs")
        for difficulty, idx in samples:
            key = f"{difficulty}/{idx}"
            print(f"\n── {key} ──")
            ok = generate_sample(difficulty, idx)
            if not ok:
                print(f"  ERROR generating {key}")
        print("\n[GENERATE] Done.")
        return

    # ── Score phase ───────────────────────────────────────────────────────
    print(f"[SCORE] {len(samples)} samples × 4 variants (loading from {OUT_DIR})")
    all_results = {}
    pass_count = fail_count = missing_count = 0

    for difficulty, idx in samples:
        key = f"{difficulty}/{idx}"
        data = score_sample(difficulty, idx)
        if data is None:
            print(f"\n  {key}: no saved renders — run with --generate first")
            missing_count += 1
            continue

        all_results[key] = data
        print_table(key, data)

        tots = [data[v]["total"] for v in VARIANTS if v in data]
        if len(tots) >= 2 and all(tots[i] >= tots[i+1] - 0.01 for i in range(len(tots)-1)):
            pass_count += 1
        else:
            fail_count += 1

    total = pass_count + fail_count
    print(f"\n{'═'*100}")
    print(f"  SUMMARY: {pass_count}/{total} samples in correct ordering", end="")
    if missing_count:
        print(f"  ({missing_count} missing — run --generate)", end="")
    print(f"\n{'═'*100}")

    # Per-metric averages
    if all_results:
        print(f"\n  Per-metric averages across {len(all_results)} samples:")
        for v in VARIANTS:
            avgs = {}
            for m in METRIC_COLS:
                vals = [all_results[k][v][m] for k in all_results if v in all_results[k]]
                avgs[m] = sum(vals)/len(vals) if vals else 0.0
            row = f"    {v:<12}" + "".join(f" {avgs.get(c,0):>10.3f}" for c in METRIC_COLS)
            print(row)

    report_path = OUT_DIR / "reward_test_report.json"
    report_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Report → {report_path}")


if __name__ == "__main__":
    main()
