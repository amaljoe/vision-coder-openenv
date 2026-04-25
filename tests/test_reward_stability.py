#!/usr/bin/env python3
"""Reward stability test suite for VisionCoder OpenEnv.

Test cases in data/tests/<num>/ (0-14, mapping easy/0-4, medium/0-4, hard/0-4).
Each case has:
  reference.html           — committed (from data/{difficulty}/{idx}.html)
  variants/*.html          — committed (generated once, deterministic from reference)
  expected_scores.json     — committed (ideal target scores per variant)
  renders/                 — gitignored, auto-generated on first run

Usage:
    # First run: scaffold + render + score
    python tests/test_reward_stability.py --render

    # Force re-render even if renders exist
    python tests/test_reward_stability.py --render --force

    # Score only (renders must exist)
    python tests/test_reward_stability.py

    # Score specific cases
    python tests/test_reward_stability.py --cases 0,1,5

    # After tuning reward functions, adopt current scores as new expected
    python tests/test_reward_stability.py --update-expected

    # Run as pytest (renders must exist; skips cases without renders)
    python -m pytest tests/test_reward_stability.py -v
    python -m pytest tests/test_reward_stability.py -v -k "easy"
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(Path.home() / "playwright-browsers"))

# ── Reward functions (same as environment.py — keep in sync) ─────────────────
from vcoder.rewards.format_rewards import format_reward
from vcoder.rewards.validity_rewards import html_validity_reward
from vcoder.rewards.structural_rewards import structural_similarity_reward
from vcoder.rewards.color_rewards import color_reward
from vcoder.rewards.visual_rewards import clip_visual_reward

WEIGHTS: dict[str, float] = {
    "format":     1.0,
    "validity":   1.0,
    "structural": 0.5,
    "text_block": 3.0,
    "position":   1.0,
    "color":      1.0,
    "clip":       2.0,
}
WEIGHT_SUM = sum(WEIGHTS.values())  # 9.5

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_SRC  = _ROOT / "data"
TESTS_DIR = _ROOT / "data" / "tests"

# ── Case index: 0-4=easy, 5-9=medium, 10-14=hard ─────────────────────────────
CASE_SOURCES: dict[int, tuple[str, int]] = {
    **{i:      ("easy",   i)     for i in range(5)},
    **{i + 5:  ("medium", i)     for i in range(5)},
    **{i + 10: ("hard",   i)     for i in range(5)},
}

# ── Variants (ordered from best to worst) ─────────────────────────────────────
VARIANTS = ["perfect", "minor_diff", "bad_colors", "half_styled", "no_layout", "no_style", "blank"]

BLANK_HTML = (
    "<!DOCTYPE html><html><head><title>Page</title></head>"
    "<body style=\"background:#fff;\"></body></html>"
)

# ── Canonical target scores (what a well-calibrated reward should output) ─────
# These are the *ideal* values. Correlation between actual and these indicates
# how well the reward function distinguishes quality levels.
CANONICAL_EXPECTED: dict[str, float] = {
    "perfect":    0.92,
    "minor_diff": 0.85,
    "bad_colors": 0.65,
    "half_styled": 0.62,
    "no_layout":  0.52,
    "no_style":   0.35,
    "blank":      0.00,
}

# ── Pass/fail thresholds ──────────────────────────────────────────────────────
MIN_SPEARMAN_PER_CASE = 0.80  # rank correlation must exceed this per case
MIN_SPEARMAN_GLOBAL   = 0.85  # rank correlation across all (case, variant) pairs
MAX_BLANK_SCORE       = 0.05  # blank variants must score below this
MIN_PERFECT_SCORE     = 0.80  # perfect variants must score above this


# ─────────────────────────────────────────────────────────────────────────────
# HTML variant generation
# ─────────────────────────────────────────────────────────────────────────────

_LAYOUT_PROPS = {
    "padding", "margin", "border-radius", "box-shadow", "display",
    "align-items", "justify-content", "min-height", "min-width",
    "position", "top", "right", "bottom", "left", "transform",
    "flex", "grid", "float", "overflow", "vertical-align",
    "width", "height", "box-sizing",
}


def make_variants(ref_html: str) -> dict[str, str]:
    """Generate 7 quality-level HTML variants from a reference HTML string."""
    perfect = ref_html

    minor = ref_html
    minor = re.sub(r"(background(?:-color)?:\s*)(#[0-9a-fA-F]{6})", r"\g<1>#888888", minor, count=1)
    minor = re.sub(r"font-size:(\d+)px",
                   lambda m: f"font-size:{max(8, int(m.group(1)) - 4)}px",
                   minor, count=2)

    def _invert(m: re.Match) -> str:
        r = 255 - int(m.group(1), 16)
        g = 255 - int(m.group(2), 16)
        b = 255 - int(m.group(3), 16)
        return f"#{r:02x}{g:02x}{b:02x}"
    bad_colors = re.sub(r'#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})', _invert, ref_html)

    def _strip_layout(m: re.Match) -> str:
        kept = [p.strip() for p in m.group(1).split(";")
                if p.strip() and not any(p.strip().lower().startswith(lp) for lp in _LAYOUT_PROPS)]
        return f'style="{"; ".join(kept)}"'
    no_layout = re.sub(r'style="([^"]*)"', _strip_layout, ref_html)

    def _keep_half(m: re.Match) -> str:
        props = [p.strip() for p in m.group(1).split(";") if p.strip()]
        return f'style="{"; ".join(props[::2])}"'
    half_styled = re.sub(r'style="([^"]*)"', _keep_half, ref_html)

    no_style = re.sub(r'\s+style="[^"]*"', "", ref_html)
    no_style = re.sub(r'\s+class="[^"]*"', "", no_style)

    return {
        "perfect":    perfect,
        "minor_diff": minor,
        "bad_colors": bad_colors,
        "half_styled": half_styled,
        "no_layout":  no_layout,
        "no_style":   no_style,
        "blank":      BLANK_HTML,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scaffolding — create data/tests/<num>/ structure (no Playwright needed)
# ─────────────────────────────────────────────────────────────────────────────

def scaffold_test_case(num: int, overwrite: bool = False) -> Path:
    """Create data/tests/<num>/ with reference.html, variants/, expected_scores.json."""
    difficulty, idx = CASE_SOURCES[num]
    src_html_path = DATA_SRC / difficulty / f"{idx}.html"
    case_dir = TESTS_DIR / str(num)

    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "renders").mkdir(exist_ok=True)

    ref_html = src_html_path.read_text()

    ref_dest = case_dir / "reference.html"
    if overwrite or not ref_dest.exists():
        ref_dest.write_text(ref_html)

    meta = {"source": f"{difficulty}/{idx}", "difficulty": difficulty, "idx": idx}
    meta_dest = case_dir / "meta.json"
    if overwrite or not meta_dest.exists():
        meta_dest.write_text(json.dumps(meta, indent=2))

    expected_dest = case_dir / "expected_scores.json"
    if overwrite or not expected_dest.exists():
        expected_dest.write_text(json.dumps(CANONICAL_EXPECTED, indent=2))

    variants_dir = case_dir / "variants"
    variants_dir.mkdir(exist_ok=True)
    variants = make_variants(ref_html)
    for name, html in variants.items():
        dest = variants_dir / f"{name}.html"
        if overwrite or not dest.exists():
            dest.write_text(html)

    return case_dir


def scaffold_all(overwrite: bool = False):
    """Scaffold all 15 test cases."""
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    for num in range(15):
        scaffold_test_case(num, overwrite=overwrite)


# ─────────────────────────────────────────────────────────────────────────────
# Rendering — Playwright (slow, do once)
# ─────────────────────────────────────────────────────────────────────────────

def _render_pw(html: str, width: int = 640, height: int = 480) -> Image.Image | None:
    import io
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = browser.new_page(viewport={"width": width, "height": height})
            page.set_content(html, wait_until="networkidle")
            png = page.screenshot(full_page=True)
            browser.close()
        return Image.open(io.BytesIO(png)).convert("RGB")
    except Exception as exc:
        print(f"    render failed: {exc}", file=sys.stderr)
        return None


def _extract_blocks_pw(html: str, width: int = 640, height: int = 480) -> list[dict]:
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
                        .map(n => n.textContent.trim()).join(' ').trim();
                    if (!directText) continue;
                    const rect = node.getBoundingClientRect();
                    if (rect.width <= 0 || rect.height <= 0) continue;
                    if (rect.top < 0 || rect.left < 0) continue;
                    results.push({text: directText,
                                  x: rect.left, y: rect.top,
                                  width: rect.width, height: rect.height});
                }
                return results;
            }""")
            browser.close()
        return blocks or []
    except Exception as exc:
        print(f"    block extraction failed: {exc}", file=sys.stderr)
        return []


def render_test_case(num: int, force: bool = False) -> bool:
    """Generate renders for one test case. Returns True on success."""
    case_dir = TESTS_DIR / str(num)
    renders_dir = case_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    ref_html = (case_dir / "reference.html").read_text()
    ref_png = renders_dir / "reference.png"
    ref_blocks_json = renders_dir / "reference_blocks.json"

    if force or not ref_png.exists():
        print("  reference render …", end=" ", flush=True)
        img = _render_pw(ref_html)
        if img is None:
            print("FAILED"); return False
        img.save(ref_png)
        blocks = _extract_blocks_pw(ref_html)
        ref_blocks_json.write_text(json.dumps(blocks))
        print(f"ok ({len(blocks)} blocks)")

    for name in VARIANTS:
        html_path = case_dir / "variants" / f"{name}.html"
        png_path  = renders_dir / f"{name}.png"
        blk_path  = renders_dir / f"{name}_blocks.json"

        if not force and png_path.exists():
            continue

        html = html_path.read_text()
        print(f"  [{name}] render …", end=" ", flush=True)
        img = _render_pw(html)
        if img is None:
            img = Image.new("RGB", (640, 480), (255, 255, 255))
        img.save(png_path)
        blocks = _extract_blocks_pw(html)
        blk_path.write_text(json.dumps(blocks))
        print(f"ok ({len(blocks)} blocks)")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Scoring — pure Python, no Playwright (uses saved renders + blocks)
# ─────────────────────────────────────────────────────────────────────────────

def _text_block_score(ref_blocks: list[dict], pred_blocks: list[dict]) -> float:
    from difflib import SequenceMatcher
    from scipy.optimize import linear_sum_assignment

    if not ref_blocks:
        return 1.0 if not pred_blocks else 0.5
    if not pred_blocks:
        return 0.0

    n_ref, n_pred = len(ref_blocks), len(pred_blocks)
    cost = np.zeros((n_ref, n_pred))
    for r, rb in enumerate(ref_blocks):
        ax1, ay1 = rb["x"], rb["y"]
        ax2, ay2 = ax1 + rb["width"], ay1 + rb["height"]
        for p, pb in enumerate(pred_blocks):
            bx1, by1 = pb["x"], pb["y"]
            bx2, by2 = bx1 + pb["width"], by1 + pb["height"]
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = rb["width"]*rb["height"] + pb["width"]*pb["height"] - inter
                cost[r, p] = 1.0 - (inter / union if union > 0 else 0.0)
            else:
                cost[r, p] = 1.0

    row_ind, col_ind = linear_sum_assignment(cost)
    matched, text_scores = 0, []
    for r, p in zip(row_ind, col_ind):
        iou = 1.0 - cost[r, p]
        if iou > 0.1:
            matched += 1
            a, b = ref_blocks[r]["text"], pred_blocks[p]["text"]
            sim = SequenceMatcher(None, a, b).ratio() if (a and b) else (1.0 if not a and not b else 0.0)
            text_scores.append(sim)

    block_match = matched / n_ref
    text_sim    = sum(text_scores) / n_ref if text_scores else 0.0
    return 0.5 * block_match + 0.5 * text_sim


def _position_score(ref_blocks: list[dict], pred_blocks: list[dict]) -> float:
    from scipy.optimize import linear_sum_assignment

    if not ref_blocks:
        return 1.0 if not pred_blocks else 0.5
    if not pred_blocks:
        return 0.0

    DIAG = math.sqrt(640**2 + 480**2)
    n_ref, n_pred = len(ref_blocks), len(pred_blocks)
    cost = np.zeros((n_ref, n_pred))
    for r, rb in enumerate(ref_blocks):
        rcx, rcy = rb["x"] + rb["width"] / 2, rb["y"] + rb["height"] / 2
        for p, pb in enumerate(pred_blocks):
            pcx, pcy = pb["x"] + pb["width"] / 2, pb["y"] + pb["height"] / 2
            cost[r, p] = math.sqrt((rcx - pcx)**2 + (rcy - pcy)**2) / DIAG

    row_ind, col_ind = linear_sum_assignment(cost)
    pos_scores = [1.0 - cost[r, p] for r, p in zip(row_ind, col_ind)]
    if len(pos_scores) < n_ref:
        pos_scores += [0.0] * (n_ref - len(pos_scores))
    return max(0.0, sum(pos_scores) / n_ref)


def _content_factor(pred_img: Image.Image, ref_img: Image.Image) -> float:
    """Multiplier → 0 when prediction is blank while reference has content."""
    SIZE = (32, 32)
    pred_arr = np.array(pred_img.convert("RGB").resize(SIZE))
    ref_arr  = np.array(ref_img.convert("RGB").resize(SIZE))
    pred_nw = float(((pred_arr < 240).any(axis=-1)).mean())
    ref_nw  = float(((ref_arr  < 240).any(axis=-1)).mean())
    if ref_nw > 0.01 and pred_nw < 0.005:
        return pred_nw / 0.005
    return 1.0


def score_variant(
    pred_html: str,
    ref_html: str,
    pred_img: Image.Image,
    ref_img: Image.Image,
    pred_blocks: list[dict],
    ref_blocks: list[dict],
) -> dict[str, float]:
    """Score one variant against its reference using the reward functions."""
    completions = [[{"content": pred_html}]]

    fmt    = format_reward(completions)[0]
    val    = html_validity_reward(completions)[0]
    struct = structural_similarity_reward(completions, solution=[ref_html])[0]
    col    = color_reward(completions, image=[ref_img], pred_image=[pred_img])[0]
    clip_s = clip_visual_reward(completions, image=[ref_img], pred_image=[pred_img])[0]
    tb     = _text_block_score(ref_blocks, pred_blocks)
    pos    = _position_score(ref_blocks, pred_blocks)

    scores = {
        "format": fmt, "validity": val, "structural": struct,
        "text_block": tb, "position": pos, "color": col, "clip": clip_s,
    }
    raw_total = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS) / WEIGHT_SUM
    scores["total"] = raw_total * _content_factor(pred_img, ref_img)
    return scores


def score_test_case(num: int) -> dict | None:
    """Load saved renders and score all variants. Returns None if renders missing."""
    case_dir = TESTS_DIR / str(num)
    renders_dir = case_dir / "renders"
    ref_png = renders_dir / "reference.png"

    if not ref_png.exists():
        return None

    ref_html   = (case_dir / "reference.html").read_text()
    ref_img    = Image.open(ref_png).convert("RGB")
    ref_blocks = json.loads((renders_dir / "reference_blocks.json").read_text())

    results = {}
    for name in VARIANTS:
        png_path = renders_dir / f"{name}.png"
        blk_path = renders_dir / f"{name}_blocks.json"
        html_path = case_dir / "variants" / f"{name}.html"
        if not png_path.exists():
            continue

        pred_img    = Image.open(png_path).convert("RGB")
        pred_html   = html_path.read_text() if html_path.exists() else ""
        pred_blocks = json.loads(blk_path.read_text()) if blk_path.exists() else []

        t0 = time.time()
        scores = score_variant(pred_html, ref_html, pred_img, ref_img, pred_blocks, ref_blocks)
        scores["_elapsed"] = round(time.time() - t0, 2)
        results[name] = scores

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Statistics — correlation between actual and expected
# ─────────────────────────────────────────────────────────────────────────────

def _spearman(x: list[float], y: list[float]) -> float:
    from scipy.stats import spearmanr
    if len(x) < 3:
        return 1.0
    rho, _ = spearmanr(x, y)
    return float(rho) if not math.isnan(rho) else 0.0


def _pearson(x: list[float], y: list[float]) -> float:
    from scipy.stats import pearsonr
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return 1.0 if np.allclose(x, y) else 0.0
    r, _ = pearsonr(x, y)
    return float(r) if not math.isnan(r) else 0.0


def compute_stats(
    case_results: dict[int, dict],  # num → {variant: scores}
    case_expected: dict[int, dict[str, float]],  # num → {variant: stored expected (baseline)}
) -> dict:
    """Compute per-case and global correlation statistics.

    Two comparison axes:
      - vs stored expected (case_expected): regression — checks scores are stable
      - vs CANONICAL_EXPECTED: quality — checks reward distinguishes quality levels

    Pass/fail is based on:
      - Spearman ρ vs canonical ≥ MIN_SPEARMAN_PER_CASE (quality rank ordering)
      - blank ≤ MAX_BLANK_SCORE
      - perfect ≥ MIN_PERFECT_SCORE
    """
    per_case = {}
    global_actual, global_canon, global_stored = [], [], []

    for num, results in case_results.items():
        stored   = case_expected[num]
        actual_tots   = [results[v]["total"]  for v in VARIANTS if v in results]
        stored_tots   = [stored.get(v, CANONICAL_EXPECTED[v]) for v in VARIANTS if v in results]
        canon_tots    = [CANONICAL_EXPECTED[v]                for v in VARIANTS if v in results]

        if len(actual_tots) < 2:
            continue

        rho_stored = _spearman(actual_tots, stored_tots)
        rho_canon  = _spearman(actual_tots, canon_tots)
        r_stored   = _pearson(actual_tots, stored_tots)
        rmse_stored = float(np.sqrt(np.mean([(a - e)**2 for a, e in zip(actual_tots, stored_tots)])))
        rmse_canon  = float(np.sqrt(np.mean([(a - e)**2 for a, e in zip(actual_tots, canon_tots)])))

        # Anchor ordering: perfect > minor_diff > no_style > blank
        anchors = ["perfect", "minor_diff", "no_style", "blank"]
        anchor_tots = [results[v]["total"] for v in anchors if v in results]
        ordering_ok = all(anchor_tots[i] >= anchor_tots[i+1] - 0.01 for i in range(len(anchor_tots)-1))

        blank_ok   = results.get("blank",   {}).get("total", 0.0) <= MAX_BLANK_SCORE
        perfect_ok = results.get("perfect", {}).get("total", 1.0) >= MIN_PERFECT_SCORE

        per_case[num] = {
            "spearman":       rho_canon,   # vs canonical (quality indicator)
            "spearman_stored": rho_stored,  # vs baseline (regression indicator)
            "pearson":        r_stored,
            "rmse":           rmse_stored,
            "rmse_canon":     rmse_canon,
            "ordering_ok":    ordering_ok,
            "blank_ok":       blank_ok,
            "perfect_ok":     perfect_ok,
            "pass": rho_canon >= MIN_SPEARMAN_PER_CASE and blank_ok and perfect_ok,
            "actual":   {v: results[v]["total"] for v in VARIANTS if v in results},
            "expected": stored_tots,
        }

        global_actual.extend(actual_tots)
        global_canon.extend(canon_tots)
        global_stored.extend(stored_tots)

    global_rho_canon  = _spearman(global_actual, global_canon)
    global_rho_stored = _spearman(global_actual, global_stored)
    global_r          = _pearson(global_actual, global_stored)
    global_rmse_stored = float(np.sqrt(np.mean([(a-e)**2 for a,e in zip(global_actual, global_stored)]))) if global_actual else 0.0
    global_rmse_canon  = float(np.sqrt(np.mean([(a-e)**2 for a,e in zip(global_actual, global_canon)])))  if global_actual else 0.0

    return {
        "per_case": per_case,
        "global": {
            "spearman":        global_rho_canon,   # quality
            "spearman_stored": global_rho_stored,  # regression
            "pearson":         global_r,
            "rmse":            global_rmse_stored,
            "rmse_canon":      global_rmse_canon,
            "n_pairs":         len(global_actual),
            "pass": global_rho_canon >= MIN_SPEARMAN_GLOBAL,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

METRIC_COLS = ["format", "validity", "structural", "text_block", "position", "color", "clip", "total"]


def print_case_table(num: int, results: dict, expected: dict[str, float], stats: dict):
    meta = json.loads((TESTS_DIR / str(num) / "meta.json").read_text())
    print(f"\n{'─'*110}")
    print(f"  Case {num:2d}  [{meta['source']}]   "
          f"ρ(quality)={stats['spearman']:+.3f}  "
          f"ρ(regress)={stats['spearman_stored']:+.3f}  "
          f"RMSE={stats['rmse']:.3f}  "
          f"RMSE(canon)={stats['rmse_canon']:.3f}  "
          f"{'PASS' if stats['pass'] else 'FAIL'}")
    print(f"{'─'*110}")

    header = f"  {'variant':<12}" + "".join(f" {c:>10}" for c in METRIC_COLS) + "  baseline   Δ(canon)"
    print(header)
    print(f"  {'─'*12}" + "  ─────────" * len(METRIC_COLS) + "  ────────  ────────")

    for v in VARIANTS:
        if v not in results:
            continue
        s = results[v]
        baseline = expected.get(v, CANONICAL_EXPECTED[v])
        canon    = CANONICAL_EXPECTED[v]
        delta_bl = s["total"] - baseline
        delta_ca = s["total"] - canon
        row = (f"  {v:<12}"
               + "".join(f" {s.get(c, 0):>10.3f}" for c in METRIC_COLS)
               + f"  {baseline:.4f}  {delta_ca:+.3f}")
        print(row)

    issues = []
    if not stats["blank_ok"]:
        issues.append(f"blank score {results['blank']['total']:.3f} > {MAX_BLANK_SCORE}")
    if not stats["perfect_ok"]:
        issues.append(f"perfect score {results['perfect']['total']:.3f} < {MIN_PERFECT_SCORE}")
    if not stats["ordering_ok"]:
        issues.append("anchor ordering violated (perfect > minor_diff > no_style > blank)")
    if stats["spearman"] < MIN_SPEARMAN_PER_CASE:
        issues.append(f"quality ρ {stats['spearman']:.3f} < {MIN_SPEARMAN_PER_CASE}")
    for issue in issues:
        print(f"  ⚠  {issue}")


def print_summary(stats: dict, all_results: dict[int, dict]):
    global_s = stats["global"]
    per_case = stats["per_case"]
    passes = sum(1 for s in per_case.values() if s["pass"])
    total  = len(per_case)

    print(f"\n{'═'*110}")
    print(f"  GLOBAL  ρ(quality)={global_s['spearman']:+.3f}  "
          f"ρ(regress)={global_s['spearman_stored']:+.3f}  "
          f"RMSE(baseline)={global_s['rmse']:.4f}  "
          f"RMSE(canon)={global_s['rmse_canon']:.3f}  "
          f"n={global_s['n_pairs']}  "
          f"{'PASS' if global_s['pass'] else 'FAIL'}")
    print(f"  Per-case: {passes}/{total} PASS")
    print(f"  ρ(quality): correlation with canonical targets — measures reward calibration quality")
    print(f"  ρ(regress): correlation with stored baseline — measures reward stability (regression)")
    print(f"{'═'*110}")

    print(f"\n  Per-metric averages across {len(all_results)} cases:")
    header = f"  {'variant':<12}" + "".join(f" {c:>10}" for c in METRIC_COLS)
    print(header)
    print(f"  {'─'*12}" + "  ─────────" * len(METRIC_COLS))
    for v in VARIANTS:
        avgs = {}
        for m in METRIC_COLS:
            vals = [all_results[n][v][m] for n in all_results if v in all_results[n]]
            avgs[m] = sum(vals) / len(vals) if vals else 0.0
        row = f"  {v:<12}" + "".join(f" {avgs.get(c, 0):>10.3f}" for c in METRIC_COLS)
        print(row)


# ─────────────────────────────────────────────────────────────────────────────
# Pytest-compatible test functions
# ─────────────────────────────────────────────────────────────────────────────

def _load_case_expected(num: int) -> dict[str, float]:
    expected_path = TESTS_DIR / str(num) / "expected_scores.json"
    if expected_path.exists():
        return json.loads(expected_path.read_text())
    return CANONICAL_EXPECTED.copy()


def _require_renders(num: int):
    """Skip pytest test if renders are missing for this case."""
    import pytest
    ref_png = TESTS_DIR / str(num) / "renders" / "reference.png"
    if not ref_png.exists():
        pytest.skip(f"Renders for case {num} not generated. "
                    "Run: python tests/test_reward_stability.py --render")


try:
    import pytest

    @pytest.mark.parametrize("num", list(range(15)))
    def test_ordering(num: int):
        """Variants must be ordered: perfect > minor_diff > no_style > blank."""
        _require_renders(num)
        results = score_test_case(num)
        assert results is not None, f"Scoring failed for case {num}"

        anchors = ["perfect", "minor_diff", "no_style", "blank"]
        tots = [results[v]["total"] for v in anchors if v in results]
        for i in range(len(tots) - 1):
            assert tots[i] >= tots[i+1] - 0.01, (
                f"Case {num}: anchor ordering violated: "
                + " > ".join(f"{v}={t:.3f}" for v, t in zip(anchors, tots))
            )

    @pytest.mark.parametrize("num", list(range(15)))
    def test_blank_score_near_zero(num: int):
        """Blank pages must score below 0.05."""
        _require_renders(num)
        results = score_test_case(num)
        assert results is not None
        blank_score = results.get("blank", {}).get("total", 0.0)
        assert blank_score <= MAX_BLANK_SCORE, (
            f"Case {num}: blank score {blank_score:.3f} > {MAX_BLANK_SCORE}"
        )

    @pytest.mark.parametrize("num", list(range(15)))
    def test_perfect_score_high(num: int):
        """Perfect variants must score above 0.80."""
        _require_renders(num)
        results = score_test_case(num)
        assert results is not None
        perfect_score = results.get("perfect", {}).get("total", 0.0)
        assert perfect_score >= MIN_PERFECT_SCORE, (
            f"Case {num}: perfect score {perfect_score:.3f} < {MIN_PERFECT_SCORE}"
        )

    @pytest.mark.parametrize("num", list(range(15)))
    def test_spearman_per_case(num: int):
        """Spearman rank correlation with expected scores must exceed threshold."""
        _require_renders(num)
        results = score_test_case(num)
        assert results is not None
        expected = _load_case_expected(num)
        actual_tots   = [results[v]["total"]               for v in VARIANTS if v in results]
        expected_tots = [expected.get(v, CANONICAL_EXPECTED[v]) for v in VARIANTS if v in results]
        rho = _spearman(actual_tots, expected_tots)
        assert rho >= MIN_SPEARMAN_PER_CASE, (
            f"Case {num}: Spearman ρ={rho:.3f} < {MIN_SPEARMAN_PER_CASE}"
        )

    def test_global_spearman():
        """Global Spearman ρ across all scored (case, variant) pairs must exceed threshold."""
        scored = {}
        for num in range(15):
            ref_png = TESTS_DIR / str(num) / "renders" / "reference.png"
            if not ref_png.exists():
                continue
            results = score_test_case(num)
            if results:
                scored[num] = results

        if not scored:
            import pytest
            pytest.skip("No rendered test cases found. Run --render first.")

        all_actual, all_expected = [], []
        for num, results in scored.items():
            expected = _load_case_expected(num)
            for v in VARIANTS:
                if v in results:
                    all_actual.append(results[v]["total"])
                    all_expected.append(expected.get(v, CANONICAL_EXPECTED[v]))

        rho = _spearman(all_actual, all_expected)
        assert rho >= MIN_SPEARMAN_GLOBAL, (
            f"Global Spearman ρ={rho:.3f} < {MIN_SPEARMAN_GLOBAL} "
            f"across {len(all_actual)} (case, variant) pairs"
        )

except ImportError:
    pass  # pytest not available — CLI mode only


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Reward stability test suite")
    p.add_argument("--render", action="store_true",
                   help="Run Playwright to generate renders (slow, do once)")
    p.add_argument("--force", action="store_true",
                   help="Force re-render even if renders exist")
    p.add_argument("--scaffold-only", action="store_true",
                   help="Only scaffold data/tests/ structure (no rendering/scoring)")
    p.add_argument("--update-expected", action="store_true",
                   help="After scoring, write actual scores to expected_scores.json")
    p.add_argument("--cases", metavar="N,...",
                   help="Comma-separated case numbers to run (default: all)")
    p.add_argument("--min-spearman", type=float, default=MIN_SPEARMAN_GLOBAL,
                   help=f"Minimum global Spearman ρ to pass (default: {MIN_SPEARMAN_GLOBAL})")
    return p.parse_args()


def main():
    args = parse_args()
    global MIN_SPEARMAN_GLOBAL
    MIN_SPEARMAN_GLOBAL = args.min_spearman

    if args.cases:
        case_nums = [int(x) for x in args.cases.split(",")]
    else:
        case_nums = list(range(15))

    # ── Scaffold ─────────────────────────────────────────────────────────────
    print(f"[SCAFFOLD] Checking data/tests/ structure …")
    needs_scaffold = any(not (TESTS_DIR / str(n) / "reference.html").exists() for n in case_nums)
    if needs_scaffold or args.scaffold_only:
        print(f"  Creating {len(case_nums)} test cases …")
        for n in case_nums:
            scaffold_test_case(n)
        print(f"  Done.")
    else:
        print(f"  All {len(case_nums)} cases already scaffolded.")

    if args.scaffold_only:
        return

    # ── Render ────────────────────────────────────────────────────────────────
    if args.render or args.force:
        print(f"\n[RENDER] {len(case_nums)} cases × {len(VARIANTS) + 1} renders each …")
        for n in case_nums:
            meta = json.loads((TESTS_DIR / str(n) / "meta.json").read_text())
            print(f"\n── case {n:2d}  [{meta['source']}] ──")
            ok = render_test_case(n, force=args.force)
            if not ok:
                print(f"  ERROR rendering case {n}")
        print(f"\n[RENDER] Done.")
    else:
        # Check if any case is missing renders
        missing = [n for n in case_nums
                   if not (TESTS_DIR / str(n) / "renders" / "reference.png").exists()]
        if missing:
            print(f"\n⚠  Cases without renders: {missing}")
            print(f"   Run: python tests/test_reward_stability.py --render")

    # ── Score ─────────────────────────────────────────────────────────────────
    print(f"\n[SCORE] Scoring {len(case_nums)} cases …")
    all_results: dict[int, dict] = {}
    case_expected: dict[int, dict[str, float]] = {}

    for n in case_nums:
        results = score_test_case(n)
        if results is None:
            print(f"  case {n:2d}: no renders — skipping")
            continue
        all_results[n] = results
        case_expected[n] = _load_case_expected(n)

    if not all_results:
        print("No results to report. Generate renders first with --render.")
        return

    stats = compute_stats(all_results, case_expected)

    for n in sorted(all_results):
        if n in stats["per_case"]:
            print_case_table(n, all_results[n], case_expected[n], stats["per_case"][n])

    print_summary(stats, all_results)

    # ── Update expected ───────────────────────────────────────────────────────
    if args.update_expected:
        print(f"\n[UPDATE-EXPECTED] Writing actual scores to expected_scores.json …")
        for n, results in all_results.items():
            expected_path = TESTS_DIR / str(n) / "expected_scores.json"
            new_expected = {v: round(results[v]["total"], 4)
                            for v in VARIANTS if v in results}
            expected_path.write_text(json.dumps(new_expected, indent=2))
            print(f"  case {n:2d}: {new_expected}")
        print(f"  Done.")

    # ── Save report ───────────────────────────────────────────────────────────
    def _to_python(obj):
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_python(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    report = _to_python({
        "stats": stats,
        "results": {str(n): {v: {k: round(s[k], 4) for k in s if not k.startswith("_")}
                              for v, s in res.items()}
                   for n, res in all_results.items()},
    })
    report_path = TESTS_DIR / "last_run_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Full report → {report_path}")

    # ── Exit code ─────────────────────────────────────────────────────────────
    if not stats["global"]["pass"]:
        print(f"\n  FAIL: global Spearman ρ={stats['global']['spearman']:.3f} "
              f"< {MIN_SPEARMAN_GLOBAL}")
        sys.exit(1)
    print(f"\n  PASS: global Spearman ρ={stats['global']['spearman']:.3f}")


if __name__ == "__main__":
    main()
