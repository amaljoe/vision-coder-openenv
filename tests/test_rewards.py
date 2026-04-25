#!/usr/bin/env python3
"""Reward correlation tests.

Scores 7 quality-level variants per test case and asserts Spearman ρ ≥ 0.80
against each case's expected_scores.json.  Renders are auto-generated via
Playwright the first time they are needed; pass --force-render to redo them.

    pytest tests/test_rewards.py              # auto-renders if missing
    pytest tests/test_rewards.py --force-render

    python tests/test_rewards.py              # CLI: score + report
    python tests/test_rewards.py --force-render
    python tests/test_rewards.py --update-expected
    python tests/test_rewards.py --cases 0,1,5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(Path.home() / "playwright-browsers"))

from openenv.server.rewards.format_rewards import format_reward
from openenv.server.rewards.validity_rewards import html_validity_reward
from openenv.server.rewards.structural_rewards import structural_similarity_reward
from openenv.server.rewards.color_rewards import color_reward
from openenv.server.rewards.visual_rewards import clip_visual_reward
from openenv.server.rewards.ssim_reward import ssim_reward

# ── Constants ─────────────────────────────────────────────────────────────────

WEIGHTS: dict[str, float] = {
    "format":     0.5,
    "validity":   0.5,
    "structural": 0.5,
    "text_block": 3.0,
    "position":   1.0,
    "color":      1.5,
    "clip":       2.5,
    "ssim":       1.5,
}
WEIGHT_SUM = sum(WEIGHTS.values())  # 11.0

TESTS_DIR = _ROOT / "data" / "tests"
DATA_SRC  = _ROOT / "data"

CASE_SOURCES: dict[int, tuple[str, int]] = {
    **{i:      ("easy",   i) for i in range(5)},
    **{i + 5:  ("medium", i) for i in range(5)},
    **{i + 10: ("hard",   i) for i in range(5)},
}
VARIANTS = ["perfect", "minor_diff", "bad_colors", "half_styled", "no_layout", "no_style", "blank"]

BLANK_HTML = (
    "<!DOCTYPE html><html><head><title>Page</title></head>"
    "<body style=\"background:#fff;\"></body></html>"
)

CANONICAL_EXPECTED: dict[str, float] = {
    "perfect":    0.95,
    "minor_diff": 0.88,
    "bad_colors": 0.68,
    "half_styled": 0.60,
    "no_layout":  0.50,
    "no_style":   0.38,
    "blank":      0.00,
}

MIN_SPEARMAN_PER_CASE = 0.80
MIN_SPEARMAN_GLOBAL   = 0.85


# ── Variant generation ────────────────────────────────────────────────────────

import re as _re

_LAYOUT_PROPS = {
    "padding", "margin", "border-radius", "box-shadow", "display",
    "align-items", "justify-content", "min-height", "min-width",
    "position", "top", "right", "bottom", "left", "transform",
    "flex", "grid", "float", "overflow", "vertical-align",
    "width", "height", "box-sizing",
}


def make_variants(ref_html: str) -> dict[str, str]:
    minor = ref_html
    minor = _re.sub(r"(background(?:-color)?:\s*)(#[0-9a-fA-F]{6})", r"\g<1>#888888", minor, count=1)
    minor = _re.sub(r"font-size:(\d+)px",
                    lambda m: f"font-size:{max(8, int(m.group(1)) - 4)}px", minor, count=2)

    def _invert(m):
        r = 255 - int(m.group(1), 16)
        g = 255 - int(m.group(2), 16)
        b = 255 - int(m.group(3), 16)
        return f"#{r:02x}{g:02x}{b:02x}"
    bad_colors = _re.sub(r'#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})', _invert, ref_html)

    def _strip_layout(m):
        kept = [p.strip() for p in m.group(1).split(";")
                if p.strip() and not any(p.strip().lower().startswith(lp) for lp in _LAYOUT_PROPS)]
        return f'style="{"; ".join(kept)}"'
    no_layout = _re.sub(r'style="([^"]*)"', _strip_layout, ref_html)

    def _keep_half(m):
        props = [p.strip() for p in m.group(1).split(";") if p.strip()]
        return f'style="{"; ".join(props[::2])}"'
    half_styled = _re.sub(r'style="([^"]*)"', _keep_half, ref_html)

    no_style = _re.sub(r'\s+style="[^"]*"', "", ref_html)
    no_style = _re.sub(r'\s+class="[^"]*"', "", no_style)

    return {
        "perfect":    ref_html,
        "minor_diff": minor,
        "bad_colors": bad_colors,
        "half_styled": half_styled,
        "no_layout":  no_layout,
        "no_style":   no_style,
        "blank":      BLANK_HTML,
    }


# ── Scaffolding ───────────────────────────────────────────────────────────────

def scaffold_test_case(num: int) -> Path:
    difficulty, idx = CASE_SOURCES[num]
    case_dir = TESTS_DIR / str(num)
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "renders").mkdir(exist_ok=True)

    ref_html = (DATA_SRC / difficulty / f"{idx}.html").read_text()

    for path, content in [
        (case_dir / "reference.html", ref_html),
        (case_dir / "meta.json", json.dumps(
            {"source": f"{difficulty}/{idx}", "difficulty": difficulty, "idx": idx}, indent=2)),
        (case_dir / "expected_scores.json", json.dumps(CANONICAL_EXPECTED, indent=2)),
    ]:
        if not path.exists():
            path.write_text(content)

    variants_dir = case_dir / "variants"
    variants_dir.mkdir(exist_ok=True)
    for name, html in make_variants(ref_html).items():
        dest = variants_dir / f"{name}.html"
        if not dest.exists():
            dest.write_text(html)

    return case_dir


# ── Playwright rendering ──────────────────────────────────────────────────────

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
                    document.body || document.documentElement, NodeFilter.SHOW_ELEMENT, null);
                let node;
                while ((node = walker.nextNode())) {
                    const directText = Array.from(node.childNodes)
                        .filter(n => n.nodeType === Node.TEXT_NODE)
                        .map(n => n.textContent.trim()).join(' ').trim();
                    if (!directText) continue;
                    const rect = node.getBoundingClientRect();
                    if (rect.width <= 0 || rect.height <= 0 || rect.top < 0 || rect.left < 0) continue;
                    results.push({text: directText, x: rect.left, y: rect.top,
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
    case_dir = TESTS_DIR / str(num)
    renders_dir = case_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    ref_html = (case_dir / "reference.html").read_text()
    ref_png = renders_dir / "reference.png"

    if force or not ref_png.exists():
        print(f"  [{num}] reference …", end=" ", flush=True)
        img = _render_pw(ref_html)
        if img is None:
            print("FAILED"); return False
        img.save(ref_png)
        (renders_dir / "reference_blocks.json").write_text(json.dumps(_extract_blocks_pw(ref_html)))
        print("ok")

    for name in VARIANTS:
        png_path = renders_dir / f"{name}.png"
        if not force and png_path.exists():
            continue
        html = (case_dir / "variants" / f"{name}.html").read_text()
        print(f"  [{num}] {name} …", end=" ", flush=True)
        img = _render_pw(html) or Image.new("RGB", (640, 480), (255, 255, 255))
        img.save(png_path)
        (renders_dir / f"{name}_blocks.json").write_text(json.dumps(_extract_blocks_pw(html)))
        print("ok")

    return True


def _ensure_renders(num: int, force: bool = False) -> bool:
    """Return True if renders are ready; auto-render if missing."""
    ref_png = TESTS_DIR / str(num) / "renders" / "reference.png"
    if not force and ref_png.exists():
        return True
    scaffold_test_case(num)
    return render_test_case(num, force=force)


# ── Scoring ───────────────────────────────────────────────────────────────────

def _text_block_score(ref_blocks: list[dict], pred_blocks: list[dict]) -> float:
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

    return 0.5 * (matched / n_ref) + 0.5 * (sum(text_scores) / n_ref if text_scores else 0.0)


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
        rcx, rcy = rb["x"] + rb["width"]/2, rb["y"] + rb["height"]/2
        for p, pb in enumerate(pred_blocks):
            pcx, pcy = pb["x"] + pb["width"]/2, pb["y"] + pb["height"]/2
            cost[r, p] = math.sqrt((rcx-pcx)**2 + (rcy-pcy)**2) / DIAG

    row_ind, col_ind = linear_sum_assignment(cost)
    pos_scores = [1.0 - cost[r, p] for r, p in zip(row_ind, col_ind)]
    if len(pos_scores) < n_ref:
        pos_scores += [0.0] * (n_ref - len(pos_scores))
    return max(0.0, sum(pos_scores) / n_ref)


def _content_factor(pred_img: Image.Image, ref_img: Image.Image) -> float:
    SIZE = (32, 32)
    pred_arr = np.array(pred_img.convert("RGB").resize(SIZE))
    ref_arr  = np.array(ref_img.convert("RGB").resize(SIZE))
    pred_nw = float(((pred_arr < 240).any(axis=-1)).mean())
    ref_nw  = float(((ref_arr  < 240).any(axis=-1)).mean())
    if ref_nw > 0.01 and pred_nw < 0.005:
        return pred_nw / 0.005
    return 1.0


def score_variant(pred_html, ref_html, pred_img, ref_img, pred_blocks, ref_blocks) -> dict[str, float]:
    completions = [[{"content": pred_html}]]
    scores = {
        "format":     format_reward(completions)[0],
        "validity":   html_validity_reward(completions)[0],
        "structural": structural_similarity_reward(completions, solution=[ref_html])[0],
        "color":      color_reward(completions, image=[ref_img], pred_image=[pred_img])[0],
        "clip":       clip_visual_reward(completions, image=[ref_img], pred_image=[pred_img])[0],
        "ssim":       ssim_reward(completions, image=[ref_img], pred_image=[pred_img])[0],
        "text_block": _text_block_score(ref_blocks, pred_blocks),
        "position":   _position_score(ref_blocks, pred_blocks),
    }
    raw_total = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS) / WEIGHT_SUM
    scores["total"] = raw_total * _content_factor(pred_img, ref_img)
    return scores


def score_test_case(num: int) -> dict | None:
    case_dir = TESTS_DIR / str(num)
    renders_dir = case_dir / "renders"
    if not (renders_dir / "reference.png").exists():
        return None

    ref_html   = (case_dir / "reference.html").read_text()
    ref_img    = Image.open(renders_dir / "reference.png").convert("RGB")
    ref_blocks = json.loads((renders_dir / "reference_blocks.json").read_text())

    results = {}
    for name in VARIANTS:
        png_path  = renders_dir / f"{name}.png"
        blk_path  = renders_dir / f"{name}_blocks.json"
        html_path = case_dir / "variants" / f"{name}.html"
        if not png_path.exists():
            continue
        results[name] = score_variant(
            pred_html   = html_path.read_text() if html_path.exists() else "",
            ref_html    = ref_html,
            pred_img    = Image.open(png_path).convert("RGB"),
            ref_img     = ref_img,
            pred_blocks = json.loads(blk_path.read_text()) if blk_path.exists() else [],
            ref_blocks  = ref_blocks,
        )
    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _spearman(x: list[float], y: list[float]) -> float:
    from scipy.stats import spearmanr
    if len(x) < 3:
        return 1.0
    rho, _ = spearmanr(x, y)
    return float(rho) if not math.isnan(rho) else 0.0


def _load_case_expected(num: int) -> dict[str, float]:
    p = TESTS_DIR / str(num) / "expected_scores.json"
    return json.loads(p.read_text()) if p.exists() else CANONICAL_EXPECTED.copy()


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("num", list(range(15)))
def test_spearman_per_case(num: int, force_render):
    if not _ensure_renders(num, force=force_render):
        pytest.skip(f"Playwright unavailable; cannot render case {num}")
    results = score_test_case(num)
    assert results is not None
    expected = _load_case_expected(num)
    actual = [results[v]["total"]               for v in VARIANTS if v in results]
    target = [expected.get(v, CANONICAL_EXPECTED[v]) for v in VARIANTS if v in results]
    rho = _spearman(actual, target)
    assert rho >= MIN_SPEARMAN_PER_CASE, (
        f"Case {num}: Spearman ρ={rho:.3f} < {MIN_SPEARMAN_PER_CASE}\n"
        + "  " + "  ".join(f"{v}={results[v]['total']:.3f}" for v in VARIANTS if v in results)
    )


def test_global_spearman(force_render):
    all_actual, all_expected = [], []
    for num in range(15):
        if not _ensure_renders(num, force=force_render):
            continue
        results = score_test_case(num)
        if not results:
            continue
        expected = _load_case_expected(num)
        for v in VARIANTS:
            if v in results:
                all_actual.append(results[v]["total"])
                all_expected.append(expected.get(v, CANONICAL_EXPECTED[v]))

    if not all_actual:
        pytest.skip("No rendered test cases available")

    rho = _spearman(all_actual, all_expected)
    assert rho >= MIN_SPEARMAN_GLOBAL, (
        f"Global Spearman ρ={rho:.3f} < {MIN_SPEARMAN_GLOBAL} "
        f"across {len(all_actual)} (case, variant) pairs"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

METRIC_COLS = ["format", "validity", "structural", "text_block", "position", "color", "clip", "ssim", "total"]


def _print_case_table(num, results, stats):
    meta = json.loads((TESTS_DIR / str(num) / "meta.json").read_text())
    print(f"\n{'─'*110}")
    print(f"  Case {num:2d}  [{meta['source']}]   ρ={stats['rho']:+.3f}  {'PASS' if stats['pass'] else 'FAIL'}")
    print(f"{'─'*110}")
    print(f"  {'variant':<12}" + "".join(f" {c:>10}" for c in METRIC_COLS) + "  Δ(canon)")
    for v in VARIANTS:
        if v not in results:
            continue
        s = results[v]
        delta = s["total"] - CANONICAL_EXPECTED.get(v, 0)
        print(f"  {v:<12}" + "".join(f" {s.get(c, 0):>10.3f}" for c in METRIC_COLS) + f"  {delta:+.3f}")


def main():
    p = argparse.ArgumentParser(description="Reward correlation test suite")
    p.add_argument("--force-render",    action="store_true", help="Re-render even if PNGs exist")
    p.add_argument("--update-expected", action="store_true", help="Write actual scores to expected_scores.json")
    p.add_argument("--cases", metavar="N,...", help="Comma-separated case numbers (default: all)")
    args = p.parse_args()

    case_nums = [int(x) for x in args.cases.split(",")] if args.cases else list(range(15))

    all_results = {}
    for n in case_nums:
        if not _ensure_renders(n, force=args.force_render):
            print(f"  case {n:2d}: render failed — skipping")
            continue
        r = score_test_case(n)
        if r:
            all_results[n] = r

    if not all_results:
        print("No results.")
        return

    all_actual, all_expected_flat = [], []
    per_case_stats = {}
    for num, results in all_results.items():
        expected = _load_case_expected(num)
        actual = [results[v]["total"]               for v in VARIANTS if v in results]
        target = [expected.get(v, CANONICAL_EXPECTED[v]) for v in VARIANTS if v in results]
        rho = _spearman(actual, target)
        per_case_stats[num] = {"rho": rho, "pass": rho >= MIN_SPEARMAN_PER_CASE}
        all_actual.extend(actual)
        all_expected_flat.extend(target)

    for n in sorted(all_results):
        _print_case_table(n, all_results[n], per_case_stats[n])

    global_rho = _spearman(all_actual, all_expected_flat)
    passes = sum(1 for s in per_case_stats.values() if s["pass"])
    print(f"\n{'═'*110}")
    print(f"  GLOBAL  ρ={global_rho:+.3f}  {passes}/{len(per_case_stats)} PASS")
    print(f"{'═'*110}")

    if args.update_expected:
        print("\n[UPDATE-EXPECTED] Writing actual scores …")
        for n, results in all_results.items():
            new_exp = {v: round(results[v]["total"], 4) for v in VARIANTS if v in results}
            (TESTS_DIR / str(n) / "expected_scores.json").write_text(json.dumps(new_exp, indent=2))
            print(f"  case {n:2d}: {new_exp}")

    (TESTS_DIR / "last_run_report.json").write_text(json.dumps({
        "global_spearman": global_rho,
        "per_case": {str(n): {v: round(r["total"], 4) for v, r in res.items()}
                     for n, res in all_results.items()},
    }, indent=2))

    if global_rho < MIN_SPEARMAN_GLOBAL:
        print(f"\n  FAIL: global ρ={global_rho:.3f} < {MIN_SPEARMAN_GLOBAL}")
        sys.exit(1)
    print(f"\n  PASS: global ρ={global_rho:.3f}")


if __name__ == "__main__":
    main()
