#!/usr/bin/env python3
"""VisionCoder reward tests — unit tests + stability/correlation tests in one file.

Unit tests (fast, mocked Playwright — run always):
    pytest tests/test_rewards.py -m "not integration"

Integration tests (real Playwright rendering):
    pytest tests/test_rewards.py

Stability tests (require pre-rendered PNGs in data/tests/):
    pytest tests/test_rewards.py -k "stability"

CLI — render then score:
    python tests/test_rewards.py --render        # generate renders (needs Playwright)
    python tests/test_rewards.py                 # score using cached renders
    python tests/test_rewards.py --update-expected  # lock in new baselines
    python tests/test_rewards.py --cases 0,1,5   # specific cases only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from unittest.mock import MagicMock, patch

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

# ── Stability test constants ──────────────────────────────────────────────────

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
    **{i:      ("easy",   i)     for i in range(5)},
    **{i + 5:  ("medium", i)     for i in range(5)},
    **{i + 10: ("hard",   i)     for i in range(5)},
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
MAX_BLANK_SCORE       = 0.05
MIN_PERFECT_SCORE     = 0.80


# ── Unit test helpers ─────────────────────────────────────────────────────────

SIMPLE_HTML = """<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>
  <h1>Hello World</h1>
  <p>This is a paragraph.</p>
  <div style="color:blue;">Blue text</div>
</body>
</html>"""

EMPTY_HTML = ""
MINIMAL_HTML = "<html><body><p>Hi</p></body></html>"
DIFFERENT_HTML = """<!DOCTYPE html>
<html>
<head><title>Other</title></head>
<body>
  <h2>Goodbye World</h2>
  <span>Different content entirely</span>
</body>
</html>"""
MALFORMED_HTML = "<div><p>Not closed"


def _make_completion(html: str) -> list[list[dict]]:
    return [[{"content": html}]]


def _white_image(w: int = 640, h: int = 480) -> Image.Image:
    return Image.new("RGB", (w, h), color=(255, 255, 255))


def _solid_image(color: tuple, w: int = 640, h: int = 480) -> Image.Image:
    return Image.new("RGB", (w, h), color=color)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractHtml:
    def test_extracts_from_fenced_markdown(self):
        from openenv.server.rewards import extract_html
        assert extract_html("```html\n<html><body>hi</body></html>\n```") == "<html><body>hi</body></html>"

    def test_strips_think_blocks(self):
        from openenv.server.rewards import extract_html
        assert extract_html("<think>reasoning</think>\n```html\n<p>ok</p>\n```") == "<p>ok</p>"

    def test_handles_unclosed_fence(self):
        from openenv.server.rewards import extract_html
        assert "<p>truncated" in extract_html("```html\n<p>truncated")

    def test_passthrough_plain_html(self):
        from openenv.server.rewards import extract_html
        assert extract_html(SIMPLE_HTML) == SIMPLE_HTML


class TestFormatReward:
    def _run(self, html: str) -> float:
        return format_reward(_make_completion(html))[0]

    def test_perfect_score(self):
        assert self._run("```html\n<!DOCTYPE html><html><body></body></html>\n```") == 1.0

    def test_no_fence_no_doctype(self):
        assert self._run("<p>bare</p>") == 0.0

    def test_fence_but_no_doctype(self):
        assert self._run("```html\n<p>no doctype</p>\n```") == 0.5

    def test_doctype_but_no_fence(self):
        assert self._run("<!DOCTYPE html><html></html>") == 0.5

    def test_batch(self):
        completions = [
            [{"content": "```html\n<!DOCTYPE html><html></html>\n```"}],
            [{"content": "bare"}],
        ]
        scores = format_reward(completions)
        assert scores[0] == 1.0
        assert scores[1] == 0.0


class TestValidityReward:
    def _run(self, html: str) -> float:
        return html_validity_reward(_make_completion(html))[0]

    def test_full_score(self):
        html = "<html><head></head><body><p>a</p><div>b</div><span>c</span><ul><li>d</li></ul><h1>e</h1></body></html>"
        assert self._run(html) == 1.0

    def test_no_structure_tags(self):
        assert self._run("<p>bare paragraph</p>") < 0.5

    def test_empty_html(self):
        assert self._run(EMPTY_HTML) == 0.0

    def test_scores_in_range(self):
        assert 0.0 <= self._run(SIMPLE_HTML) <= 1.0


class TestStructuralReward:
    def _run(self, html: str, ref: str) -> float:
        return structural_similarity_reward(_make_completion(html), solution=[ref])[0]

    def test_identical_html(self):
        assert self._run(SIMPLE_HTML, SIMPLE_HTML) == 1.0

    def test_empty_vs_ref(self):
        score = self._run(EMPTY_HTML, SIMPLE_HTML)
        assert 0.0 <= score <= 0.5

    def test_different_html(self):
        assert 0.0 <= self._run(DIFFERENT_HTML, SIMPLE_HTML) <= 1.0


class TestTextBlockReward:
    def _run_with_blocks(self, ref_blocks, pred_blocks, ref_html=SIMPLE_HTML) -> float:
        from openenv.server.rewards import text_block_rewards
        with patch.object(text_block_rewards, "_get_text_blocks") as mock_get:
            mock_get.side_effect = [ref_blocks, pred_blocks]
            return text_block_rewards.text_block_reward(_make_completion(SIMPLE_HTML), solution=[ref_html])[0]

    def test_perfect_match(self):
        b = [{"text": "Hello", "x": 10, "y": 10, "width": 100, "height": 20}]
        assert self._run_with_blocks(b, b) == pytest.approx(1.0, abs=0.01)

    def test_no_pred_blocks(self):
        ref = [{"text": "Hello", "x": 10, "y": 10, "width": 100, "height": 20}]
        assert self._run_with_blocks(ref, []) == 0.0

    def test_no_ref_no_pred(self):
        assert self._run_with_blocks([], []) == 1.0

    def test_no_solution_returns_zero(self):
        from openenv.server.rewards.text_block_rewards import text_block_reward
        assert text_block_reward(_make_completion(SIMPLE_HTML), solution=None)[0] == 0.0


class TestBboxIou:
    def test_identical(self):
        from openenv.server.rewards.text_block_rewards import _bbox_iou
        b = {"x": 0, "y": 0, "width": 100, "height": 100}
        assert _bbox_iou(b, b) == pytest.approx(1.0)

    def test_non_overlapping(self):
        from openenv.server.rewards.text_block_rewards import _bbox_iou
        a = {"x": 0, "y": 0, "width": 10, "height": 10}
        b = {"x": 100, "y": 100, "width": 10, "height": 10}
        assert _bbox_iou(a, b) == 0.0

    def test_partial_overlap(self):
        from openenv.server.rewards.text_block_rewards import _bbox_iou
        a = {"x": 0, "y": 0, "width": 20, "height": 20}
        b = {"x": 10, "y": 10, "width": 20, "height": 20}
        assert 0.0 < _bbox_iou(a, b) < 1.0


class TestColorReward:
    def _run_with_images(self, ref_img, pred_img) -> float:
        from openenv.server.rewards import color_rewards
        with patch.object(color_rewards, "_render_html", return_value=pred_img):
            return color_rewards.color_reward(_make_completion(SIMPLE_HTML), image=[ref_img])[0]

    def test_identical_images_high_score(self):
        img = _solid_image((100, 150, 200))
        assert self._run_with_images(img, img) > 0.95

    def test_no_reference_returns_neutral(self):
        assert color_reward(_make_completion(SIMPLE_HTML), image=None)[0] == 0.5

    def test_score_in_range(self):
        assert 0.0 <= self._run_with_images(_solid_image((200, 100, 50)), _solid_image((180, 120, 70))) <= 1.0


class TestVisualRewards:
    def test_pred_image_identical_high_score(self):
        from openenv.server.rewards import visual_rewards
        img = _solid_image((128, 64, 32))
        with patch.object(visual_rewards, "_clip_similarity", return_value=0.97):
            scores = visual_rewards.clip_visual_reward(_make_completion(SIMPLE_HTML), image=[img], pred_image=[img])
        assert scores[0] == pytest.approx(0.97)

    def test_pred_image_provided_skips_render(self):
        from openenv.server.rewards import visual_rewards
        img = _solid_image((10, 20, 30))
        with (
            patch.object(visual_rewards, "_render_html") as mock_render,
            patch.object(visual_rewards, "_clip_similarity", return_value=0.8),
        ):
            visual_rewards.clip_visual_reward(_make_completion(SIMPLE_HTML), image=[img], pred_image=[img])
        mock_render.assert_not_called()

    def test_no_reference_returns_neutral(self):
        from openenv.server.rewards import visual_rewards
        img = _white_image()
        assert visual_rewards.clip_visual_reward(_make_completion(SIMPLE_HTML), image=None, pred_image=[img])[0] == 0.5

    def test_pil_similarity_identical(self):
        from openenv.server.rewards.visual_rewards import _pil_similarity
        img = _solid_image((100, 200, 50))
        assert _pil_similarity(img, img) == pytest.approx(1.0)


class TestEnvironmentWeights:
    def test_weight_sum(self):
        from openenv.server import environment as env_mod
        assert sum(env_mod.REWARD_WEIGHTS.values()) == pytest.approx(env_mod._WEIGHT_SUM)
        assert env_mod._WEIGHT_SUM == pytest.approx(11.0)

    def test_all_phases_present(self):
        from openenv.server import environment as env_mod
        for key in ("format", "validity", "structural", "text_block", "position", "color", "clip", "ssim"):
            assert key in env_mod.REWARD_WEIGHTS


# ═══════════════════════════════════════════════════════════════════════════════
# STABILITY TESTS — correlate actual vs expected scores for render+reference pairs
# ═══════════════════════════════════════════════════════════════════════════════

# ── HTML variant generation ───────────────────────────────────────────────────

import re as _re

_LAYOUT_PROPS = {
    "padding", "margin", "border-radius", "box-shadow", "display",
    "align-items", "justify-content", "min-height", "min-width",
    "position", "top", "right", "bottom", "left", "transform",
    "flex", "grid", "float", "overflow", "vertical-align",
    "width", "height", "box-sizing",
}


def make_variants(ref_html: str) -> dict[str, str]:
    """Generate 7 quality-level HTML variants from a reference HTML string."""
    minor = ref_html
    minor = _re.sub(r"(background(?:-color)?:\s*)(#[0-9a-fA-F]{6})", r"\g<1>#888888", minor, count=1)
    minor = _re.sub(r"font-size:(\d+)px",
                   lambda m: f"font-size:{max(8, int(m.group(1)) - 4)}px",
                   minor, count=2)

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

def scaffold_test_case(num: int, overwrite: bool = False) -> Path:
    difficulty, idx = CASE_SOURCES[num]
    case_dir = TESTS_DIR / str(num)
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "renders").mkdir(exist_ok=True)

    ref_html = (DATA_SRC / difficulty / f"{idx}.html").read_text()

    ref_dest = case_dir / "reference.html"
    if overwrite or not ref_dest.exists():
        ref_dest.write_text(ref_html)

    meta_dest = case_dir / "meta.json"
    if overwrite or not meta_dest.exists():
        meta_dest.write_text(json.dumps({"source": f"{difficulty}/{idx}", "difficulty": difficulty, "idx": idx}, indent=2))

    expected_dest = case_dir / "expected_scores.json"
    if overwrite or not expected_dest.exists():
        expected_dest.write_text(json.dumps(CANONICAL_EXPECTED, indent=2))

    variants_dir = case_dir / "variants"
    variants_dir.mkdir(exist_ok=True)
    for name, html in make_variants(ref_html).items():
        dest = variants_dir / f"{name}.html"
        if overwrite or not dest.exists():
            dest.write_text(html)

    return case_dir


def scaffold_all(overwrite: bool = False):
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    for num in range(15):
        scaffold_test_case(num, overwrite=overwrite)


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
        print("  reference render …", end=" ", flush=True)
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
        print(f"  [{name}] render …", end=" ", flush=True)
        img = _render_pw(html) or Image.new("RGB", (640, 480), (255, 255, 255))
        img.save(png_path)
        (renders_dir / f"{name}_blocks.json").write_text(json.dumps(_extract_blocks_pw(html)))
        print("ok")

    return True


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
    fmt    = format_reward(completions)[0]
    val    = html_validity_reward(completions)[0]
    struct = structural_similarity_reward(completions, solution=[ref_html])[0]
    col    = color_reward(completions, image=[ref_img], pred_image=[pred_img])[0]
    clip_s = clip_visual_reward(completions, image=[ref_img], pred_image=[pred_img])[0]
    ssim_s = ssim_reward(completions, image=[ref_img], pred_image=[pred_img])[0]
    tb     = _text_block_score(ref_blocks, pred_blocks)
    pos    = _position_score(ref_blocks, pred_blocks)
    scores = {"format": fmt, "validity": val, "structural": struct,
              "text_block": tb, "position": pos, "color": col, "clip": clip_s, "ssim": ssim_s}
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
        scores = score_variant(
            pred_html   = html_path.read_text() if html_path.exists() else "",
            ref_html    = ref_html,
            pred_img    = Image.open(png_path).convert("RGB"),
            ref_img     = ref_img,
            pred_blocks = json.loads(blk_path.read_text()) if blk_path.exists() else [],
            ref_blocks  = ref_blocks,
        )
        results[name] = scores
    return results


# ── Statistics ────────────────────────────────────────────────────────────────

def _spearman(x: list[float], y: list[float]) -> float:
    from scipy.stats import spearmanr
    if len(x) < 3:
        return 1.0
    rho, _ = spearmanr(x, y)
    return float(rho) if not math.isnan(rho) else 0.0


def _load_case_expected(num: int) -> dict[str, float]:
    p = TESTS_DIR / str(num) / "expected_scores.json"
    return json.loads(p.read_text()) if p.exists() else CANONICAL_EXPECTED.copy()


# ── Pytest stability tests ────────────────────────────────────────────────────

def _require_renders(num: int):
    if not (TESTS_DIR / str(num) / "renders" / "reference.png").exists():
        pytest.skip(f"Renders for case {num} not generated. Run: python tests/test_rewards.py --render")


@pytest.mark.parametrize("num", list(range(15)))
def test_stability_ordering(num: int):
    """Variants must be ordered: perfect > minor_diff > no_style > blank."""
    _require_renders(num)
    results = score_test_case(num)
    assert results is not None
    anchors = ["perfect", "minor_diff", "no_style", "blank"]
    tots = [results[v]["total"] for v in anchors if v in results]
    for i in range(len(tots) - 1):
        assert tots[i] >= tots[i+1] - 0.01, (
            f"Case {num}: ordering violated: "
            + " > ".join(f"{v}={t:.3f}" for v, t in zip(anchors, tots))
        )


@pytest.mark.parametrize("num", list(range(15)))
def test_stability_blank_near_zero(num: int):
    """Blank pages must score below 0.05."""
    _require_renders(num)
    results = score_test_case(num)
    assert results is not None
    blank_score = results.get("blank", {}).get("total", 0.0)
    assert blank_score <= MAX_BLANK_SCORE, f"Case {num}: blank score {blank_score:.3f} > {MAX_BLANK_SCORE}"


@pytest.mark.parametrize("num", list(range(15)))
def test_stability_perfect_score_high(num: int):
    """Perfect variants must score above 0.80."""
    _require_renders(num)
    results = score_test_case(num)
    assert results is not None
    perfect = results.get("perfect", {}).get("total", 0.0)
    assert perfect >= MIN_PERFECT_SCORE, f"Case {num}: perfect score {perfect:.3f} < {MIN_PERFECT_SCORE}"


@pytest.mark.parametrize("num", list(range(15)))
def test_stability_spearman_per_case(num: int):
    """Spearman ρ(actual, expected) ≥ 0.80 per case."""
    _require_renders(num)
    results = score_test_case(num)
    assert results is not None
    expected = _load_case_expected(num)
    actual  = [results[v]["total"]               for v in VARIANTS if v in results]
    target  = [expected.get(v, CANONICAL_EXPECTED[v]) for v in VARIANTS if v in results]
    rho = _spearman(actual, target)
    assert rho >= MIN_SPEARMAN_PER_CASE, f"Case {num}: Spearman ρ={rho:.3f} < {MIN_SPEARMAN_PER_CASE}"


def test_stability_global_spearman():
    """Global Spearman ρ across all scored (case, variant) pairs ≥ 0.85."""
    scored = {}
    for num in range(15):
        if (TESTS_DIR / str(num) / "renders" / "reference.png").exists():
            r = score_test_case(num)
            if r:
                scored[num] = r

    if not scored:
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


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestTextBlockRewardIntegration:
    def test_same_html_high_score(self):
        from openenv.server.rewards.text_block_rewards import text_block_reward
        scores = text_block_reward(_make_completion(SIMPLE_HTML), solution=[SIMPLE_HTML])
        assert scores[0] > 0.5

    def test_empty_pred_returns_zero(self):
        from openenv.server.rewards.text_block_rewards import text_block_reward
        assert text_block_reward(_make_completion(EMPTY_HTML), solution=[SIMPLE_HTML])[0] == 0.0


@pytest.mark.integration
class TestColorRewardIntegration:
    def test_same_render_high_score(self):
        from openenv.server.rewards.visual_rewards import _render_html
        ref_image = _render_html(SIMPLE_HTML)
        if ref_image is None:
            pytest.skip("Playwright not available")
        assert color_reward(_make_completion(SIMPLE_HTML), image=[ref_image])[0] > 0.7


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

METRIC_COLS = ["format", "validity", "structural", "text_block", "position", "color", "clip", "ssim", "total"]


def _print_case_table(num, results, expected, stats):
    meta = json.loads((TESTS_DIR / str(num) / "meta.json").read_text())
    print(f"\n{'─'*110}")
    print(f"  Case {num:2d}  [{meta['source']}]   "
          f"ρ={stats['spearman']:+.3f}  "
          f"{'PASS' if stats['pass'] else 'FAIL'}")
    print(f"{'─'*110}")
    header = f"  {'variant':<12}" + "".join(f" {c:>10}" for c in METRIC_COLS) + "  Δ(canon)"
    print(header)
    for v in VARIANTS:
        if v not in results:
            continue
        s = results[v]
        delta = s["total"] - CANONICAL_EXPECTED.get(v, 0)
        row = f"  {v:<12}" + "".join(f" {s.get(c, 0):>10.3f}" for c in METRIC_COLS) + f"  {delta:+.3f}"
        print(row)


def main():
    p = argparse.ArgumentParser(description="Reward stability test suite")
    p.add_argument("--render", action="store_true", help="Generate renders with Playwright")
    p.add_argument("--force",  action="store_true", help="Force re-render even if renders exist")
    p.add_argument("--update-expected", action="store_true", help="Write actual scores to expected_scores.json")
    p.add_argument("--cases", metavar="N,...", help="Comma-separated case numbers (default: all)")
    args = p.parse_args()

    case_nums = [int(x) for x in args.cases.split(",")] if args.cases else list(range(15))

    print("[SCAFFOLD] Checking data/tests/ structure …")
    for n in case_nums:
        scaffold_test_case(n)
    print("  Done.")

    if args.render or args.force:
        print(f"\n[RENDER] {len(case_nums)} cases …")
        for n in case_nums:
            meta = json.loads((TESTS_DIR / str(n) / "meta.json").read_text())
            print(f"\n── case {n:2d}  [{meta['source']}] ──")
            render_test_case(n, force=args.force)
        print("[RENDER] Done.")

    print(f"\n[SCORE] Scoring {len(case_nums)} cases …")
    all_results = {}
    case_expected = {}
    for n in case_nums:
        r = score_test_case(n)
        if r is None:
            print(f"  case {n:2d}: no renders — skipping")
            continue
        all_results[n] = r
        case_expected[n] = _load_case_expected(n)

    if not all_results:
        print("No results. Run --render first.")
        return

    all_actual, all_expected_flat = [], []
    per_case_stats = {}
    for num, results in all_results.items():
        stored = case_expected[num]
        actual = [results[v]["total"]                       for v in VARIANTS if v in results]
        target = [stored.get(v, CANONICAL_EXPECTED[v])     for v in VARIANTS if v in results]
        canon  = [CANONICAL_EXPECTED.get(v, 0)             for v in VARIANTS if v in results]
        rho_q = _spearman(actual, canon)
        blank_ok   = results.get("blank",   {}).get("total", 0.0) <= MAX_BLANK_SCORE
        perfect_ok = results.get("perfect", {}).get("total", 1.0) >= MIN_PERFECT_SCORE
        per_case_stats[num] = {
            "spearman": rho_q,
            "pass": rho_q >= MIN_SPEARMAN_PER_CASE and blank_ok and perfect_ok,
        }
        all_actual.extend(actual)
        all_expected_flat.extend(canon)

    for n in sorted(all_results):
        _print_case_table(n, all_results[n], case_expected[n], per_case_stats[n])

    global_rho = _spearman(all_actual, all_expected_flat)
    passes = sum(1 for s in per_case_stats.values() if s["pass"])
    print(f"\n{'═'*110}")
    print(f"  GLOBAL  ρ(quality)={global_rho:+.3f}  {passes}/{len(per_case_stats)} PASS")
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
