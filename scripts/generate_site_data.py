"""Generate docs/data.json and copy renders into docs/images/.

Computes all 8 sub-reward scores for every variant of every test case
using pre-rendered PNGs + pre-computed blocks.json — no Playwright needed.
"""
from __future__ import annotations

import json
import math
import pathlib
import shutil
import sys

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT = pathlib.Path(__file__).parent.parent

# ── reward imports ────────────────────────────────────────────────────
from openenv.server.rewards.format_rewards import format_reward
from openenv.server.rewards.validity_rewards import html_validity_reward
from openenv.server.rewards.structural_rewards import structural_similarity_reward
from openenv.server.rewards.color_rewards import color_reward
from openenv.server.rewards.visual_rewards import clip_visual_reward
from openenv.server.rewards.ssim_reward import ssim_reward

# ── constants (mirror environment.py) ────────────────────────────────
WEIGHTS = {
    "format": 0.5, "validity": 0.5, "structural": 0.5,
    "text_block": 3.0, "position": 1.0, "color": 1.5,
    "clip": 2.5, "ssim": 1.5,
}
WEIGHT_SUM = sum(WEIGHTS.values())
_VIEWPORT_W, _VIEWPORT_H = 640, 480
_VIEWPORT_DIAG = math.sqrt(_VIEWPORT_W**2 + _VIEWPORT_H**2)
_IOU_MATCH_THRESHOLD = 0.05

VARIANTS = ["perfect", "minor_diff", "bad_colors", "half_styled",
            "no_layout", "no_style", "blank"]
TASK_IDS = list(range(15))


# ── block-based rewards (no Playwright — use pre-computed blocks.json) ──

def _bbox_iou(a: dict, b: dict) -> float:
    ax1, ay1 = a["x"] - a["width"] / 2, a["y"] - a["height"] / 2
    ax2, ay2 = a["x"] + a["width"] / 2, a["y"] + a["height"] / 2
    bx1, by1 = b["x"] - b["width"] / 2, b["y"] - b["height"] / 2
    bx2, by2 = b["x"] + b["width"] / 2, b["y"] + b["height"] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _text_sim(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _score_blocks(ref_blocks: list, pred_blocks: list) -> tuple[float, float]:
    """Return (text_block_score, position_score) from pre-computed block lists."""
    if not ref_blocks:
        tb = 1.0 if not pred_blocks else 0.5
        pos = 1.0 if not pred_blocks else 0.5
        return tb, pos
    if not pred_blocks:
        return 0.0, 0.0

    n_ref, n_pred = len(ref_blocks), len(pred_blocks)
    iou_cost = np.zeros((n_ref, n_pred))
    dist_cost = np.zeros((n_ref, n_pred))

    for r, rb in enumerate(ref_blocks):
        ref_cx = rb["x"] + rb["width"] / 2
        ref_cy = rb["y"] + rb["height"] / 2
        for p, pb in enumerate(pred_blocks):
            iou_cost[r, p] = 1.0 - _bbox_iou(rb, pb)
            pred_cx = pb["x"] + pb["width"] / 2
            pred_cy = pb["y"] + pb["height"] / 2
            dist = math.sqrt((ref_cx - pred_cx) ** 2 + (ref_cy - pred_cy) ** 2)
            dist_cost[r, p] = dist / _VIEWPORT_DIAG

    row_ind, col_ind = linear_sum_assignment(iou_cost)

    # text_block score
    matched, text_scores = 0, []
    for r, p in zip(row_ind, col_ind):
        iou = 1.0 - iou_cost[r, p]
        if iou > _IOU_MATCH_THRESHOLD:
            matched += 1
            text_scores.append(_text_sim(ref_blocks[r]["text"], pred_blocks[p]["text"]))
    tb = 0.5 * (matched / n_ref) + 0.5 * (sum(text_scores) / n_ref if text_scores else 0.0)

    # position score (use same matching)
    row_ind2, col_ind2 = linear_sum_assignment(dist_cost)
    pos_scores = [1.0 - dist_cost[r, p] for r, p in zip(row_ind2, col_ind2)]
    if len(pos_scores) < n_ref:
        pos_scores += [0.0] * (n_ref - len(pos_scores))
    pos = max(0.0, sum(pos_scores) / n_ref)

    return tb, pos


def _load_blocks(path: pathlib.Path) -> list:
    if path.exists():
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    return []


# ── main ──────────────────────────────────────────────────────────────

def compute_rewards(
    variant_html: str,
    ref_html: str,
    ref_img: Image.Image,
    pred_img: Image.Image,
    ref_blocks: list,
    pred_blocks: list,
) -> dict:
    comp = [[{"content": variant_html}]]
    sol = [ref_html]
    imgs = [ref_img]
    pred_imgs = [pred_img]

    fmt   = format_reward(comp)[0]
    val   = html_validity_reward(comp)[0]
    struct = structural_similarity_reward(comp, solution=sol)[0]
    col   = color_reward(comp, image=imgs, pred_image=pred_imgs)[0]
    clip  = clip_visual_reward(comp, image=imgs, pred_image=pred_imgs)[0]
    ssim  = ssim_reward(comp, image=imgs, pred_image=pred_imgs)[0]
    tb, pos = _score_blocks(ref_blocks, pred_blocks)

    raw = (WEIGHTS["format"] * fmt + WEIGHTS["validity"] * val
           + WEIGHTS["structural"] * struct + WEIGHTS["text_block"] * tb
           + WEIGHTS["position"] * pos + WEIGHTS["color"] * col
           + WEIGHTS["clip"] * clip + WEIGHTS["ssim"] * ssim)

    # content multiplier (blank check on pred at 32×32)
    small = pred_img.resize((32, 32)).convert("RGB")
    arr = np.array(small)
    nonwhite = np.mean(arr < 240)
    ref_small = ref_img.resize((32, 32)).convert("RGB")
    ref_nonwhite = np.mean(np.array(ref_small) < 240)
    if ref_nonwhite > 0.01 and nonwhite < 0.005:
        multiplier = nonwhite / 0.005
        raw *= multiplier

    total = raw / WEIGHT_SUM
    return {
        "format": round(fmt, 4), "validity": round(val, 4),
        "structural": round(struct, 4), "text_block": round(tb, 4),
        "position": round(pos, 4), "color": round(col, 4),
        "clip": round(clip, 4), "ssim": round(ssim, 4),
        "total": round(total, 4),
    }


def run():
    docs_dir = ROOT / "docs"
    img_dir = docs_dir / "images" / "tests"
    img_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    for tid in TASK_IDS:
        test_dir = ROOT / "data" / "tests" / str(tid)
        renders_dir = test_dir / "renders"
        variants_dir = test_dir / "variants"

        meta = json.loads((test_dir / "meta.json").read_text())
        ref_html = (test_dir / "reference.html").read_text()

        ref_img_path = renders_dir / "reference.png"
        ref_img = Image.open(ref_img_path).convert("RGB")
        ref_blocks = _load_blocks(renders_dir / "reference_blocks.json")

        # Copy reference image
        case_img_dir = img_dir / str(tid)
        case_img_dir.mkdir(exist_ok=True)
        shutil.copy(ref_img_path, case_img_dir / "reference.png")

        print(f"\n[{tid}] {meta['difficulty']}/{meta['idx']}")
        variant_records = []

        for vname in VARIANTS:
            pred_png = renders_dir / f"{vname}.png"
            pred_html_path = variants_dir / f"{vname}.html"
            if not pred_png.exists() or not pred_html_path.exists():
                print(f"  skip {vname} (missing)")
                continue

            pred_img = Image.open(pred_png).convert("RGB")
            pred_html = pred_html_path.read_text()
            pred_blocks = _load_blocks(renders_dir / f"{vname}_blocks.json")

            rewards = compute_rewards(pred_html, ref_html, ref_img, pred_img,
                                      ref_blocks, pred_blocks)
            print(f"  {vname:12s}  total={rewards['total']:.3f}  "
                  f"clip={rewards['clip']:.2f}  ssim={rewards['ssim']:.2f}  "
                  f"tb={rewards['text_block']:.2f}")

            shutil.copy(pred_png, case_img_dir / f"{vname}.png")
            variant_records.append({
                "name": vname,
                "rewards": rewards,
                "image": f"images/tests/{tid}/{vname}.png",
                "html": pred_html,
            })

        # Sort descending by total (blank always last)
        variant_records.sort(key=lambda v: v["rewards"]["total"], reverse=True)

        cases.append({
            "id": tid,
            "difficulty": meta["difficulty"],
            "source": meta["source"],
            "reference_image": f"images/tests/{tid}/reference.png",
            "reference_html": ref_html,
            "variants": variant_records,
        })

    out = docs_dir / "data.json"
    out.write_text(json.dumps(cases, indent=2))
    print(f"\nWrote {out} ({out.stat().st_size // 1024} KB)")
    print(f"Images in {img_dir}")


if __name__ == "__main__":
    run()
