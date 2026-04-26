"""Generate assets/reward_grid.png and assets/reward_grid.jsonl.

Layout: 3 rows (easy/medium/hard) × 4 columns (reference + 3 randomly sampled
variants sorted descending by reward score). No column headers.
"""
import json
import random
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).parent.parent

TASKS = [
    {"difficulty": "Easy",   "task_id": 0},
    {"difficulty": "Medium", "task_id": 5},
    {"difficulty": "Hard",   "task_id": 10},
]
POOL = ["perfect", "minor_diff", "bad_colors", "half_styled", "no_layout", "no_style"]

BG   = "#0d1117"
TICK = "#8b949e"
ANNO = "#e6edf3"

def score_color(score):
    """Red (0) → yellow (0.5) → green (1)."""
    if score >= 0.7:
        r, g = int((1 - score) * 2 * 255), 180
    else:
        r, g = 220, int(score / 0.7 * 160)
    return (r / 255, g / 255, 60 / 255)


def run(seed: int = 42):
    rng = random.Random(seed)
    records = []

    for task in TASKS:
        tid = task["task_id"]
        scores_path = ROOT / f"data/tests/{tid}/expected_scores.json"
        scores = json.loads(scores_path.read_text())
        sampled = rng.sample(POOL, 3)
        variants = sorted(
            [{"name": v, "score": round(scores[v], 4)} for v in sampled],
            key=lambda x: x["score"],
            reverse=True,   # descending: best first
        )
        records.append({
            "difficulty": task["difficulty"],
            "task_id": tid,
            "variants": variants,
        })

    # Write JSONL
    out_jsonl = ROOT / "assets/reward_grid.jsonl"
    out_jsonl.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    print(f"Wrote {out_jsonl}")

    # ── Plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), facecolor=BG)
    fig.subplots_adjust(hspace=0.08, wspace=0.04)

    for row_idx, record in enumerate(records):
        tid = record["task_id"]
        renders_dir = ROOT / f"data/tests/{tid}/renders"

        # Column 0 — reference
        ax = axes[row_idx, 0]
        img = Image.open(renders_dir / "reference.png").convert("RGB")
        ax.imshow(np.array(img))
        ax.set_facecolor(BG)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
            spine.set_linewidth(0.8)
        # Row label as text inside the axis (top-left corner)
        ax.text(-0.18, 0.5, record["difficulty"], transform=ax.transAxes,
                color=ANNO, fontsize=13, fontweight="bold",
                va="center", ha="center", rotation=90)

        # Columns 1-3 — sampled variants descending
        for col_idx, variant in enumerate(record["variants"], start=1):
            ax = axes[row_idx, col_idx]
            png = renders_dir / f"{variant['name']}.png"
            img = Image.open(png).convert("RGB")
            ax.imshow(np.array(img))
            ax.set_facecolor(BG)
            ax.axis("off")
            color = score_color(variant["score"])
            # Score badge at bottom-center of image
            ax.text(0.5, -0.04, f"reward: {variant['score']:.2f}",
                    transform=ax.transAxes, color=color,
                    fontsize=10, fontweight="bold",
                    va="top", ha="center")

    plt.suptitle("Reward Function Evaluation — Reference vs Sampled Quality Levels",
                 color=ANNO, fontsize=12, y=0.998)

    out_png = ROOT / "assets/reward_grid.png"
    plt.savefig(out_png, dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    run()
