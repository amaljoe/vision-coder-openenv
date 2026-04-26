"""Generate training_curve.png from assets/train.jsonl."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
DATA = ROOT / "assets" / "train.jsonl"
OUT  = ROOT / "assets" / "training_curve.png"


def smooth(vals, w=3):
    out = []
    for i in range(len(vals)):
        sl = vals[max(0, i - w):i + w + 1]
        out.append(sum(sl) / len(sl))
    return out


def main():
    rows = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]

    iters  = [r["iter"]   for r in rows]
    easy   = [r["easy"]   for r in rows]
    medium = [r["medium"] for r in rows]
    hard   = [r["hard"]   for r in rows]
    mean   = [r["mean"]   for r in rows]
    loss   = [r.get("loss") for r in rows]

    colors = {"easy": "#3b82f6", "medium": "#22c55e", "hard": "#ef4444", "mean": "#facc15"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), facecolor="#0d1117")
    fig.subplots_adjust(hspace=0.35)

    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.title.set_color("#e6edf3")
        ax.grid(True, alpha=0.15, color="#30363d")

    ax1.plot(iters, easy,          color=colors["easy"],   linewidth=1.4, label="Easy",   alpha=0.85)
    ax1.plot(iters, medium,        color=colors["medium"], linewidth=1.4, label="Medium", alpha=0.85)
    ax1.plot(iters, hard,          color=colors["hard"],   linewidth=1.4, label="Hard",   alpha=0.85)
    ax1.plot(iters, smooth(mean),  color=colors["mean"],   linewidth=2.2, linestyle="--", label="Mean (smoothed)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Training Reward Progression")
    ax1.legend(framealpha=0.2, labelcolor="white", facecolor="#161b22", edgecolor="#30363d")
    ax1.set_xlim(0, max(iters))

    loss_iters = [iters[i] for i, v in enumerate(loss) if v is not None]
    loss_vals  = [v for v in loss if v is not None]
    ax2.plot(loss_iters, loss_vals, color="#a78bfa", linewidth=1.4, marker="o", markersize=3, label="GRPO loss")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training Loss (GRPO)")
    ax2.legend(framealpha=0.2, labelcolor="white", facecolor="#161b22", edgecolor="#30363d")
    ax2.set_xlim(0, max(iters))

    plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
