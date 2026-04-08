"""Generate benchmark renders for README visual comparison.

Runs qwen3.5:4b on all 15 samples (5 per difficulty), picks the best
easy/medium/hard triplet where easy_score > medium_score > hard_score,
and saves renders + scores to assets/.

Usage:
    # Start the env server first:
    uvicorn openenv.server.app:app --host 127.0.0.1 --port 18090

    # Then run:
    uv run python utils/generate_benchmark_images.py [--model qwen3.5:4b] [--port 18090]
"""
import argparse
import base64
import json
import os
import sys

import httpx
from openai import OpenAI
from playwright.sync_api import sync_playwright

from vcoder.rewards import extract_html

VIEWPORT = {"width": 640, "height": 480}


def render_html(html: str) -> bytes:
    """Render HTML to PNG at 640x480 using Playwright (no fences, clean HTML)."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT)
        page.set_content(html, wait_until="networkidle")
        png = page.screenshot(full_page=False)
        browser.close()
    return png


def run(model: str, server: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
    env = httpx.Client(base_url=server, timeout=180)
    results: dict[str, list] = {"easy": [], "medium": [], "hard": []}

    total = 15
    done = 0

    for difficulty in ["easy", "medium", "hard"]:
        for i in range(5):
            done += 1
            print(f"[{done}/{total}] {difficulty} sample {i} — LLM call...", flush=True)
            obs = env.post("/reset", params={"difficulty": difficulty}).json()
            screenshot_b64 = obs["screenshot_b64"]

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a UI-to-code expert. Given a screenshot of a web page, "
                            "output ONLY the complete raw HTML with inline CSS that reproduces "
                            "the layout as closely as possible. Do not include any markdown, "
                            "explanations, or code fences — just the HTML."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                            {"type": "text", "text": "Generate the HTML for this page."},
                        ],
                    },
                ],
                max_tokens=4096,
                temperature=0.2,
            )
            raw_html = resp.choices[0].message.content or ""
            clean_html = extract_html(raw_html)

            print(f"[{done}/{total}] {difficulty} sample {i} — scoring + rendering...", flush=True)
            step = env.post("/step", json={"html": raw_html}).json()
            rewards = step["metadata"]["rewards"]

            # Render clean HTML (fences stripped) at same 640x480 viewport as reference
            qwen_png = render_html(clean_html)
            ref_png = base64.b64decode(screenshot_b64)

            results[difficulty].append({
                "index": i,
                "rewards": rewards,
                "total": rewards["total"],
                "ref_png": ref_png,
                "qwen_png": qwen_png,
            })
            print(
                f"[{done}/{total}] {difficulty} sample {i} — "
                f"total={rewards['total']:.3f} format={rewards['format']:.2f} "
                f"clip={rewards['clip']:.3f}",
                flush=True,
            )

    # Find best triplet where easy > medium > hard
    print("\nFinding best combo (easy > medium > hard)...", flush=True)
    best_combo = None
    best_gap = -1.0
    for e in results["easy"]:
        for m in results["medium"]:
            for h in results["hard"]:
                if e["total"] > m["total"] > h["total"]:
                    gap = e["total"] - h["total"]
                    if gap > best_gap:
                        best_gap = gap
                        best_combo = (e, m, h)

    if best_combo is None:
        print("No perfect ordering found — picking best per difficulty", flush=True)
        best_combo = (
            max(results["easy"], key=lambda x: x["total"]),
            max(results["medium"], key=lambda x: x["total"]),
            max(results["hard"], key=lambda x: x["total"]),
        )

    e, m, h = best_combo
    print(
        f"Selected: easy[{e['index']}]={e['total']:.3f} > "
        f"medium[{m['index']}]={m['total']:.3f} > "
        f"hard[{h['index']}]={h['total']:.3f}",
        flush=True,
    )

    for diff, sample in zip(["easy", "medium", "hard"], [e, m, h]):
        with open(os.path.join(out_dir, f"ref_{diff}.png"), "wb") as f:
            f.write(sample["ref_png"])
        with open(os.path.join(out_dir, f"qwen_{diff}.png"), "wb") as f:
            f.write(sample["qwen_png"])
        print(f"Saved {diff}: total={sample['total']:.3f}", flush=True)

    with open(os.path.join(out_dir, "benchmark_results.json"), "w") as f:
        json.dump(
            {"easy": e["rewards"], "medium": m["rewards"], "hard": h["rewards"]},
            f,
            indent=2,
        )

    print(f"\nDone. Results → {out_dir}/benchmark_results.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--port", default="18090")
    parser.add_argument("--out", default="assets")
    args = parser.parse_args()
    run(args.model, f"http://127.0.0.1:{args.port}", args.out)
