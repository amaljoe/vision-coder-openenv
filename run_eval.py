"""
Evaluate qwen3.5:4b on all 5 samples per difficulty.
Records reward scores, finds best representative samples, and saves PNG pairs.
"""
import json
import base64
import time
import httpx
from openai import OpenAI
from playwright.sync_api import sync_playwright
from PIL import Image
import io
import os

BASE_URL = "http://127.0.0.1:18090"
OLLAMA_BASE = "http://localhost:11434/v1"
MODEL = "qwen3.5:4b"
ASSETS_DIR = "/Users/amaljoe/Desktop/Workspace/AI/vision-coder-openenv/assets"
DATA_DIR = "/Users/amaljoe/Desktop/Workspace/AI/vision-coder-openenv/data"

os.makedirs(ASSETS_DIR, exist_ok=True)

client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE)

SYSTEM_PROMPT = (
    "You are a UI-to-code expert. Given a screenshot of a web page, output ONLY "
    "the complete raw HTML with inline CSS that reproduces the layout as closely as possible. "
    "Do not include any markdown, explanations, or code fences — just the HTML."
)


def render_html(html: str) -> bytes:
    """Render HTML to 640x480 PNG via Playwright."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 640, "height": 480})
        page.set_content(html, wait_until="networkidle")
        png = page.screenshot(full_page=False)
        browser.close()
    return png


def reset_to_sample(difficulty: str, target_idx: int) -> dict:
    """Reset server to a specific sample index by calling reset multiple times."""
    # First get current state
    r = httpx.get(f"{BASE_URL}/state", timeout=30)
    state = r.json()

    # Call reset with difficulty to cycle through
    # The server increments by 1 each time, wrapping around 5
    # We need to call it enough times to land on target_idx
    # Strategy: call reset 5 times to ensure we know position, then target

    # Reset once to get current index
    result = httpx.post(f"{BASE_URL}/reset", json={"difficulty": difficulty}, timeout=60).json()
    current_idx = result["metadata"]["sample_index"]

    # How many more resets needed to reach target_idx?
    # current is at current_idx, next reset will be (current_idx + 1) % 5
    # We want to be at target_idx after resets
    needed = (target_idx - (current_idx + 1)) % 5
    for _ in range(needed):
        result = httpx.post(f"{BASE_URL}/reset", json={"difficulty": difficulty}, timeout=60).json()

    # Now do final reset to land on target
    if needed > 0:
        result = httpx.post(f"{BASE_URL}/reset", json={"difficulty": difficulty}, timeout=60).json()

    actual_idx = result["metadata"]["sample_index"]
    if actual_idx != target_idx:
        print(f"  WARNING: wanted idx={target_idx}, got idx={actual_idx}")

    return result


def call_qwen(screenshot_b64: str) -> str:
    """Call qwen3.5:4b with a screenshot and return generated HTML."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                {"type": "text", "text": "Generate the HTML for this page."}
            ]}
        ],
        max_tokens=4096,
        temperature=0.2,
    )
    return response.choices[0].message.content


def get_step_rewards(html: str) -> dict:
    """Submit HTML to /step and get reward breakdown."""
    r = httpx.post(f"{BASE_URL}/step", json={"html": html}, timeout=120)
    result = r.json()
    return {
        "total": result["reward"],
        "detail": result["metadata"]["rewards"]
    }


def evaluate_all_samples():
    """Run qwen3.5:4b on all 5 samples per difficulty, record rewards."""
    difficulties = ["easy", "medium", "hard"]
    all_results = {}

    for difficulty in difficulties:
        print(f"\n{'='*60}")
        print(f"Evaluating {difficulty} samples...")
        print('='*60)

        samples_results = []

        for idx in range(5):
            print(f"\n  Sample {idx}:")

            # Reset to this specific sample
            print(f"    Resetting to sample {idx}...")
            obs = reset_to_sample(difficulty, idx)
            actual_idx = obs["metadata"]["sample_index"]
            screenshot_b64 = obs["screenshot_b64"]

            print(f"    Got sample_index={actual_idx}")

            # Call qwen
            print(f"    Calling qwen3.5:4b...")
            try:
                html = call_qwen(screenshot_b64)
                print(f"    Got HTML ({len(html)} chars)")
            except Exception as e:
                print(f"    ERROR calling qwen: {e}")
                html = "<html><body><p>Error</p></body></html>"

            # Get rewards via /step
            print(f"    Getting rewards...")
            try:
                rewards = get_step_rewards(html)
                print(f"    Total reward: {rewards['total']:.4f}")
                print(f"    Detail: {rewards['detail']}")
            except Exception as e:
                print(f"    ERROR getting rewards: {e}")
                rewards = {"total": 0.0, "detail": {}}

            samples_results.append({
                "sample_index": actual_idx,
                "screenshot_b64": screenshot_b64,
                "generated_html": html,
                "total_reward": rewards["total"],
                "detail_rewards": rewards["detail"],
            })

        all_results[difficulty] = samples_results

        # Print summary for this difficulty
        print(f"\n  {difficulty} summary:")
        for r in samples_results:
            print(f"    idx={r['sample_index']} total={r['total_reward']:.4f}")

    return all_results


def find_best_samples(all_results: dict) -> dict:
    """Find best representative sample per difficulty.

    Goal: easy_mean > medium_mean > hard_mean
    Best sample = one closest to expected difficulty ordering.
    """
    best = {}

    # Compute means per difficulty
    means = {}
    for diff, samples in all_results.items():
        means[diff] = sum(s["total_reward"] for s in samples) / len(samples)

    print(f"\nMean rewards: easy={means.get('easy',0):.4f} medium={means.get('medium',0):.4f} hard={means.get('hard',0):.4f}")

    # For "best representative", pick sample closest to the mean for its difficulty
    # but also considering it satisfies easy > medium > hard ordering
    for diff, samples in all_results.items():
        mean = means[diff]
        # Pick sample closest to mean
        best_sample = min(samples, key=lambda s: abs(s["total_reward"] - mean))
        best[diff] = best_sample
        print(f"Best {diff} sample: idx={best_sample['sample_index']} total={best_sample['total_reward']:.4f}")

    return best


def save_assets(best_samples: dict):
    """Save reference and qwen PNG pairs to assets/."""
    for diff, sample in best_samples.items():
        print(f"\nSaving assets for {diff}...")

        # Save reference screenshot
        ref_png_data = base64.b64decode(sample["screenshot_b64"])
        ref_path = os.path.join(ASSETS_DIR, f"ref_{diff}.png")
        with open(ref_path, "wb") as f:
            f.write(ref_png_data)
        print(f"  Saved {ref_path}")

        # Render qwen HTML to PNG
        print(f"  Rendering qwen HTML...")
        try:
            qwen_png_data = render_html(sample["generated_html"])
            qwen_path = os.path.join(ASSETS_DIR, f"qwen_{diff}.png")
            with open(qwen_path, "wb") as f:
                f.write(qwen_png_data)
            print(f"  Saved {qwen_path}")
        except Exception as e:
            print(f"  ERROR rendering qwen HTML: {e}")


def main():
    print("Starting evaluation of qwen3.5:4b on all samples...")
    print(f"Server: {BASE_URL}")
    print(f"Model: {MODEL}")

    # Check server health
    r = httpx.get(f"{BASE_URL}/health", timeout=10)
    print(f"Server health: {r.json()}")

    # Run evaluation
    all_results = evaluate_all_samples()

    # Find best representative samples
    print("\n" + "="*60)
    print("Finding best representative samples...")
    best_samples = find_best_samples(all_results)

    # Save assets
    print("\n" + "="*60)
    print("Saving assets...")
    save_assets(best_samples)

    # Build results summary
    results_json = {
        "model": MODEL,
        "all_samples": {
            diff: [
                {
                    "sample_index": s["sample_index"],
                    "total_reward": s["total_reward"],
                    "detail_rewards": s["detail_rewards"],
                }
                for s in samples
            ]
            for diff, samples in all_results.items()
        },
        "best_samples": {
            diff: {
                "sample_index": s["sample_index"],
                "total_reward": s["total_reward"],
                "detail_rewards": s["detail_rewards"],
            }
            for diff, s in best_samples.items()
        }
    }

    results_path = os.path.join(ASSETS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)

    ordering_ok = True
    totals = {}

    for diff in ["easy", "medium", "hard"]:
        s = best_samples[diff]
        d = s["detail_rewards"]
        total = s["total_reward"]
        totals[diff] = total

        fmt = d.get("format", "N/A")
        val = d.get("validity", "N/A")
        struct = d.get("structural", "N/A")
        tb = d.get("text_block", "N/A")
        pos = d.get("position", "N/A")
        col = d.get("color", "N/A")
        clip = d.get("clip", "N/A")

        def fmt_val(v):
            if isinstance(v, float):
                return f"{v:.3f}"
            return str(v)

        print(f"{diff:6s} sample={s['sample_index']} total={total:.3f} "
              f"format={fmt_val(fmt)} validity={fmt_val(val)} structural={fmt_val(struct)} "
              f"text_block={fmt_val(tb)} position={fmt_val(pos)} color={fmt_val(col)} clip={fmt_val(clip)}")

    print()
    easy_t = totals.get("easy", 0)
    med_t = totals.get("medium", 0)
    hard_t = totals.get("hard", 0)

    if easy_t > med_t > hard_t:
        print(f"✓ Ordering confirmed: easy({easy_t:.3f}) > medium({med_t:.3f}) > hard({hard_t:.3f})")
    else:
        print(f"✗ Ordering NOT as expected: easy={easy_t:.3f} medium={med_t:.3f} hard={hard_t:.3f}")
        print(f"  (easy > medium: {easy_t > med_t}, medium > hard: {med_t > hard_t})")

    return results_json


if __name__ == "__main__":
    main()
