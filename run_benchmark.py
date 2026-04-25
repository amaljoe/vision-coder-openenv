"""
Benchmark script: run qwen3.5:4b and nemotron-3-nano:4b on easy/medium/hard,
collect reward breakdowns, render to PNG, and save assets.
"""
from __future__ import annotations

import base64
import os
import json
import re
import time
import httpx
from openai import OpenAI
from playwright.sync_api import sync_playwright

SERVER = "http://127.0.0.1:18090"
OLLAMA_BASE = "http://localhost:11434/v1"
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

DIFFICULTIES = ["easy", "medium", "hard"]


def render_html_to_png(html: str) -> bytes:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 960})
        page.set_content(html, wait_until="networkidle")
        png = page.screenshot(full_page=False)
        browser.close()
    return png


def clean_html(text: str) -> str:
    """Strip markdown fences and thinking tags if present."""
    # Remove <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        # remove first line (```html or ```)
        lines = lines[1:]
        # remove last ``` if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def generate_html_qwen(screenshot_b64: str) -> str:
    client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE)
    response = client.chat.completions.create(
        model="qwen3.5:4b",
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
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                    },
                    {"type": "text", "text": "Generate the HTML for this page."},
                ],
            },
        ],
        max_tokens=4096,
        temperature=0.2,
    )
    return clean_html(response.choices[0].message.content or "")


def generate_html_nemotron(screenshot_b64: str, prompt: str) -> str:
    client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE)

    # Try vision first
    try:
        response = client.chat.completions.create(
            model="nemotron-3-nano:4b",
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
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                        },
                        {"type": "text", "text": "Generate the HTML for this page."},
                    ],
                },
            ],
            max_tokens=4096,
            temperature=0.2,
        )
        html = clean_html(response.choices[0].message.content or "")
        if html:
            print("  [nemotron] vision mode succeeded")
            return html
    except Exception as e:
        print(f"  [nemotron] vision failed ({e}), falling back to text-only")

    # Fallback: text-only
    response = client.chat.completions.create(
        model="nemotron-3-nano:4b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a UI-to-code expert. Output ONLY the complete raw HTML with "
                    "inline CSS that reproduces the described layout. Do not include any "
                    "markdown, explanations, or code fences — just the HTML."
                ),
            },
            {
                "role": "user",
                "content": f"Generate a complete HTML page with the following description: {prompt}",
            },
        ],
        max_tokens=4096,
        temperature=0.2,
    )
    return clean_html(response.choices[0].message.content or "")


def run_step(html: str) -> dict:
    resp = httpx.post(f"{SERVER}/step", json={"html": html}, timeout=120)
    resp.raise_for_status()
    result = resp.json()
    rewards = result.get("metadata", {}).get("rewards", {})
    return rewards


def main():
    results = {}

    for difficulty in DIFFICULTIES:
        print(f"\n{'='*60}")
        print(f"  Difficulty: {difficulty}")
        print(f"{'='*60}")

        # Reset and get reference screenshot
        print("  [reset] fetching reference screenshot...")
        resp = httpx.post(f"{SERVER}/reset", params={"difficulty": difficulty}, timeout=30)
        resp.raise_for_status()
        obs = resp.json()
        screenshot_b64 = obs["screenshot_b64"]
        prompt = obs.get("prompt", "")
        print(f"  [reset] got screenshot, prompt: {prompt[:80]}...")

        # Save reference PNG
        ref_png = base64.b64decode(screenshot_b64)
        ref_path = os.path.join(ASSETS_DIR, f"ref_{difficulty}.png")
        with open(ref_path, "wb") as f:
            f.write(ref_png)
        print(f"  [saved] {ref_path}")

        results[difficulty] = {}

        # --- Qwen ---
        print(f"\n  [qwen3.5:4b] generating HTML...")
        t0 = time.time()
        qwen_html = generate_html_qwen(screenshot_b64)
        print(f"  [qwen3.5:4b] generated in {time.time()-t0:.1f}s, {len(qwen_html)} chars")

        # Re-reset to ensure clean state for scoring
        resp = httpx.post(f"{SERVER}/reset", params={"difficulty": difficulty}, timeout=30)
        resp.raise_for_status()

        qwen_rewards = run_step(qwen_html)
        print(f"  [qwen3.5:4b] rewards: {json.dumps(qwen_rewards, indent=None)}")
        results[difficulty]["qwen"] = qwen_rewards

        # Render qwen HTML to PNG
        try:
            qwen_png = render_html_to_png(qwen_html if qwen_html else "<html><body></body></html>")
            qwen_path = os.path.join(ASSETS_DIR, f"qwen_{difficulty}.png")
            with open(qwen_path, "wb") as f:
                f.write(qwen_png)
            print(f"  [saved] {qwen_path}")
        except Exception as e:
            print(f"  [qwen] render failed: {e}")

        # --- Nemotron ---
        print(f"\n  [nemotron-3-nano:4b] generating HTML...")
        t0 = time.time()
        nemotron_html = generate_html_nemotron(screenshot_b64, prompt)
        print(f"  [nemotron-3-nano:4b] generated in {time.time()-t0:.1f}s, {len(nemotron_html)} chars")

        # Re-reset to ensure clean state for scoring
        resp = httpx.post(f"{SERVER}/reset", params={"difficulty": difficulty}, timeout=30)
        resp.raise_for_status()

        nemotron_rewards = run_step(nemotron_html)
        print(f"  [nemotron-3-nano:4b] rewards: {json.dumps(nemotron_rewards, indent=None)}")
        results[difficulty]["nemotron"] = nemotron_rewards

        # Render nemotron HTML to PNG
        try:
            nem_png = render_html_to_png(nemotron_html if nemotron_html else "<html><body></body></html>")
            nem_path = os.path.join(ASSETS_DIR, f"nemotron_{difficulty}.png")
            with open(nem_path, "wb") as f:
                f.write(nem_png)
            print(f"  [saved] {nem_path}")
        except Exception as e:
            print(f"  [nemotron] render failed: {e}")

    # Print summary
    print("\n\n" + "="*70)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*70)
    for diff in DIFFICULTIES:
        print(f"\n  {diff.upper()}")
        for model in ["qwen", "nemotron"]:
            r = results[diff].get(model, {})
            print(f"    {model:15s}: total={r.get('total',0):.4f}  "
                  f"fmt={r.get('format',0):.3f}  "
                  f"val={r.get('validity',0):.3f}  "
                  f"struct={r.get('structural',0):.3f}  "
                  f"text={r.get('text_block',0):.3f}  "
                  f"pos={r.get('position',0):.3f}  "
                  f"color={r.get('color',0):.3f}  "
                  f"clip={r.get('clip',0):.3f}")

    # Save results JSON
    results_path = os.path.join(ASSETS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    return results


if __name__ == "__main__":
    main()
