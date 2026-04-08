"""Re-render qwen_hard.png with extract_html applied (strip fences)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from openai import OpenAI
from playwright.sync_api import sync_playwright
from vcoder.rewards import extract_html

VIEWPORT = {"width": 640, "height": 480}
SERVER = "http://127.0.0.1:18090"
MODEL = "qwen3.5:4b"
OUT = "assets/qwen_hard.png"

# Hard sample 4 — advance to index 4 by resetting 5 times
client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
env = httpx.Client(base_url=SERVER, timeout=180)

print("Advancing to hard sample 4...", flush=True)
obs = None
for i in range(5):
    obs = env.post("/reset", params={"difficulty": "hard"}).json()
    print(f"  reset {i} done", flush=True)

screenshot_b64 = obs["screenshot_b64"]
prompt = obs.get("prompt", "Generate the HTML for this page.")

print("Running LLM...", flush=True)
resp = client.chat.completions.create(
    model=MODEL,
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
                {"type": "text", "text": prompt},
            ],
        },
    ],
    max_tokens=4096,
    temperature=0.2,
)
raw_html = resp.choices[0].message.content or ""
clean_html = extract_html(raw_html)

print(f"Raw starts with: {raw_html[:60]!r}", flush=True)
print(f"Clean starts with: {clean_html[:60]!r}", flush=True)

print("Rendering...", flush=True)
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport=VIEWPORT)
    page.set_content(clean_html, wait_until="networkidle")
    png = page.screenshot(full_page=False)
    browser.close()

with open(OUT, "wb") as f:
    f.write(png)

print(f"Saved {OUT} ({len(png)} bytes)", flush=True)
env.close()
