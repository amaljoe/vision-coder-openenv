"""VisionCoder OpenEnv — baseline inference script.

Runs one episode per task difficulty (easy / medium / hard), using an
OpenAI-compatible vision model to generate HTML from each screenshot.

Required environment variables:
  API_BASE_URL  — OpenAI-compatible LLM endpoint
  MODEL_NAME    — Model ID (must support vision/image inputs)
  HF_TOKEN      — Hugging Face token (for WebSight dataset download)

Output format (strictly followed for automated scoring):
  [START] episode_id=<uuid> difficulty=<level>
  [STEP]  step=1 reward=<float> format=<f> validity=<f> structural=<f> clip=<f>
  [END]   episode_id=<uuid> total_reward=<float>

Usage:
  python inference.py
"""
from __future__ import annotations

import base64
import io
import logging
import os
import sys
import threading
import time
import urllib.request

import uvicorn
from openai import OpenAI

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
SERVER_PORT = int(os.environ.get("INFERENCE_SERVER_PORT", "18080"))
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"

TASKS = ["easy", "medium", "hard"]
SYSTEM_PROMPT = (
    "You are a UI-to-code expert. Given a screenshot of a web page, output ONLY the "
    "complete raw HTML with inline CSS that reproduces the layout as closely as possible. "
    "Do not include any markdown, explanations, or code fences — just the HTML."
)


# ---------------------------------------------------------------------------
# Embedded server startup
# ---------------------------------------------------------------------------

def _start_server():
    """Start the FastAPI environment server in a background thread."""
    from openenv.server.app import app  # noqa: import inside thread is fine

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=SERVER_PORT,
        log_level="error",
    )
    server = uvicorn.Server(config)
    server.run()


def _wait_for_server(timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{SERVER_URL}/health", timeout=2)
            return
        except Exception:
            time.sleep(1.0)
    raise RuntimeError(f"Server did not start within {timeout}s")


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _generate_html(client: OpenAI, screenshot_b64: str, prompt: str) -> str:
    """Call the vision LLM with the screenshot and return generated HTML."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot_b64}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        max_tokens=4096,
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference() -> None:
    import httpx

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-placeholder"), base_url=API_BASE_URL)
    env_client = httpx.Client(base_url=SERVER_URL, timeout=180.0)

    all_rewards: list[float] = []

    for difficulty in TASKS:
        # ---- reset ----
        resp = env_client.post("/reset", params={"difficulty": difficulty})
        resp.raise_for_status()
        obs = resp.json()

        episode_id = obs.get("metadata", {}).get("episode_id", "unknown")
        screenshot_b64 = obs.get("screenshot_b64", "")
        prompt = obs.get("prompt", "")

        print(f"[START] episode_id={episode_id} difficulty={difficulty}", flush=True)

        # ---- generate ----
        html = _generate_html(client, screenshot_b64, prompt)

        # ---- step ----
        step_resp = env_client.post("/step", json={"html": html})
        step_resp.raise_for_status()
        result = step_resp.json()

        rewards = result.get("metadata", {}).get("rewards", {})
        total = result.get("reward", 0.0)
        all_rewards.append(total)

        print(
            f"[STEP]  step=1 "
            f"reward={total:.4f} "
            f"format={rewards.get('format', 0.0):.4f} "
            f"validity={rewards.get('validity', 0.0):.4f} "
            f"structural={rewards.get('structural', 0.0):.4f} "
            f"clip={rewards.get('clip', 0.0):.4f}",
            flush=True,
        )
        print(f"[END]   episode_id={episode_id} total_reward={total:.4f}", flush=True)

    env_client.close()
    mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    print(f"\nMean reward across {len(TASKS)} tasks: {mean_reward:.4f}", flush=True)


def main() -> None:
    # Start environment server in background
    t = threading.Thread(target=_start_server, daemon=True)
    t.start()

    print("Waiting for environment server to start …", flush=True)
    _wait_for_server()
    print("Server ready.", flush=True)

    run_inference()


if __name__ == "__main__":
    main()
