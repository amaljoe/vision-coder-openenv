"""VisionCoder OpenEnv — inference script.

Runs one episode per task difficulty (easy / medium / hard), using an
OpenAI-compatible vision model to generate HTML from each screenshot.

Required environment variables:
  API_BASE_URL  — OpenAI-compatible LLM endpoint
  MODEL_NAME    — Model ID (must support vision/image inputs)
  HF_TOKEN      — Hugging Face / API key (also checked as API_KEY)

STDOUT FORMAT (mandatory):
  [START] task=<difficulty> env=vision-coder model=<model>
  [STEP]  step=<n> action=<truncated_html> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Usage:
  python inference.py
"""
from __future__ import annotations

import io
import logging
import os
import sys
import threading
import time
import urllib.request
from typing import List, Optional

import uvicorn
from openai import OpenAI

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — matches inference-sample.py conventions
# ---------------------------------------------------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or "sk-placeholder"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-VL-72B-Instruct"
SERVER_PORT = int(os.environ.get("INFERENCE_SERVER_PORT", "18080"))
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"

TASKS = ["easy", "medium", "hard"]
BENCHMARK = "vision-coder"
SUCCESS_SCORE_THRESHOLD = 0.1

SYSTEM_PROMPT = (
    "You are a UI-to-code expert. Given a screenshot of a web page, output ONLY the "
    "complete raw HTML with inline CSS that reproduces the layout as closely as possible. "
    "Do not include any markdown, explanations, or code fences — just the HTML."
)


# ---------------------------------------------------------------------------
# Logging helpers — mandatory stdout format
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Truncate action to avoid flooding stdout
    action_summary = action[:80].replace("\n", " ").strip() if action else "null"
    print(
        f"[STEP] step={step} action={action_summary} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Embedded environment server
# ---------------------------------------------------------------------------

def _start_server() -> None:
    from openenv.server.app import app
    config = uvicorn.Config(app, host="127.0.0.1", port=SERVER_PORT, log_level="error")
    uvicorn.Server(config).run()


def _wait_for_server(timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{SERVER_URL}/health", timeout=2)
            return
        except Exception:
            time.sleep(1.0)
    raise RuntimeError(f"Environment server did not start within {timeout}s")


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _generate_html(client: OpenAI, screenshot_b64: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
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

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    env_client = httpx.Client(base_url=SERVER_URL, timeout=180.0)

    all_rewards: List[float] = []

    for difficulty in TASKS:
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False
        error_msg: Optional[str] = None

        log_start(task=difficulty, env=BENCHMARK, model=MODEL_NAME)

        try:
            resp = env_client.post("/reset", params={"difficulty": difficulty})
            resp.raise_for_status()
            obs = resp.json()

            screenshot_b64 = obs.get("screenshot_b64", "")
            prompt = obs.get("prompt", "")

            try:
                html = _generate_html(client, screenshot_b64, prompt)
            except Exception as exc:
                print(f"[DEBUG] LLM call failed: {exc}", flush=True)
                html = "<!DOCTYPE html><html><head></head><body><p>Generation failed.</p></body></html>"
                error_msg = str(exc)[:120]

            step_resp = env_client.post("/step", json={"html": html})
            step_resp.raise_for_status()
            result = step_resp.json()

            reward = result.get("reward", 0.0)
            rewards.append(reward)
            steps_taken = 1
            all_rewards.append(reward)

            log_step(
                step=1,
                action=html,
                reward=reward,
                done=True,
                error=error_msg,
            )

            score = reward  # single-step episode; reward already in [0, 1]
            success = score >= SUCCESS_SCORE_THRESHOLD

        except Exception as exc:
            error_msg = str(exc)[:120]
            print(f"[DEBUG] Episode error: {exc}", flush=True)
            if not rewards:
                rewards.append(0.0)
                steps_taken = 1
            score = 0.0
            success = False

        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    env_client.close()
    mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    print(f"\nMean reward across {len(TASKS)} tasks: {mean_reward:.4f}", flush=True)


def main() -> None:
    t = threading.Thread(target=_start_server, daemon=True)
    t.start()

    print("Waiting for environment server to start …", flush=True)
    try:
        _wait_for_server()
    except RuntimeError as exc:
        print(f"[DEBUG] Server startup failed: {exc}", flush=True)
        # Emit failed END lines for all tasks so the evaluator sees output
        for difficulty in TASKS:
            log_start(task=difficulty, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.0, rewards=[0.0])
        sys.exit(1)

    print("Server ready.", flush=True)

    try:
        run_inference()
    except Exception as exc:
        print(f"[DEBUG] Unhandled error in run_inference: {exc}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
