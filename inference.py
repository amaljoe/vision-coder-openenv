"""VisionCoder OpenEnv — Round 2 inference script.

Multi-step, multi-agent loop:
  Developer (fast, tool-calling) → step() → Critic (thinking) → repeat ≤ MAX_STEPS

Required environment variables:
  API_BASE_URL  — OpenAI-compatible LLM endpoint
  MODEL_NAME    — Model ID (must support vision + tool use)
  HF_TOKEN      — Hugging Face / API key

STDOUT FORMAT (mandatory):
  [START] task=<difficulty> env=vision-coder model=<model>
  [STEP]  step=<n> action=<truncated_html> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""
from __future__ import annotations

import json
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
# Configuration
# ---------------------------------------------------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or "sk-placeholder"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen3.5-35B-A3B"
SERVER_PORT = int(os.environ.get("INFERENCE_SERVER_PORT", "18080"))
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"

TASKS = ["easy", "medium", "hard"]
BENCHMARK = "vision-coder"
SUCCESS_SCORE_THRESHOLD = 0.1
MAX_STEPS = int(os.environ.get("MAX_STEPS", "5"))  # max developer turns per episode

FALLBACK_HTML = "<!DOCTYPE html><html><head></head><body><p>Generation failed.</p></body></html>"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
DEVELOPER_SYSTEM = (
    "You are a UI-to-code expert. Given a reference screenshot of a web page, "
    "generate complete HTML with inline CSS that reproduces the layout as accurately as possible.\n\n"
    "You have access to a render_html tool. Call it to preview your HTML before finalising — "
    "compare the render to the reference and adjust before outputting your final answer.\n\n"
    "Output ONLY raw HTML — no markdown fences, no explanations."
)

CRITIC_SYSTEM = (
    "You are a precise UI reviewer. You will be shown a reference screenshot and the current "
    "rendered HTML output. Your job is to identify specific visual differences between them.\n\n"
    "Be concrete: name elements, colors, sizes, positions that differ. "
    "Focus on what matters most visually.\n\n"
    "If the render closely matches the reference with no significant differences, "
    "output exactly: DONE\n\n"
    "Otherwise output a concise, actionable critique the developer can act on."
)

RENDER_TOOL = {
    "type": "function",
    "function": {
        "name": "render_html",
        "description": "Render HTML to an image for visual self-check. Returns a low-res preview.",
        "parameters": {
            "type": "object",
            "properties": {
                "html": {"type": "string", "description": "Complete HTML to render"}
            },
            "required": ["html"],
        },
    },
}

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_summary = action[:80].replace("\n", " ").strip() if action else "null"
    print(
        f"[STEP] step={step} action={action_summary} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment server
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
# Developer agent
# ---------------------------------------------------------------------------

def _handle_render_tool_call(tool_call, env_client) -> dict:
    """Call /render and return the tool result message with the low-res image."""
    try:
        args = json.loads(tool_call.function.arguments)
        html = args.get("html", "")
        resp = env_client.post("/render", json={"html": html}, timeout=60)
        resp.raise_for_status()
        image_low_b64 = resp.json()["image_low_b64"]
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": [
                {"type": "text", "text": "Rendered. Compare this to the reference and adjust your HTML:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_low_b64}"}},
            ],
        }
    except Exception as exc:
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": f"Render failed: {exc}. Proceed with your best estimate.",
        }


def developer_turn(
    client: OpenAI,
    env_client,
    ref_b64: str,
    current_html: str,
    critique: Optional[str],
) -> str:
    """Developer generates or refines HTML, optionally calling render tool to self-check."""
    messages = [{"role": "system", "content": DEVELOPER_SYSTEM}]

    # Initial user message: reference image + task
    user_content = [
        {"type": "text", "text": "Reference screenshot (reproduce this UI):"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_b64}"}},
    ]
    if current_html and critique:
        user_content.append({
            "type": "text",
            "text": (
                f"\n\nYour previous HTML:\n```html\n{current_html[:3000]}\n```\n\n"
                f"Critic feedback to address:\n{critique}\n\n"
                "Revise the HTML to fix the issues above. "
                "Use render_html to preview your changes before outputting the final HTML."
            ),
        })
    else:
        user_content.append({
            "type": "text",
            "text": (
                "\n\nGenerate complete HTML with inline CSS to reproduce this screenshot. "
                "Use render_html to preview your output before finalising."
            ),
        })

    messages.append({"role": "user", "content": user_content})

    # Agentic loop: handle tool calls until the model outputs final HTML
    for _ in range(3):  # max 3 render tool calls per developer turn
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=[RENDER_TOOL],
            tool_choice="auto",
            max_tokens=4096,
            temperature=0.7,
        )
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                if tc.function.name == "render_html":
                    messages.append(_handle_render_tool_call(tc, env_client))
            # Continue loop — model will see the render and produce final HTML
        else:
            return choice.message.content or FALLBACK_HTML

    # Fallback if tool loop exhausted without text output
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=4096,
        temperature=0.7,
    )
    return response.choices[0].message.content or FALLBACK_HTML


# ---------------------------------------------------------------------------
# Critic agent
# ---------------------------------------------------------------------------

def critic_turn(
    client: OpenAI,
    ref_b64: str,
    render_prev_b64: Optional[str],
    critique_prev: Optional[str],
    render_curr_b64: str,
) -> str:
    """Critic compares reference vs current render and returns feedback or DONE."""
    content = [
        {"type": "text", "text": "Reference screenshot:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_b64}"}},
    ]

    if render_prev_b64 and critique_prev:
        content += [
            {"type": "text", "text": f"Previous render (after your last critique: \"{critique_prev[:200]}\"):"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{render_prev_b64}"}},
        ]

    content += [
        {"type": "text", "text": "Current render:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{render_curr_b64}"}},
        {
            "type": "text",
            "text": (
                "Compare the current render to the reference. "
                "List specific visual differences to fix. "
                "Output exactly DONE if the render is a close match."
            ),
        },
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": content},
        ],
        max_tokens=1024,
        temperature=0.1,  # critic should be precise, not creative
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference() -> None:
    import httpx

    llm = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    env_client = httpx.Client(base_url=SERVER_URL, timeout=180.0)
    all_rewards: List[float] = []

    for difficulty in TASKS:
        episode_rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        log_start(task=difficulty, env=BENCHMARK, model=MODEL_NAME)

        try:
            resp = env_client.post("/reset", params={"difficulty": difficulty})
            resp.raise_for_status()
            obs = resp.json()
            session_id = obs["session_id"]
            ref_b64 = obs["screenshot_b64"]

            current_html = ""
            critique: Optional[str] = None
            render_prev: Optional[str] = None
            error_msg: Optional[str] = None

            for step_i in range(MAX_STEPS):
                error_msg = None

                # --- Developer turn ---
                try:
                    current_html = developer_turn(llm, env_client, ref_b64, current_html, critique)
                except Exception as exc:
                    error_msg = str(exc)[:120]
                    current_html = FALLBACK_HTML

                # --- Step the environment ---
                step_resp = env_client.post(
                    "/step",
                    json={"html": current_html, "session_id": session_id},
                )
                step_resp.raise_for_status()
                result = step_resp.json()

                reward = float(result.get("reward", 0.0))
                done = bool(result.get("done", False))
                render_full = result.get("render_full")

                episode_rewards.append(reward)
                steps_taken = step_i + 1
                log_step(steps_taken, current_html, reward, done, error_msg)

                if done:
                    break

                # --- Critic turn ---
                try:
                    critique = critic_turn(llm, ref_b64, render_prev, critique, render_full)
                    if "DONE" in critique:
                        break
                except Exception as exc:
                    # Critic failure is non-fatal — continue without feedback
                    critique = None
                    logger.warning("Critic failed: %s", exc)

                render_prev = render_full

            score = episode_rewards[-1] if episode_rewards else 0.0
            success = score >= SUCCESS_SCORE_THRESHOLD

        except Exception as exc:
            error_msg = str(exc)[:120]
            print(f"[DEBUG] Episode error ({difficulty}): {exc}", flush=True)
            if not episode_rewards:
                episode_rewards.append(0.0)
                steps_taken = max(steps_taken, 1)
            score = 0.0
            success = False

        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=episode_rewards)
            all_rewards.extend(episode_rewards)

    env_client.close()
    mean = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    print(f"\nMean reward across {len(TASKS)} tasks: {mean:.4f}", flush=True)


def main() -> None:
    t = threading.Thread(target=_start_server, daemon=True)
    t.start()

    print("Waiting for environment server to start …", flush=True)
    try:
        _wait_for_server()
    except RuntimeError as exc:
        print(f"[DEBUG] Server startup failed: {exc}", flush=True)
        for difficulty in TASKS:
            log_start(task=difficulty, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=1, score=0.0, rewards=[0.0])
        sys.exit(1)

    print("Server ready.", flush=True)

    try:
        run_inference()
    except Exception as exc:
        print(f"[DEBUG] Unhandled error: {exc}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
