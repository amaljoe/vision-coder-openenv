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
from datetime import datetime
from pathlib import Path
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
DEBUG = bool(os.environ.get("DEBUG", ""))

FALLBACK_HTML = "<!DOCTYPE html><html><head></head><body><p>Generation failed.</p></body></html>"

# ---------------------------------------------------------------------------
# Episode debugger — writes a self-contained .md per episode when DEBUG=1
# ---------------------------------------------------------------------------

class EpisodeDebugger:
    """Logs the full Developer↔Critic conversation to outputs/<run>/<difficulty>.md.

    Images are saved as separate PNGs in outputs/<run>/images/ and referenced
    with relative paths — works in GitHub markdown and keeps the .md readable.
    Instantiate once per episode; call the log_* methods in order.
    """

    OUTPUT_DIR = Path("outputs")

    def __init__(self, run_id: str, difficulty: str, model: str):
        import base64 as _b64
        self._b64 = _b64
        self._run_id = run_id
        self._difficulty = difficulty
        self._model = model
        self._out = self.OUTPUT_DIR / run_id
        self._img_dir = self._out / "images"
        self._img_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._out / f"{difficulty}.md"
        self._f = self._path.open("w", encoding="utf-8")
        self._step = 0
        self._img_counter = 0
        self._write(
            f"# Episode: {difficulty}  \n"
            f"**Model:** `{model}`  **Run:** `{run_id}`  "
            f"**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

    # ------------------------------------------------------------------

    def log_reference(self, ref_b64: str) -> None:
        self._write("## Reference\n\n")
        self._write(self._save_img(ref_b64, "reference") + "\n\n---\n\n")

    def log_developer_input(self, current_html: str, critique: Optional[str]) -> None:
        self._step += 1
        self._write(f"## Step {self._step} — Developer\n\n")
        if critique:
            self._write(f"**Critic feedback received:**\n\n> {critique.strip()}\n\n")
        if current_html:
            self._write(
                f"**Previous HTML ({len(current_html)} chars):**\n\n"
                f"```html\n{current_html[:2000]}"
                f"{'…' if len(current_html) > 2000 else ''}\n```\n\n"
            )

    def log_developer_render_call(self, html: str, render_b64: str) -> None:
        self._write(
            f"**Developer called render_html** ({len(html)} chars):\n\n"
            f"```html\n{html[:1000]}{'…' if len(html) > 1000 else ''}\n```\n\n"
            f"Preview: {self._save_img(render_b64, f'step{self._step}_dev_preview')}\n\n"
        )

    def log_developer_output(self, html: str) -> None:
        self._write(
            f"**Developer final HTML ({len(html)} chars):**\n\n"
            f"```html\n{html[:3000]}{'…' if len(html) > 3000 else ''}\n```\n\n"
        )

    def log_step_result(self, reward: float, done: bool, render_full_b64: Optional[str], sub_rewards: Optional[dict] = None) -> None:
        self._write(f"**Reward: `{reward:.4f}`** | done: `{done}`\n\n")
        if sub_rewards:
            rows = " | ".join(f"{k}: {v:.3f}" for k, v in sub_rewards.items())
            self._write(f"*Sub-rewards:* {rows}\n\n")
        if render_full_b64:
            self._write(f"**Rendered output:**\n\n{self._save_img(render_full_b64, f'step{self._step}_rendered')}\n\n")

    def log_critic_input(self, ref_b64: str, render_prev_b64: Optional[str], critique_prev: Optional[str], render_curr_b64: str) -> None:
        self._write(f"### Critic\n\n**Reference:** {self._save_img(ref_b64, 'reference', dedup=True)}\n\n")
        if render_prev_b64 and critique_prev:
            self._write(
                f"**Previous render** *(after critique: \"{critique_prev[:120].strip()}\")*:\n\n"
                f"{self._save_img(render_prev_b64, f'step{self._step}_prev_render')}\n\n"
            )
        self._write(f"**Current render:** {self._save_img(render_curr_b64, f'step{self._step}_curr_render', dedup=True)}\n\n")

    def log_critic_output(self, critique: str) -> None:
        verdict = "✅ DONE" if "DONE" in critique else "🔁 Feedback"
        self._write(f"**Critic says ({verdict}):**\n\n> {critique.strip()}\n\n---\n\n")

    def log_summary(self, steps: int, score: float, rewards: List[float]) -> None:
        self._write(
            f"## Summary\n\n"
            f"- **Steps:** {steps}\n"
            f"- **Final score:** {score:.4f}\n"
            f"- **All rewards:** {', '.join(f'{r:.4f}' for r in rewards)}\n"
        )
        self._f.close()
        print(f"[DEBUG] Episode log → {self._path}", flush=True)

    # ------------------------------------------------------------------

    def _write(self, text: str) -> None:
        self._f.write(text)
        self._f.flush()

    def _save_img(self, b64: str, name: str, dedup: bool = False) -> str:
        """Save b64 PNG to images/<name>.png, return relative markdown image tag.

        dedup=True reuses the file path without re-saving (for repeated references
        to the same image, e.g. reference shown in both Developer and Critic sections).
        """
        fname = f"{self._difficulty}_{name}.png"
        fpath = self._img_dir / fname
        if not dedup or not fpath.exists():
            fpath.write_bytes(self._b64.b64decode(b64))
        rel = f"images/{fname}"
        return f"![{name}]({rel})"

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

def _handle_render_tool_call(tool_call, env_client, dbg: Optional[EpisodeDebugger] = None) -> dict:
    """Call /render and return the tool result message with the low-res image."""
    try:
        args = json.loads(tool_call.function.arguments)
        html = args.get("html", "")
        resp = env_client.post("/render", json={"html": html}, timeout=60)
        resp.raise_for_status()
        image_low_b64 = resp.json()["image_low_b64"]
        if dbg:
            dbg.log_developer_render_call(html, image_low_b64)
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
    dbg: Optional[EpisodeDebugger] = None,
) -> str:
    """Developer generates or refines HTML, optionally calling render tool to self-check."""
    if dbg:
        dbg.log_developer_input(current_html, critique)

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
                    messages.append(_handle_render_tool_call(tc, env_client, dbg))
            # Continue loop — model will see the render and produce final HTML
        else:
            html_out = choice.message.content or FALLBACK_HTML
            if dbg:
                dbg.log_developer_output(html_out)
            return html_out

    # Fallback if tool loop exhausted without text output
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=4096,
        temperature=0.7,
    )
    html_out = response.choices[0].message.content or FALLBACK_HTML
    if dbg:
        dbg.log_developer_output(html_out)
    return html_out


# ---------------------------------------------------------------------------
# Critic agent
# ---------------------------------------------------------------------------

def critic_turn(
    client: OpenAI,
    ref_b64: str,
    render_prev_b64: Optional[str],
    critique_prev: Optional[str],
    render_curr_b64: str,
    dbg: Optional[EpisodeDebugger] = None,
) -> str:
    """Critic compares reference vs current render and returns feedback or DONE."""
    if dbg:
        dbg.log_critic_input(ref_b64, render_prev_b64, critique_prev, render_curr_b64)

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
    critique_out = response.choices[0].message.content or ""
    if dbg:
        dbg.log_critic_output(critique_out)
    return critique_out


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference() -> None:
    import httpx

    llm = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    env_client = httpx.Client(base_url=SERVER_URL, timeout=180.0)
    all_rewards: List[float] = []

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for difficulty in TASKS:
        episode_rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        dbg: Optional[EpisodeDebugger] = (
            EpisodeDebugger(run_id, difficulty, MODEL_NAME) if DEBUG else None
        )

        log_start(task=difficulty, env=BENCHMARK, model=MODEL_NAME)

        try:
            resp = env_client.post("/reset", params={"difficulty": difficulty})
            resp.raise_for_status()
            obs = resp.json()
            session_id = obs["session_id"]
            ref_b64 = obs["screenshot_b64"]

            if dbg:
                dbg.log_reference(ref_b64)

            current_html = ""
            critique: Optional[str] = None
            render_prev: Optional[str] = None
            error_msg: Optional[str] = None

            for step_i in range(MAX_STEPS):
                error_msg = None

                # --- Developer turn ---
                try:
                    current_html = developer_turn(llm, env_client, ref_b64, current_html, critique, dbg)
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
                sub_rewards = result.get("metadata", {}).get("rewards")

                if dbg:
                    dbg.log_step_result(reward, done, render_full, sub_rewards)

                episode_rewards.append(reward)
                steps_taken = step_i + 1
                log_step(steps_taken, current_html, reward, done, error_msg)

                if done:
                    break

                # --- Critic turn ---
                try:
                    critique = critic_turn(llm, ref_b64, render_prev, critique, render_full, dbg)
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
            if dbg:
                dbg.log_summary(steps_taken, score, episode_rewards)
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
