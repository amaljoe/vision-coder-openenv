"""VisionCoder OpenEnv — Round 2 inference script.

Multi-step, multi-agent loop:
  Developer (fast, tool-calling) → step() → Critic (thinking, TODO-list) → repeat ≤ MAX_STEPS

Required environment variables:
  API_BASE_URL  — OpenAI-compatible LLM endpoint
  MODEL_NAME    — Model ID (must support vision + tool use)
  HF_TOKEN      — Hugging Face / API key

STDOUT FORMAT (mandatory):
  [START] task=<difficulty> env=vision-coder model=<model>
  [STEP]  step=<n> action=<truncated_html> reward=<0.00> done=<true|false> error=<msg|null>
  [CRITIC] step=<n> reward=<0.00> → <critique_preview>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""
from __future__ import annotations

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
MAX_STEPS = int(os.environ.get("MAX_STEPS", "5"))
DEBUG = bool(os.environ.get("DEBUG", ""))

# ---------------------------------------------------------------------------
# Episode debugger — writes a self-contained .md per episode when DEBUG=1
# ---------------------------------------------------------------------------

class EpisodeDebugger:
    """Logs the full Developer↔Critic conversation to outputs/<run>/<difficulty>.md.

    Images are saved as separate PNGs in outputs/<run>/images/ and referenced
    with relative paths — works in GitHub markdown and keeps the .md readable.
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
        self._write(
            f"# Episode: {difficulty}  \n"
            f"**Model:** `{model}`  **Run:** `{run_id}`  "
            f"**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

    def log_reference(self, ref_b64: str) -> None:
        self._write("## Reference\n\n")
        self._write(self._save_img(ref_b64, "reference") + "\n\n---\n\n")

    def log_developer_input(self, current_html: str, critique: Optional[str]) -> None:
        self._step += 1
        self._write(f"## Step {self._step} — Developer\n\n")
        if critique:
            self._write(f"**Critic feedback received:**\n\n```\n{critique.strip()}\n```\n\n")
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
                f"**Previous render** *(prior critique)*:\n\n"
                f"{self._save_img(render_prev_b64, f'step{self._step}_prev_render')}\n\n"
            )
        self._write(f"**Current render:** {self._save_img(render_curr_b64, f'step{self._step}_curr_render', dedup=True)}\n\n")

    def log_critic_output(self, critique: str, todo=None) -> None:
        from vcoder.agents import TodoList
        all_done = isinstance(todo, TodoList) and todo.all_done()
        pending = todo.pending_count() if isinstance(todo, TodoList) else None
        verdict = "✅ ALL DONE" if all_done else f"🔁 {pending} item(s) remaining" if pending is not None else "🔁 Feedback"
        self._write(f"**Critic says ({verdict}):**\n\n```\n{critique.strip()}\n```\n\n---\n\n")

    def log_summary(self, steps: int, score: float, rewards: List[float]) -> None:
        self._write(
            f"## Summary\n\n"
            f"- **Steps:** {steps}\n"
            f"- **Final score:** {score:.4f}\n"
            f"- **All rewards:** {', '.join(f'{r:.4f}' for r in rewards)}\n"
        )
        self._f.close()
        print(f"[DEBUG] Episode log → {self._path}", flush=True)

    def _write(self, text: str) -> None:
        self._f.write(text)
        self._f.flush()

    def _save_img(self, b64: str, name: str, dedup: bool = False) -> str:
        fname = f"{self._difficulty}_{name}.png"
        fpath = self._img_dir / fname
        if not dedup or not fpath.exists():
            fpath.write_bytes(self._b64.b64decode(b64))
        return f"![{name}](images/{fname})"


# ---------------------------------------------------------------------------
# Logging helpers (mandatory stdout format for evaluator)
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
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference() -> None:
    import httpx
    from vcoder.agents import AgentConfig, run_episode

    config = AgentConfig(
        api_key=API_KEY,
        api_base=API_BASE_URL,
        model=MODEL_NAME,
        max_steps=MAX_STEPS,
    )
    env_client = httpx.Client(base_url=SERVER_URL, timeout=180.0)
    all_rewards: List[float] = []

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for difficulty in TASKS:
        episode_rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False
        error_msg: Optional[str] = None

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

            def _on_step(sr) -> None:
                episode_rewards.append(sr.reward)
                nonlocal steps_taken
                steps_taken = sr.step
                log_step(sr.step, sr.html, sr.reward, sr.done, sr.error)

            run_episode(env_client, config, session_id, ref_b64, dbg, on_step=_on_step)

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
