"""Benchmark three inference approaches on 15 examples (5 per difficulty).

Approaches:
  A — Multi-agent: Developer (full-res ref + Critic TODO) + Critic (full-res ref + renders)
  B — Long-horizon Developer: full-res ref + all previous renders + all previous HTML, no Critic
  C — Short-horizon Developer: full-res ref + only last render + only last HTML, no Critic

Usage:
  export API_BASE_URL=http://localhost:8001/v1
  export MODEL_NAME=qwen35
  export HF_TOKEN=sk-local
  export MAX_STEPS=5
  export PLAYWRIGHT_BROWSERS_PATH=~/playwright-browsers
  cd ~/workspace/vision-coder-openenv
  /dev/shm/qwen35/bin/python benchmark.py 2>&1 | tee benchmark_results.txt
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import uvicorn

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or "sk-placeholder"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen3.5-35B-A3B"
SERVER_PORT = int(os.environ.get("INFERENCE_SERVER_PORT", "18080"))
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"
MAX_STEPS = int(os.environ.get("MAX_STEPS", "5"))
NUM_EPISODES = int(os.environ.get("NUM_EPISODES", "5"))   # per difficulty
DIFFICULTIES = ["easy", "medium", "hard"]

# Partial results written here after every episode so a crash loses nothing
_RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
PARTIAL_PATH = Path(f"benchmark_partial_{_RUN_ID}.json")
_partial: dict = {}  # keyed by "A/easy/1", etc.


def _flush_partial() -> None:
    PARTIAL_PATH.write_text(json.dumps(_partial, indent=2))


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
    raise RuntimeError(f"Env server did not start within {timeout}s")


def _run_approach(
    label: str,
    approach_id: str,
    env_client,
    config,
) -> Tuple[List[float], float]:
    """Run approach on all examples. Returns (rewards_per_episode, total_wall_time)."""
    from openenv.agents import run_episode, run_episode_long_dev, run_episode_short_dev, run_episode_d

    runners = {
        "A": run_episode,
        "B": run_episode_long_dev,
        "C": run_episode_short_dev,
        "D": run_episode_d,
    }
    runner = runners[approach_id]

    # Reset dataset indices so every approach sees the same samples
    env_client.post("/reset_dataset")

    all_rewards: List[float] = []
    approach_start = time.time()

    for difficulty in DIFFICULTIES:
        for ep in range(1, NUM_EPISODES + 1):
            ep_start = time.time()
            final_reward = 0.0
            all_step_rewards: List[float] = []

            try:
                resp = env_client.post("/reset", params={"difficulty": difficulty})
                resp.raise_for_status()
                obs = resp.json()
                session_id = obs["session_id"]
                ref_b64 = obs["screenshot_b64"]

                results = runner(
                    env_client, config, session_id, ref_b64,
                    dbg=None, on_step=None,
                )
                if results:
                    final_reward = results[-1].reward
                    all_step_rewards = [r.reward for r in results]

            except Exception as exc:
                print(f"[BENCH] ERROR approach={approach_id} difficulty={difficulty} ep={ep}: {exc}", flush=True)

            ep_time = time.time() - ep_start
            all_rewards.append(final_reward)

            # Save final render as PNG
            if results and results[-1].render_full_b64:
                import base64 as _b64
                renders_dir = Path("benchmark_renders") / _RUN_ID / approach_id
                renders_dir.mkdir(parents=True, exist_ok=True)
                png_path = renders_dir / f"{difficulty}_ep{ep}_r{final_reward:.3f}.png"
                png_path.write_bytes(_b64.b64decode(results[-1].render_full_b64))

            # Persist after every episode
            key = f"{approach_id}/{difficulty}/{ep}"
            _partial[key] = {
                "approach": approach_id,
                "difficulty": difficulty,
                "episode": ep,
                "final_reward": final_reward,
                "step_rewards": all_step_rewards,
                "time_s": round(ep_time, 2),
            }
            _flush_partial()

            print(
                f"[BENCH] approach={approach_id} difficulty={difficulty} ep={ep} "
                f"reward={final_reward:.4f} time={ep_time:.1f}s",
                flush=True,
            )

    total_time = time.time() - approach_start
    return all_rewards, total_time


def _print_table(results: Dict[str, Tuple[str, List[float], float]]) -> None:
    """Print per-approach summary and per-difficulty breakdown."""
    print("\n" + "=" * 70, flush=True)
    print("BENCHMARK RESULTS", flush=True)
    print("=" * 70, flush=True)

    header = f"{'Approach':<32} {'Mean Rwd':>9} {'Total Rwd':>10} {'Time':>8}"
    print(header, flush=True)
    print("-" * 70, flush=True)

    for approach_id, (label, rewards, total_time) in results.items():
        mean_r = sum(rewards) / len(rewards) if rewards else 0.0
        total_r = sum(rewards)
        print(
            f"  {label:<30} {mean_r:>9.4f} {total_r:>10.4f} {total_time:>7.1f}s",
            flush=True,
        )

    print("\nPer-difficulty breakdown (mean reward):", flush=True)
    print(f"{'Approach':<32} {'easy':>8} {'medium':>8} {'hard':>8}", flush=True)
    print("-" * 70, flush=True)

    for approach_id, (label, rewards, _) in results.items():
        # rewards: [easy×5, medium×5, hard×5]
        easy_r   = sum(rewards[0:5])   / 5 if len(rewards) >= 5  else 0.0
        medium_r = sum(rewards[5:10])  / 5 if len(rewards) >= 10 else 0.0
        hard_r   = sum(rewards[10:15]) / 5 if len(rewards) >= 15 else 0.0
        print(
            f"  {label:<30} {easy_r:>8.4f} {medium_r:>8.4f} {hard_r:>8.4f}",
            flush=True,
        )

    print("=" * 70, flush=True)


def main() -> None:
    import httpx
    from openenv.agents import AgentConfig

    # Start env server only if not already running
    try:
        urllib.request.urlopen(f"{SERVER_URL}/health", timeout=2)
        print("Environment server already running — skipping startup.", flush=True)
    except Exception:
        t = threading.Thread(target=_start_server, daemon=True)
        t.start()
        print("Waiting for environment server …", flush=True)
        try:
            _wait_for_server()
        except RuntimeError as exc:
            print(f"[BENCH] Server startup failed: {exc}", flush=True)
            sys.exit(1)
        print("Server ready.", flush=True)

    config = AgentConfig(
        api_key=API_KEY,
        api_base=API_BASE_URL,
        model=MODEL_NAME,
        max_steps=MAX_STEPS,
    )
    env_client = httpx.Client(base_url=SERVER_URL, timeout=180.0)

    _only = os.getenv("ONLY_APPROACHES", "").upper().split(",") if os.getenv("ONLY_APPROACHES") else None
    APPROACHES = [
        (aid, lbl) for aid, lbl in [
            ("A", "A: Multi-agent (Dev+Critic)"),
            ("B", "B: Long-horizon Developer"),
            ("C", "C: Short-horizon Developer"),
            ("D", "D: LongDev(low-res)+SimpleCritic"),
        ] if _only is None or aid in _only
    ]

    print(f"[BENCH] Partial results → {PARTIAL_PATH} (flushed after every episode)", flush=True)

    results: Dict[str, Tuple[str, List[float], float]] = {}
    for approach_id, label in APPROACHES:
        print(f"\n{'='*60}", flush=True)
        print(f"Running approach {label}  ({NUM_EPISODES} eps × {len(DIFFICULTIES)} difficulties = {NUM_EPISODES * len(DIFFICULTIES)} episodes)", flush=True)
        print(f"{'='*60}", flush=True)
        rewards, total_time = _run_approach(label, approach_id, env_client, config)
        results[approach_id] = (label, rewards, total_time)
        mean_r = sum(rewards) / len(rewards) if rewards else 0.0
        print(f"[BENCH] Approach {approach_id} done — mean={mean_r:.4f} time={total_time:.1f}s", flush=True)

    env_client.close()
    _print_table(results)

    # Write final summary JSON
    summary_path = Path(f"benchmark_summary_{_RUN_ID}.json")
    summary = {
        approach_id: {
            "label": label,
            "mean_reward": round(sum(rw) / len(rw), 6) if rw else 0.0,
            "total_reward": round(sum(rw), 6),
            "total_time_s": round(tt, 2),
            "rewards": rw,
        }
        for approach_id, (label, rw, tt) in results.items()
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[BENCH] Final summary → {summary_path}", flush=True)


if __name__ == "__main__":
    main()
