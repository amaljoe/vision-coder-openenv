"""Evaluate trained LoRA vs base model on the VisionCoder environment.

Usage:
    /dev/shm/qwen35/bin/python eval_lora.py \
        --lora-path checkpoints/run2/developer_final \
        --model ~/models/Qwen3.5-2B \
        --episodes 3

Prints per-difficulty rewards for base (LoRA disabled) and trained (LoRA enabled).
Starts its own environment server on INFERENCE_SERVER_PORT (default 18081).
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import statistics
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

import httpx
import torch
from PIL import Image

SERVER_PORT = int(os.environ.get("INFERENCE_SERVER_PORT", "18081"))
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"
DIFFICULTIES = ["easy", "medium", "hard"]
MAX_STEPS = 2


def _b64_to_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def run_episode_eval(model, processor, env_client: httpx.Client, difficulty: str, use_lora: bool) -> float:
    from qwen_vl_utils import process_vision_info

    device = str(next(model.parameters()).device)

    if use_lora:
        model.enable_adapter_layers()
    else:
        model.disable_adapter_layers()

    resp = env_client.post("/reset", params={"difficulty": difficulty})
    resp.raise_for_status()
    obs = resp.json()
    session_id = obs["session_id"]
    ref_image = _b64_to_image(obs["screenshot_b64"])

    best_reward = 0.0
    current_html = ""

    for _ in range(MAX_STEPS):
        messages = [
            {"role": "system", "content": (
                "You are a UI-to-code expert. Given a reference screenshot, "
                "generate complete HTML with inline CSS. Output ONLY raw HTML."
            )},
            {"role": "user", "content": [
                {"type": "image", "image": ref_image},
                {"type": "text", "text": (
                    "Generate complete HTML with inline CSS to reproduce this screenshot."
                    if not current_html else
                    f"Improve this HTML:\n```html\n{current_html[:2000]}\n```\nOutput revised HTML only."
                )},
            ]},
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            prompt_len = inputs["input_ids"].shape[1]
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        current_html = processor.decode(output_ids[0, prompt_len:], skip_special_tokens=True)

        step_resp = env_client.post("/step", json={"html": current_html, "session_id": session_id})
        step_resp.raise_for_status()
        result = step_resp.json()
        reward = float(result.get("reward", 0.0))
        best_reward = max(best_reward, reward)
        if result.get("done"):
            break

    return best_reward


def evaluate(model, processor, label: str, use_lora: bool, episodes: int, env_client: httpx.Client) -> dict:
    print(f"\n--- {label} (lora={'ON' if use_lora else 'OFF'}) ---")
    results = {}
    for diff in DIFFICULTIES:
        rewards = []
        for ep in range(episodes):
            r = run_episode_eval(model, processor, env_client, diff, use_lora)
            rewards.append(r)
            print(f"  {diff} ep={ep+1}: {r:.4f}")
        mean = statistics.mean(rewards)
        results[diff] = mean
        print(f"  {diff} mean: {mean:.4f}")
    overall = statistics.mean(results.values())
    print(f"  OVERALL mean: {overall:.4f}")
    results["mean"] = overall
    return results


def _start_server() -> None:
    from openenv.server.app import app
    import uvicorn
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
    raise RuntimeError(f"Server did not start within {timeout}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", required=True, help="Path to trained LoRA directory")
    parser.add_argument("--model", required=True, help="Base model path or ID")
    parser.add_argument("--episodes", type=int, default=2, help="Episodes per difficulty")
    args = parser.parse_args()

    lora_path = Path(args.lora_path)
    if not lora_path.exists():
        print(f"ERROR: LoRA path not found: {lora_path}")
        return

    # Start environment server
    print(f"Starting environment server on port {SERVER_PORT}...")
    t = threading.Thread(target=_start_server, daemon=True)
    t.start()
    _wait_for_server()
    print("Server ready.")

    env_client = httpx.Client(base_url=SERVER_URL, timeout=180.0)

    print(f"Loading model: {args.model}")
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from peft import PeftModel

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    print(f"Attaching LoRA from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, str(lora_path), is_trainable=False)
    model.eval()

    base_results = evaluate(model, processor, "BASE (no LoRA)", use_lora=False, episodes=args.episodes, env_client=env_client)
    trained_results = evaluate(model, processor, "TRAINED (with LoRA)", use_lora=True, episodes=args.episodes, env_client=env_client)

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Difficulty':<12} {'Base':>8} {'Trained':>10} {'Delta':>8} {'Δ%':>8}")
    print("-"*60)
    for diff in DIFFICULTIES + ["mean"]:
        b = base_results.get(diff, 0)
        t = trained_results.get(diff, 0)
        delta = t - b
        pct = (delta / b * 100) if b > 0 else 0
        symbol = "+" if delta >= 0 else ""
        print(f"{diff:<12} {b:>8.4f} {t:>10.4f} {symbol}{delta:>7.4f} {symbol}{pct:>6.1f}%")

    print("\n# Blog-ready markdown table:")
    print("| Difficulty | Base 2B | Trained 2B (GRPO) | Delta |")
    print("|---|---|---|---|")
    for diff in DIFFICULTIES + ["**mean**"]:
        key = diff.strip("*")
        b = base_results.get(key, 0)
        t = trained_results.get(key, 0)
        delta = t - b
        symbol = "+" if delta >= 0 else ""
        print(f"| {diff} | {b:.3f} | **{t:.3f}** | {symbol}{delta:.3f} |")

    # Save results JSON for blog update
    import json
    results = {
        "base": base_results,
        "trained": trained_results,
        "lora_path": str(lora_path),
        "model": args.model,
        "episodes": args.episodes,
    }
    out = Path("checkpoints/eval_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")

    env_client.close()


if __name__ == "__main__":
    main()
