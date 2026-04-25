"""VisionCoder OpenEnv — Round 2 RL training.

Full-episode GRPO with shaped reward for Developer and Critic agents.
Alternating training phases: Developer (critic frozen) → Critic (developer frozen) → repeat.

Reward design:
  R_total(t) = R_terminal + λ · Σ(r_s - r_{s-1}  for s = t+1 .. n)
  λ = 0.2 — shaped signal stays subordinate to terminal reward

Usage:
  python train.py --phase developer --episodes 200 --k-rollouts 4
  python train.py --phase critic    --episodes 200 --k-rollouts 4
  python train.py --phase alternate --episodes-per-phase 200 --k-rollouts 4 --num-phases 4

Requirements:
  pip install peft transformers accelerate
  (torch already installed — CPU build for Mac, use GPU build for training)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

from openenv.prompts import DEVELOPER_TRAIN_SYSTEM, CRITIC_TRAIN_SYSTEM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("TRAIN_MODEL", "Qwen/Qwen3.5-9B")
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))
SERVER_PORT = int(os.environ.get("TRAIN_SERVER_PORT", "18081"))
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"
LAMBDA_SHAPED = 0.2   # weight for shaped improvement reward
MAX_STEPS = 5         # max developer turns per episode (must match environment.py)
DIFFICULTIES = ["easy", "medium", "hard"]

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

LR = 2e-5
MAX_GRAD_NORM = 1.0
MAX_NEW_TOKENS = 1024   # capped for training — reduces OOM in gradient forward pass
CRITIC_MAX_TOKENS = 512

DEVELOPER_SYSTEM = DEVELOPER_TRAIN_SYSTEM
CRITIC_SYSTEM = CRITIC_TRAIN_SYSTEM


class Phase(Enum):
    DEVELOPER = "developer"
    CRITIC = "critic"
    COMBINED = "combined"   # train both agents simultaneously


# ---------------------------------------------------------------------------
# Rollout data structures
# ---------------------------------------------------------------------------

@dataclass
class TurnData:
    """One agent turn: stored tokens + reward for log-prob recomputation."""
    phase: Phase                               # which agent generated this
    input_ids: torch.Tensor                    # prompt tokens [seq_len]
    pixel_values: Optional[torch.Tensor]       # image pixels (may be None)
    image_grid_thw: Optional[torch.Tensor]     # Qwen3-VL image grid positions
    mm_token_type_ids: Optional[torch.Tensor]  # Qwen3-VL multimodal token types
    generated_ids: torch.Tensor                # generated tokens [gen_len]
    text_output: str                           # decoded text
    reward_after: Optional[float] = None       # env reward after developer turn (None for critic turns)
    step_idx: int = 0


@dataclass
class EpisodeRollout:
    turns: List[TurnData] = field(default_factory=list)
    developer_rewards: List[float] = field(default_factory=list)  # one per developer turn

    @property
    def R_terminal(self) -> float:
        return self.developer_rewards[-1] if self.developer_rewards else 0.0


# ---------------------------------------------------------------------------
# Shaped return computation
# ---------------------------------------------------------------------------

def compute_step_returns(rewards: List[float], lambda_shaped: float = LAMBDA_SHAPED) -> List[float]:
    """Compute R_total for each developer step.

    R_total(t) = R_terminal + λ · Σ(r_s - r_{s-1}  for s = t+1 .. n)

    Telescope: Σ(delta_s for s=t+1..n) = r_n - r_t
    So R_total(t) = R_terminal + λ · (R_terminal - r_t)
    """
    R_terminal = rewards[-1]
    return [R_terminal + lambda_shaped * (R_terminal - r_t) for r_t in rewards]


def grpo_advantages(returns_per_rollout: List[List[float]]) -> List[List[float]]:
    """Group-relative advantage normalisation across K rollouts.

    For each step position, normalize across K rollout returns.
    """
    import numpy as np

    # Flatten all returns across rollouts and positions
    flat = [r for rollout in returns_per_rollout for r in rollout]
    if not flat:
        return returns_per_rollout

    mean_r = float(np.mean(flat))
    std_r = float(np.std(flat)) + 1e-8

    return [
        [(r - mean_r) / std_r for r in rollout]
        for rollout in returns_per_rollout
    ]


# ---------------------------------------------------------------------------
# Environment server
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def setup_model(model_name: str = MODEL_NAME):
    """Load Qwen3.5 VL with LoRA. Returns (model, processor)."""
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from peft import LoraConfig, get_peft_model, TaskType

    logger.info("Loading %s …", model_name)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Qwen3.5 VL: use Qwen3_5ForConditionalGeneration (handles pixel_values/image_grid_thw).
    # ignore_mismatched_sizes=True: some Q-proj weights differ between text/VL configs;
    # they're re-initialised from scratch — LoRA adapts them during training.
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()
    return model, processor


def _prepare_inputs(processor, messages: list, images: list, device: str) -> dict:
    """Apply chat template and processor (Qwen3-VL format), return input tensors."""
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in inputs.items()}


def _device(model) -> str:
    return next(model.parameters()).device


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def rollout_episode(
    model,
    processor,
    env_client,
    difficulty: str,
    training_phase: Phase,
) -> EpisodeRollout:
    """Collect one full episode (Developer + Critic alternating).

    During DEVELOPER training: LoRA ON for Developer, OFF for Critic.
    During CRITIC training:    LoRA OFF for Developer, ON for Critic.
    """
    import base64
    import io
    from PIL import Image

    device = str(_device(model))
    episode = EpisodeRollout()

    # Reset environment
    resp = env_client.post("/reset", params={"difficulty": difficulty})
    resp.raise_for_status()
    obs = resp.json()
    session_id = obs["session_id"]
    ref_b64 = obs["screenshot_b64"]
    ref_image = Image.open(io.BytesIO(base64.b64decode(ref_b64))).convert("RGB")

    current_html = ""
    critique: Optional[str] = None
    render_prev_b64: Optional[str] = None

    for step_i in range(MAX_STEPS):
        # --- Developer turn ---
        dev_messages = [{"role": "system", "content": DEVELOPER_SYSTEM}]
        user_content: list = [
            {"type": "image", "image": ref_image},
        ]
        if current_html and critique:
            user_content.append({
                "type": "text",
                "text": (
                    f"Revise your HTML to fix this critique:\n{critique}\n\n"
                    f"Previous HTML:\n```html\n{current_html[:2000]}\n```\n\n"
                    "Output only the revised raw HTML."
                ),
            })
        else:
            user_content.append({
                "type": "text",
                "text": "Generate complete HTML with inline CSS to reproduce this screenshot.",
            })
        dev_messages.append({"role": "user", "content": user_content})

        is_dev_trainable = training_phase in (Phase.DEVELOPER, Phase.COMBINED)
        if not is_dev_trainable:
            model.disable_adapter_layers()

        with torch.no_grad():
            inputs = _prepare_inputs(processor, dev_messages, [ref_image], device)
            prompt_len = inputs["input_ids"].shape[1]
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0, prompt_len:]
        current_html = processor.decode(generated_ids, skip_special_tokens=True)

        episode.turns.append(TurnData(
            phase=Phase.DEVELOPER,
            input_ids=inputs["input_ids"][0].cpu(),
            pixel_values=inputs.get("pixel_values", torch.empty(0)).cpu(),
            image_grid_thw=inputs.get("image_grid_thw", torch.empty(0)).cpu(),
            mm_token_type_ids=inputs.get("mm_token_type_ids", torch.empty(0)).cpu(),
            generated_ids=generated_ids.cpu(),
            text_output=current_html,
            step_idx=step_i,
        ))

        if not is_dev_trainable:
            model.enable_adapter_layers()

        # --- Step environment ---
        step_resp = env_client.post(
            "/step",
            json={"html": current_html, "session_id": session_id},
        )
        step_resp.raise_for_status()
        result = step_resp.json()
        reward = float(result.get("reward", 0.0))
        done = bool(result.get("done", False))
        render_full_b64 = result.get("render_full")

        episode.developer_rewards.append(reward)
        episode.turns[-1].reward_after = reward

        if done:
            break

        # --- Critic turn ---
        is_crit_trainable = training_phase in (Phase.CRITIC, Phase.COMBINED)
        if not is_crit_trainable:
            model.disable_adapter_layers()

        try:
            render_curr = Image.open(
                io.BytesIO(base64.b64decode(render_full_b64))
            ).convert("RGB") if render_full_b64 else None

            crit_messages = [{"role": "system", "content": CRITIC_SYSTEM}]
            crit_content: list = [
                {"type": "text", "text": "Reference:"},
                {"type": "image", "image": ref_image},
            ]
            if render_prev_b64:
                prev_img = Image.open(
                    io.BytesIO(base64.b64decode(render_prev_b64))
                ).convert("RGB")
                crit_content += [
                    {"type": "text", "text": f"Previous render (critique was: {critique or 'none'}):"},
                    {"type": "image", "image": prev_img},
                ]
            if render_curr:
                crit_content += [
                    {"type": "text", "text": "Current render:"},
                    {"type": "image", "image": render_curr},
                ]
            crit_content.append({
                "type": "text",
                "text": "List specific differences or output DONE.",
            })
            crit_messages.append({"role": "user", "content": crit_content})

            images_for_critic = [ref_image]
            if render_prev_b64:
                images_for_critic.append(prev_img)
            if render_curr:
                images_for_critic.append(render_curr)

            with torch.no_grad():
                crit_inputs = _prepare_inputs(processor, crit_messages, images_for_critic, device)
                crit_prompt_len = crit_inputs["input_ids"].shape[1]
                crit_output = model.generate(
                    **crit_inputs,
                    max_new_tokens=CRITIC_MAX_TOKENS,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
            crit_gen_ids = crit_output[0, crit_prompt_len:]
            critique = processor.decode(crit_gen_ids, skip_special_tokens=True)

            episode.turns.append(TurnData(
                phase=Phase.CRITIC,
                input_ids=crit_inputs["input_ids"][0].cpu(),
                pixel_values=crit_inputs.get("pixel_values", torch.empty(0)).cpu(),
                image_grid_thw=crit_inputs.get("image_grid_thw", torch.empty(0)).cpu(),
                mm_token_type_ids=crit_inputs.get("mm_token_type_ids", torch.empty(0)).cpu(),
                generated_ids=crit_gen_ids.cpu(),
                text_output=critique,
                step_idx=step_i,
            ))

            if "DONE" in critique:
                break

        except Exception as exc:
            logger.warning("Critic failed at step %d: %s", step_i, exc)
            critique = None
        finally:
            if not is_crit_trainable:
                model.enable_adapter_layers()

        render_prev_b64 = render_full_b64

    return episode


# ---------------------------------------------------------------------------
# Policy gradient loss
# ---------------------------------------------------------------------------

def compute_pg_loss(
    model,
    processor,
    episode: EpisodeRollout,
    advantages_per_dev_step: List[float],
    training_phase: Phase,
    device: str,
) -> torch.Tensor:
    """Compute GRPO policy gradient loss over trainable agent's tokens.

    Re-runs model forward pass with gradients over the stored sequences.
    """
    loss_terms: List[torch.Tensor] = []

    dev_step_idx = 0  # tracks which developer step we're on
    is_combined = (training_phase == Phase.COMBINED)

    for turn in episode.turns:
        # Combined: train all turns; otherwise only the matching phase
        if not is_combined and turn.phase != training_phase:
            dev_step_idx += (1 if turn.phase == Phase.DEVELOPER else 0)
            continue

        if turn.phase == Phase.DEVELOPER:
            advantage = advantages_per_dev_step[min(dev_step_idx, len(advantages_per_dev_step) - 1)]
            dev_step_idx += 1
        else:
            # Critic turn: use advantage of the NEXT developer step (or last)
            advantage = advantages_per_dev_step[min(dev_step_idx, len(advantages_per_dev_step) - 1)]

        if len(turn.generated_ids) == 0:
            continue

        # Reconstruct full sequence: [prompt | generated], capped to avoid OOM
        MAX_GRAD_SEQ = 512
        prompt_ids = turn.input_ids[-MAX_GRAD_SEQ:]   # keep tail of prompt (most relevant)
        gen_ids_trunc = turn.generated_ids[:MAX_GRAD_SEQ]
        full_ids = torch.cat([prompt_ids, gen_ids_trunc], dim=0).unsqueeze(0).to(device)

        # Skip vision kwargs for gradient forward pass: text-only log-probs are sufficient
        outputs = model(input_ids=full_ids)
        logits = outputs.logits[0]  # [expanded_seq_len, vocab_size]

        # Shift: generated tokens are at the tail; use truncated lengths
        gen_len = len(gen_ids_trunc)
        actual_prompt_len = logits.shape[0] - gen_len
        gen_logits = logits[actual_prompt_len - 1: actual_prompt_len - 1 + gen_len]
        gen_ids = gen_ids_trunc.to(device)

        log_probs = F.log_softmax(gen_logits, dim=-1)
        token_log_probs = log_probs.gather(1, gen_ids.unsqueeze(1)).squeeze(1)
        seq_log_prob = token_log_probs.mean()

        loss_terms.append(-advantage * seq_log_prob)

    if not loss_terms:
        return torch.tensor(0.0, requires_grad=True)
    return torch.stack(loss_terms).mean()


# ---------------------------------------------------------------------------
# Training phase
# ---------------------------------------------------------------------------

def run_phase(
    model,
    processor,
    optimizer,
    phase: Phase,
    num_episodes: int,
    k_rollouts: int,
    log_writer: csv.DictWriter,
) -> None:
    """Run one training phase (Developer or Critic) for num_episodes episodes."""
    import httpx

    env_client = httpx.Client(base_url=SERVER_URL, timeout=180.0)
    device = str(_device(model))

    episode_num = 0
    difficulty_cycle = iter(DIFFICULTIES * (num_episodes // len(DIFFICULTIES) + 1))

    while episode_num < num_episodes:
        difficulty = next(difficulty_cycle)

        # Collect K rollouts
        rollouts: List[EpisodeRollout] = []
        for k in range(k_rollouts):
            try:
                ep = rollout_episode(model, processor, env_client, difficulty, phase)
                rollouts.append(ep)
            except Exception as exc:
                logger.warning("Rollout %d failed: %s", k, exc)

        if not rollouts:
            episode_num += 1
            continue

        # Compute shaped returns per rollout
        returns_per_rollout = [
            compute_step_returns(ep.developer_rewards)
            for ep in rollouts
        ]

        # Group-relative advantages
        adv_per_rollout = grpo_advantages(returns_per_rollout)

        # Policy gradient update for each rollout
        total_loss = torch.tensor(0.0)
        valid_rollouts = 0

        for ep, adv in zip(rollouts, adv_per_rollout):
            if not ep.developer_rewards:
                continue
            try:
                loss = compute_pg_loss(model, processor, ep, adv, phase, device)
                if torch.isfinite(loss) and loss.requires_grad:
                    total_loss = total_loss + loss
                    valid_rollouts += 1
            except Exception as exc:
                logger.warning("Loss computation failed: %s", exc)

        if valid_rollouts > 0:
            avg_loss = total_loss / valid_rollouts
            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

        # Logging
        mean_terminal = float(sum(ep.R_terminal for ep in rollouts) / len(rollouts))
        mean_steps = float(sum(len(ep.developer_rewards) for ep in rollouts) / len(rollouts))
        logger.info(
            "Phase=%s ep=%d/%d diff=%s k=%d mean_R=%.4f mean_steps=%.1f loss=%.4f",
            phase.value, episode_num + 1, num_episodes, difficulty,
            len(rollouts), mean_terminal, mean_steps,
            avg_loss.item() if valid_rollouts > 0 else 0.0,
        )
        log_writer.writerow({
            "phase": phase.value,
            "episode": episode_num,
            "difficulty": difficulty,
            "mean_terminal_reward": mean_terminal,
            "mean_steps": mean_steps,
            "loss": avg_loss.item() if valid_rollouts > 0 else 0.0,
        })

        episode_num += 1

        # Checkpoint every 50 episodes
        if episode_num % 50 == 0:
            ckpt = CHECKPOINT_DIR / f"{phase.value}_ep{episode_num}"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt)
            processor.save_pretrained(ckpt)
            logger.info("Checkpoint saved: %s", ckpt)

    env_client.close()

    # Final checkpoint
    final_ckpt = CHECKPOINT_DIR / f"{phase.value}_final"
    final_ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_ckpt)
    processor.save_pretrained(final_ckpt)
    logger.info("Final checkpoint saved: %s", final_ckpt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global MODEL_NAME, CHECKPOINT_DIR

    parser = argparse.ArgumentParser(description="VisionCoder Round 2 RL training")
    parser.add_argument("--phase", choices=["developer", "critic", "alternate", "combined"],
                        default="alternate")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Episodes for single-phase training")
    parser.add_argument("--episodes-per-phase", type=int, default=200,
                        help="Episodes per phase in alternating mode")
    parser.add_argument("--k-rollouts", type=int, default=4,
                        help="Rollouts per episode for GRPO")
    parser.add_argument("--num-phases", type=int, default=4,
                        help="Number of alternating phases")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR))
    args = parser.parse_args()

    MODEL_NAME = args.model
    CHECKPOINT_DIR = Path(args.checkpoint_dir)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Start environment server
    t = threading.Thread(target=_start_server, daemon=True)
    t.start()
    logger.info("Waiting for environment server …")
    _wait_for_server()
    logger.info("Environment server ready at %s", SERVER_URL)

    # Load model
    model, processor = setup_model(MODEL_NAME)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=0.01,
    )

    # Reward log
    log_path = CHECKPOINT_DIR / "reward_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(
        log_file,
        fieldnames=["phase", "episode", "difficulty", "mean_terminal_reward", "mean_steps", "loss"],
    )
    log_writer.writeheader()

    try:
        if args.phase in ("developer", "critic", "combined"):
            phase = Phase(args.phase)
            run_phase(model, processor, optimizer, phase, args.episodes, args.k_rollouts, log_writer)
        else:
            # Alternate: Developer → Critic → Developer → ...
            phases = [Phase.DEVELOPER, Phase.CRITIC] * (args.num_phases // 2)
            if args.num_phases % 2:
                phases.append(Phase.DEVELOPER)
            for p in phases:
                logger.info("Starting phase: %s", p.value)
                run_phase(model, processor, optimizer, p,
                          args.episodes_per_phase, args.k_rollouts, log_writer)
    finally:
        log_file.close()
        logger.info("Reward log written to %s", log_path)


if __name__ == "__main__":
    main()
