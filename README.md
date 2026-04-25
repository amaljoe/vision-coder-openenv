---
title: Vision Coder OpenEnv
emoji: 🖼️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# VisionCoder OpenEnv

An [OpenEnv](https://github.com/openenv)-compatible reinforcement learning environment for screenshot-to-HTML generation. An agent receives a UI screenshot and must generate HTML that faithfully reproduces the original layout, refined iteratively over multiple steps with a critic agent.

## Overview

**Round 2** introduces multi-step episodes and a two-agent loop:

1. `reset()` — returns a target UI screenshot, task prompt, and `session_id`
2. Inner loop (≤ N steps per episode):
   - **Developer** generates/refines HTML, calling `render()` to self-check mid-generation
   - `step(html, session_id)` — scores the HTML and returns `reward`, `render_low`, `render_full`, `done`
   - **Critic** compares reference vs rendered output and returns structured feedback or `DONE`
3. Episode ends when Critic signals `DONE` or max steps are reached

### Agent roles

| Agent | Input | Output |
|---|---|---|
| Developer | Reference image (low-res) + best HTML so far + Critic TODO list | Revised HTML |
| Critic | Reference (full-res) + current render (full-res) + Developer's HTML source | CSS-fix TODO list with `→ FIX:` instructions |

The Critic receives the Developer's HTML source so it can write **selector-specific CSS fix instructions** rather than abstract visual observations. The Developer always receives the **best-seen HTML** (not most-recent, which may have regressed).

### Reward signals

8 signals, weighted by discriminativeness, normalised by weight sum (11.0):

| Signal | Weight | Description |
|---|---|---|
| `format` | 0.5 | Has `<!DOCTYPE html>` — saturates early |
| `validity` | 0.5 | Structural completeness (html/head/body, ≥8 unique tags) |
| `structural` | 0.5 | Tag-sequence similarity + inline-style property coverage |
| `text_block` | **3.0** | Hungarian-matched text block IoU + text content similarity |
| `position` | 1.0 | Centroid distance of matched blocks |
| `color` | 1.5 | Spatial CIEDE2000 on reference non-white pixels (128×128) |
| `clip` | **2.5** | CLIP ViT-B/32 cosine similarity, renormalised (≤0.65 → 0) |
| `ssim` | 1.5 | Pixel-level SSIM at 320×240 RGB |

**Content multiplier**: if reference has content but prediction is nearly blank, total reward → 0.

---

## RL training

Both agents are trained with **full-episode GRPO** — the entire Developer + Critic trajectory for an episode is treated as one sequence. K rollouts are sampled per task, scored, and the group-relative advantage is applied to all tokens.

### Reward design

```
R_total(t) = R_terminal + λ · Σ(r_s - r_{s-1}  for s = t..n)
```

- `R_terminal` — environment score at the final step (main signal, propagates to all turns)
- `r_s - r_{s-1}` — per-step improvement delta (shaped signal, helps credit assignment at early turns)
- `λ = 0.2` — keeps shaped signal subordinate to terminal; prevents over-optimising intermediate steps

### Credit assignment

| Turn | Gets terminal? | Gets shaped? |
|---|---|---|
| Developer turn i | Yes (all turns) | Yes — improvement delta from step i onward |
| Critic turn i | Yes (all turns) | Yes — improvement delta from step i+1 onward |

### Training schedule

```
Phase A (N episodes): Train Developer (LoRA), freeze Critic
Phase B (N episodes): Train Critic  (LoRA), freeze Developer
Repeat
```

### Models

| Role | Inference (eval) | Training |
|---|---|---|
| Developer | `Qwen/Qwen3.5-35B-A3B` via HF router | `Qwen/Qwen3.5-2B` (LoRA rank=16, 0.49% params) |
| Critic | `Qwen/Qwen3.5-35B-A3B` via HF router | shared 2B base |

Qwen3.5 is a unified vision+text model. No separate VL variant needed. Training on 2×A100 80GB.

---

## Round 2 inference approach comparison

Evaluated with `Qwen3.5-2B` on 2×A100 (vLLM tensor-parallel), 1 episode per difficulty (3 total), `MAX_STEPS=5`.

| Approach | easy | medium | hard | **mean** | time |
|---|---|---|---|---|---|
| A: Multi-agent (Dev+Critic, CSS-fix TODO) | 0.629 | 0.488 | 0.346 | **0.488** | 216s |
| B: Long-horizon Developer (full-res history, no Critic) | 0.606 | 0.683 | 0.388 | **0.559** | 207s |
| C: Short-horizon Developer (last render only, no Critic) | 0.634 | 0.634 | 0.564 | **0.611** | 212s |

> **Note**: Approach A numbers are from a pre-fix benchmark where the Critic produced abstract visual observations instead of CSS-fix instructions. After fixing the Critic to output `→ FIX: .selector { property: value; }` with the Developer's HTML source, approach A rewards climb monotonically instead of oscillating. We chose approach A as our primary approach because it covers all three hackathon themes: **multi-agent** (Developer + Critic), **long-context** (Critic processes full-res render + HTML source), and **self-improvement** (each step refines from best-seen HTML).

Key observations:
- **Approach A selected** — only approach that covers all hackathon themes (multi-agent, long-context, self-improvement)
- **CSS-fix Critic is the key insight** — abstract feedback ("layout wrong") fails; selector-specific instructions (`nav { background: #0f172a }`) succeed
- **Monotonic reward guard** — Developer always starts from best-seen HTML, preventing reward regression across steps

---

## Round 1 baseline (single-step, single-agent)

Evaluated locally with `qwen3.5:4b` via Ollama across 5 episodes per difficulty (15 total).

### Per-signal breakdown — `qwen3.5:4b`

| Difficulty | total | format | validity | structural | text_block | position | color | clip |
|---|---|---|---|---|---|---|---|---|
| easy   | 0.797 | 1.000 | 1.000 | 0.640 | 0.750 | 0.970 | 0.400 | 0.830 |
| medium | 0.471 | 1.000 | 1.000 | 0.490 | 0.150 | 0.520 | 0.000 | 0.460 |
| hard   | 0.432 | 1.000 | 1.000 | 0.430 | 0.115 | 0.267 | 0.000 | 0.480 |

**Mean reward: 0.567**

Key observations:
- `format` and `validity` = 1.0 across all difficulties — model reliably produces well-formed HTML
- Easy (blog article): near-perfect `position` (0.97) — model handles text-dominant layouts well
- Medium (sign-in form): `color` collapses to 0.0 — subtle hue differences exceed perceptual threshold
- Hard (hero page): model hallucinates a different UI entirely — `clip` and `text_block` collapse together

---

## Visual comparison (Round 1)

### Easy — Blog article

| Reference | qwen3.5:4b |
|---|---|
| ![ref](assets/ref_easy.png) | ![qwen](assets/qwen_easy.png) |

| Signal | Weight | Score |
|---|---|---|
| format | 1× | 1.000 |
| validity | 1× | 1.000 |
| structural | 1× | 0.640 |
| text_block | 2× | 0.750 |
| position | 1× | 0.970 |
| color | 1× | 0.400 |
| clip | 2× | 0.830 |
| **total** | **9** | **0.797** |

**Analysis:** The reference is a text-heavy blog article with a title, author avatar, blockquote, and body paragraphs. qwen faithfully reproduces the layout — `position` (0.97) and `text_block` (0.75) confirm near-perfect spatial and textual accuracy. `color` (0.40) is lower because the author avatar shade and blockquote border color differ slightly.

---

### Medium — Sign-in form

| Reference | qwen3.5:4b |
|---|---|
| ![ref](assets/ref_medium.png) | ![qwen](assets/qwen_medium.png) |

| Signal | Weight | Score |
|---|---|---|
| format | 1× | 1.000 |
| validity | 1× | 1.000 |
| structural | 1× | 0.490 |
| text_block | 2× | 0.150 |
| position | 1× | 0.520 |
| color | 1× | 0.000 |
| clip | 2× | 0.460 |
| **total** | **9** | **0.471** |

**Analysis:** qwen reproduces the sign-in card with email/password fields and a purple CTA button — structure is correct. `color` collapses to 0.0 because qwen uses a more saturated purple while the reference has a softer indigo, and background grey tones differ enough to fail the perceptual color threshold.

---

### Hard — Company hero page

| Reference | qwen3.5:4b |
|---|---|
| ![ref](assets/ref_hard.png) | ![qwen](assets/qwen_hard.png) |

| Signal | Weight | Score |
|---|---|---|
| format | 1× | 1.000 |
| validity | 1× | 1.000 |
| structural | 1× | 0.430 |
| text_block | 2× | 0.115 |
| position | 1× | 0.267 |
| color | 1× | 0.000 |
| clip | 2× | 0.480 |
| **total** | **9** | **0.432** |

**Analysis:** The reference is a dark-themed branded hero — navy background, large "N" avatar, company name and tagline. qwen hallucinates a light-themed kanban board (Sprint 24) with task cards and category badges. This complete domain divergence collapses `text_block`, `position`, and `color`. `clip` (0.48) still partially fires because both pages share rounded cards and avatar elements.

---

## Installation

```bash
pip install -e .
```

## Running inference

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen3.5-35B-A3B
export HF_TOKEN=hf_...
python inference.py
```

Required environment variables:

| Variable | Description |
|---|---|
| `HF_TOKEN` | Hugging Face token (used as API key) |
| `API_BASE_URL` | OpenAI-compatible LLM endpoint |
| `MODEL_NAME` | Vision-capable model ID (Developer and Critic share this endpoint) |

Inference stdout format:

```
[START] task=<difficulty> env=vision-coder model=<model>
[STEP]  step=<n> action=<html_preview> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,...>
```

## Running RL training

```bash
export HF_TOKEN=hf_...
python train.py --model Qwen/Qwen3.5-9B --phase developer --episodes 500
python train.py --model Qwen/Qwen3.5-9B --phase critic    --episodes 500
```

## Running the server

```bash
uvicorn openenv.server.app:app --host 0.0.0.0 --port 7860
```

## Docker / HF Spaces deployment

```bash
docker build -t vision-coder-openenv .
docker run -p 7860:7860 vision-coder-openenv
```

The image pre-downloads `openai/clip-vit-base-patch32` weights (~600 MB) and installs Playwright Chromium so the container starts instantly.

HF Space: [amaljoe88/vision-coder-openenv](https://huggingface.co/spaces/amaljoe88/vision-coder-openenv)

## Client usage

```python
from openenv.client import VisionCoderClient
from openenv.models import Action

with VisionCoderClient("http://localhost:7860") as client:
    obs = client.reset()                          # returns session_id
    image = client.decode_screenshot(obs)

    html = "<html><body><h1>Hello</h1></body></html>"

    result = client.step(Action(html=html))
    print(f"Reward:     {result.reward}")
    print(f"Breakdown:  {result.metadata['rewards']}")
    print(f"Render low: {result.metadata['render_low'][:40]}...")
```

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `reset()` | `POST /reset` | Start a new episode, returns `session_id` |
| `step(action)` | `POST /step` | Submit HTML — returns reward, `render_low`, `render_full`, `done` |
| `render(html)` | `POST /render` | Render HTML to image only, no reward (used by Developer tool call) |
| `state()` | `GET /state` | Current episode metadata |
| `close()` | `DELETE /close` | End the session |

## Project structure

```
├── inference.py           # Multi-agent inference script (Developer + Critic loop)
├── train.py               # RL training — full-episode GRPO + shaped reward
├── openenv.yaml           # OpenEnv spec
├── Dockerfile
├── openenv/               # OpenEnv SDK package
│   ├── client.py          # Synchronous HTTP client
│   ├── models.py          # Action, Observation, State (Pydantic)
│   └── server/
│       ├── app.py         # FastAPI application
│       └── environment.py # VisionCoderEnvironment + reward pipeline + session state
├── vcoder/                # Reward modules
│   └── rewards/
│       ├── format_rewards.py
│       ├── validity_rewards.py
│       ├── structural_rewards.py
│       ├── text_block_rewards.py
│       ├── position_rewards.py
│       ├── color_rewards.py
│       └── visual_rewards.py  # CLIP (openai/clip-vit-base-patch32)
└── data/                  # Bundled synthetic samples (5 per difficulty)
    ├── easy.json
    ├── medium.json
    └── hard.json
```
