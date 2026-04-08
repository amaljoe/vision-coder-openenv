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

An [OpenEnv](https://github.com/openenv)-compatible reinforcement learning environment for screenshot-to-HTML generation. An agent receives a UI screenshot and is rewarded based on how accurately its generated HTML reproduces the original layout.

## Overview

Each episode:
1. `reset()` — returns a target UI screenshot (base64 PNG) and a task prompt
2. `step(action)` — submits HTML code, receives a composite reward score

### Reward signals

7 signals across 4 phases, normalised to [0, 1] by dividing by the weight sum (9.0):

| Signal | Weight | Phase | Description |
|---|---|---|---|
| `format` | 1× | 0 | Markdown fencing, `<html>` / doctype tags present |
| `validity` | 1× | 0 | HTML parseability, structural tags, tag diversity (≥5 unique) |
| `structural` | 1× | 0 | DOM tag-sequence similarity + CSS-class Jaccard overlap |
| `text_block` | 2× | 1 | Text block match rate + text content similarity (Hungarian matching on IoU) |
| `position` | 1× | 2 | Spatial layout accuracy — normalised centre-to-centre distance of matched blocks |
| `color` | 1× | 3 | Perceptual color accuracy via CIEDE2000 on sampled non-white pixels |
| `clip` | 2× | 4 | CLIP cosine similarity after Playwright render (`openai/clip-vit-base-patch32`, CPU) |

## Baseline results

Evaluated locally with `inference.py` (one episode per difficulty — easy / medium / hard).  
Model: `qwen3.5:4b` via Ollama. Reward pipeline weight sum = 9.0.

### Overall scores

| Model | easy | medium | hard | **mean** |
|---|---|---|---|---|
| `qwen3.5:4b` | 0.739 | 0.686 | 0.469 | **0.631** |
| `nemotron-3-nano:4b` | 0.732 | 0.617 | 0.640 | **0.663** |

> Note: scores dropped vs. the previous 4-signal baseline (mean ~0.81) because `text_block` and `position` add genuine penalties for layout mismatches that pixel-diff alone missed.

### Per-signal breakdown — `qwen3.5:4b`

| Difficulty | total | format | validity | structural | text_block | position | color | clip |
|---|---|---|---|---|---|---|---|---|
| easy   | 0.739 | 1.000 | 1.000 | 0.614 | 0.333 | 0.972 | 0.499 | 0.948 |
| medium | 0.686 | 1.000 | 1.000 | 0.504 | 0.508 | 0.518 | 0.291 | 0.920 |
| hard   | 0.469 | 0.500 | 1.000 | 0.345 | 0.051 | 0.053 | 0.888 | 0.666 |

**Key observations:**
- `text_block` and `position` collapse on hard tasks (0.05) — the model reproduces visual style but places elements incorrectly in complex layouts
- `color` is unusually high on hard (0.89) despite low CLIP (0.67) — the dark dashboard palette is replicated but at wrong positions
- `format` penalty on hard (0.5) reflects qwen3.5 wrapping output in markdown fences; the underlying HTML is valid
- `validity` = 1.0 across all difficulties — generated HTML is always well-formed

## Installation

```bash
pip install -e .
```

## Running inference

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-VL-72B-Instruct
export HF_TOKEN=hf_...
python inference.py
```

Required environment variables:

| Variable | Description |
|---|---|
| `HF_TOKEN` | Hugging Face token (used as API key) |
| `API_BASE_URL` | OpenAI-compatible LLM endpoint |
| `MODEL_NAME` | Vision-capable model ID |

Inference stdout format:

```
[START] task=<difficulty> env=vision-coder model=<model>
[STEP]  step=<n> action=<html_preview> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,...>
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
    obs = client.reset()

    # Decode the target screenshot
    image = client.decode_screenshot(obs)

    # Run your model inference here ...
    html = "<html><body><h1>Hello</h1></body></html>"

    result = client.step(Action(html=html))
    print(f"Reward: {result.reward}")
    print(f"Breakdown: {result.metadata['rewards']}")
```

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `reset()` | `POST /reset` | Start a new episode |
| `step(action)` | `POST /step` | Submit HTML and receive reward |
| `state()` | `GET /state` | Current episode metadata |
| `close()` | `DELETE /close` | End the session |

## Project structure

```
├── inference.py           # Baseline inference script (runs 3 episodes)
├── openenv.yaml           # OpenEnv spec
├── Dockerfile
├── openenv/               # OpenEnv SDK package
│   ├── client.py          # Synchronous HTTP client
│   ├── models.py          # Action, Observation, State (Pydantic)
│   └── server/
│       ├── app.py         # FastAPI application
│       └── environment.py # VisionCoderEnvironment + reward pipeline
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
