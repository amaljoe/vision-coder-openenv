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

Evaluated locally via Ollama on one episode per difficulty (easy / medium / hard). Reward pipeline weight sum = 9.0. Renders are shown in the [Visual comparison](#visual-comparison) section below.

### Overall scores

| Model | easy | medium | hard | **mean** |
|---|---|---|---|---|
| `qwen3.5:4b` | 0.436 | 0.449 | 0.390 | **0.425** |
| `nemotron-3-nano:4b` | 0.482 | 0.424 | 0.393 | **0.433** |

### Per-signal breakdown — `qwen3.5:4b`

| Difficulty | total | format | validity | structural | text_block | position | color | clip |
|---|---|---|---|---|---|---|---|---|
| easy   | 0.436 | 0.500 | 1.000 | 0.507 | 0.136 | 0.677 | 0.225 | 0.369 |
| medium | 0.449 | 0.500 | 1.000 | 0.406 | 0.086 | 0.355 | 0.310 | 0.647 |
| hard   | 0.390 | 0.500 | 1.000 | 0.349 | 0.098 | 0.159 | 0.108 | 0.600 |

**Key observations:**
- `format` = 0.5 across all difficulties — qwen3.5 wraps output in markdown fences which are stripped, but the penalty applies
- `validity` = 1.0 across all difficulties — generated HTML is always well-formed
- `structural` and `clip` degrade progressively from easy → hard as layout complexity increases
- Hard task (Pulsar dashboard): model correctly renders the dark sidebar but leaves the main content area empty — explains high `color` but low `text_block`/`position`

### Per-signal breakdown — `nemotron-3-nano:4b`

| Difficulty | total | format | validity | structural | text_block | position | color | clip |
|---|---|---|---|---|---|---|---|---|
| easy   | 0.482 | 0.500 | 1.000 | 0.726 | 0.079 | 0.394 | 0.190 | 0.686 |
| medium | 0.424 | 0.500 | 1.000 | 0.689 | 0.068 | 0.159 | 0.326 | 0.501 |
| hard   | 0.393 | 0.500 | 1.000 | 0.573 | 0.017 | 0.085 | 0.295 | 0.527 |

**Key observations:**
- nemotron-3-nano:4b supports vision inputs via Ollama — image was passed directly (no text-only fallback needed)
- `structural` is consistently higher than qwen3.5:4b (0.73 vs 0.51 on easy) — nemotron produces cleaner tag hierarchies despite shorter outputs
- `text_block` collapses across all difficulties (0.02–0.08) — the model generates mostly structural scaffolding without reproducing actual text content
- `position` and `clip` are lower than qwen3.5:4b, reflecting shorter/simpler HTML that misses detail at higher difficulty

---

## Visual comparison

### Easy — Sign-in form

| Reference | qwen3.5:4b | nemotron-3-nano:4b |
|---|---|---|
| ![ref](assets/ref_easy.png) | ![qwen](assets/qwen_easy.png) | ![nemotron](assets/nemotron_easy.png) |

| Signal | Weight | qwen3.5:4b | nemotron-3-nano:4b |
|---|---|---|---|
| format | 1× | 0.500 | 0.500 |
| validity | 1× | 1.000 | 1.000 |
| structural | 1× | 0.507 | 0.726 |
| text_block | 2× | 0.136 | 0.079 |
| position | 1× | 0.677 | 0.394 |
| color | 1× | 0.225 | 0.190 |
| clip | 2× | 0.369 | 0.686 |
| **total** | **9** | **0.436** | **0.482** |

**Analysis:** qwen correctly reproduces the sign-in card layout with email/password fields and the purple CTA button — `position` (0.68) and `text_block` (0.14) reflect this fidelity. nemotron generates a completely generic "My Page" skeleton unrelated to the reference, yet scores higher overall (0.482) because the clean tag structure inflates `structural` (0.73) and CLIP picks up the similar background color (0.69).

---

### Medium — SaaS landing page

| Reference | qwen3.5:4b | nemotron-3-nano:4b |
|---|---|---|
| ![ref](assets/ref_medium.png) | ![qwen](assets/qwen_medium.png) | ![nemotron](assets/nemotron_medium.png) |

| Signal | Weight | qwen3.5:4b | nemotron-3-nano:4b |
|---|---|---|---|
| format | 1× | 0.500 | 0.500 |
| validity | 1× | 1.000 | 1.000 |
| structural | 1× | 0.406 | 0.689 |
| text_block | 2× | 0.086 | 0.068 |
| position | 1× | 0.355 | 0.159 |
| color | 1× | 0.310 | 0.326 |
| clip | 2× | 0.647 | 0.501 |
| **total** | **9** | **0.449** | **0.424** |

**Analysis:** qwen reproduces the "Streamline / Ship faster, break nothing." hero with correct nav links, CTA buttons, and "NOW IN BETA" badge — `clip` (0.65) confirms strong visual similarity. The lavender hero background is missing (explains lower `color`). nemotron again outputs a generic "Welcome / This is a sample page." template — the only reward it earns is structural validity and coincidental color overlap.

---

### Hard — CI/CD dashboard

| Reference | qwen3.5:4b | nemotron-3-nano:4b |
|---|---|---|
| ![ref](assets/ref_hard.png) | ![qwen](assets/qwen_hard.png) | ![nemotron](assets/nemotron_hard.png) |

| Signal | Weight | qwen3.5:4b | nemotron-3-nano:4b |
|---|---|---|---|
| format | 1× | 0.500 | 0.500 |
| validity | 1× | 1.000 | 1.000 |
| structural | 1× | 0.349 | 0.573 |
| text_block | 2× | 0.098 | 0.017 |
| position | 1× | 0.159 | 0.085 |
| color | 1× | 0.108 | 0.295 |
| clip | 2× | 0.600 | 0.527 |
| **total** | **9** | **0.390** | **0.393** |

**Analysis:** qwen gets the dark sidebar (Pulsar logo, nav links) right but leaves the main content area completely empty — no stats cards, no deployments table. This explains the pattern: `clip` is decent (0.60, picks up the dark theme) but `text_block` (0.10) and `position` (0.16) collapse because the right-side content is absent. nemotron again outputs a generic template, scoring comparably due to structural cleanliness despite zero visual resemblance.

---

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
