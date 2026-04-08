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

| Signal | Weight | Description |
|---|---|---|
| `format` | 1× | Markdown fencing, `<html>` / doctype tags present |
| `validity` | 1× | HTML parseability, structure, tag diversity |
| `structural` | 1× | DOM tag-sequence and CSS-class overlap vs. reference |
| `clip` | 3× | CLIP cosine similarity after rendering (`openai/clip-vit-base-patch32`, CPU) |

The total reward is the weighted sum normalised to [0, 1] by dividing by 6.

## Baseline results

Evaluated locally with `inference.py` (3 episodes — easy / medium / hard).

| Model | easy | medium | hard | **mean** |
|---|---|---|---|---|
| `qwen3.5:4b` (Ollama) | 0.917 | 0.878 | 0.641 | **0.812** |
| `nemotron-3-nano:4b` (Ollama) | 0.732 | 0.617 | 0.640 | **0.663** |

Reward breakdown for `qwen3.5:4b`:

| Difficulty | format | validity | structural | clip |
|---|---|---|---|---|
| easy | 1.000 | 1.000 | 0.614 | 0.963 |
| medium | 1.000 | 1.000 | 0.560 | 0.902 |
| hard | 0.500 | 1.000 | 0.332 | 0.671 |

## Installation

```bash
pip install -e .
```

Dependencies: `fastapi`, `uvicorn`, `httpx`, `pillow`, `pydantic`

## Running the server

```bash
uvicorn openenv.server.app:app --host 0.0.0.0 --port 8080
```

## Client usage

```python
from openenv.client import VisionCoderClient
from openenv.models import Action

with VisionCoderClient("http://localhost:8080") as client:
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
openenv/
├── __init__.py
├── client.py          # Synchronous HTTP client
├── models.py          # Action, Observation, State (Pydantic)
└── server/
    ├── app.py         # FastAPI application
    └── environment.py # VisionCoderEnvironment logic
```
