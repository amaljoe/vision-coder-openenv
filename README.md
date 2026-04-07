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
| `clip` | 3× | CLIP image-image similarity after rendering |

The total reward is the weighted sum across all four signals.

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
