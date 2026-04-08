# Round 1 Requirements

Source: Scaler x Meta PyTorch OpenEnv Hackathon dashboard

---

## Core Task

> "Build a complete, real-world OpenEnv environment that an AI agent can learn from through the standard step() / reset() / state() API"

---

## Mandatory Requirements

### 1. Real-World Tasks (no games/toys)
- Must simulate real-world tasks
- Games and toy problems explicitly prohibited
- **Our status:** ✅ Screenshot-to-HTML generation — a genuine UI-engineering task

### 2. OpenEnv Specification Compliance
- Valid `openenv.yaml` in repo root
- Typed Pydantic models for Action, Observation, State
- Endpoints: `POST /reset`, `POST /step`, `GET /state`
- `openenv validate` must pass
- **Our status:** ✅ All present and passing

### 3. Minimum 3 Tasks (Easy → Medium → Hard)
- At least 3 tasks with distinct difficulty levels
- Agent graders implemented for each
- Scores/rewards must be in [0.0, 1.0]
- **Our status:** ✅ easy / medium / hard with 5 samples each

### 4. Meaningful Reward Functions
- Must include partial progress signals (not binary 0/1)
- Reward range strictly [0.0, 1.0]
- **Our status:** ✅ 7 signals (format, validity, structural, text_block, position, color, clip), all graded

### 5. Baseline Inference Script
- Must be named `inference.py` and placed in repo root
- Must complete error-free and produce scores
- Must use OpenAI Client for all LLM calls
- Required env vars: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- Runtime under 20 minutes
- Structured stdout: `[START]`, `[STEP]`, `[END]` tags
- **Our status:** ✅ fully compliant — see stdout format below

### 6. Hugging Face Spaces Deployment
- Functional Dockerfile that builds successfully
- Space must respond to `/reset` with HTTP 200
- **Our status:** ✅ `amaljoe88/vision-coder-openenv`

### 7. Comprehensive README
- Must document the environment, tasks, reward logic, and usage
- **Our status:** ✅ README covers reward signals, baseline results, API reference, installation

---

## Inference Script stdout Format

Mandatory format (must match exactly):

```
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
```

Rules:
- One `[START]` per episode, at episode begin
- One `[STEP]` per step, immediately after `env.step()` returns
- One `[END]` after episode ends — **always emitted, even on exception** (via `finally`)
- `reward` and `rewards` formatted to 2 decimal places
- `done` and `success` lowercase booleans: `true` or `false`
- `error` = raw exception string, or `null` if none
- All fields on a single line, no internal newlines

---

## Required Environment Variables

| Variable | Purpose | Our Default |
|---|---|---|
| `HF_TOKEN` | Hugging Face / API key for LLM | read first (also `API_KEY`, `OPENAI_API_KEY`) |
| `API_BASE_URL` | OpenAI-compatible LLM endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Vision-capable model identifier | `Qwen/Qwen2.5-VL-72B-Instruct` |

---

## Infrastructure Constraints

| Constraint | Limit |
|---|---|
| Inference script runtime | < 20 minutes |
| Min RAM | 8 GB |
| Min vCPU | 2 |

---

## Submission Rules

- Only team leads can submit final entries
- Submissions can be revised before the deadline (April 8, 11:59 PM)
- Both GitHub repo URL and HF Space URL must be provided
- HF Space URL format: `https://huggingface.co/spaces/<owner>/<space>`
