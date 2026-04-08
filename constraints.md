# Hackathon Constraints & Compliance

Source: Scaler x Meta PyTorch OpenEnv Hackathon

---

## Runtime

| Constraint | Limit | Our Status |
|---|---|---|
| Inference script runtime | < 20 minutes | ~3 min (3 episodes × ~1 min each) |
| Min RAM | 8 GB | ~3–4 GB (CLIP ~1.5 GB, app ~500 MB, Playwright ~300 MB) |
| Min vCPU | 2 | HF free Spaces provides 2 vCPU |
| HF Space HTTP response | 200 on `/reset` | ✅ |

---

## Inference Script (`inference.py`)

| Constraint | Requirement | Our Status |
|---|---|---|
| Filename | `inference.py` in repo root | ✅ |
| LLM client | Must use OpenAI client | ✅ `from openai import OpenAI` |
| API key env var | `HF_TOKEN` (also `API_KEY`) | ✅ reads `HF_TOKEN → API_KEY → OPENAI_API_KEY` |
| Base URL env var | `API_BASE_URL` | ✅ default: `https://router.huggingface.co/v1` |
| Model env var | `MODEL_NAME` | ✅ default: `Qwen/Qwen2.5-VL-72B-Instruct` |
| Must not crash | Exit 0 always | ✅ try/except around LLM call, episode, server startup; `[END]` always emitted via `finally` |

### Mandatory stdout format

```
[START] task=<difficulty> env=<benchmark> model=<model>
[STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
```

Rules:
- One `[START]` per episode
- One `[STEP]` per step, immediately after `env.step()` returns
- One `[END]` after episode ends — **always emitted, even on exception**
- `reward` and `rewards` formatted to 2 decimal places
- `done` and `success` are lowercase booleans: `true` or `false`
- `error` is the exception string or `null` if none
- All fields on a single line, no newlines within a line
- Score must be in [0, 1]

---

## Environment Server

| Constraint | Requirement | Our Status |
|---|---|---|
| Deployment | HuggingFace Spaces (Docker) | ✅ `amaljoe88/vision-coder-openenv` |
| `openenv.yaml` | Must exist in repo root | ✅ |
| `openenv validate` | Must pass | ✅ `[OK] vision-coder-openenv: Ready for multi-mode deployment` |
| Dockerfile | Must build successfully | ✅ |
| Required endpoints | `POST /reset`, `POST /step`, `GET /state` | ✅ |
| Task count | Minimum 3 | ✅ easy / medium / hard |
| Difficulty progression | Easy → Medium → Hard | ✅ |
| Reward range | Strictly [0.0, 1.0] | ✅ normalised ÷ 9.0 |
| Reward signals | Meaningful partial progress (not binary) | ✅ 7 signals with gradations |
| Task type | Real-world tasks (games/toys prohibited) | ✅ screenshot-to-HTML generation |

---

## Reward Signals

7 signals, weight sum = 9.0, all outputs clamped to [0, 1]:

| Signal | Weight | Phase | Description |
|---|---|---|---|
| `format` | 1 | 0 | Markdown fencing + `<html>`/DOCTYPE tags |
| `validity` | 1 | 0 | BS4 parse + structural tags + ≥5 unique tag types |
| `structural` | 1 | 0 | Tag-sequence difflib + CSS-class Jaccard |
| `text_block` | 2 | 1 | Text block IoU matching + text content similarity |
| `position` | 1 | 2 | Normalised centre-to-centre distance of matched blocks |
| `color` | 1 | 3 | CIEDE2000 perceptual color distance on sampled pixels |
| `clip` | 2 | 4 | CLIP cosine similarity (`openai/clip-vit-base-patch32`, CPU) |

---

## Submission

| Constraint | Requirement | Our Status |
|---|---|---|
| GitHub repo | Public, submitted URL | ✅ `github.com/amaljoe/vision-coder-openenv` |
| HF Space URL | `huggingface.co/spaces/<owner>/<space>` format | ✅ `amaljoe88/vision-coder-openenv` |
| Round 1 deadline | April 8, 11:59 PM | ✅ submitted |
| Submitter | Team lead only | ✅ solo team |

---

## Known Risks

- **Text block / position Playwright launches**: each `step()` triggers 4 DOM extractions (2 for `text_block`, 2 for `position`) + 1 screenshot render. On 2 vCPU this is ~8–12s overhead per episode.
- **CLIP on CPU**: first inference loads ~600 MB of weights; subsequent calls reuse the singleton. Cold-start adds ~5s.
- **HF Space cold start**: Space sleeps after inactivity. Evaluator pings `/reset` which wakes it; may need 30–60s before responding.
