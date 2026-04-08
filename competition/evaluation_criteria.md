# Evaluation Criteria

Source: Scaler x Meta PyTorch OpenEnv Hackathon dashboard

---

## Pre-Submission Validation (Automated — Must Pass)

These are run by the `validate-submission.sh` script and the hackathon evaluator:

| Check | What is verified | Our status |
|---|---|---|
| HF Space ping | `POST /reset` returns HTTP 200 | ✅ |
| OpenEnv compliance | `openenv.yaml`, typed models, `step()`/`reset()`/`state()` endpoints | ✅ |
| Dockerfile build | Docker image builds successfully from repo | ✅ |
| Inference script runs | `inference.py` completes without errors, produces `[START]/[STEP]/[END]` output | ✅ (try/except guards all failure paths) |
| Task graders | ≥3 tasks enumerated, rewards in [0.0, 1.0] | ✅ |

---

## Agentic Evaluation (Automated)

The evaluator runs `inference.py` from `/tmp/workspace/` with their own LLM credentials:

- Sets `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME` from their infrastructure
- Runs the script and checks exit code (must be 0)
- Parses `[START]`, `[STEP]`, `[END]` stdout lines
- Scores are extracted from `score=` field in `[END]`

**Our status:** ✅ Script always exits 0; `[END]` emitted via `finally` block even on LLM failure

---

## Assessment Dimensions (Human + LLM Judging)

| Dimension | Description | Our approach |
|---|---|---|
| **Runtime Correctness** | Executes without errors, produces valid scores | 7 reward signals, all guarded with try/except |
| **Interface Compliance** | Adheres to OpenEnv standards (yaml, endpoints, models) | Full spec compliance, `openenv validate` passes |
| **Task Design** | Clear, realistic, testable real-world tasks | Screenshot-to-HTML — UI engineering task with 3 difficulties |
| **Grading Logic** | Sensible reward systems with partial progress | 7 signals: format, validity, structural, text_block, position, color, CLIP |

---

## Our Reward Signals

All signals graded continuously in [0, 1] — no binary scoring:

| Signal | Weight | What it measures |
|---|---|---|
| `format` | 1× | Output format discipline — fences + doctype |
| `validity` | 1× | HTML well-formedness and structural completeness |
| `structural` | 1× | DOM-level similarity to reference |
| `text_block` | 2× | Text content and block placement matching |
| `position` | 1× | Spatial layout accuracy |
| `color` | 1× | Perceptual color fidelity (CIEDE2000) |
| `clip` | 2× | Visual similarity (CLIP ViT-Base/32 cosine sim) |
| **Total weight** | **9** | Normalised: raw_sum ÷ 9.0 |

---

## Baseline Results (qwen3.5:4b via Ollama)

| Difficulty | total | format | validity | structural | text_block | position | color | clip |
|---|---|---|---|---|---|---|---|---|
| easy   | 0.739 | 1.000 | 1.000 | 0.614 | 0.333 | 0.972 | 0.499 | 0.948 |
| medium | 0.686 | 1.000 | 1.000 | 0.504 | 0.508 | 0.518 | 0.291 | 0.920 |
| hard   | 0.469 | 0.500 | 1.000 | 0.345 | 0.051 | 0.053 | 0.888 | 0.666 |
| **mean** | **0.631** | | | | | | | |

---

## Round 2 (Finals) — April 25–26, Bangalore

- 48-hour on-campus hackathon at Scaler School of Technology
- Mentorship from Meta engineers
- Judging by Meta's global team
- Specific evaluation criteria not yet published (post Round 1 results on April 10)
