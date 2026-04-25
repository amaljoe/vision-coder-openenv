# VisionCoder OpenEnv — Claude Code Guide

## Project
Screenshot-to-HTML RL environment for the Scaler x Meta PyTorch Hackathon.
OpenEnv-compatible HTTP API: `reset()` / `step()` / `render()` / `state()`.

**Round 1** (backup branch `round1`): single-step, single-agent inference.
**Round 2** (current `main`): multi-step iterative environment + multi-agent (Developer + Critic) + RL training.

## Package structure
- `openenv/` — mapped to repo root (`__init__.py`, `client.py`, `models.py`)
- `openenv.server` — mapped to `server/` (`app.py`, `environment.py`)
- `vcoder/` — reward pipeline and data loading
- `data/` — bundled synthetic samples (5 per difficulty, ~40KB each)
- `data/tests/` — reward stability test cases (0-14; committed HTML + expected scores; renders gitignored)
- `tests/test_reward_stability.py` — reward stability test suite (see "Reward stability tests" section)

## Running on rmgpu006 (cluster)

### Step 1 — start vLLM (tmux session: `vllm`)
```bash
~/.local/bin/tmux new-session -s vllm
# inside that session:
apptainer exec --nv ~/apptainer-images/cuda-custom-amal_latest.sif bash -c \
  'export LD_PRELOAD=/dev/shm/qwen35/lib/libstdc++.so.6;
   /dev/shm/qwen35/bin/python -m vllm.entrypoints.openai.api_server \
   --model ~/models/Qwen3.5-2B --served-model-name qwen35 \
   --tensor-parallel-size 2 --port 8001 --host 0.0.0.0 \
   --max-model-len 65536 --enable-auto-tool-choice --tool-call-parser hermes' \
   2>&1 | tee ~/vllm_qwen35.log
```
**`--enable-auto-tool-choice --tool-call-parser hermes` is mandatory** — without it every Developer call fails with 400 Bad Request and falls back to FALLBACK_HTML.

### Step 2 — start env server (tmux session: `openenv`, no apptainer needed)
```bash
~/.local/bin/tmux new-session -s openenv
# inside that session:
export PLAYWRIGHT_BROWSERS_PATH=~/playwright-browsers
cd ~/workspace/vision-coder-openenv
/dev/shm/qwen35/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 18080
```

### Step 3 — run inference (same openenv session or a new window)
```bash
export API_BASE_URL=http://localhost:8001/v1
export MODEL_NAME=qwen35
export HF_TOKEN=sk-local
export MAX_STEPS=2
export PLAYWRIGHT_BROWSERS_PATH=~/playwright-browsers
cd ~/workspace/vision-coder-openenv
/dev/shm/qwen35/bin/python inference.py
```

### First-time setup
```bash
# Install package (once per env build)
/dev/shm/qwen35/bin/pip install -e .

# Download model (once, needs proxy sourced)
source ~/proxy-setup/scripts/proxy_env.sh
/dev/shm/qwen35/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-2B', local_dir='$HOME/models/Qwen3.5-2B')
"
```

## Running locally (generic)
```bash
pip install -e .
uvicorn openenv.server.app:app --host 0.0.0.0 --port 7860
```

## Running inference (HF router)
```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen3.5-35B-A3B
export HF_TOKEN=hf_...
python inference.py
```

## Environment variables
- `API_BASE_URL` — OpenAI-compatible LLM endpoint (required)
- `MODEL_NAME` — vision-capable model ID (required); Developer and Critic share this endpoint
- `HF_TOKEN` — Hugging Face token / API key for LLM calls (primary auth key)
- `MAX_STEPS` — max developer turns per episode (default: 5)
- `INFERENCE_SERVER_PORT` — env server port (default: 18080)
- `DEBUG` — set to `1` to enable full episode debug logging. Creates `outputs/<run_id>/` with:
  - `<difficulty>.md` — per-episode markdown log (reference image, all step renders, HTML, rewards, critic text)
  - `images/` — PNGs saved separately (reference, each step's rendered output, critic comparison views)
  - Run: `DEBUG=1 API_BASE_URL=... MODEL_NAME=... /dev/shm/qwen35/bin/python inference.py`
- `ONE_SHOT` — set to `1` to prepend 1-shot examples to Developer and Critic prompts.
  Developer example shows the render→adjust→output-HTML loop; Critic examples show when to give feedback vs DONE.
  Helps smaller models (2B) stay on-format. Toggle off to reduce context usage.

## Python version
Use `python3.13` locally (pip maps to python3.13 here, not `python3`).
On rmgpu006: use `/dev/shm/qwen35/bin/python`.

---

## Reward function

Composite of 7 sub-rewards, weighted and normalised to [0, 1]:

| Reward | Weight | What it measures |
|---|---|---|
| `format` | 1.0 | Has `\`\`\`html` fence + `<!DOCTYPE html>` |
| `validity` | 1.0 | Structural completeness (`html`/`head`/`body`, diverse tags) |
| `structural` | 0.5 | Tag-sequence similarity + inline-style property coverage |
| `text_block` | 3.0 | Hungarian-matched text block IoU + text similarity |
| `position` | 1.0 | Hungarian-matched centroid distance |
| `color` | 1.0 | Spatial CIEDE2000 on reference non-white pixels |
| `clip` | 2.0 | CLIP ViT-B/32 cosine similarity, renormalised (threshold 0.65) |

**Weight sum = 9.5.** `text_block` weight is highest (3×) — most discriminative signal; blank/wrong layout → 0.

**Content multiplier:** applied to the weighted total. If reference has content but prediction is nearly blank (< 0.5% non-white pixels at 32×32), multiplier scales linearly from 0 to 1. Ensures blank predictions score 0.0 even if individual sub-rewards are nonzero.

**CLIP renormalisation:** raw cosine ≤ 0.65 → score 0; 1.0 → 1.0. Makes blank pages (raw ~0.45) and unstyled pages (raw ~0.76) meaningfully separated.

**Observed scores on 15 test cases (averages):**
```
perfect    0.947    minor_diff  0.887    bad_colors  0.792
half_styled 0.534   no_layout   0.474    no_style    0.409    blank 0.000
```
Global Spearman ρ vs canonical targets = 0.956 (15/15 PASS).

---

## Reward stability tests

Test suite at `tests/test_reward_stability.py`. Test data in `data/tests/<num>/` (0-14, mapping easy/0-4, medium/0-4, hard/0-4).

Each case has:
- `reference.html`, `variants/*.html` (7 quality levels) — committed
- `expected_scores.json` — per-case baseline scores — committed
- `renders/` — PNG renders + block JSONs — **gitignored**, auto-generated

```bash
# First run on a new machine (needs apptainer for Playwright on rmgpu006)
apptainer exec ~/apptainer-images/cuda-custom-amal_latest.sif bash -c \
  'export PLAYWRIGHT_BROWSERS_PATH=~/playwright-browsers
   /dev/shm/qwen35/bin/python tests/test_reward_stability.py --render'

# Score only (fast, uses cached renders — works outside apptainer)
/dev/shm/qwen35/bin/python tests/test_reward_stability.py

# Re-render specific cases
/dev/shm/qwen35/bin/python tests/test_reward_stability.py --render --cases 0,1,5

# After changing reward functions, lock in new baseline
/dev/shm/qwen35/bin/python tests/test_reward_stability.py --update-expected

# As pytest
/dev/shm/qwen35/bin/python -m pytest tests/test_reward_stability.py -v
```

Two correlation axes reported:
- **ρ(quality)** vs `CANONICAL_EXPECTED` in code — measures reward calibration quality
- **ρ(regress)** vs `expected_scores.json` — regression test, catches reward function drift

Pass criteria: quality ρ ≥ 0.80 per case, blank ≤ 0.05, perfect ≥ 0.80.

---

## Round 2 architecture

### Models
| Role | Inference (eval) | Training |
|---|---|---|
| Developer | `Qwen/Qwen3.5-35B-A3B` via HF router | `Qwen/Qwen3.5-9B` with LoRA |
| Critic | `Qwen/Qwen3.5-35B-A3B` via HF router | `Qwen/Qwen3.5-9B` with LoRA, thinking on |

- Qwen3.5 is unified vision+text — no separate VL variant needed
- Native tool calling via Qwen3-Coder XML format — reliable at 4-9B scale
- 35B-A3B is a MoE that activates 3B params at runtime: fast + high quality
- 9B fits 40GB A100 with LoRA; use 4B for 24GB GPU

### Environment changes (server/)
- `reset()` returns task + `session_id`; session state lives in-memory per episode
- `step(html, session_id)` → `{reward, render_low (base64), render_full (base64), done}`
- `render(html)` → `{image (base64)}` — renders only, no reward (used by Developer tool call)
- `done=true` when max steps reached

### Multi-agent inference loop (inference.py)
```
for each task (episode):
    state = env.reset()                        # → session_id, reference_image
    code, critique, render_prev = "", None, None

    for step i in range(MAX_STEPS):
        # Developer: fast mode, render() tool available
        # Input: reference_image (low-res) + code + critique
        # Calls render(new_html) mid-generation to self-check
        code = developer.generate(ref_image, code, critique)

        result = env.step(code, session_id)    # → reward, render_low, render_full, done
        log_step(i, code, result.reward, result.done)
        if result.done: break

        # Critic: thinking mode on
        # Input: reference_image (full-res) + render_{i-1} + critique_{i-1} + render_i
        critique = critic.review(ref_image, render_prev, critique, result.render_full)
        if "DONE" in critique: break

        render_prev = result.render_full

log_end(...)
```

### RL training (train.py)

**Reward function (per turn t in trajectory):**
```
R_total(t) = R_terminal + λ · Σ(r_s - r_{s-1}  for s = t..n)

R_terminal = environment score at final step n    ← main signal
r_s - r_{s-1} = per-step improvement delta        ← shaped signal
λ = 0.2                                           ← keeps shaped signal subordinate
```

- `R_terminal` propagates backward to all turns (solves long-horizon credit assignment)
- Shaped reward gives additional gradient signal at early turns without dominating
- Both Developer and Critic tokens receive this advantage; Critic's shaped reward
  is the improvement delta from step i+1 onward (first step after its critique)

**Training algorithm: full-episode GRPO**
```
for each task:
    sample K full trajectories τ_1..τ_K  (different temperatures/seeds)
    score each trajectory: R_terminal_k + shaped deltas
    compute group-relative advantage: A_t = (G_t - mean_k) / std_k
    update: ∇ log π(a_t | s_t) · A_t  for all tokens in trajectory
```

**Alternating training schedule:**
```
Phase A (N episodes): Train Developer (LoRA), freeze Critic
Phase B (N episodes): Train Critic   (LoRA, thinking on), freeze Developer
Repeat until convergence
```

### inference.py output format (unchanged from Round 1)
```
[START] task=<difficulty> env=vision-coder model=<model>
[STEP]  step=<n> action=<html_preview> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,...>
```

---

## Git remotes
Two remotes must both be kept in sync:
- `origin` → GitHub (`https://github.com/amaljoe/vision-coder-openenv`)
- `hf` → HuggingFace Spaces (`https://huggingface.co/spaces/amaljoe88/vision-coder-openenv`)

**A `post-push` hook auto-pushes to `hf` whenever you push to `origin`.** Never push only to origin and forget HF — that's what caused multiple Phase 2 failures (HF Space was 10 commits behind for most of the day).

To deploy manually use the `/deploy` skill which pushes both, waits for build, and health-checks `/reset`.

---

## Submission workflow

### 1. Push code
```bash
git push origin main   # post-push hook auto-syncs to hf
```

### 2. Verify HF Space is live
```bash
curl -s "https://huggingface.co/api/spaces/amaljoe88/vision-coder-openenv" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['runtime']['stage'], d['sha'][:8])"
```
Wait for `stage: RUNNING`. Docker build with CLIP takes 5–10 minutes.

### 3. Health check
```bash
curl -X POST "https://amaljoe88-vision-coder-openenv.hf.space/reset?difficulty=easy" -o /dev/null -w "%{http_code}"
# Must return 200
```

### 4. Submit via dashboard
Only the team lead submits at: `https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard`
- GitHub URL: `https://github.com/amaljoe/vision-coder-openenv`
- HF Space URL: `https://huggingface.co/spaces/amaljoe88/vision-coder-openenv`

### 5. Check submission status
Use the `/check-hackathon-status` skill or:
```bash
curl -s "https://www.scaler.com/meta-pytorch-hackathon/api/v1/submissions/status?round=1" \
  -H "Authorization: Bearer BAh7B0kiDHVzZXJfaWQGOgZFVGkCcp1JIg5pc3N1ZWRfYXQGOwBUbCsHLfLDaQ==--bf98a96719f439decefc69bb1a42fe0d28fa29f82fb956d8546f326225af310d" \
  | python3 -m json.tool
```
Check `validation.agentic_evaluation.status` — target is `"success"` with all 5 steps `"pass"`.

---

## How evaluation works

The evaluator does NOT use the HF Space to run inference. The flow is:
1. **HF Space ping** — pre-submission only: `POST /reset` must return 200
2. **Agentic eval**:
   - Clones GitHub repo to `/tmp/workspace/`
   - `docker build` from `Dockerfile`
   - Runs `inference.py` inside Docker with `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME` set
   - Parses `[START]`/`[STEP]`/`[END]` stdout
   - Validates task scores

Eval pipeline: `docker_build` → `inference` → `parse_output` → `task_validation` → `llm_check`

---

## inference.py requirements (critical)

Rules enforced by evaluator parser:
- `action=` must be a plain string — **no `!r` repr quoting** (caused one failure)
- `done` and `success` are lowercase: `true`/`false`
- `error` is `null` if none
- `[END]` must always be emitted — put it in a `finally` block
- Exit code must be 0 — wrap everything in try/except
- LLM call must be inside try/except with a fallback HTML so it never crashes

Auth variable priority (evaluator sets `HF_TOKEN`, not `OPENAI_API_KEY`):
```python
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or "sk-placeholder"
```

---

## Challenges faced and lessons learned

### 1. HF Space drifted 10 commits behind
**What happened:** `git push origin main` only updates GitHub. HF Spaces has a separate `hf` remote that needs an explicit `git push hf main`. We pushed all code fixes to GitHub but forgot to push to HF, so the Space ran month-old code for the entire evaluation window.

**Fix:** Added `.git/hooks/post-push` that auto-pushes to `hf` whenever `origin` is pushed.

---

### 2. Auth failure — wrong API key env var
**What happened:** Old `inference.py` read `OPENAI_API_KEY` but the evaluator only sets `HF_TOKEN`. This caused every LLM call to use `"sk-placeholder"` → 401 auth error → crash.

**Fix:** Read `HF_TOKEN` first, fall back through `API_KEY` → `OPENAI_API_KEY` → placeholder.

---

### 3. Timing race — 3-minute window between commit and submission
**What happened:** Commit `4dc307c` (inference.py rewrite with auth fix) was pushed at 11:49 AM. User submitted at 11:52 AM — only 3 minutes later. The evaluator cloned the repo before the push fully propagated and ran the older version without proper try/except. The submission failed despite the fix being "live".

**Lesson:** After pushing a critical fix, wait at least 5–10 minutes before resubmitting to avoid the evaluator picking up stale code.

---

### 4. `!r` repr quoting on action field broke output parsing
**What happened:** `log_step` used `f"action={action!r}"` which wraps the HTML string in Python quotes: `action='<!DOCTYPE...'`. The evaluator parser expected `action=<!DOCTYPE...` (no quotes). Found by comparing against the sample inference script.

**Fix:** Changed `{action_summary!r}` to `{action_summary}`.

---

### 5. transformers v5 API change for CLIP
**What happened:** `model.get_image_features()` in transformers v5 returns a dataclass (`BaseModelOutputWithPooling`), not a raw tensor. Code that did `features / features.norm()` crashed with `'BaseModelOutputWithPooling' object has no attribute 'norm'`.

**Fix:**
```python
out = model.get_image_features(pixel_values=pv)
features = out.pooler_output if hasattr(out, "pooler_output") else out
```

---

### 6. Docker build downloads ~1.2GB — fragile on slow networks
The Dockerfile downloads: `torch` CPU (~600MB), CLIP model weights (~600MB), Playwright Chromium (~400MB). If any download fails or times out during `docker build`, the whole eval fails at `docker_build` step.

**Mitigation:** Keep these RUN layers cached by not changing requirements.txt or the download commands unnecessarily.

---

### 7. 4 Playwright browser launches per step (performance)
**What happened:** `text_block_reward`, `position_reward`, `color_reward`, and `clip_visual_reward` each launched Playwright independently — 4–6 browser sessions per `step()` call.

**Fix:** Render the predicted HTML once in `environment.py`, pass the `PIL.Image` as `pred_image` parameter to both `color_reward` and `clip_visual_reward` to skip duplicate renders.

---

### 9. `color_reward` false positive on white-background pages
**What happened:** Original implementation sampled non-white pixels independently from each image, then compared them. A blank white prediction vs a mostly-white reference (e.g. `#f0f2f5` login form) both sampled near-white pixels → CIEDE2000 ≈ 0 → score 0.80–0.96 (false high).

**Fix:** Spatial comparison — resize both images to 128×128, compute per-pixel CIEDE2000, average only over positions where the **reference** is non-white. Blank prediction at those positions gets correct high ΔE. Falls back to full mean when reference is nearly all white (< 2% non-white).

---

### 10. `structural_reward` trivially inflated for inline-style HTML
**What happened:** All 15 reference HTMLs use inline `style=""` attributes, not CSS classes. The CSS class overlap term always returned 1.0 (neither ref nor pred had classes), making blank pages score 0.50 on structural instead of ~0.25.

**Fix:** When ref has no CSS classes, fall back to inline style **property name** coverage: `len(pred_props ∩ ref_props) / len(ref_props)`. Blank page has 1–2 style props vs 15–25 in ref → score 0.08–0.25. Perfect match (same HTML) → same props → 1.0.

---

### 11. `validity_reward` too generous on blank pages
**What happened:** `_MIN_DIVERSE_TAGS` was 5. A blank page with 4 tags (`html`, `head`, `title`, `body`) scored 4/5 = 0.80 on diversity, giving total validity ≈ 0.90.

**Fix:** Raised to 8. Blank page now scores 4/8 = 0.50 on diversity → total validity ≈ 0.75.

---

### 12. Per-step GRPO loses long-horizon signal (Round 2 lesson)
**What happened (design trap):** Applying GRPO independently at each step means Dev_0 only sees `r_0` — the final reward never flows back. Early turns get misguided signal regardless of episode outcome.

**Fix:** Full-episode GRPO — sample K complete trajectories, apply group-relative advantage to all tokens uniformly. Augment with shaped improvement-delta reward (λ=0.2) for early-turn credit assignment. See `train.py`.

---

## Self-Update Protocol

**This file is a living document. Claude must keep it current.**

Whenever you discover something not already recorded here — a new flag, env var, command, lesson, bug, or architectural decision — add it immediately. Do not wait to be asked.

### What to update and where

| Discovery | Where to add |
|---|---|
| New `--flag` for vLLM / uvicorn / inference | Relevant step in "Running on rmgpu006" |
| New environment variable | "Environment variables" section |
| New bug or production incident | "Challenges faced and lessons learned" (next numbered entry) |
| New architectural decision | Relevant subsection under "Round 2 architecture" |
| Change to submission / eval behavior | "Submission workflow" or "How evaluation works" |
| New debug flag or mode | "Environment variables" section with a note on what it enables |

### Rules
- Write lessons in past tense under "Challenges faced" — **What happened**, **Fix** format.
- Keep commands copy-pasteable and tested; update them if they change.
- After updating this file, commit it: `git add CLAUDE.md && git commit -m "docs: update CLAUDE.md — <one-line summary>"`
