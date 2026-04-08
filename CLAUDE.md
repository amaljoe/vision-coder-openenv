# VisionCoder OpenEnv — Claude Code Guide

## Project
Screenshot-to-HTML RL environment for the Scaler x Meta PyTorch Hackathon.
OpenEnv-compatible HTTP API: `reset()` / `step()` / `state()`.

## Package structure
- `openenv/` — mapped to repo root (`__init__.py`, `client.py`, `models.py`)
- `openenv.server` — mapped to `server/` (`app.py`, `environment.py`)
- `vcoder/` — reward pipeline and data loading
- `data/` — bundled synthetic samples (5 per difficulty, ~40KB each)

## Running locally
```bash
pip install -e .
uvicorn openenv.server.app:app --host 0.0.0.0 --port 7860
```

## Running inference
```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-VL-72B-Instruct
export HF_TOKEN=hf_...
python inference.py
```

## Environment variables
- `API_BASE_URL` — OpenAI-compatible LLM endpoint (required)
- `MODEL_NAME` — vision-capable model ID (required)
- `HF_TOKEN` — Hugging Face token / API key for LLM calls (primary auth key)

## Key design decisions
- Bundled `data/*.json` for instant `reset()` — no runtime dataset download
- Real CLIP (`openai/clip-vit-base-patch32`, CPU) for visual reward — pre-downloaded in Dockerfile
- 7 reward signals: format(1×) + validity(1×) + structural(1×) + text_block(2×) + position(1×) + color(1×) + clip(2×), normalised ÷ 9.0
- Docker image ~3.5GB (python:3.11-slim + torch CPU + CLIP + Playwright Chromium)
- HF Space: `amaljoe88/vision-coder-openenv` (port 7860)

## Python version
Use `python3.13` locally (pip maps to python3.13 here, not `python3`).

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

## How Phase 2 evaluation works

The evaluator does NOT use the HF Space to run inference. The flow is:
1. **HF Space ping** — pre-submission only: `POST /reset` must return 200
2. **Phase 2 agentic eval**:
   - Clones GitHub repo to `/tmp/workspace/`
   - `docker build` from `Dockerfile`
   - Runs `inference.py` inside Docker with `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME` set
   - Parses `[START]`/`[STEP]`/`[END]` stdout
   - Validates task scores

Phase 2 steps: `docker_build` → `inference` → `parse_output` → `task_validation` → `llm_check`

---

## inference.py requirements (critical)

Must match the sample inference script exactly:

```
[START] task=<difficulty> env=vision-coder model=<model>
[STEP]  step=<n> action=<html_preview> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,...>
```

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
The Dockerfile downloads: `torch` CPU (~600MB), CLIP model weights (~600MB), Playwright Chromium (~400MB). If any download fails or times out during `docker build`, the whole Phase 2 fails at `docker_build` step.

**Mitigation:** Keep these RUN layers cached by not changing requirements.txt or the download commands unnecessarily.

---

### 7. 4 Playwright browser launches per step (performance)
**What happened:** `text_block_reward`, `position_reward`, `color_reward`, and `clip_visual_reward` each launched Playwright independently — 4–6 browser sessions per `step()` call.

**Fix:** Render the predicted HTML once in `environment.py`, pass the `PIL.Image` as `pred_image` parameter to both `color_reward` and `clip_visual_reward` to skip duplicate renders.
