# Round 2 Progress — VisionCoder OpenEnv
**Last updated:** 2026-04-25  
**Branch:** `main` | Round 1 backup: `round1`

---

## Status: IN PROGRESS

---

## Completed

### Architecture & Planning
- [x] Round 1 branched and backed up (`round1`)
- [x] Round 2 architecture finalized: multi-step env + Developer + Critic + RL training
- [x] Model selected: Qwen3.5 (unified VLM, native tool use, 256K ctx)
  - Inference: `Qwen/Qwen3.5-35B-A3B` (MoE, fast)
  - Training: `Qwen/Qwen3.5-9B` with LoRA (fits 40GB A100)
- [x] RL design finalized: full-episode GRPO + shaped reward
  - `R_total(t) = R_terminal + λ·Σ(r_s - r_{s-1} for s=t+1..n)`
  - λ = 0.2, alternating Developer / Critic training phases

### Environment (server/)
- [x] Session management: `_sessions: dict[session_id → _Session]`
- [x] `reset()` returns `session_id` in Observation and metadata
- [x] `step()` accepts `session_id`, tracks `step_count`, returns `render_low` (320×240) + `render_full` (640×480) as base64 PNG alongside reward
- [x] `done=True` at `MAX_STEPS = 5`
- [x] New `render()` method: renders HTML to image, no reward (Developer tool call)
- [x] Backward compat: `_last_session_id` fallback when `session_id` omitted

### API (server/app.py)
- [x] `POST /render` endpoint wired to `env.render()`
- [x] Updated step logging with `step/max_steps` and `done`

### Models (models.py)
- [x] `Action.session_id: Optional[str]`
- [x] `Observation.render_low`, `render_full`, `session_id`
- [x] `State.session_id`, `max_steps`
- [x] New `RenderRequest`, `RenderResponse`

### Client (client.py)
- [x] `reset(difficulty=)` param
- [x] `render(html)` method
- [x] `decode_image()` helper

### Inference (inference.py)
- [x] Full multi-agent loop rewritten for Round 2
- [x] Developer: fast mode, calls `/render` tool mid-generation to self-check, up to 3 render calls per turn
- [x] Critic: low temperature (0.1), compares reference + render_prev + render_curr, outputs critique or `DONE`
- [x] Episode ends on `DONE` from critic OR `done=True` from env (max steps)
- [x] Default model updated to `Qwen/Qwen3.5-35B-A3B`
- [x] `[START]`/`[STEP]`/`[END]` format preserved for evaluator

### Docs
- [x] `README.md` rewritten with Round 2 workflow, RL design, agent roles
- [x] `CLAUDE.md` updated with finalized architecture, training pseudocode, model choices

---

## In Progress

- [ ] **Environment full test** — Playwright renders too slowly on Mac; run on server (see "How to test" below)
- [x] **train.py** — Written, syntax OK, not yet run end-to-end

---

## TODO

### Immediate — run on server (Playwright too slow on Mac)

**Environment test** — validates all Round 2 changes end-to-end:
```bash
python3 -c "
from openenv.models import Action, RenderRequest
from openenv.server.environment import VisionCoderEnvironment, MAX_STEPS

env = VisionCoderEnvironment()
obs = env.reset(difficulty='easy')
assert obs.session_id and not obs.done and obs.screenshot_b64
print(f'PASS reset  session={obs.session_id[:8]}  max_steps={MAX_STEPS}')

sid = obs.session_id
html = '<html><body><h1>Test</h1></body></html>'
for i in range(MAX_STEPS):
    r = env.step(Action(html=html, session_id=sid))
    assert r.render_low and r.render_full and len(r.render_full) > len(r.render_low)
    assert r.done == (i == MAX_STEPS - 1)
    print(f'PASS step {i+1}  reward={r.reward:.4f}  done={r.done}')

rr = env.render(RenderRequest(html=html))
assert len(rr.image_b64) > len(rr.image_low_b64)
print(f'PASS render()  full={len(rr.image_b64)}ch  low={len(rr.image_low_b64)}ch')

env2 = VisionCoderEnvironment()
env2.reset(difficulty='hard')
r2 = env2.step(Action(html=html))
print(f'PASS backward-compat  reward={r2.reward:.4f}')

env3 = VisionCoderEnvironment()
obsA = env3.reset(difficulty='easy')
obsB = env3.reset(difficulty='medium')
rA = env3.step(Action(html=html, session_id=obsA.session_id))
rB = env3.step(Action(html=html, session_id=obsB.session_id))
assert obsA.session_id != obsB.session_id
print(f'PASS concurrent  A={rA.reward:.4f}  B={rB.reward:.4f}')

from fastapi.testclient import TestClient
from openenv.server.app import app
tc = TestClient(app)
r = tc.post('/render', json={'html': html})
assert r.status_code == 200 and 'image_b64' in r.json()
print(f'PASS /render  status={r.status_code}')

r = tc.post('/reset', params={'difficulty':'easy'})
sid2 = r.json()['session_id']
for i in range(MAX_STEPS):
    r = tc.post('/step', json={'html': html, 'session_id': sid2})
    res = r.json()
    print(f'PASS /step {i+1}  reward={res[\"reward\"]:.4f}  done={res[\"done\"]}')
    if res['done']: break

print('ALL TESTS PASSED')
"
```

**Inference test** — needs HF_TOKEN + API_BASE_URL:
```bash
export HF_TOKEN=hf_...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen3.5-35B-A3B
python inference.py
```
Expected output: `[START]` → multiple `[STEP]` lines (up to 5 per difficulty) → `[END]`

**Training** — needs GPU (40GB A100 for 9B, 24GB for 4B):
```bash
# Install training deps if not present
pip install peft accelerate

# Single-phase (Developer only, quick test)
python train.py --phase developer --episodes 10 --k-rollouts 2

# Full alternating training
python train.py --alternate --episodes-per-phase 200 --k-rollouts 4 --num-phases 4
```
Reward log saved to `checkpoints/reward_log.csv`. Checkpoints every 50 episodes.

### Remaining work
- [ ] Fix any bugs found during server-side testing
- [ ] Benchmark: compare multi-agent vs single-agent (Round 1) reward
- [ ] Run training, verify reward curve improves over episodes
- [ ] Push fine-tuned checkpoint to HF Hub, update `MODEL_NAME` in inference.py
- [ ] Deploy to HF Space, verify `/reset` returns 200
- [ ] Submit via Scaler dashboard

---

## Key files

| File | Role |
|---|---|
| `server/environment.py` | Multi-step env, session state, reward pipeline |
| `server/app.py` | FastAPI routes incl. `/render` |
| `inference.py` | Multi-agent loop (Developer + Critic), evaluator entry point |
| `train.py` | RL training — GRPO + shaped reward (TODO) |
| `models.py` | Pydantic models — Action, Observation, State, RenderRequest/Response |
| `client.py` | HTTP client |
| `vcoder/rewards/` | 7 reward signals (format, validity, structural, text_block, position, color, clip) |
| `data/` | Bundled samples (5 per difficulty) |

---

## Reward signals (unchanged)

7 signals, weight sum = 9.0:
`format(1) + validity(1) + structural(1) + text_block(2) + position(1) + color(1) + clip(2)`

---

## Round 1 baseline (single-step, qwen3.5:4b via Ollama)

| Difficulty | Score |
|---|---|
| easy   | 0.797 |
| medium | 0.471 |
| hard   | 0.432 |
| **mean** | **0.567** |

Round 2 target: beat this with multi-agent iterative refinement (untrained), then further with RL.

---

## Known pitfalls (from Round 1)

1. Push to BOTH `origin` (GitHub) and `hf` (HF Spaces) — post-push hook handles this automatically
2. Wait ≥5 min after push before resubmitting — evaluator clones GitHub, needs propagation time
3. `action=` field: no `!r` repr quoting — plain string only
4. `[END]` must always emit — use `finally` block
5. Auth: `HF_TOKEN` → `API_KEY` → `OPENAI_API_KEY` priority order
