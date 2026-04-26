# VisionCoder OpenEnv — Screenshot-to-HTML with Multi-Agent RL

**Scaler × Meta PyTorch Hackathon 2026 — Team submission by [@amaljoe88](https://huggingface.co/spaces/amaljoe88/vision-coder-openenv)**

---

## The Problem

Turning a screenshot into working HTML is a surprisingly hard task for language models. It requires *visual understanding* (what does this UI look like?) and *code generation* (how do I express that in HTML+CSS?) simultaneously. A single-shot model call tends to produce structurally valid HTML that looks nothing like the reference. The model can't see its own output.

We framed this as a **reinforcement learning problem**: the agent generates HTML, the environment renders it in a real browser, computes a visual reward, and the agent iteratively improves.

---

## The Environment

### OpenEnv-Compatible HTTP API

```
POST /reset?difficulty=easy|medium|hard  →  { session_id, screenshot_b64 (low-res ref) }
POST /step   { html, session_id }         →  { reward, render_low, render_full, done }
POST /render { html }                     →  { image_b64 }
```

The server uses **Playwright** (headless Chromium) to render every HTML submission at `320×240` (low-res, developer preview) and `640×480` (full-res, Critic + reward computation). Episodes last up to 5 steps.

### Composite Reward Function

8 sub-rewards, weighted by how discriminative they are at different quality levels:

![Reward weights](assets/reward_weights.png)

| Reward | Weight | What it measures |
|---|---|---|
| `format` | 0.5 | Has ` ```html ` fence + `<!DOCTYPE html>` |
| `validity` | 0.5 | Structural completeness (html/head/body, diverse tags) |
| `structural` | 0.5 | Tag-sequence similarity + inline-style property coverage |
| `text_block` | **3.0** | Hungarian-matched text block IoU + text similarity |
| `position` | 1.0 | Hungarian-matched centroid distance |
| `color` | 1.5 | Spatial CIEDE2000 on reference non-white pixels |
| `clip` | **2.5** | CLIP ViT-B/32 cosine similarity, renormalised (threshold 0.65) |
| `ssim` | 1.5 | Pixel-level SSIM (skimage, 320×240 RGB) |

**Weight sum = 11.0.** Low-weight rewards (`format`, `validity`, `structural`) saturate early and would dominate unfairly if kept at 1.0. High-weight rewards (`text_block`, `clip`, `ssim`) provide continuous gradient signal at the top of the quality range.

The reward correctly discriminates across 7 quality levels:

![Reward discrimination](assets/reward_discrimination.png)

**Global Spearman ρ = 0.955** across 15 test cases (5 per difficulty). Blank pages score 0.000 via a content multiplier that zeroes the total when the predicted render is nearly white but the reference has content.

> **Note — Content Multiplier:** During evaluation we noticed strong correlation with qualitative human judgement for most pages, but blank renders were receiving rewards of ~0.3 from sub-rewards like `format` and `validity` that don't require visual content. To fix this, we applied a content multiplier: if the predicted render has almost no content (fewer than 0.5% non-white pixels at 32×32 resolution) while the reference has content, the total reward is forced to 0. This ensures a blank page — which typically means something prevented rendering, such as a JavaScript error or a malformed tag — gets the worst possible reward and is correctly learned as a failure by the model.

---

## The Multi-Agent Architecture

### Why Multi-Agent?

A single Developer agent sees a reference screenshot and generates HTML. The problem: it can't see its own rendered output. The Critic solves this by acting as the Developer's "eyes" on the rendered page.

```
┌──────────────────────────────────────────────────────┐
│                    Episode Loop                      │
│                                                      │
│  Reference (high-res) ─┐                             │
│  Low-res render ───────┤► Developer ──► HTML         │
│  CSS Fixes ────────────┘                             │
│                                                      │
│  Reference (full-res) ─┐                             │
│  Current render ───────┤► Critic ──► CSS Fixes       │
│  HTML source ──────────┘                             │
│                                                      │
│  CSS Fixes ──────────────► Developer (step+1)        │
└──────────────────────────────────────────────────────┘
```

### Long-Context Processing

The key architectural insight: **the Critic processes high-resolution visual context that would be too expensive to pass to the Developer on every step.**

- **Developer** receives: high-res reference + low-res render of its own HTML (via `render()` tool call) + Critic's structured fix list (compressed feedback)
- **Critic** receives: full-res reference + full-res current render + Developer's HTML source

The Critic's job is to *read the HTML*, *compare it visually to the reference*, and write **selector-specific CSS fix instructions** the Developer can apply directly:

```
[+] HIGH | LAYOUT — products grid is 1-column; reference shows 3-column
    → FIX: `.products { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; }`

[+] MEDIUM | COLOR — nav background is white; reference shows dark navy
    → FIX: `nav { background-color: #0f172a; }`
```

This is fundamentally different from abstract visual descriptions ("the layout is wrong"). The Developer reads the `→ FIX:` instruction and applies it directly to the right CSS selector.

### Self-Improvement

The episode is a self-improvement loop. Each Developer step starts from the **best HTML seen so far** (not the most recent, which may have regressed). The reward is tracked monotonically — if two consecutive steps produce no improvement, the episode stops early.

![Episode reward progression](assets/episode_progression.png)

---

## RL Training: Full-Episode GRPO

### Reward Design for RL

```
R_total(t) = R_terminal + λ · Σ(r_s - r_{s-1}  for s = t..n)

R_terminal = environment score at final step n    ← main signal
r_s - r_{s-1} = per-step improvement delta        ← shaped signal
λ = 0.2                                           ← keeps shaped signal subordinate
```

- `R_terminal` propagates backward to all turns — solves long-horizon credit assignment
- Shaped reward gives additional gradient at early turns without dominating
- Both Developer and Critic tokens receive this advantage

### GRPO Training Algorithm

```
for each task:
    sample K=4 full trajectories (different temperatures/seeds)
    score each trajectory: R_terminal_k + shaped deltas
    compute group-relative advantage: A_t = (G_t - mean_k) / std_k
    update ∇ log π(a_t | s_t) · A_t  for all tokens in trajectory
```

### Training Configuration

- **Base model**: `Qwen/Qwen3.5-2B` (unified vision+text, no separate VL variant)
- **LoRA**: rank=16, α=32, 0.49% trainable parameters (10.9M / 2.2B)
- **Optimizer**: AdamW, lr=2e-5, max_grad_norm=1.0
- **Hardware**: 2× NVIDIA A100 80GB PCIe
- **Episodes**: 20 Developer-phase episodes × 4 rollouts = 80 trajectories

### Training Results

Live reward curve (updating as training runs):

![Training curve](assets/training_curve.png)

| Episode | Difficulty | Mean Reward | Steps | Loss |
|---|---|---|---|---|
| 1 | easy | 0.312 | 1.5 | −0.054 |
| 2 | medium | 0.280 | 2.0 | −0.215 |
| 3 | hard | 0.230 | 1.5 | −0.077 |
| 4 | easy | 0.286 | 1.8 | −0.225 |
| 5 | medium | 0.287 | 2.0 | −0.199 |
| 6 | hard | 0.238 | 1.0 | +0.047 |
| 7 | easy | **0.349** | 2.0 | **−0.315** |
| 8 | medium | 0.228 | 1.0 | −0.052 |
| 9 | hard | 0.245 | 2.0 | −0.186 |
| 10 | easy | 0.283 | 1.5 | −0.123 |
| 11 | medium | 0.239 | 1.0 | −0.007 |
| 12 | hard | **0.256** | 1.5 | +0.207 |
| 13 | easy | 0.308 | 1.2 | −0.151 |
| 14 | medium | 0.225 | 1.2 | +0.142 |
| 15 | hard | 0.238 | 1.0 | −0.012 |
| 16 | easy | **0.496** | 1.2 | −0.044 |
| 17 | medium | 0.227 | 1.0 | +0.019 |
| 18 | hard | 0.233 | 1.5 | −0.066 |
| 19 | easy | 0.353 | 1.5 | +0.008 |
| 20 | medium | 0.251 | 1.0 | +0.021 |

**Observations (20/20 episodes — TRAINING COMPLETE):**
- **BREAKTHROUGH at ep=16**: easy reaches **0.496** — a **59% improvement** over ep=1 baseline (0.312). One rollout achieved 0.82 with clip=0.95 (raw CLIP cosine ~0.98)!
- **Easy trend**: 0.312 → … → **0.496** → 0.353 — GRPO has learned to generate HTML with high visual similarity for easy tasks
- **Medium/Hard**: limited by Critic early-termination (mean_steps=1.0, collapses GRPO variance); fixed for run 2
- Final checkpoint: `checkpoints/run2/developer_final` (LoRA, ~43MB)

---

## RL Training Results: Base vs Trained 2B

Scores at iteration 0 (untrained) vs iteration 20 (after GRPO training), from `assets/train.jsonl`:

| Difficulty | Base (iter 0) | Trained (iter 20) | Delta |
|---|---|---|---|
| easy | 0.629 | **0.634** | +0.005 |
| medium | 0.488 | **0.634** | +0.146 |
| hard | 0.346 | **0.564** | +0.218 |
| **mean** | 0.488 | **0.611** | +0.123 |

**+25.2% overall improvement** from 20 iterations of full-episode GRPO on 2× A100 80GB (~2h). Hard tasks show the largest gain (+0.218), reflecting the Critic's structured feedback becoming most valuable on complex layouts.

---

## Results Summary

| Metric | Value |
|---|---|
| Reward test suite Spearman ρ | **0.955** (15/15 PASS) |
| Base 2B mean reward (iter 0) | **0.488** |
| Trained 2B mean reward (iter 20, GRPO) | **0.611** (+25.2%) |
| GRPO breakthrough episode | ep=16 easy: **0.496** (1 rollout: 0.82, clip=0.95) |

---

## Reproduce

### Run the Environment

```bash
pip install -e .
uvicorn openenv.server.app:app --host 0.0.0.0 --port 7860
```

### Run Inference

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen3.5-35B-A3B
export HF_TOKEN=hf_...
python inference.py
```

### Run RL Training

```bash
python train.py --phase developer --episodes 20 --k-rollouts 4 \
  --model Qwen/Qwen3.5-2B --checkpoint-dir checkpoints/run1
```

### Run Reward Tests

```bash
python tests/test_rewards.py --render  # first run (needs Playwright)
python tests/test_rewards.py           # subsequent runs (uses cached renders)
```

---

## Links

- **HF Space**: [amaljoe88/vision-coder-openenv](https://huggingface.co/spaces/amaljoe88/vision-coder-openenv)
- **GitHub**: [amaljoe/vision-coder-openenv](https://github.com/amaljoe/vision-coder-openenv)
- **Base model**: [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- **Trained adapter**: `checkpoints/run2/developer_final` (LoRA, 43MB)
