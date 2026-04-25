# Environment Setup — vision-coder-openenv (rmgpu006)

Qwen3.5 inference and RL training on 2× A100 80GB PCIe. CUDA 12.8 inside apptainer.

---

## Quick restore (< 1 min) — USE THIS on new nodes

```bash
restorevllm35   # ~/envs/qwen35.tar.zst → /dev/shm/qwen35
job35           # new tmux window: apptainer + qwen35 env activated
```

After any env change, save the snapshot:
```bash
savevllm35      # /dev/shm/qwen35 → ~/envs/qwen35.tar.zst
```

---

## Server specifics (rmgpu006)

| Item | Value |
|------|-------|
| GPUs | 2× NVIDIA A100 80GB PCIe |
| CUDA driver | 550.54.14 |
| CUDA toolkit (container) | 12.8 |
| /dev/shm | 94GB RAM disk |
| Apptainer image | `~/apptainer-images/cuda-custom-amal_latest.sif` |
| tmux binary | `~/.local/bin/tmux` (native, built from source — no FUSE) |
| micromamba | `~/.local/bin/micromamba` |

---

## Key package versions

| Package | Version | Why pinned |
|---------|---------|-----------|
| torch | 2.10+ (cu128) | vllm 0.17+ hard requirement |
| vllm | 0.19.1 | Qwen3.5 GDN architecture support (added in 0.17.0) |
| transformers | main branch | No stable release supports Qwen3.5 GDN yet |
| python | 3.10 | Matches apptainer container |
| CUDA | 12.8 | Container CUDA; use `cu128` wheels |

**Do NOT use vllm < 0.17.0** — Qwen3.5 uses Gated Delta Networks (GDN), a hybrid linear attention/sparse-MoE architecture not present in earlier vllm.
**vllm 0.20.0 does not exist on PyPI** — latest stable is 0.19.1 (as of 2026-04-25).
**Do NOT use stable transformers** — use `git+https://github.com/huggingface/transformers.git@main`.

---

## Env location

- **vision-coder-openenv**: `/dev/shm/qwen35` (torch 2.10+, vllm 0.20)
- **ixbrl-tagging**: `/dev/shm/vllm` (torch 2.9.1, vllm 0.15.0) — separate project, incompatible versions

---

## Build from scratch (one-time per reboot, ~10-20 min)

```bash
# In an SSH terminal (not via Claude Code — needs TTY for tmux attach)
~/.local/bin/tmux new-session -s build
apptainer exec --nv ~/apptainer-images/cuda-custom-amal_latest.sif \
    bash ~/workspace/vision-coder-openenv/scripts/build_qwen35_env.sh 2>&1 | tee ~/qwen35_build.log
```

---

## Required ~/.bashrc additions

```bash
# micromamba
export MAMBA_EXE="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
alias mamba='micromamba'

# Apptainer (--nv for GPU pass-through)
alias app='apptainer exec --nv ~/apptainer-images/cuda-custom-amal_latest.sif bash'

# qwen35 env
alias savevllm35='bash ~/workspace/vision-coder-openenv/scripts/save_env.sh'
alias restorevllm35='bash ~/workspace/vision-coder-openenv/scripts/restore_env.sh'
alias job35='bash ~/workspace/vision-coder-openenv/scripts/job.sh work /dev/shm/qwen35'
alias jobvllm='bash ~/workspace/vision-coder-openenv/scripts/job.sh work /dev/shm/vllm'

# ixbrl env
alias savevllm='bash ~/workspace/ixbrl-tagging/scripts/save_env.sh'
alias restorevllm='bash ~/workspace/ixbrl-tagging/scripts/restore_env.sh'

# Playwright
export PLAYWRIGHT_BROWSERS_PATH="$HOME/playwright-browsers"
export APPTAINERENV_PLAYWRIGHT_BROWSERS_PATH="$HOME/playwright-browsers"

# libstdc++ fix (set after env is ready)
# export LD_PRELOAD=/dev/shm/qwen35/lib/libstdc++.so.6
```

---

## Starting vLLM

```bash
# Qwen3.5-9B (fits in 1× A100 80GB; use both GPUs for throughput)
~/.local/bin/tmux new-window -t job -n vllm
# In that window:
app
micromamba activate /dev/shm/qwen35
export LD_PRELOAD=/dev/shm/qwen35/lib/libstdc++.so.6
bash ~/workspace/vision-coder-openenv/scripts/vllm_qwen35.sh Qwen/Qwen3.5-9B-Instruct chandra

# Or use the alias after job35:
bash scripts/vllm_qwen35.sh
```

Serves at `http://0.0.0.0:8000` (OpenAI-compatible). Set in env:
```bash
export VLLM_API_BASE=http://localhost:8000/v1
export VLLM_MODEL_NAME=chandra
```

---

## Tmux workflow

```
job session (managed by job.sh / job35 alias):
  window 0: default shell
  window vllm: vLLM server
  window work: training / inference
```

```bash
# Open a new apptainer+env window (the job skill):
job35              # qwen35 env
jobvllm            # ixbrl vllm env
bash scripts/job.sh my-window  # custom name

# Attach from SSH:
~/.local/bin/tmux attach -t job
```

---

## Known issues and workarounds

| Issue | Fix |
|-------|-----|
| `CXXABI_1.3.15 not found` on vllm/training start | `export LD_PRELOAD=/dev/shm/qwen35/lib/libstdc++.so.6` (already in vllm_qwen35.sh) |
| `tmux new-session` fails with FUSE error | Use `~/.local/bin/tmux` (native binary built from source, not AppImage) |
| `apptainer exec` fails — can't start inside Claude Code shell | Use a real SSH TTY; `apptainer exec --nv ... bash script.sh` works non-interactively |
| SSH → GitHub times out (SOCKS5 proxy) | Use HTTPS + `$GITHUB_TOKEN`: `git clone https://${GITHUB_TOKEN}@github.com/...` |
| `tee` / `sed` / standard tools missing in tmux pane | Old tmux AppImage issue — fixed by installing native tmux 3.6a |
| CUDA graph capture assertion error | Reduce `--max-cudagraph-capture-size` in vllm server |
| OOM on long contexts (Qwen3.5 default 262K) | Set `--max-model-len 32768` in vllm_qwen35.sh |
| stable `transformers` import fails for Qwen3.5 | Install from main: `pip install git+https://github.com/huggingface/transformers.git@main` |
| vllm downloads stale cached torch 2.10+ | Use `--torch-backend=auto` with uv; always verify `torch.__version__` after install |

---

## Verification

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import vllm; print(vllm.__version__)"
python -c "from transformers import AutoConfig; print('transformers main ok')"
python -c "from openai import OpenAI; print('openai ok')"
```

Expected: `torch 2.10.x+cu128`, `cuda=True`, `devices=2`, `vllm 0.19.1`.

---

## Typical session layout

```
~/.local/bin/tmux attach -t job

  job:0      (shell)    — general work
  job:vllm   (vllm)     — vLLM server: bash scripts/vllm_qwen35.sh
  job:train  (train)    — PYTHONPATH=. accelerate launch ... train.py
  job:work   (work)     — inference / eval
```
