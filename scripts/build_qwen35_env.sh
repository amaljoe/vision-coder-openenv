#!/bin/bash
# Builds /dev/shm/qwen35 — vllm 0.20+ with Qwen3.5 GDN support
# Run inside apptainer: apptainer exec --nv ~/apptainer-images/cuda-custom-amal_latest.sif bash <this_script>
set -euo pipefail

export MAMBA_EXE="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"

ENV=/dev/shm/qwen35

echo "=== Creating $ENV ==="
micromamba create -p $ENV python=3.10 -y
micromamba activate $ENV

echo "=== Installing uv ==="
pip install uv -q

echo "=== Installing vllm 0.19.1 (latest; Qwen3.5 GDN support since 0.17.0) ==="
# --torch-backend=auto picks cu128 torch build matching container's CUDA 12.8
uv pip install vllm==0.19.1 --torch-backend=auto

echo "=== Installing transformers from main (Qwen3.5 GDN support not in stable) ==="
uv pip install "git+https://github.com/huggingface/transformers.git@main"

echo "=== Installing training stack ==="
uv pip install accelerate peft trl datasets bitsandbytes einops

echo "=== Installing project requirements ==="
uv pip install fastapi "uvicorn[standard]" httpx pillow pydantic \
    beautifulsoup4 openai playwright scipy scikit-image \
    numpy pandas tqdm requests pyarrow streamlit \
    pydantic-settings python-dotenv jupyter tensorboard qwen-vl-utils

echo "=== Fixing libstdc++ (for vLLM CXXABI) ==="
micromamba install -p $ENV -c conda-forge libstdcxx-ng -y

echo "=== Installing Playwright browsers ==="
PLAYWRIGHT_BROWSERS_PATH=$HOME/playwright-browsers python -m playwright install chromium || echo "WARN: playwright install failed — run manually"

echo "=== Verifying ==="
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"
python -c "import vllm; print('vllm:', vllm.__version__)"
python -c "from transformers import AutoConfig; print('transformers main ok')"

echo ""
echo "=== Done! Env ready at $ENV ==="
echo "Save with: savevllm35"
