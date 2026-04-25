#!/bin/bash
# Start vLLM server for Qwen3.5
# Usage: bash scripts/vllm_qwen35.sh [model] [name]
# Defaults: Qwen/Qwen3.5-9B-Instruct, served as "chandra"
set -euo pipefail

MODEL="${1:-Qwen/Qwen3.5-9B-Instruct}"
NAME="${2:-qwen35}"
GPUS="${GPUS:-2}"
PORT="${PORT:-8000}"

export MAMBA_EXE="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate /dev/shm/qwen35

# libstdc++ fix for CXXABI_1.3.15
export LD_PRELOAD=/dev/shm/qwen35/lib/libstdc++.so.6

echo "Starting vLLM: $MODEL → served as '$NAME' on port $PORT (${GPUS} GPUs)"
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$NAME" \
    --tensor-parallel-size "$GPUS" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len 65536
