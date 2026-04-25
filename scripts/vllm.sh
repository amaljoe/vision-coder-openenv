#!/bin/bash
# Start vLLM server for Qwen3.5
# Usage: bash scripts/vllm.sh [model] [served-name]
#   model: 0.8b | 2b | 9b | <local-path> | <hf-model-id>  (default: 2b)
#   served-name: name clients use in API calls               (default: qwen35)
# Env overrides: PORT=8001 GPUS=2 MAX_LEN=65536
set -euo pipefail

RAW_MODEL="${1:-2b}"
NAME="${2:-qwen35}"
PORT="${PORT:-8001}"
MAX_LEN="${MAX_LEN:-65536}"

ENFORCE_EAGER=""
case "$RAW_MODEL" in
    0.8b|0.8B) MODEL="$HOME/models/Qwen3.5-0.8B"; GPUS="${GPUS:-1}"; ENFORCE_EAGER="--enforce-eager" ;;
    2b|2B)     MODEL="$HOME/models/Qwen3.5-2B";   GPUS="${GPUS:-2}" ;;
    9b|9B)     MODEL="$HOME/models/Qwen3.5-9B";   GPUS="${GPUS:-2}" ;;
    *)         MODEL="$RAW_MODEL";                 GPUS="${GPUS:-2}" ;;
esac

export LD_PRELOAD=/dev/shm/qwen35/lib/libstdc++.so.6
# glibc headers for triton cuda_utils compilation (glibc-devel not installed on AlmaLinux 8.9)
export CPATH="$HOME/glibc-headers${CPATH:+:$CPATH}"

echo "Starting vLLM: $MODEL → '$NAME' on port $PORT (${GPUS} GPU(s))"
/dev/shm/qwen35/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$NAME" \
    --tensor-parallel-size "$GPUS" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --max-model-len "$MAX_LEN" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    $ENFORCE_EAGER
