#!/usr/bin/env bash
# Run after training completes to evaluate trained LoRA vs base and start run 2.
# Usage: bash scripts/eval_after_training.sh [lora_dir]
#
# Steps:
#   1. Start vLLM with base model (qwen35) + trained LoRA (qwen35-trained)
#   2. Run inference.py with base model → measure baseline
#   3. Run inference.py with trained model → measure improvement
#   4. Print comparison and write to checkpoints/eval_results.txt
#   5. Prompt whether to start run 2 (resume from trained LoRA)

set -euo pipefail

LORA_DIR="${1:-checkpoints/run2/developer_final}"
MODEL_PATH="$HOME/models/Qwen3.5-2B"
VLLM_LOG="$HOME/vllm_eval.log"
RESULTS_FILE="checkpoints/eval_results.txt"
PYTHON="/dev/shm/qwen35/bin/python"

echo "=== Post-Training Evaluation ==="
echo "LoRA: $LORA_DIR"
echo "Model: $MODEL_PATH"
echo ""

# Verify LoRA exists
if [ ! -d "$LORA_DIR" ]; then
    echo "ERROR: LoRA dir not found: $LORA_DIR"
    exit 1
fi

# Kill any lingering vLLM processes
if pgrep -f "vllm.entrypoints" > /dev/null 2>&1; then
    echo "Killing existing vLLM process..."
    pkill -f "vllm.entrypoints" || true
    sleep 5
fi

echo "Starting vLLM with base + trained LoRA (tensor-parallel=2, port 8001)..."
apptainer exec --nv ~/apptainer-images/cuda-custom-amal_latest.sif bash -c "
    export LD_PRELOAD=/dev/shm/qwen35/lib/libstdc++.so.6
    /dev/shm/qwen35/bin/python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH \
        --served-model-name qwen35 \
        --tensor-parallel-size 2 \
        --port 8001 \
        --host 0.0.0.0 \
        --max-model-len 65536 \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --enable-lora \
        --lora-modules qwen35-trained=$LORA_DIR
" 2>&1 | tee "$VLLM_LOG" &
VLLM_PID=$!

echo "Waiting for vLLM to be ready..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo "vLLM ready (${i}s)"
        break
    fi
    sleep 5
done

echo ""
echo "--- BASELINE (base model, no LoRA) ---"
export API_BASE_URL="http://localhost:8001/v1"
export MODEL_NAME="qwen35"
export HF_TOKEN="sk-local"
export MAX_STEPS="3"
export INFERENCE_SERVER_PORT="18082"

$PYTHON inference.py 2>&1 | tee /tmp/eval_base.log
BASE_SCORE=$(grep "\[END\]" /tmp/eval_base.log | grep -oP "score=\K[\d.]+" | paste -sd+ | bc -l 2>/dev/null || echo "0")

echo ""
echo "--- TRAINED (with LoRA) ---"
export MODEL_NAME="qwen35-trained"

$PYTHON inference.py 2>&1 | tee /tmp/eval_trained.log
TRAINED_SCORE=$(grep "\[END\]" /tmp/eval_trained.log | grep -oP "score=\K[\d.]+" | paste -sd+ | bc -l 2>/dev/null || echo "0")

echo ""
echo "=== COMPARISON ==="
echo "Base mean:    $BASE_SCORE"
echo "Trained mean: $TRAINED_SCORE"

{
    echo "=== Evaluation Results $(date) ==="
    echo "LoRA: $LORA_DIR"
    echo ""
    echo "Base inference log:"
    grep "\[END\]\|\[START\]" /tmp/eval_base.log
    echo ""
    echo "Trained inference log:"
    grep "\[END\]\|\[START\]" /tmp/eval_trained.log
} > "$RESULTS_FILE"

echo "Results saved to $RESULTS_FILE"

# Kill vLLM
kill $VLLM_PID 2>/dev/null || pkill -f "vllm.entrypoints" || true
wait $VLLM_PID 2>/dev/null || true

echo ""
echo "To start run 2 (resume from trained LoRA with fixed CRITIC + MAX_NEW_TOKENS=2048):"
echo "  apptainer exec --nv ~/apptainer-images/cuda-custom-amal_latest.sif bash -c \\"
echo "    'export LD_PRELOAD=/dev/shm/qwen35/lib/libstdc++.so.6; \\"
echo "     /dev/shm/qwen35/bin/python train.py \\"
echo "       --phase developer \\"
echo "       --episodes 10 \\"
echo "       --k-rollouts 4 \\"
echo "       --model $MODEL_PATH \\"
echo "       --checkpoint-dir checkpoints/run3 \\"
echo "       --resume-from $LORA_DIR' \\"
echo "    2>&1 | tee checkpoints/train_run3.log"
