#!/bin/bash
# job.sh — open a new tmux window inside apptainer with qwen35 mamba env activated
# Usage: job [window-name] [env]
#   window-name  default: "work"
#   env          default: /dev/shm/qwen35  (use "vllm" for /dev/shm/vllm)
#
# Aliases: job35 (qwen35 env), jobvllm (ixbrl vllm env)

TMUX_BIN="${HOME}/.local/bin/tmux"
SESSION="${TMUX_SESSION:-job}"
WINDOW="${1:-work}"
ENV_PATH="${2:-/dev/shm/qwen35}"

if [ "$ENV_PATH" = "vllm" ]; then
    ENV_PATH=/dev/shm/vllm
fi

INIT_CMD="source ~/.bashrc; apptainer exec --nv ~/apptainer-images/cuda-custom-amal_latest.sif bash --rcfile <(echo 'source ~/.bashrc; export MAMBA_EXE=\$HOME/.local/bin/micromamba; export MAMBA_ROOT_PREFIX=\$HOME/micromamba; eval \"\$(\$MAMBA_EXE shell hook --shell bash --root-prefix \$MAMBA_ROOT_PREFIX 2>/dev/null)\"; micromamba activate ${ENV_PATH}; export LD_PRELOAD=${ENV_PATH}/lib/libstdc++.so.6; cd ~/workspace; echo \"[job] Inside apptainer, env: ${ENV_PATH}\"')"

# Create session if needed
$TMUX_BIN has-session -t "$SESSION" 2>/dev/null || $TMUX_BIN new-session -d -s "$SESSION"

# Open new window with apptainer + env
$TMUX_BIN new-window -t "$SESSION" -n "$WINDOW"
$TMUX_BIN send-keys -t "$SESSION:$WINDOW" "$INIT_CMD" Enter

echo "Opened tmux window '$WINDOW' in session '$SESSION' with env: $ENV_PATH"
echo "Attach with: tmux attach -t $SESSION"
