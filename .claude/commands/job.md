Open a tmux window named `job:<window>` inside apptainer with the qwen35 mamba env activated.

Usage: `/job <window-name>`  
Examples: `/job vllm`, `/job work`, `/job train`

$ARGUMENTS is the window name (e.g. "vllm", "work", "train"). Defaults to "work" if omitted.

```bash
TMUX_BIN="$HOME/.local/bin/tmux"
SESSION="job"
WINDOW="${1:-work}"
ENV_PATH="/dev/shm/qwen35"

# Create session if it doesn't exist
$TMUX_BIN has-session -t "$SESSION" 2>/dev/null || $TMUX_BIN new-session -d -s "$SESSION" -x 220 -y 50

# Reuse existing window or create new one
if $TMUX_BIN list-windows -t "$SESSION" -F "#{window_name}" 2>/dev/null | grep -qx "$WINDOW"; then
    echo "Attaching to existing window: $SESSION:$WINDOW"
else
    $TMUX_BIN new-window -t "$SESSION" -n "$WINDOW"
    INIT_CMD="apptainer exec --nv ~/apptainer-images/cuda-custom-amal_latest.sif bash --rcfile <(echo 'source ~/.bashrc; export MAMBA_EXE=\$HOME/.local/bin/micromamba; export MAMBA_ROOT_PREFIX=\$HOME/micromamba; eval \"\$(\$MAMBA_EXE shell hook --shell bash --root-prefix \$MAMBA_ROOT_PREFIX 2>/dev/null)\"; micromamba activate ${ENV_PATH}; export LD_PRELOAD=${ENV_PATH}/lib/libstdc++.so.6; cd ~/workspace/vision-coder-openenv; echo \"[job] apptainer ready | env: ${ENV_PATH} | session: ${SESSION}:${WINDOW}\"')"
    $TMUX_BIN send-keys -t "$SESSION:$WINDOW" "$INIT_CMD" Enter
    echo "Created window $SESSION:$WINDOW — apptainer + qwen35 env starting"
fi

echo "Attach with: ~/.local/bin/tmux attach -t $SESSION"
```
