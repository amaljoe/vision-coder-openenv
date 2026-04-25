Open a tmux window named `job:<window>` inside apptainer with the qwen35 mamba env activated.

Usage:
  `/job <window-name>`                    — open window, drop into shell
  `/job <window-name>: <command>`         — open window and run command immediately

Examples:
  `/job work`
  `/job train`
  `/job train-alt: python train.py --phase alternate --episodes-per-phase 20`

`$ARGUMENTS` is parsed as `<window>` or `<window>: <command>`.
Window name defaults to "work" if omitted.

```bash
TMUX_BIN="$HOME/.local/bin/tmux"
SESSION="job"
ENV_PATH="/dev/shm/qwen35"

# Parse arguments: "window-name: command to run"  OR  just "window-name"
ARGS="$ARGUMENTS"
if [[ "$ARGS" == *": "* ]]; then
    WINDOW="${ARGS%%: *}"
    COMMAND="${ARGS#*: }"
else
    WINDOW="${ARGS:-work}"
    COMMAND=""
fi

# Create session if it doesn't exist
$TMUX_BIN has-session -t "$SESSION" 2>/dev/null || $TMUX_BIN new-session -d -s "$SESSION" -x 220 -y 50

# Reuse existing window or create new one
IS_NEW=false
if $TMUX_BIN list-windows -t "$SESSION" -F "#{window_name}" 2>/dev/null | grep -qx "$WINDOW"; then
    echo "Reusing existing window: $SESSION:$WINDOW"
else
    IS_NEW=true
    $TMUX_BIN new-window -t "$SESSION" -n "$WINDOW"
    INIT_CMD="apptainer exec --nv ~/apptainer-images/cuda-custom-amal_latest.sif bash --rcfile <(echo 'source ~/.bashrc; export MAMBA_EXE=\$HOME/.local/bin/micromamba; export MAMBA_ROOT_PREFIX=\$HOME/micromamba; eval \"\$(\$MAMBA_EXE shell hook --shell bash --root-prefix \$MAMBA_ROOT_PREFIX 2>/dev/null)\"; micromamba activate ${ENV_PATH}; export LD_PRELOAD=${ENV_PATH}/lib/libstdc++.so.6; cd ~/workspace/vision-coder-openenv; echo \"[job] apptainer ready | env: ${ENV_PATH} | session: ${SESSION}:${WINDOW}\"')"
    $TMUX_BIN send-keys -t "$SESSION:$WINDOW" "$INIT_CMD" Enter
    echo "Created window $SESSION:$WINDOW — apptainer + qwen35 env starting"
fi

# Send command if provided (wait for apptainer to boot on new windows)
if [[ -n "$COMMAND" ]]; then
    if $IS_NEW; then
        sleep 7
    fi
    $TMUX_BIN send-keys -t "$SESSION:$WINDOW" "$COMMAND" Enter
    echo "Sent to $SESSION:$WINDOW: $COMMAND"
fi

echo "Attach with: ~/.local/bin/tmux attach -t $SESSION"
```
