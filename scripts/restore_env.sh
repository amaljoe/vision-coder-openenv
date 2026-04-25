#!/bin/bash
# Restore ~/envs/qwen35.tar.zst → /dev/shm/qwen35
set -euo pipefail

SRC="$HOME/envs/qwen35.tar.zst"
DST=/dev/shm/qwen35

[ -f "$SRC" ] || { echo "ERROR: $SRC not found. Run savevllm35 on a built node."; exit 1; }
[ -d "$DST" ] && { echo "WARNING: $DST exists — removing..."; rm -rf "$DST"; }

echo "Restoring $SRC ($(du -sh "$SRC" | cut -f1)) → $DST"
START=$(date +%s)
zstd -T0 -d --progress "$SRC" --stdout | tar xf - -C /dev/shm
END=$(date +%s)

echo "Extracted in $((END-START))s. Env: $(du -sh "$DST" | cut -f1)"

[ -f "$DST/lib/libstdc++.so.6" ] || micromamba install -p "$DST" -c conda-forge libstdcxx-ng -y --no-banner

export LD_PRELOAD="$DST/lib/libstdc++.so.6"
export APPTAINERENV_LD_PRELOAD="$DST/lib/libstdc++.so.6"
echo "Ready. Run: job35"
