#!/bin/bash
# Save /dev/shm/qwen35 → ~/envs/qwen35.tar.zst
set -euo pipefail

SRC=/dev/shm/qwen35
DST="$HOME/envs/qwen35.tar.zst"
TMP="${DST}.tmp"

[ -d "$SRC" ] || { echo "ERROR: $SRC not found."; exit 1; }

mkdir -p "$(dirname "$DST")"
SIZE_BYTES=$(du -sb "$SRC" | cut -f1)
SIZE_HUMAN=$(du -sh "$SRC" | cut -f1)
echo "Saving $SRC ($SIZE_HUMAN) → $DST"

START=$(date +%s)
tar cf - -C /dev/shm qwen35 | zstd -T0 -3 --progress --size-hint="$SIZE_BYTES" -o "$TMP"
mv "$TMP" "$DST"
END=$(date +%s)

echo "Done in $((END-START))s. Archive: $(du -sh "$DST" | cut -f1)"
echo "Restore with: restorevllm35"
