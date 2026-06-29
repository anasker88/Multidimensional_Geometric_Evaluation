#!/usr/bin/env bash
# Multi-GPU activation patching launcher.
# Splits pairs across NUM_GPUS processes (one per GPU), then merges.
#
# Usage:
#   bash scripts/patch_launch_multigpu.sh [PAIRS_JSON] [OUT_DIR] [NUM_GPUS]
#
# Examples:
#   bash scripts/patch_launch_multigpu.sh   # defaults below
#   bash scripts/patch_launch_multigpu.sh output/patch_pairs/qwen35_9b_aligned.json output/patch_run/qwen35_9b 4

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PAIRS="${1:-output/patch_pairs/qwen35_9b_aligned.json}"
OUT="${2:-output/patch_run/qwen35_9b}"
NUM_GPUS="${3:-4}"

if [ ! -f "$PAIRS" ]; then
  echo "ERROR: pairs file not found: $PAIRS"
  echo "Run patch_pairs.py first:"
  echo "  .venv-patch/bin/python validation/patch_pairs.py --balance 40 --balance-types 1,2,3 --aligned-only --out $PAIRS"
  exit 1
fi

mkdir -p "$OUT"
echo "=== patch_launch_multigpu ==="
echo "pairs : $PAIRS"
echo "out   : $OUT"
echo "GPUs  : $NUM_GPUS"
echo ""

PIDS=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
  LOG="$OUT/shard${i}.log"
  echo "Starting shard $i on GPU $i -> log: $LOG"
  CUDA_VISIBLE_DEVICES=$i .venv-patch/bin/python validation/patch_run.py \
    --pairs "$PAIRS" \
    --out "$OUT" \
    --shard-id "$i" \
    --num-shards "$NUM_GPUS" \
    > "$LOG" 2>&1 &
  PIDS+=($!)
done

echo ""
echo "Waiting for ${#PIDS[@]} shards (PIDs: ${PIDS[*]})..."
FAILED=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  if wait "$pid"; then
    echo "  shard $i (PID $pid): done"
  else
    echo "  shard $i (PID $pid): FAILED (see $OUT/shard${i}.log)"
    FAILED=$((FAILED + 1))
  fi
done

if [ "$FAILED" -gt 0 ]; then
  echo "ERROR: $FAILED shard(s) failed. Check logs in $OUT/"
  exit 1
fi

echo ""
echo "=== Merging $NUM_GPUS shards ==="
.venv-patch/bin/python validation/patch_merge.py \
  --num-shards "$NUM_GPUS" \
  --in-dir "$OUT"

echo ""
echo "=== Done ==="
echo "Results: $OUT/patch_results.json"
echo "Plots  : $OUT/plots/"
