#!/usr/bin/env bash
# gemma-2-27b-it patching — Phase 1 (resid sweep) + Phase 2 (component/attn+MLP) IN PARALLEL.
# Phase 1 and Phase 2 are independent (Phase 3 depends on Phase 2's attn·last peak, not on Phase 1),
# so we run both concurrently across the 4× A100 80GB. Each phase is sharded across 2 GPUs; each of
# the 4 processes loads the 27B model once (bf16, ~51GB — fits a single 80GB GPU, parallel loads).
#   Phase 1 (patch_run):        GPU 0,1  -> shards 0,1 of 2
#   Phase 2 (patch_components): GPU 2,3  -> shards 0,1 of 2
# 27B in fp32 does not fit -> --dtype bfloat16 (patching indicators are dtype-robust per roadmap).
# Same 425-pair set across all phases (family recorded per-pair for post-hoc stratification; no cap,
# matching the tested gemma-2-9b methodology).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
source .venv/bin/activate
export HF_TOKEN="$(cat ~/.cache/huggingface/token)" HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

MODEL="google/gemma-2-27b-it"
PAIRS="results/patching/pairs/gemma2_27b_aligned.json"
RUN_OUT="results/patching/run/gemma2_27b"
CMP_OUT="results/patching/components/gemma2_27b"
DTYPE="bfloat16"; POS="edit,last"; N=2
mkdir -p "$RUN_OUT" "$CMP_OUT"
ts(){ date -u '+%F %T UTC'; }
echo "[$(ts)] === Phase1+2 parallel START (gemma-2-27b, dtype=$DTYPE) ==="

PIDS=()
# Phase 1 — residual-stream sweep (GPU 0,1)
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python patching/patch_run.py \
    --model-name "$MODEL" --pairs "$PAIRS" --dtype "$DTYPE" \
    --positions "$POS" --out "$RUN_OUT" \
    --num-shards $N --shard-id $i > "$RUN_OUT/shard${i}.log" 2>&1 &
  PIDS+=($!); echo "[$(ts)] P1 shard $i -> GPU $i (pid $!)"
done
# Phase 2 — component (attn/MLP) patching (GPU 2,3)
for i in 0 1; do
  gpu=$((i+2))
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python patching/patch_components.py \
    --model-name "$MODEL" --pairs "$PAIRS" --dtype "$DTYPE" \
    --positions "$POS" --out "$CMP_OUT" \
    --num-shards $N --shard-id $i > "$CMP_OUT/shard${i}.log" 2>&1 &
  PIDS+=($!); echo "[$(ts)] P2 shard $i -> GPU $gpu (pid $!)"
done

echo "[$(ts)] waiting for ${#PIDS[@]} shards: ${PIDS[*]}"
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || { echo "[$(ts)] shard pid $pid FAILED"; FAIL=1; }; done
[ "$FAIL" = 0 ] || { echo "[$(ts)] ABORT: a shard failed (see shard*.log)"; exit 1; }

echo "[$(ts)] === merging shards ==="
.venv/bin/python patching/patch_merge.py --num-shards $N --in-dir "$RUN_OUT"
.venv/bin/python patching/patch_merge.py --num-shards $N --in-dir "$CMP_OUT"
echo "[$(ts)] === Phase1+2 DONE ==="
echo "P12_DONE_MARKER"
