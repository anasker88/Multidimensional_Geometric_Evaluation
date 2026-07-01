#!/bin/bash
# One-off: add google/gemma-2-27b-it to the final_20260701 sweep on the updated
# dataset. GPUs are now A100 80GB, so 27B fits in bf16 on a single GPU (TP=1) —
# this avoids the old TP=2 NCCL failure and yields valid confidence (unlike the
# patch_candidates_others_20260630 run, which fell back to HF with conf blank).
# Same config as scripts/run_all.sh: greedy + rep1.0 defaults, simple_prompt, maxtok=16.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
source .venv/bin/activate
HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

TS="final_20260701"
MODEL="google/gemma-2-27b-it"
SAFE="${MODEL//\//_}"
LOG="logs/run_${SAFE}_${TS}.log"

echo "[$(date -u '+%F %T UTC')] >>> START $MODEL (GPU=0 TP=1 bf16 simple_prompt maxtok=16)" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate.py \
  --models "$MODEL" --dims "2,3,4" \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
  --dtype bfloat16 --max-new-tokens 16 --prompt-type simple_prompt \
  --results-root results/eval --timestamp "$TS" \
  >>"$LOG" 2>&1
echo "[$(date -u '+%F %T UTC')] <<< DONE $MODEL (exit $?)" | tee -a "$LOG"
