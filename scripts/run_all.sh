#!/bin/bash
# Re-run all 9 previously-evaluated models on the corrected augmented datasets.
# Memory-safe: TP=1 models paired one-per-GPU; 70B-class run TP=2 sequentially.
# Disk-safe: 70B caches removed after use. Resilient: a model failure is logged,
# the run continues. Detached (setsid+nohup) so it survives SSH disconnect.

cd /home/tota_abe/Multidimensional_Geometric_Evaluation || exit 1
source .venv/bin/activate
HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

DIMS="2,3,4"
TS="${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"
ML="logs/run_master_${TS}.log"
HUB="$HOME/.cache/huggingface/hub"

log(){ echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$ML"; }
disk(){ df -h /home | tail -1 | tee -a "$ML"; }

log "=== RUN START (timestamp=$TS) ==="
log "vLLM $(python -c 'import vllm;print(vllm.__version__)')  | results -> results/$TS/"
disk

# ---- helper: single evaluate.py run (one model) ----
run1(){ # args: GPUS MODEL TP GPUMEM PROMPT MML EXTRA_LABEL
  local gpus="$1" model="$2" tp="$3" mem="$4" prompt="$5" mml="$6" lbl="$7"
  local safe="${model//\//_}"
  local extra=""; [ -n "$mml" ] && extra="--max-model-len $mml"
  log ">>> START $model (GPU=$gpus TP=$tp mem=$mem prompt=$prompt ${mml:+mml=$mml}) ${lbl}"
  CUDA_VISIBLE_DEVICES=$gpus python evaluate.py \
    --models "$model" --dims "$DIMS" \
    --tensor-parallel-size "$tp" --gpu-memory-utilization "$mem" \
    --dtype bfloat16 --max-new-tokens 16 --prompt-type "$prompt" $extra \
    --timestamp "$TS" \
    >>"logs/rerun_${safe}_${TS}.log" 2>&1
  log "<<< DONE  $model (exit $?)"
}

# ============ STAGE 1: TP=1 models, paired one per GPU ============
log "STAGE 1: single-GPU models (2 in parallel)"

run1 0 "Qwen/Qwen3.5-9B"  1 0.85 simple_prompt ""       & P0=$!
run1 1 "Qwen/Qwen3.5-27B" 1 0.85 simple_prompt 200000   & P1=$!
wait $P0; wait $P1

run1 0 "google/gemma-4-12B-it" 1 0.85 simple_prompt ""   & P0=$!
run1 1 "google/gemma-4-31B-it" 1 0.90 simple_prompt 8192 & P1=$!
wait $P0; wait $P1

run1 0 "Qwen/Qwen3-30B-A3B" 1 0.85 simple_prompt ""      & P0=$!
run1 1 "Qwen/Qwen3-32B"     1 0.90 simple_prompt 8192    & P1=$!
wait $P0; wait $P1
disk

# ============ STAGE 2: TP=2 models, sequential (both GPUs) ============
log "STAGE 2: 70B-class models (TP=2, sequential)"

run1 0,1 "Qwen/Qwen3.5-35B-A3B" 2 0.85 simple_prompt ""

run1 0,1 "meta-llama/Llama-3.3-70B-Instruct" 2 0.92 simple_prompt_strict 4096
log "Freeing Llama-70B cache for disk headroom..."
rm -rf "$HUB/models--meta-llama--Llama-3.3-70B-Instruct"; disk

run1 0,1 "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" 2 0.92 simple_prompt 4096
log "Freeing DeepSeek-70B cache..."
rm -rf "$HUB/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B"; disk

log "=== RUN COMPLETE ==="
log "Per-model status:"
grep -hE "^\[.*\] (>>> START|<<< DONE)" "$ML" | tee -a "$ML" >/dev/null
echo "ALL_DONE_MARKER $TS" >> "$ML"
