#!/bin/bash
# Full curated sweep: 11 models on the corrected datasets, with per-answer
# confidence + numeric_mc + empty tracking. Writes to results/<RUN_TS>/<model>/.
#
# Scheduling:
#   - 100B-class models run TP=4 (all GPUs), sequentially.
#   - mid models run TP=2 paired across GPU {0,1} and {2,3}; small models TP=1.
# Disk-safety (host runs ~95% full): the four 100B-class caches are deleted
# right after their run, and the giants run FIRST so their freed space covers
# the one model that still needs downloading (Qwen3-Next-80B). gpt-oss needs the
# venv-activated ninja on PATH (handled by `source .venv/bin/activate`).
#
# Detached, SSH-disconnect-safe launch:
#   RUN_TS=final_$(date -u +%Y%m%d) setsid nohup bash scripts/run_all.sh >/dev/null 2>&1 &

cd /home/tota_abe/Multidimensional_Geometric_Evaluation || exit 1
source .venv/bin/activate
HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

DIMS="2,3,4"
TS="${RUN_TS:-final_$(date -u +%Y%m%d)}"
ML="logs/run_master_${TS}.log"
HUB="$HOME/.cache/huggingface/hub"

log(){ echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$ML"; }
disk(){ df -h /home | tail -1 | tee -a "$ML"; }

run1(){  # gpus model tp mem prompt maxtok [extra args...]
  local gpus="$1" model="$2" tp="$3" mem="$4" prompt="$5" maxtok="$6"; shift 6
  local safe="${model//\//_}"
  log ">>> START $model (GPU=$gpus TP=$tp prompt=$prompt maxtok=$maxtok $*)"
  CUDA_VISIBLE_DEVICES="$gpus" python evaluate.py \
    --models "$model" --dims "$DIMS" \
    --tensor-parallel-size "$tp" --gpu-memory-utilization "$mem" \
    --dtype bfloat16 --max-new-tokens "$maxtok" --prompt-type "$prompt" "$@" \
    --results-root results --timestamp "$TS" \
    >>"logs/run_${safe}_${TS}.log" 2>&1
  log "<<< DONE  $model (exit $?)"
}
del_cache(){ rm -rf "$HUB/models--${1//\//--}"; log "freed cache: $1"; disk; }

log "=== RUN START ($TS) — results/$TS/ ==="
log "vLLM $(python -c 'import vllm;print(vllm.__version__)')"
disk

# ---- Stage A: 100B-class, TP=4, sequential. Giants first to free disk. ----
run1 0,1,2,3 "Qwen/Qwen3-235B-A22B-FP8"          4 0.85 simple_prompt        16
del_cache "Qwen/Qwen3-235B-A22B-FP8"
run1 0,1,2,3 "Qwen/Qwen3.5-122B-A10B"            4 0.85 simple_prompt        16
del_cache "Qwen/Qwen3.5-122B-A10B"
run1 0,1,2,3 "Qwen/Qwen3-Next-80B-A3B-Instruct"  4 0.85 simple_prompt        16
del_cache "Qwen/Qwen3-Next-80B-A3B-Instruct"
run1 0,1,2,3 "openai/gpt-oss-120b"               4 0.85 simple_prompt_strict 512 --reasoning-effort low
del_cache "openai/gpt-oss-120b"

# ---- Stage B: mid models, TP=2 paired across GPU {0,1} and {2,3}. ----
run1 0,1 "Qwen/Qwen3.5-35B-A3B" 2 0.85 simple_prompt 16 & A=$!
run1 2,3 "Qwen/Qwen3.5-27B"     2 0.85 simple_prompt 16 & B=$!
wait $A; wait $B
run1 0,1 "Qwen/Qwen3-30B-A3B"   2 0.85 simple_prompt 16 & A=$!
run1 2,3 "Qwen/Qwen3-32B"       2 0.85 simple_prompt 16 & B=$!
wait $A; wait $B

# ---- Stage C: gemma-31B (TP=2) ∥ small models (TP=1) ----
run1 0,1 "google/gemma-4-31B-it" 2 0.85 simple_prompt 16 & A=$!
run1 2   "Qwen/Qwen3.5-9B"       1 0.85 simple_prompt 16 & B=$!
run1 3   "google/gemma-4-12B-it" 1 0.85 simple_prompt 16 & C=$!
wait $A; wait $B; wait $C

log "=== RUN COMPLETE ($TS) ==="
echo "ALL_DONE_MARKER $TS" >> "$ML"