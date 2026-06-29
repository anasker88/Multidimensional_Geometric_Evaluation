#!/bin/bash
# Full curated sweep: 15 in-scope models on the corrected datasets, with
# per-answer confidence + numeric_mc + empty tracking. Writes to
# results/<RUN_TS>/<model>/.
#
# Sampling uses the corrected defaults baked into evaluate.py (greedy +
# repetition_penalty=1.0) — no sampling flags passed here. rep1.1 was shown to
# be the sole cause of the MC primacy aversion / A-slot underestimate.
#
# Scope: Qwen3-235B-A22B-FP8 (FP8 quant) and gpt-oss-120b (harmony/512tok) are
# excluded — axis-unstable for cross-model comparison. Families covered:
# Qwen3.5 (2B/4B/9B/27B/35B-A3B/122B-A10B), Qwen3 (30B-A3B/32B/Next-80B),
# gemma-4 (E4B/12B/26B-A4B/31B), OLMo-3-7B (Instruct/RL-Zero-Math).
#
# Scheduling: 100B-class TP=4 sequential (cache-freed after, run first); mid
# models TP=2 paired across GPU {0,1}/{2,3}; small models TP=1 one per GPU.
#
# Detached, SSH-disconnect-safe launch:
#   RUN_TS=final_$(date -u +%Y%m%d) setsid nohup bash scripts/run_all.sh >/dev/null 2>&1 &

cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
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
  CUDA_VISIBLE_DEVICES="$gpus" python evaluation/evaluate.py \
    --models "$model" --dims "$DIMS" \
    --tensor-parallel-size "$tp" --gpu-memory-utilization "$mem" \
    --dtype bfloat16 --max-new-tokens "$maxtok" --prompt-type "$prompt" "$@" \
    --results-root results --timestamp "$TS" \
    >>"logs/run_${safe}_${TS}.log" 2>&1
  log "<<< DONE  $model (exit $?)"
}
del_cache(){ rm -rf "$HUB/models--${1//\//--}"; log "freed cache: $1"; disk; }

log "=== RUN START ($TS) — results/$TS/ — greedy + rep1.0 (defaults) ==="
log "vLLM $(python -c 'import vllm;print(vllm.__version__)')"
disk

# ---- Stage A: 100B-class, TP=4, sequential, cache-freed after each. ----
run1 0,1,2,3 "Qwen/Qwen3.5-122B-A10B"            4 0.85 simple_prompt 16
del_cache "Qwen/Qwen3.5-122B-A10B"
run1 0,1,2,3 "Qwen/Qwen3-Next-80B-A3B-Instruct"  4 0.85 simple_prompt 16
del_cache "Qwen/Qwen3-Next-80B-A3B-Instruct"

# ---- Stage B: mid models (TP=2) paired across GPU {0,1} and {2,3}. ----
run1 0,1 "Qwen/Qwen3.5-35B-A3B"      2 0.85 simple_prompt 16 & A=$!
run1 2,3 "Qwen/Qwen3.5-27B"          2 0.85 simple_prompt 16 & B=$!
wait $A; wait $B
run1 0,1 "Qwen/Qwen3-30B-A3B"        2 0.85 simple_prompt 16 & A=$!
run1 2,3 "Qwen/Qwen3-32B"            2 0.85 simple_prompt 16 & B=$!
wait $A; wait $B
run1 0,1 "google/gemma-4-31B-it"     2 0.85 simple_prompt 16 & A=$!
run1 2,3 "google/gemma-4-26B-A4B-it" 2 0.85 simple_prompt 16 & B=$!
wait $A; wait $B

# ---- Stage C: small models (TP=1), one per GPU. ----
run1 0 "Qwen/Qwen3.5-9B"     1 0.85 simple_prompt 16 & A=$!
run1 1 "Qwen/Qwen3.5-4B"     1 0.85 simple_prompt 16 & B=$!
run1 2 "Qwen/Qwen3.5-2B"     1 0.85 simple_prompt 16 & C=$!
run1 3 "google/gemma-4-12B-it" 1 0.85 simple_prompt 16 & D=$!
wait $A; wait $B; wait $C; wait $D
run1 0 "google/gemma-4-E4B-it"           1 0.85 simple_prompt 16 & A=$!
run1 1 "allenai/Olmo-3-7B-Instruct"      1 0.85 simple_prompt 16 & B=$!
run1 2 "allenai/Olmo-3-7B-RL-Zero-Math"  1 0.85 simple_prompt 16 & C=$!
wait $A; wait $B; wait $C

log "=== RUN COMPLETE ($TS) ==="
echo "ALL_DONE_MARKER $TS" >> "$ML"
