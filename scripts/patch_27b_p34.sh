#!/usr/bin/env bash
# gemma-2-27b-it patching — Phase 3 (per-head z) + Phase 4 (top-k ablation).
# Sequential by data dependency: P3 needs the attn·last peak layer (from Phase 2), P4 needs P3's
# head ranking. Both single-process on GPU 0, bf16. The Phase-3 layer window is auto-derived from
# the Phase-2 merged components json (attn:denoise:last pooled 'all' argmax ± 2, clamped).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
source .venv/bin/activate
export HF_TOKEN="$(cat ~/.cache/huggingface/token)" HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

MODEL="google/gemma-2-27b-it"
PAIRS="results/patching/pairs/gemma2_27b_aligned.json"
CMP="results/patching/components/gemma2_27b/patch_results.json"
HEADS_OUT="results/patching/heads/gemma2_27b"
ABL_OUT="results/patching/ablate/gemma2_27b"
DTYPE="bfloat16"; GPU=0
mkdir -p "$HEADS_OUT" "$ABL_OUT"
ts(){ date -u '+%F %T UTC'; }

# --- derive Phase-3 layer window from Phase-2 attn·last peak ---
WIN=$(.venv/bin/python - "$CMP" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); agg=d['aggregate']; layers=d['metadata']['layers']
m=agg['attn:denoise:last']['all']['mean']
pk=max(range(len(m)),key=lambda i:m[i]); peakL=layers[pk]
lo=max(0,peakL-2); hi=min(layers[-1],peakL+2)
win=[L for L in layers if lo<=L<=hi]
sys.stderr.write(f"attn:denoise:last peak = L{peakL} (val {m[pk]:+.4f}); window {win}\n")
print(",".join(map(str,win)))
PY
)
echo "[$(ts)] Phase-3 layer window: $WIN"

echo "[$(ts)] === Phase 3 (patch_heads) on GPU $GPU ==="
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python patching/patch_heads.py \
  --model-name "$MODEL" --pairs "$PAIRS" --dtype "$DTYPE" \
  --layers "$WIN" --out "$HEADS_OUT" 2>&1 | tail -40
echo "[$(ts)] Phase 3 done -> $HEADS_OUT/patch_heads_results.json"

echo "[$(ts)] === Phase 4 (patch_ablate) on GPU $GPU ==="
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python patching/patch_ablate.py \
  --model-name "$MODEL" --pairs "$PAIRS" --dtype "$DTYPE" \
  --heads-json "$HEADS_OUT/patch_heads_results.json" \
  --out "$ABL_OUT" 2>&1 | tail -40
echo "[$(ts)] === Phase 3+4 DONE ==="
echo "P34_DONE_MARKER"
