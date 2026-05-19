#!/usr/bin/env bash
set -euo pipefail

# Use GPU 3 by default; allow caller override.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
# Run validate.py with on-demand activation generation.

MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-7B-Instruct"}
MODEL_NAME_OVERRIDE=${MODEL_NAME_OVERRIDE:-"Qwen/Qwen2.5-7B-Instruct"}
SAE_RELEASE=${SAE_RELEASE:-"../SAE/sae"}
SAE_ID=${SAE_ID:-"Qwen/Qwen2.5-7B-Instruct/17"}
POOLING=${POOLING:-"max"}
BATCH_SIZE=${BATCH_SIZE:-8}
REASONING=${REASONING:-"without"}
OUTPUT_DIR=${OUTPUT_DIR:-"sae_activations"}
DEVICE=${DEVICE:-"auto"}
ACTIVATION_FALLBACK_CPU=${ACTIVATION_FALLBACK_CPU:-0}

SINGULAR_DIMS=${SINGULAR_DIMS:-"4,3"}
TOPK=${TOPK:-10}
SINGULAR_METHOD=${SINGULAR_METHOD:-"both"}
ENTROPY_ALPHA=${ENTROPY_ALPHA:-0.7}
PLOT_REASON_SCORE=${PLOT_REASON_SCORE:-1}
PLOT_QUANTILE=${PLOT_QUANTILE:-0.997}
FILTER_CORRECT_DIR=${FILTER_CORRECT_DIR:-""}

MODEL_DIR=${MODEL_NAME//\//_}
VALIDATE_RESULTS_ROOT=${VALIDATE_RESULTS_ROOT:-"output/validate"}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-"$(date +%Y%m%d_%H%M%S)"}
SINGULAR_DIMS_LABEL=${SINGULAR_DIMS//,/vs}
RESULTS_DIR=${RESULTS_DIR:-"$VALIDATE_RESULTS_ROOT/${RUN_TIMESTAMP}_${MODEL_DIR}_dim${SINGULAR_DIMS_LABEL}"}

LAYER_LABEL=${LAYER_LABEL:-""}
if [[ -z "$LAYER_LABEL" ]]; then
	SAE_ID_LAST=${SAE_ID##*/}
	if [[ "$SAE_ID_LAST" =~ ^[0-9]+$ ]]; then
		LAYER_LABEL="layer_${SAE_ID_LAST}"
	elif [[ "$SAE_ID_LAST" == layer_* ]]; then
		LAYER_LABEL="$SAE_ID_LAST"
	fi
fi

VALIDATE_ARGS=(
	--output-dir "$OUTPUT_DIR"
	--results-dir "$RESULTS_DIR"
	--model-name "$MODEL_NAME"
	--sae-release "$SAE_RELEASE"
	--sae-id "$SAE_ID"
	--activation-pooling "$POOLING"
	--activation-reasoning "$REASONING"
	--activation-batch-size "$BATCH_SIZE"
	--activation-device "$DEVICE"
	--activation-model-name-override "$MODEL_NAME_OVERRIDE"
	--singular-dims "$SINGULAR_DIMS"
	--topk "$TOPK"
	--singular-method "$SINGULAR_METHOD"
	--entropy-alpha "$ENTROPY_ALPHA"
	--plot-quantile "$PLOT_QUANTILE"
)

if [[ "$ACTIVATION_FALLBACK_CPU" == "1" ]]; then
	VALIDATE_ARGS+=(--activation-fallback-cpu)
fi

if [[ -n "$LAYER_LABEL" ]]; then
	VALIDATE_ARGS+=(--layer "$LAYER_LABEL")
fi

if [[ "$PLOT_REASON_SCORE" == "1" ]]; then
	VALIDATE_ARGS+=(--plot-reason-score)
fi

if [[ -n "$FILTER_CORRECT_DIR" ]]; then
	VALIDATE_ARGS+=(--filter-correct-dir "$FILTER_CORRECT_DIR")
fi

mkdir -p "$RESULTS_DIR"
cat >"$RESULTS_DIR/run_conditions_shell.json" <<EOF
{
	"run_timestamp": "$RUN_TIMESTAMP",
	"model_name": "$MODEL_NAME",
	"model_name_override": "$MODEL_NAME_OVERRIDE",
	"sae_release": "$SAE_RELEASE",
	"sae_id": "$SAE_ID",
	"pooling": "$POOLING",
	"batch_size": $BATCH_SIZE,
	"reasoning": "$REASONING",
	"output_dir": "$OUTPUT_DIR",
	"device": "$DEVICE",
	"activation_fallback_cpu": "$ACTIVATION_FALLBACK_CPU",
	"singular_dims": "$SINGULAR_DIMS",
	"topk": $TOPK,
	"singular_method": "$SINGULAR_METHOD",
	"entropy_alpha": $ENTROPY_ALPHA,
	"plot_reason_score": "$PLOT_REASON_SCORE",
	"plot_quantile": $PLOT_QUANTILE,
	"filter_correct_dir": "$FILTER_CORRECT_DIR",
	"results_dir": "$RESULTS_DIR"
}
EOF

echo "Validation outputs will be saved to: $RESULTS_DIR"

python cli/validate.py "${VALIDATE_ARGS[@]}"
