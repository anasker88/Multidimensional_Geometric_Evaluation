#!/usr/bin/env bash
set -euo pipefail

# Run save_activation.py then validate.py in sequence (skip save if results exist).

MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-7B-Instruct"}
MODEL_NAME_OVERRIDE=${MODEL_NAME_OVERRIDE:-"Qwen/Qwen2.5-7B-Instruct"}
SAE_RELEASE=${SAE_RELEASE:-"../SAE/sae"}
SAE_ID=${SAE_ID:-"Qwen/Qwen2.5-7B-Instruct/17"}
POOLING=${POOLING:-"max"}
BATCH_SIZE=${BATCH_SIZE:-8}
REASONING=${REASONING:-"without"}
OUTPUT_DIR=${OUTPUT_DIR:-"sae_activations"}
DEVICE=${DEVICE:-"auto"}

SINGULAR_DIMS=${SINGULAR_DIMS:-"3,2"}
TOPK=${TOPK:-10}
SINGULAR_METHOD=${SINGULAR_METHOD:-"both"}
ENTROPY_ALPHA=${ENTROPY_ALPHA:-0.7}
PLOT_REASON_SCORE=${PLOT_REASON_SCORE:-1}
PLOT_QUANTILE=${PLOT_QUANTILE:-0.997}

MODEL_DIR=${MODEL_NAME//\//_}
HAS_ACTIVATION_FILES=0

LAYER_LABEL=${LAYER_LABEL:-""}
if [[ -z "$LAYER_LABEL" ]]; then
	SAE_ID_LAST=${SAE_ID##*/}
	if [[ "$SAE_ID_LAST" =~ ^[0-9]+$ ]]; then
		LAYER_LABEL="layer_${SAE_ID_LAST}"
	elif [[ "$SAE_ID_LAST" == layer_* ]]; then
		LAYER_LABEL="$SAE_ID_LAST"
	fi
fi

if [[ -f "$OUTPUT_DIR/metadata.json" ]]; then
	HAS_ACTIVATION_FILES=1
elif ls "$OUTPUT_DIR"/feature_activations_dim*_*.pt >/dev/null 2>&1; then
	HAS_ACTIVATION_FILES=1
elif [[ -d "$OUTPUT_DIR/$MODEL_DIR" ]]; then
	if find "$OUTPUT_DIR/$MODEL_DIR" -maxdepth 2 -type f \( -name "metadata.json" -o -name "feature_activations_dim*_*.pt" \) -print -quit 2>/dev/null | grep -q .; then
		HAS_ACTIVATION_FILES=1
	fi
fi

if [[ "$HAS_ACTIVATION_FILES" != "1" ]]; then
	python save_activation.py \
		--model-name "$MODEL_NAME" \
		--model-name-override "$MODEL_NAME_OVERRIDE" \
		--sae-release "$SAE_RELEASE" \
		--sae-id "$SAE_ID" \
		--pooling "$POOLING" \
		--reasoning "$REASONING" \
		--batch-size "$BATCH_SIZE" \
		--output-dir "$OUTPUT_DIR" \
		--device "$DEVICE"
else
	echo "Detected existing activation files; skipping save_activation.py"
fi

VALIDATE_ARGS=(
	--output-dir "$OUTPUT_DIR"
	--model-name "$MODEL_NAME"
	--singular-dims "$SINGULAR_DIMS"
	--topk "$TOPK"
	--singular-method "$SINGULAR_METHOD"
	--entropy-alpha "$ENTROPY_ALPHA"
	--plot-quantile "$PLOT_QUANTILE"
)

if [[ -n "$LAYER_LABEL" ]]; then
	VALIDATE_ARGS+=(--layer "$LAYER_LABEL")
fi

if [[ "$PLOT_REASON_SCORE" == "1" ]]; then
	VALIDATE_ARGS+=(--plot-reason-score)
fi

python validate.py "${VALIDATE_ARGS[@]}"
