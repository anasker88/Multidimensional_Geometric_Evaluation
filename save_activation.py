import argparse
import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM
from sae_lens import SAE, HookedSAETransformer

from prompting import make_prompt_mc, make_prompt_numeric

# -------------------------------------------------------------
# Data container
# -------------------------------------------------------------
@dataclass
class PromptItem:
	uid: str
	source: str
	dimension: int | None
	qtype: str | None
	question: str
	prompt_type: str


def _read_questions_augmented(csv_path: str, reasoning: bool) -> List[PromptItem]:
	# Load augmented MC questions and convert into prompts
	items: List[PromptItem] = []
	with open(csv_path, "r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		for i, row in enumerate(reader):
			qtype = (row.get("type") or "").strip() or None
			for dim_col, dim in [("2D", 2), ("3D", 3), ("4D", 4)]:
				q = (row.get(dim_col) or "").strip()
				if not q or q == "-":
					continue
				uid = f"questions_augmented:{i}:{dim_col}"
				prompt = make_prompt_mc(q, qtype or "1", reasoning=reasoning)
				items.append(
					PromptItem(
						uid=uid,
						source="questions_augmented",
						dimension=dim,
						qtype=qtype,
						question=prompt,
						prompt_type="with_reasoning" if reasoning else "without_reasoning",
					)
				)
	return items


def _read_numeric_augmented(csv_path: str, reasoning: bool) -> List[PromptItem]:
	# Load augmented numeric questions and convert into prompts
	items: List[PromptItem] = []
	with open(csv_path, "r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		for i, row in enumerate(reader):
			q = (row.get("question") or "").strip()
			if not q:
				continue
			dim = None
			try:
				dim = int(row.get("dimension", "").strip())
			except Exception:
				dim = None
			uid = f"numeric_augmented:{i}"
			prompt = make_prompt_numeric(q, reasoning=reasoning)
			items.append(
				PromptItem(
					uid=uid,
					source="numeric_augmented",
					dimension=dim,
					qtype="numeric",
					question=prompt,
					prompt_type="with_reasoning" if reasoning else "without_reasoning",
				)
			)
	return items


def _pool_acts(acts: torch.Tensor, mode: str) -> torch.Tensor:
	# acts shape: (batch, seq, d_sae)
	if mode == "last":
		return acts[:, -1, :]
	if mode == "mean":
		return acts.mean(dim=1)
	if mode == "max":
		return acts.max(dim=1).values
	raise ValueError(f"Unknown pooling mode: {mode}")


@torch.no_grad()
def compute_feature_activations(
	model: HookedSAETransformer,
	sae: SAE,
	prompts: List[str],
	batch_size: int,
	pooling: str,
) -> torch.Tensor:
	# Run model+SAE, then pool activations over sequence positions
	all_vecs: List[torch.Tensor] = []
	# Resolve hook name for different SAE config types (e.g., Gemma-Scope)
	hook_name = getattr(sae.cfg, "hook_name", None)
	if hook_name is None and getattr(sae.cfg, "metadata", None) is not None:
		hook_name = getattr(sae.cfg.metadata, "hook_name", None)
	if hook_name is None:
		raise ValueError("Could not resolve SAE hook name from config.")

	hook_point = hook_name + ".hook_sae_acts_post"
	hook_layer = getattr(sae.cfg, "hook_layer", None)
	if hook_layer is None:
		m = re.search(r"blocks\.(\d+)\.", hook_name)
		if m:
			hook_layer = int(m.group(1))
	stop_at_layer = hook_layer + 1 if hook_layer is not None else None
	for i in range(0, len(prompts), batch_size):
		batch = prompts[i : i + batch_size]
		kwargs = {"names_filter": [hook_point]}
		if stop_at_layer is not None:
			kwargs["stop_at_layer"] = stop_at_layer
		_, cache = model.run_with_cache_with_saes(batch, saes=[sae], **kwargs)
		acts = cache[hook_point]
		vecs = _pool_acts(acts, pooling).to("cpu")
		all_vecs.append(vecs)
		del cache, acts, vecs
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
	return torch.cat(all_vecs, dim=0)


def main() -> None:
	# CLI entrypoint
	parser = argparse.ArgumentParser(
		description="Compute SAE feature activations for augmented geometry questions."
	)
	parser.add_argument(
		"--questions-csv",
		default="data/questions_augmented.csv",
		help="Path to questions_augmented.csv",
	)
	parser.add_argument(
		"--numeric-csv",
		default="data/numeric_augmented.csv",
		help="Path to numeric_augmented.csv",
	)
	parser.add_argument(
		"--model-name",
		default="google/gemma-2-9b",
		help="HF model name for HookedSAETransformer",
	)
	parser.add_argument(
		"--model-name-override",
		default=None,
		help="Optional override model name for loading base model",
	)
	parser.add_argument(
		"--sae-release",
		default="gemma-scope-9b-pt-mlp-canonical",
		help="SAE release name (gemma-scope)",
	)
	parser.add_argument(
		"--sae-id",
		default="layer_20/width_16k/canonical",
		help="SAE id within the release",
	)
	parser.add_argument(
		"--pooling",
		default="max",
		choices=["max", "mean", "last"],
		help="Pooling over sequence positions",
	)
	parser.add_argument(
		"--reasoning",
		default="without",
		choices=["with", "without", "both"],
		help="Prompt style: with reasoning, without reasoning, or both",
	)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument(
		"--output-dir",
		default="sae_activations",
		help="Output directory",
	)
	parser.add_argument(
		"--device",
		default="auto",
		choices=["auto", "cuda", "cpu", "mps"],
		help="Device for model/SAE (auto picks available accelerator)",
	)
	parser.add_argument(
		"--fallback-cpu",
		action="store_true",
		help="Fallback to CPU if CUDA OOM occurs",
	)
	args = parser.parse_args()

	if args.device == "auto":
		device = (
			"cuda"
			if torch.cuda.is_available()
			else "mps" if torch.backends.mps.is_available() else "cpu"
		)
	else:
		device = args.device

	# Use explicit override for model loading if provided
	model_name_for_load = args.model_name_override or args.model_name
	use_hf_model = args.model_name_override is not None and args.model_name_override != args.model_name
	hf_model = None
	if use_hf_model:
		print(f"Loading HF model from {args.model_name} (override name: {model_name_for_load})")
		hf_model = AutoModelForCausalLM.from_pretrained(args.model_name)

	# Resolve SAE source (local disk vs HF release)
	sae_from_disk = False
	sae_disk_path = None
	if args.sae_release and os.path.isdir(args.sae_release):
		sae_disk_path = os.path.join(args.sae_release, args.sae_id)
		if os.path.isdir(sae_disk_path):
			sae_from_disk = True
		else:
			raise FileNotFoundError(
				f"Local SAE path not found: {sae_disk_path}"
			)

	# Load SAE first to access any recommended model kwargs
	if sae_from_disk:
		sae = SAE.load_from_disk(sae_disk_path, device=device)
	else:
		sae = SAE.from_pretrained(args.sae_release, args.sae_id, device=device)

	# Load base model (prefer from_pretrained_no_processing when available)
	model_kwargs = getattr(sae.cfg, "model_from_pretrained_kwargs", None) or {}
	if hf_model is not None:
		model_kwargs = {**model_kwargs, "hf_model": hf_model}
	model_loader = getattr(HookedSAETransformer, "from_pretrained_no_processing", None)
	if callable(model_loader):
		model = model_loader(
			model_name_for_load,
			device=device,
			**model_kwargs,
		)
	else:
		model = HookedSAETransformer.from_pretrained(
			model_name_for_load,
			device=device,
			hf_model=hf_model,
		)

	# Resolve SAE layer for output path
	hook_layer = getattr(sae.cfg, "hook_layer", None)
	if hook_layer is None:
		hook_name = getattr(sae.cfg, "hook_name", None)
		if hook_name is None and getattr(sae.cfg, "metadata", None) is not None:
			hook_name = getattr(sae.cfg.metadata, "hook_name", None)
		if hook_name is not None:
			m = re.search(r"blocks\.(\d+)\.", hook_name)
			if m:
				hook_layer = int(m.group(1))
	layer_label = f"layer_{hook_layer}" if hook_layer is not None else "layer_unknown"

	# Build output directory: sae_activations/{model_name}/{layer}
	model_dir = args.model_name.replace("/", "_")
	os.makedirs(args.output_dir, exist_ok=True)
	out_dir = os.path.join(args.output_dir, model_dir, layer_label)
	os.makedirs(out_dir, exist_ok=True)

	reasoning_modes = []
	if args.reasoning in ("with", "both"):
		reasoning_modes.append(True)
	if args.reasoning in ("without", "both"):
		reasoning_modes.append(False)

	# Results: {reasoning_mode: {dimension: top10 list}}
	results: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
	all_items: List[PromptItem] = []
	saved_activation_files: List[str] = []

	for reasoning in reasoning_modes:
		mode_key = "with_reasoning" if reasoning else "without_reasoning"
		items = _read_questions_augmented(args.questions_csv, reasoning=reasoning)
		items.extend(_read_numeric_augmented(args.numeric_csv, reasoning=reasoning))
		all_items.extend(items)

		# Group prompts by dimension
		by_dim: Dict[int, List[PromptItem]] = {}
		for it in items:
			if it.dimension is None:
				continue
			by_dim.setdefault(it.dimension, []).append(it)

		results[mode_key] = {}
		for dim, dim_items in sorted(by_dim.items()):
			# Compute SAE activations for this dimension
			prompts = [it.question for it in dim_items]
			acts = compute_feature_activations(
				model=model,
				sae=sae,
				prompts=prompts,
				batch_size=args.batch_size,
				pooling=args.pooling,
			)

			# Save per-dimension activation tensor
			activation_path = os.path.join(
				out_dir, f"feature_activations_dim{dim}_{mode_key}.pt"
			)
			torch.save(acts, activation_path)
			saved_activation_files.append(os.path.relpath(activation_path, out_dir))
			# Top-10 features by mean activation
			feature_mean = acts.mean(dim=0)
			vals, inds = torch.topk(feature_mean, k=min(10, feature_mean.numel()))
			top10 = [
				{"feature_index": int(idx), "mean_activation": float(val)}
				for idx, val in zip(inds.tolist(), vals.tolist())
			]
			results[mode_key][str(dim)] = top10

			# Write per-dimension top10 summary
			with open(
				os.path.join(out_dir, f"top10_features_dim{dim}_{mode_key}.txt"),
				"w",
				encoding="utf-8",
			) as f:
				f.write("rank\tfeature_index\tmean_activation\n")
				for rank, entry in enumerate(top10, start=1):
					f.write(
						f"{rank}\t{entry['feature_index']}\t{entry['mean_activation']:.6f}\n"
					)

	# Save metadata for reproducibility
	with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
		json.dump(
			{
				"model_name": args.model_name,
				"sae_release": args.sae_release,
				"sae_id": args.sae_id,
				"pooling": args.pooling,
				"batch_size": args.batch_size,
				"reasoning": args.reasoning,
				"num_prompts": len(all_items),
				"top10_by_dimension": results,
				"activation_files": saved_activation_files,
				"items": [asdict(it) for it in all_items],
			},
			f,
			ensure_ascii=False,
			indent=2,
		)


if __name__ == "__main__":
	main()
