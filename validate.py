import argparse
import contextlib
import csv
import json
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from prompting import make_prompt_mc, make_prompt_numeric
from validation.activation_io import load_feature_activations
from validation.probe import run_probe_and_report, run_random_probe_baseline_and_report
from validation.singular import (
	compute_reason_scores,
	find_singular_activations,
	find_singular_activations_reason_score,
)
from validation.visualize import (
	TokenActivationVisualizer,
	build_prompt_list,
	collect_topk_prompt_data_from_scores,
	extract_question_answer_segments,
	filter_tokens_scores_to_question_answer,
	render_token_activation_plot,
	save_prompt_activation_json,
)


def _summarize(activations: Dict[int, torch.Tensor]) -> None:
	for dim in sorted(activations.keys()):
		tensor = activations[dim]
		shape = tuple(tensor.shape)
		print(f"dim{dim}: shape={shape}, dtype={tensor.dtype}")


def _resolve_prompt_type(
	prompt_type_by_dim: Dict[int, str],
	dim1: int,
	dim2: int,
) -> str:
	if dim1 not in prompt_type_by_dim or dim2 not in prompt_type_by_dim:
		raise ValueError("Missing prompt_type entry for requested dimensions.")
	ptype1 = prompt_type_by_dim[dim1]
	ptype2 = prompt_type_by_dim[dim2]
	if ptype1 != ptype2:
		raise ValueError(
			"Prompt types differ between dimensions. "
			"Specify --prompt-type to select a shared prompt type."
		)
	return ptype1


def _read_correct_prompt_sets(
	results_dir: str,
	dims: List[int],
	prompt_type: str,
) -> Dict[int, set[str]]:
	reasoning = prompt_type == "with_reasoning"
	result: Dict[int, set[str]] = {}
	for dim in dims:
		correct_path = os.path.join(results_dir, f"dim_{dim}_correct.csv")
		per_question_path = os.path.join(results_dir, f"dim_{dim}_per_question.csv")
		path = None
		if os.path.exists(correct_path):
			path = correct_path
		elif os.path.exists(per_question_path):
			path = per_question_path
		else:
			raise FileNotFoundError(
				f"No evaluate output CSV found for dim{dim} in {results_dir}. "
				"Expected dim_{dim}_correct.csv or dim_{dim}_per_question.csv."
			)

		prompts: set[str] = set()
		with open(path, "r", encoding="utf-8", newline="") as f:
			reader = csv.DictReader(f)
			for row in reader:
				if os.path.basename(path) == f"dim_{dim}_per_question.csv":
					if (row.get("is_correct") or "").strip() != "1":
						continue

				type_key = (row.get("type") or "").strip()
				question = (row.get("question") or "").strip()
				if not type_key or not question:
					continue

				if type_key == "numeric":
					prompt = make_prompt_numeric(question, reasoning=reasoning)
				else:
					prompt = make_prompt_mc(question, type_key, reasoning=reasoning)
				prompts.add(prompt)
		result[dim] = prompts

	return result


def _filter_activations_by_prompt_set(
	activations: Dict[int, torch.Tensor],
	metadata: Dict,
	prompt_type: str,
	allowed_prompts_by_dim: Dict[int, set[str]],
) -> Dict[int, torch.Tensor]:
	items = metadata.get("items") or []
	filtered: Dict[int, torch.Tensor] = {}

	for dim, tensor in activations.items():
		if dim not in allowed_prompts_by_dim:
			filtered[dim] = tensor
			continue
		allowed_prompts = allowed_prompts_by_dim.get(dim, set())
		dim_items = [
			item
			for item in items
			if item.get("dimension") == dim and item.get("prompt_type") == prompt_type
		]
		if len(dim_items) != tensor.shape[0]:
			raise ValueError(
				"Activation count does not match metadata items for filtering. "
				f"dim{dim}: activations={tensor.shape[0]}, metadata={len(dim_items)}"
			)

		keep_indices = [
			idx
			for idx, item in enumerate(dim_items)
			if (item.get("question") or "") in allowed_prompts
		]
		if not keep_indices:
			raise ValueError(
				f"No correct prompts matched activation metadata for dim{dim}. "
				"Check --filter-correct-dir and prompt type alignment."
			)
		idx_tensor = torch.tensor(keep_indices, dtype=torch.long)
		filtered[dim] = tensor.index_select(0, idx_tensor)

	return filtered


def _save_validate_run_conditions(
	results_dir: str,
	args: argparse.Namespace,
	extra: Dict | None = None,
) -> None:
	os.makedirs(results_dir, exist_ok=True)
	payload = {
		"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
		"cwd": os.getcwd(),
		"argv": sys.argv,
		"args": vars(args),
	}
	if extra:
		payload["resolved"] = extra
	path = os.path.join(results_dir, "run_conditions_validate.json")
	with open(path, "w", encoding="utf-8") as f:
		json.dump(payload, f, ensure_ascii=True, indent=2)


def _save_validation_summary(results_dir: str, payload: Dict) -> None:
	os.makedirs(results_dir, exist_ok=True)
	path = os.path.join(results_dir, "validation_summary.json")
	with open(path, "w", encoding="utf-8") as f:
		json.dump(payload, f, ensure_ascii=True, indent=2)


def _normalize_label(s: str) -> str:
	return re.sub(r"[^A-Za-z0-9]", "", (s or "")).upper()


def _extract_answer(response: str) -> str | None:
	if not response:
		return None
	tag_matches = re.findall(
		r"<final_answer>\s*([^<\n]+?)\s*</final_answer>",
		response,
		re.IGNORECASE,
	)
	if tag_matches:
		val = tag_matches[-1].strip()
		m = re.match(r"^([A-E])$", val, re.IGNORECASE)
		if m:
			return m.group(1).upper()

	last_box = None
	for m in re.finditer(r"\\boxed\{([^}]*)\}", response, re.IGNORECASE):
		last_box = m.group(1)
	if last_box is not None:
		m2 = re.search(r"[A-Z]", last_box)
		if m2:
			return m2.group(0).upper()
		m3 = re.search(r"[A-Za-z]", last_box)
		if m3:
			return m3.group(0).upper()
	return None


def _extract_numeric(response: str) -> str | None:
	if not response:
		return None
	tag_matches = re.findall(
		r"<final_answer>\s*([^<\n]+?)\s*</final_answer>",
		response,
		re.IGNORECASE,
	)
	if tag_matches:
		txt = tag_matches[-1]
		m = re.search(r"(-?\d+)", txt)
		if m:
			return m.group(1)

	last_box = None
	for m in re.finditer(r"\\boxed\{([^}]*)\}", response, re.IGNORECASE):
		last_box = m.group(1)
	if last_box is not None:
		m = re.search(r"(-?\d+)", last_box)
		if m:
			return m.group(1)

	for line in reversed([ln.strip() for ln in response.splitlines() if ln.strip()]):
		m = re.search(r"(-?\d+)", line)
		if m:
			return m.group(1)
	return None


def _build_eval_records(
	dim: int,
	reasoning: bool,
	questions_csv_path: str,
	numeric_csv_path: str,
) -> List[Dict[str, str]]:
	records: List[Dict[str, str]] = []

	with open(questions_csv_path, "r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			col_key = f"{dim}D"
			if col_key not in row:
				continue
			question = (row.get(col_key) or "").strip()
			if not question or question == "-":
				continue
			type_key = (row.get("type") or "unknown").strip() or "unknown"
			answer = (row.get("answer") or "").strip()
			if not answer:
				continue
			records.append(
				{
					"task_type": "mc",
					"question_type": type_key,
					"prompt": make_prompt_mc(question, type_key, reasoning=reasoning),
					"answer": answer,
				}
			)

	if os.path.exists(numeric_csv_path):
		with open(numeric_csv_path, "r", encoding="utf-8", newline="") as f:
			reader = csv.DictReader(f)
			for row in reader:
				try:
					row_dim = int((row.get("dimension") or "").strip())
				except Exception:
					continue
				if row_dim != dim:
					continue
				question = (row.get("question") or "").strip()
				answer = (row.get("answer") or "").strip()
				if not question or not answer:
					continue
				records.append(
					{
						"task_type": "numeric",
						"question_type": "numeric",
						"prompt": make_prompt_numeric(question, reasoning=reasoning),
						"answer": answer,
					}
				)

	return records


def _run_batched_generation_with_ablation(
	visualizer: TokenActivationVisualizer,
	prompts: List[str],
	feature_indices: List[int],
	batch_size: int,
	max_new_tokens: int,
	do_sample: bool,
	top_k: int,
	top_p: float,
	temperature: float,
	freq_penalty: float,
) -> List[str]:
	responses: List[str] = []
	unique_indices = sorted({int(i) for i in feature_indices if int(i) >= 0})

	fwd_hooks = []
	if unique_indices:
		def _ablate_hook(
			acts: torch.Tensor,
			hook,  # noqa: ANN001 - hook object type is framework-specific.
		) -> torch.Tensor:
			valid = [idx for idx in unique_indices if idx < acts.shape[-1]]
			if not valid:
				return acts
			patched = acts.clone()
			patched[..., valid] = 0.0
			return patched

		fwd_hooks = [(visualizer.hook_point, _ablate_hook)]

	for i in range(0, len(prompts), batch_size):
		batch_prompts = prompts[i : i + batch_size]
		with visualizer.model.saes(saes=[visualizer.sae], reset_saes_end=True):
			hook_ctx = (
				visualizer.model.hooks(fwd_hooks=fwd_hooks)
				if fwd_hooks
				else contextlib.nullcontext()
			)
			with hook_ctx:
				generated = visualizer.model.generate(
					batch_prompts,
					max_new_tokens=max_new_tokens,
					do_sample=do_sample,
					top_k=top_k if top_k > 0 else None,
					top_p=top_p,
					temperature=temperature,
					freq_penalty=freq_penalty,
					stop_at_eos=True,
					return_type="str",
					verbose=False,
				)

		if isinstance(generated, str):
			batch_outputs = [generated]
		else:
			batch_outputs = generated

		for prompt, output in zip(batch_prompts, batch_outputs):
			text = output
			if prompt and text.startswith(prompt):
				text = text[len(prompt) :]
			responses.append(text.strip())

		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	return responses


def _evaluate_model_accuracy_with_ablation(
	visualizer: TokenActivationVisualizer,
	dims: List[int],
	prompt_type: str,
	feature_indices: List[int],
	batch_size: int,
	max_new_tokens: int,
	do_sample: bool,
	top_k: int,
	top_p: float,
	temperature: float,
	freq_penalty: float,
	questions_csv_path: str,
	numeric_csv_path: str,
) -> Dict:
	reasoning = prompt_type == "with_reasoning"
	per_dim: List[Dict] = []
	total = 0
	total_correct = 0

	for dim in dims:
		records = _build_eval_records(
			dim=dim,
			reasoning=reasoning,
			questions_csv_path=questions_csv_path,
			numeric_csv_path=numeric_csv_path,
		)
		if not records:
			per_dim.append(
				{
					"dimension": dim,
					"num_questions": 0,
					"num_correct": 0,
					"accuracy": 0.0,
				}
			)
			continue

		prompts = [r["prompt"] for r in records]
		responses = _run_batched_generation_with_ablation(
			visualizer=visualizer,
			prompts=prompts,
			feature_indices=feature_indices,
			batch_size=batch_size,
			max_new_tokens=max_new_tokens,
			do_sample=do_sample,
			top_k=top_k,
			top_p=top_p,
			temperature=temperature,
			freq_penalty=freq_penalty,
		)

		dim_correct = 0
		for record, response in zip(records, responses):
			true_raw = record["answer"]
			true_label = _normalize_label(true_raw)

			if record["task_type"] == "numeric":
				pred = _extract_numeric(response)
				is_correct = False
				if pred is not None:
					try:
						truth = int(re.sub(r"[^0-9\-]", "", str(true_raw)))
						is_correct = int(pred) == truth
					except Exception:
						is_correct = False
			else:
				pred = _extract_answer(response)
				pred_label = None
				if pred is not None:
					pred_alpha = pred.strip().upper()
					if true_label.isdigit():
						idx = ord(pred_alpha) - ord("A")
						pred_label = str(idx + 1) if 0 <= idx < 26 else pred_alpha
					else:
						pred_label = pred_alpha
				is_correct = pred_label is not None and pred_label == true_label

			if is_correct:
				dim_correct += 1

		dim_total = len(records)
		total += dim_total
		total_correct += dim_correct
		per_dim.append(
			{
				"dimension": dim,
				"num_questions": dim_total,
				"num_correct": dim_correct,
				"accuracy": float(dim_correct / dim_total) if dim_total else 0.0,
			}
		)

	return {
		"prompt_type": prompt_type,
		"ablation_features": [int(i) for i in sorted(set(feature_indices))],
		"num_ablation_features": int(len(set(feature_indices))),
		"generation": {
			"do_sample": do_sample,
			"top_k": top_k,
			"top_p": top_p,
			"temperature": temperature,
			"freq_penalty": freq_penalty,
		},
		"overall": {
			"num_questions": total,
			"num_correct": total_correct,
			"accuracy": float(total_correct / total) if total else 0.0,
		},
		"per_dimension": per_dim,
	}


def _visualize_features(
	method_label: str,
	results: List[Tuple[int, float]],
	metadata: Dict,
	prompt_type: str,
	dim1: int,
	dim2: int,
	output_dir: str,
	topk_sentences: int,
	cmap: str,
	visualizer: TokenActivationVisualizer,
	batch_size: int,
	allowed_prompts_by_dim: Dict[int, set[str]] | None = None,
) -> None:
	if visualizer is None:
		print("Visualization skipped: model/SAE could not be loaded.")
		return
	items = metadata.get("items") or []
	if not items:
		print("Visualization skipped: metadata items are missing.")
		return

	prompts_dim1 = build_prompt_list(metadata, dim1, prompt_type)
	prompts_dim2 = build_prompt_list(metadata, dim2, prompt_type)
	if allowed_prompts_by_dim is not None:
		allowed1 = allowed_prompts_by_dim.get(dim1, set())
		allowed2 = allowed_prompts_by_dim.get(dim2, set())
		prompts_dim1 = [p for p in prompts_dim1 if p in allowed1]
		prompts_dim2 = [p for p in prompts_dim2 if p in allowed2]
	if not prompts_dim1 or not prompts_dim2:
		print("Visualization skipped: prompts could not be resolved from metadata.")
		return
	prompts_dim1 = extract_question_answer_segments(prompts_dim1)
	prompts_dim2 = extract_question_answer_segments(prompts_dim2)

	for rank, (feature_idx, score) in enumerate(results, start=1):
		try:
			left_prompt_scores = visualizer.pooled_feature_activations(
				prompts_dim1,
				feature_idx,
				batch_size=batch_size,
				pooling="max",
			)
			right_prompt_scores = visualizer.pooled_feature_activations(
				prompts_dim2,
				feature_idx,
				batch_size=batch_size,
				pooling="max",
			)
			left_data = collect_topk_prompt_data_from_scores(
				prompts_dim1,
				left_prompt_scores,
				topk_sentences,
			)
			right_data = collect_topk_prompt_data_from_scores(
				prompts_dim2,
				right_prompt_scores,
				topk_sentences,
			)
		except ValueError as e:
			print(f"Visualization skipped for feature {feature_idx}: {e}")
			continue

		left_prompts = [entry["prompt"] for entry in left_data]
		right_prompts = [entry["prompt"] for entry in right_data]
		left_offsets = visualizer.token_offsets_for_prompts(left_prompts)
		right_offsets = visualizer.token_offsets_for_prompts(right_prompts)
		left_scores = visualizer.token_activations(
			left_prompts, feature_idx, batch_size=batch_size
		)
		right_scores = visualizer.token_activations(
			right_prompts, feature_idx, batch_size=batch_size
		)
		left_tokens, left_scores = filter_tokens_scores_to_question_answer(
			left_prompts,
			left_offsets,
			left_scores,
		)
		right_tokens, right_scores = filter_tokens_scores_to_question_answer(
			right_prompts,
			right_offsets,
			right_scores,
		)

		file_base = (
			f"feature_{feature_idx}_rank_{rank}_dim{dim1}_dim{dim2}"
		)
		png_path = os.path.join(output_dir, "visualize", method_label, file_base + ".png")
		json_path = os.path.join(
			output_dir, "visualize", method_label, file_base + ".json"
		)
		title = (
			f"{method_label} feature {feature_idx} "
			f"(dim{dim1} vs dim{dim2}, {prompt_type})"
		)
		render_token_activation_plot(
			left_label=f"dim{dim1}",
			left_tokens=left_tokens,
			left_scores=left_scores,
			right_label=f"dim{dim2}",
			right_tokens=right_tokens,
			right_scores=right_scores,
			title=title,
			output_path=png_path,
			cmap=cmap,
		)
		save_prompt_activation_json(
			json_path,
			{
				"method": method_label,
				"feature_index": feature_idx,
				"rank": rank,
				"prompt_type": prompt_type,
				"dim1": dim1,
				"dim2": dim2,
				"score": score,
				"top_prompts": {
					f"dim{dim1}": left_data,
					f"dim{dim2}": right_data,
				},
				"token_data": {
					f"dim{dim1}": {
						"tokens": left_tokens,
						"scores": left_scores,
					},
					f"dim{dim2}": {
						"tokens": right_tokens,
						"scores": right_scores,
					},
				},
			},
		)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Load and validate feature activation tensors saved by save_activation.py"
	)
	parser.add_argument(
		"--output-dir",
		default="sae_activations",
		help="Base output directory or full activation directory",
	)
	parser.add_argument(
		"--results-dir",
		default="output/validate",
		help="Directory to save plots and visualization outputs",
	)
	parser.add_argument(
		"--model-name",
		default=None,
		help="Model name used in save_activation.py (required if output-dir is base)",
	)
	parser.add_argument(
		"--layer",
		default=None,
		help="Layer label (e.g., layer_20) used in save_activation.py",
	)
	parser.add_argument(
		"--print-metadata",
		action="store_true",
		help="Print metadata.json content if available",
	)
	parser.add_argument(
		"--singular-dims",
		default="4,3",
		help="Compute singular activations between two dims, e.g. '2,3'",
	)
	parser.add_argument(
		"--topk",
		type=int,
		default=10,
		help="Top-k singular activations to print",
	)
	parser.add_argument(
		"--singular-method",
		default="both",
		choices=["diff", "reason", "both"],
		help="Singular feature criterion: diff (mean diff), reason (ReasonScore), both",
	)
	parser.add_argument(
		"--prompt-type",
		default="auto",
		choices=["auto", "with_reasoning", "without_reasoning"],
		help="Prompt type to align with activation files",
	)
	parser.add_argument(
		"--entropy-alpha",
		type=float,
		default=0.7,
		help="Alpha for entropy term in ReasonScore",
	)
	parser.add_argument(
		"--plot-reason-score",
		action="store_true",
		help="Plot distribution of ReasonScore for all features",
	)
	parser.add_argument(
		"--plot-out",
		default=None,
		help="Output path for ReasonScore plot (e.g., reason_score_rank.png)",
	)
	parser.add_argument(
		"--plot-quantile",
		type=float,
		default=0.997,
		help="Quantile to mark on ReasonScore plot (e.g., 0.997)",
	)
	parser.add_argument(
		"--no-probe",
		action="store_true",
		help="Disable probing with singular activations",
	)
	parser.add_argument(
		"--no-visualize",
		action="store_true",
		help="Disable visualization for top-k singular features",
	)
	parser.add_argument(
		"--visualize-topk-sentences",
		type=int,
		default=6,
		help="Number of top prompts per dimension to visualize",
	)
	parser.add_argument(
		"--visualize-cmap",
		default="Reds",
		help="Colormap name for visualization",
	)
	parser.add_argument(
		"--visualize-batch-size",
		type=int,
		default=4,
		help="Batch size when computing token-level activations",
	)
	parser.add_argument(
		"--visualize-device",
		default="auto",
		choices=["auto", "cuda", "cpu", "mps"],
		help="Device for token-level visualization (auto picks available accelerator)",
	)
	parser.add_argument(
		"--probe-folds",
		type=int,
		default=10,
		help="Number of folds for probing (default: 10)",
	)
	parser.add_argument(
		"--probe-epochs",
		type=int,
		default=200,
		help="Training epochs for probing classifier",
	)
	parser.add_argument(
		"--probe-lr",
		type=float,
		default=1e-2,
		help="Learning rate for probing classifier",
	)
	parser.add_argument(
		"--probe-weight-decay",
		type=float,
		default=0.0,
		help="Weight decay for probing classifier",
	)
	parser.add_argument(
		"--probe-seed",
		type=int,
		default=0,
		help="Random seed for probing splits",
	)
	parser.add_argument(
		"--probe-random-samples",
		type=int,
		default=10,
		help="Number of random feature sets for baseline probing",
	)
	parser.add_argument(
		"--filter-correct-dir",
		default=None,
		help=(
			"Directory containing evaluate.py outputs (dim_{d}_correct.csv or "
			"dim_{d}_per_question.csv). If set, feature selection and visualization "
			"use only evaluate-correct questions."
		),
	)
	parser.add_argument(
		"--ablation-eval",
		action="store_true",
		help=(
			"Evaluate model accuracy while ablating selected top-k SAE features "
			"(derived from --singular-method results)."
		),
	)
	parser.add_argument(
		"--ablation-eval-dims",
		default="auto",
		help=(
			"Comma-separated dimensions for ablation accuracy evaluation (e.g., '2,3'). "
			"Use 'auto' to evaluate on the --singular-dims pair."
		),
	)
	parser.add_argument(
		"--ablation-questions-csv",
		default="data/questions_augmented.csv",
		help="Path to MC evaluation CSV used for ablation accuracy evaluation.",
	)
	parser.add_argument(
		"--ablation-numeric-csv",
		default="data/numeric_augmented.csv",
		help="Path to numeric evaluation CSV used for ablation accuracy evaluation.",
	)
	parser.add_argument(
		"--ablation-batch-size",
		type=int,
		default=4,
		help="Batch size for ablation accuracy evaluation generation.",
	)
	parser.add_argument(
		"--ablation-max-new-tokens",
		type=int,
		default=512,
		help="Max new tokens for ablation accuracy evaluation generation.",
	)
	parser.add_argument(
		"--ablation-greedy",
		action="store_true",
		help="Use greedy decoding for ablation evaluation (equivalent to do_sample=False).",
	)
	parser.add_argument(
		"--ablation-temperature",
		type=float,
		default=0.1,
		help="Sampling temperature for ablation evaluation.",
	)
	parser.add_argument(
		"--ablation-top-p",
		type=float,
		default=0.9,
		help="Top-p sampling for ablation evaluation.",
	)
	parser.add_argument(
		"--ablation-top-k",
		type=int,
		default=0,
		help="Top-k sampling for ablation evaluation (0 means no top-k cutoff).",
	)
	parser.add_argument(
		"--ablation-freq-penalty",
		type=float,
		default=0.0,
		help="Frequency penalty for ablation evaluation generation.",
	)
	args = parser.parse_args()
	os.makedirs(args.results_dir, exist_ok=True)

	activations, metadata, prompt_type_by_dim = load_feature_activations(
		output_dir=args.output_dir,
		model_name=args.model_name,
		layer=args.layer,
		prompt_type=args.prompt_type,
	)
	_summarize(activations)
	if args.print_metadata and metadata:
		print(json.dumps(metadata, ensure_ascii=False, indent=2))

	if args.singular_dims:
		try:
			d1_str, d2_str = [s.strip() for s in args.singular_dims.split(",", 1)]
			dim1, dim2 = int(d1_str), int(d2_str)
		except Exception as e:
			raise ValueError("--singular-dims must be like '2,3'") from e

		visualizer = None
		if not args.no_visualize:
			try:
				visualizer = TokenActivationVisualizer.from_metadata(
					metadata, device=args.visualize_device
				)
			except Exception as e:
				print(f"Visualization disabled: {e}")
				visualizer = None

		prompt_type = _resolve_prompt_type(prompt_type_by_dim, dim1, dim2)
		allowed_prompts_by_dim = None
		if args.filter_correct_dir:
			allowed_prompts_by_dim = _read_correct_prompt_sets(
				results_dir=args.filter_correct_dir,
				dims=[dim1, dim2],
				prompt_type=prompt_type,
			)
			activations = _filter_activations_by_prompt_set(
				activations=activations,
				metadata=metadata,
				prompt_type=prompt_type,
				allowed_prompts_by_dim=allowed_prompts_by_dim,
			)
			print("Filtered activations to evaluate-correct questions only:")
			_summarize(activations)

		_save_validate_run_conditions(
			results_dir=args.results_dir,
			args=args,
			extra={
				"dim1": dim1,
				"dim2": dim2,
				"resolved_prompt_type": prompt_type,
				"prompt_type_by_dim": {str(k): v for k, v in prompt_type_by_dim.items()},
				"activation_rows_after_filter": {
					str(k): int(v.shape[0]) for k, v in activations.items()
				},
			},
		)
		summary_payload: Dict = {
			"dim1": dim1,
			"dim2": dim2,
			"prompt_type": prompt_type,
			"topk": args.topk,
			"singular_method": args.singular_method,
			"entropy_alpha": args.entropy_alpha,
			"filter_correct_dir": args.filter_correct_dir,
			"methods": {},
			"probe": {},
			"artifacts": {},
		}
		probe_feature_counts: List[int] = []
		ablation_feature_sets: Dict[str, List[int]] = {}

		if args.singular_method in ("diff", "both"):
			results = find_singular_activations(activations, dim1, dim2, k=args.topk)
			summary_payload["methods"]["diff"] = [
				{"rank": rank, "feature_index": idx, "score": score}
				for rank, (idx, score) in enumerate(results, start=1)
			]
			print(f"Top {len(results)} singular activations (diff, dim{dim1} vs dim{dim2}):")
			for rank, (idx, score) in enumerate(results, start=1):
				print(f"{rank}\tfeature={idx}\t|Δmean|={score:.6f}")
			if not args.no_visualize:
				_visualize_features(
					method_label="diff",
					results=results,
					metadata=metadata,
					prompt_type=prompt_type,
					dim1=dim1,
					dim2=dim2,
					output_dir=args.results_dir,
					topk_sentences=args.visualize_topk_sentences,
					cmap=args.visualize_cmap,
					visualizer=visualizer,
					batch_size=args.visualize_batch_size,
					allowed_prompts_by_dim=allowed_prompts_by_dim,
				)
			feature_indices = [idx for idx, _ in results]
			ablation_feature_sets["diff"] = feature_indices
			if not args.no_probe:
				probe_feature_counts.append(len(feature_indices))
				probe_stats = run_probe_and_report(
					activations,
					dim1,
					dim2,
					feature_indices,
					label="diff",
					args=args,
				)
				summary_payload["probe"]["diff"] = probe_stats

		if args.singular_method in ("reason", "both"):
			results = find_singular_activations_reason_score(
				activations,
				dim1,
				dim2,
				k=args.topk,
				alpha=args.entropy_alpha,
			)
			summary_payload["methods"]["reason"] = [
				{"rank": rank, "feature_index": idx, "score": score}
				for rank, (idx, score) in enumerate(results, start=1)
			]
			print(
				f"Top {len(results)} singular activations (ReasonScore, dim{dim1} vs dim{dim2}):"
			)
			for rank, (idx, score) in enumerate(results, start=1):
				print(f"{rank}\tfeature={idx}\tscore={score:.6f}")
			if not args.no_visualize:
				_visualize_features(
					method_label="reason",
					results=results,
					metadata=metadata,
					prompt_type=prompt_type,
					dim1=dim1,
					dim2=dim2,
					output_dir=args.results_dir,
					topk_sentences=args.visualize_topk_sentences,
					cmap=args.visualize_cmap,
					visualizer=visualizer,
					batch_size=args.visualize_batch_size,
					allowed_prompts_by_dim=allowed_prompts_by_dim,
				)
			feature_indices = [idx for idx, _ in results]
			ablation_feature_sets["reason"] = feature_indices
			if not args.no_probe:
				probe_feature_counts.append(len(feature_indices))
				probe_stats = run_probe_and_report(
					activations,
					dim1,
					dim2,
					feature_indices,
					label="reason",
					args=args,
				)
				summary_payload["probe"]["reason"] = probe_stats

		if not args.no_probe and probe_feature_counts:
			try:
				random_probe_stats = run_random_probe_baseline_and_report(
					activations=activations,
					dim1=dim1,
					dim2=dim2,
					k=max(probe_feature_counts),
					args=args,
					n_samples=args.probe_random_samples,
				)
				summary_payload["probe"]["random-baseline"] = random_probe_stats
			except ValueError as e:
				print(f"Random probe skipped (baseline): {e}")
				summary_payload["probe"]["random-baseline"] = {
					"label": "random-baseline",
					"status": "skipped",
					"reason": str(e),
				}

		if args.ablation_eval:
			if not ablation_feature_sets:
				summary_payload["ablation_eval"] = {
					"status": "skipped",
					"reason": "No singular features were selected for the requested method.",
				}
				print("Ablation eval skipped: no singular features were selected.")
			else:
				if visualizer is None:
					visualizer = TokenActivationVisualizer.from_metadata(
						metadata, device=args.visualize_device
					)

				if args.ablation_eval_dims.strip().lower() == "auto":
					eval_dims = [dim1, dim2]
				else:
					parts = [p.strip() for p in args.ablation_eval_dims.split(",") if p.strip()]
					eval_dims = [int(p) for p in parts]
					if not eval_dims:
						raise ValueError("--ablation-eval-dims did not contain valid dims")

				baseline_stats = _evaluate_model_accuracy_with_ablation(
					visualizer=visualizer,
					dims=eval_dims,
					prompt_type=prompt_type,
					feature_indices=[],
					batch_size=args.ablation_batch_size,
					max_new_tokens=args.ablation_max_new_tokens,
					do_sample=not args.ablation_greedy,
					top_k=args.ablation_top_k,
					top_p=args.ablation_top_p,
					temperature=args.ablation_temperature,
					freq_penalty=args.ablation_freq_penalty,
					questions_csv_path=args.ablation_questions_csv,
					numeric_csv_path=args.ablation_numeric_csv,
				)

				method_stats: Dict[str, Dict] = {}
				for method_label, feature_indices in ablation_feature_sets.items():
					stats = _evaluate_model_accuracy_with_ablation(
						visualizer=visualizer,
						dims=eval_dims,
						prompt_type=prompt_type,
						feature_indices=feature_indices,
						batch_size=args.ablation_batch_size,
						max_new_tokens=args.ablation_max_new_tokens,
						do_sample=not args.ablation_greedy,
						top_k=args.ablation_top_k,
						top_p=args.ablation_top_p,
						temperature=args.ablation_temperature,
						freq_penalty=args.ablation_freq_penalty,
						questions_csv_path=args.ablation_questions_csv,
						numeric_csv_path=args.ablation_numeric_csv,
					)
					method_stats[method_label] = stats

					base_acc = baseline_stats["overall"]["accuracy"]
					abl_acc = stats["overall"]["accuracy"]
					delta = abl_acc - base_acc
					print(
						"Ablation eval ({label}): overall_acc={abl:.4f}, "
						"baseline={base:.4f}, delta={delta:+.4f}".format(
							label=method_label,
							abl=abl_acc,
							base=base_acc,
							delta=delta,
						)
					)

				summary_payload["ablation_eval"] = {
					"status": "ok",
					"evaluation_dimensions": eval_dims,
					"batch_size": args.ablation_batch_size,
					"max_new_tokens": args.ablation_max_new_tokens,
					"do_sample": not args.ablation_greedy,
					"temperature": args.ablation_temperature,
					"top_p": args.ablation_top_p,
					"top_k": args.ablation_top_k,
					"freq_penalty": args.ablation_freq_penalty,
					"questions_csv": args.ablation_questions_csv,
					"numeric_csv": args.ablation_numeric_csv,
					"baseline": baseline_stats,
					"methods": method_stats,
				}

		if args.plot_reason_score:
			scores = compute_reason_scores(
				activations,
				dim1,
				dim2,
				alpha=args.entropy_alpha,
			)
			scores_np = scores.detach().cpu().numpy()
			sorted_scores = np.sort(scores_np)[::-1]
			ranks = np.arange(1, len(sorted_scores) + 1)
			plot_path = args.plot_out or os.path.join(
				args.results_dir, f"reason_score_rank_dim{dim1}_dim{dim2}.png"
			)
			plt.figure(figsize=(8, 5))
			plt.semilogx(ranks, sorted_scores, marker="o", markersize=3, linewidth=1)
			q = min(max(args.plot_quantile, 0.0), 1.0)
			q_index = max(1, int(np.ceil(q * len(sorted_scores))))
			plt.axvline(
				x=q_index,
				color="orange",
				linestyle="--",
				label=f"{q:.3f}th Quantile",
			)
			plt.title(
				f"ReasonScore distribution (dim{dim1} vs dim{dim2}, alpha={args.entropy_alpha})"
			)
			plt.xlabel("Feature Rank")
			plt.ylabel("ReasonScore")
			plt.legend()
			plt.tight_layout()
			plt.savefig(plot_path)
			print(f"Saved ReasonScore rank plot: {plot_path}")
			summary_payload["artifacts"]["reason_score_plot"] = plot_path

		_save_validation_summary(args.results_dir, summary_payload)
		print(f"Saved validation summary: {os.path.join(args.results_dir, 'validation_summary.json')}")


if __name__ == "__main__":
	main()
