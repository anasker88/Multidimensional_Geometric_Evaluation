import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

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
	collect_topk_prompt_data,
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


def _visualize_features(
	method_label: str,
	results: List[Tuple[int, float]],
	activations: Dict[int, torch.Tensor],
	metadata: Dict,
	prompt_type: str,
	dim1: int,
	dim2: int,
	output_dir: str,
	topk_sentences: int,
	cmap: str,
	visualizer: TokenActivationVisualizer,
	batch_size: int,
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
	if not prompts_dim1 or not prompts_dim2:
		print("Visualization skipped: prompts could not be resolved from metadata.")
		return

	for rank, (feature_idx, score) in enumerate(results, start=1):
		try:
			left_data = collect_topk_prompt_data(
				activations[dim1], prompts_dim1, feature_idx, topk_sentences
			)
			right_data = collect_topk_prompt_data(
				activations[dim2], prompts_dim2, feature_idx, topk_sentences
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
	args = parser.parse_args()

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
		os.makedirs(args.results_dir, exist_ok=True)
		probe_feature_counts: List[int] = []

		if args.singular_method in ("diff", "both"):
			results = find_singular_activations(activations, dim1, dim2, k=args.topk)
			print(f"Top {len(results)} singular activations (diff, dim{dim1} vs dim{dim2}):")
			for rank, (idx, score) in enumerate(results, start=1):
				print(f"{rank}\tfeature={idx}\t|Δmean|={score:.6f}")
			if not args.no_visualize:
				_visualize_features(
					method_label="diff",
					results=results,
					activations=activations,
					metadata=metadata,
					prompt_type=prompt_type,
					dim1=dim1,
					dim2=dim2,
					output_dir=args.results_dir,
					topk_sentences=args.visualize_topk_sentences,
					cmap=args.visualize_cmap,
					visualizer=visualizer,
					batch_size=args.visualize_batch_size,
				)
			if not args.no_probe:
				feature_indices = [idx for idx, _ in results]
				probe_feature_counts.append(len(feature_indices))
				run_probe_and_report(
					activations,
					dim1,
					dim2,
					feature_indices,
					label="diff",
					args=args,
				)

		if args.singular_method in ("reason", "both"):
			results = find_singular_activations_reason_score(
				activations,
				dim1,
				dim2,
				k=args.topk,
				alpha=args.entropy_alpha,
			)
			print(
				f"Top {len(results)} singular activations (ReasonScore, dim{dim1} vs dim{dim2}):"
			)
			for rank, (idx, score) in enumerate(results, start=1):
				print(f"{rank}\tfeature={idx}\tscore={score:.6f}")
			if not args.no_visualize:
				_visualize_features(
					method_label="reason",
					results=results,
					activations=activations,
					metadata=metadata,
					prompt_type=prompt_type,
					dim1=dim1,
					dim2=dim2,
					output_dir=args.results_dir,
					topk_sentences=args.visualize_topk_sentences,
					cmap=args.visualize_cmap,
					visualizer=visualizer,
					batch_size=args.visualize_batch_size,
				)
			if not args.no_probe:
				feature_indices = [idx for idx, _ in results]
				probe_feature_counts.append(len(feature_indices))
				run_probe_and_report(
					activations,
					dim1,
					dim2,
					feature_indices,
					label="reason",
					args=args,
				)

		if not args.no_probe and probe_feature_counts:
			try:
				run_random_probe_baseline_and_report(
					activations=activations,
					dim1=dim1,
					dim2=dim2,
					k=max(probe_feature_counts),
					args=args,
					n_samples=args.probe_random_samples,
				)
			except ValueError as e:
				print(f"Random probe skipped (baseline): {e}")

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


if __name__ == "__main__":
	main()
