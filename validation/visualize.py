import json
import os
import re
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from sae_lens import SAE, HookedSAETransformer


def _question_answer_span(prompt: str) -> Tuple[int, int]:
	# Preferred format in current prompts:
	# Question:\n...\nAnswer choices: ...\nOutput ...
	m = re.search(
		r"Question:\n(?P<body>.*?Answer choices:.*?)(?:\nOutput the answer tag on the last line\.|\Z)",
		prompt,
		flags=re.DOTALL,
	)
	if m:
		return m.start("body"), m.end("body")

	# Fallback: visualize anything after Question:
	m2 = re.search(r"Question:\n(?P<body>.*)", prompt, flags=re.DOTALL)
	if m2:
		return m2.start("body"), m2.end("body")

	# Last fallback: entire prompt
	return 0, len(prompt)


def filter_tokens_scores_to_question_answer(
	prompts: List[str],
	offsets_by_prompt: List[List[Tuple[int, int]]],
	scores_by_prompt: List[List[float]],
) -> Tuple[List[List[str]], List[List[float]]]:
	filtered_tokens_all: List[List[str]] = []
	filtered_scores_all: List[List[float]] = []

	for prompt, offsets, scores in zip(prompts, offsets_by_prompt, scores_by_prompt):
		span_start, span_end = _question_answer_span(prompt)
		row_tokens: List[str] = []
		row_scores: List[float] = []

		for (start, end), score in zip(offsets, scores):
			if start == end:
				continue
			if end <= span_start or start >= span_end:
				continue

			clip_start = max(start, span_start)
			clip_end = min(end, span_end)
			token_text = prompt[clip_start:clip_end]
			if not token_text:
				continue

			row_tokens.append(token_text)
			row_scores.append(float(score))

		filtered_tokens_all.append(row_tokens)
		filtered_scores_all.append(row_scores)

	return filtered_tokens_all, filtered_scores_all


def extract_question_answer_segments(prompts: List[str]) -> List[str]:
	segments: List[str] = []
	for prompt in prompts:
		start, end = _question_answer_span(prompt)
		segment = prompt[start:end]
		segments.append(segment if segment else prompt)
	return segments


def build_prompt_list(metadata: Dict, dim: int, prompt_type: str) -> List[str]:
	items = metadata.get("items") or []
	prompts: List[str] = []
	for item in items:
		if item.get("dimension") != dim:
			continue
		if item.get("prompt_type") != prompt_type:
			continue
		prompt = item.get("question") or ""
		prompts.append(prompt)
	return prompts


def collect_topk_prompt_data(
	activations: torch.Tensor,
	prompts: List[str],
	feature_index: int,
	k: int,
) -> List[Dict[str, float | str]]:
	if activations.shape[0] != len(prompts):
		raise ValueError(
			"Activation count does not match prompt count. "
			"Ensure metadata items align with activation files."
		)
	if feature_index >= activations.shape[1]:
		raise ValueError("feature_index is out of range for activation tensor.")

	acts = activations[:, feature_index]
	k = min(k, acts.numel())
	vals, inds = torch.topk(acts, k=k)
	data: List[Dict[str, float | str]] = []
	for idx, val in zip(inds.tolist(), vals.tolist()):
		data.append({"prompt": prompts[idx], "score": float(val)})
	return data


def collect_topk_prompt_data_from_scores(
	prompts: List[str],
	scores: List[float],
	k: int,
) -> List[Dict[str, float | str]]:
	if len(prompts) != len(scores):
		raise ValueError("Prompt count does not match score count.")
	if not prompts:
		return []

	score_tensor = torch.tensor(scores, dtype=torch.float32)
	k = min(k, score_tensor.numel())
	vals, inds = torch.topk(score_tensor, k=k)
	data: List[Dict[str, float | str]] = []
	for idx, val in zip(inds.tolist(), vals.tolist()):
		data.append({"prompt": prompts[idx], "score": float(val)})
	return data


def _wrap_prompt(text: str, width: int = 80) -> str:
	return "\n".join(textwrap.wrap(text, width=width)) if text else ""


def render_prompt_activation_plot(
	left_label: str,
	left_data: List[Dict[str, float | str]],
	right_label: str,
	right_data: List[Dict[str, float | str]],
	title: str,
	output_path: str,
	cmap: str = "Reds",
) -> None:
	max_score = 0.0
	for entry in left_data + right_data:
		max_score = max(max_score, float(entry["score"]))
	if max_score <= 0:
		max_score = 1.0

	norm = Normalize(vmin=0.0, vmax=max_score)
	fig_height = max(6.0, 1.2 + 0.6 * max(len(left_data), len(right_data)))
	fig, ax = plt.subplots(figsize=(14, fig_height))
	ax.axis("off")
	ax.set_title(title, fontsize=14)
	fig.patch.set_facecolor("lightgray")

	ax.text(0.25, 0.92, left_label, fontsize=12, fontweight="bold", ha="center")
	ax.text(0.75, 0.92, right_label, fontsize=12, fontweight="bold", ha="center")

	y = 0.86
	y_step = 0.8 / max(1, max(len(left_data), len(right_data)))
	for idx in range(max(len(left_data), len(right_data))):
		if idx < len(left_data):
			entry = left_data[idx]
			color = plt.colormaps.get_cmap(cmap)(norm(float(entry["score"])))
			label = f"{entry['score']:.4f} | {_wrap_prompt(str(entry['prompt']))}"
			ax.text(
				0.02,
				y,
				label,
				fontsize=9,
				va="top",
				bbox=dict(facecolor=color, edgecolor="none", boxstyle="round,pad=0.3"),
			)
		if idx < len(right_data):
			entry = right_data[idx]
			color = plt.colormaps.get_cmap(cmap)(norm(float(entry["score"])))
			label = f"{entry['score']:.4f} | {_wrap_prompt(str(entry['prompt']))}"
			ax.text(
				0.52,
				y,
				label,
				fontsize=9,
				va="top",
				bbox=dict(facecolor=color, edgecolor="none", boxstyle="round,pad=0.3"),
			)
		y -= y_step

	sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
	sm.set_array(np.array([]))
	cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.03)
	cbar.set_label("Activation", fontsize=11)

	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	plt.tight_layout()
	plt.savefig(output_path)
	plt.close(fig)


def save_prompt_activation_json(
	output_path: str,
	payload: Dict,
) -> None:
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(payload, f, ensure_ascii=True, indent=2)


def _resolve_hook_info(sae: SAE) -> Tuple[str, int | None]:
	hook_name = getattr(sae.cfg, "hook_name", None)
	if hook_name is None and getattr(sae.cfg, "metadata", None) is not None:
		hook_name = getattr(sae.cfg.metadata, "hook_name", None)
	if hook_name is None:
		raise ValueError("Could not resolve SAE hook name from config.")
	hook_layer = getattr(sae.cfg, "hook_layer", None)
	if hook_layer is None:
		m = re.search(r"blocks\.(\d+)\.", hook_name)
		if m:
			hook_layer = int(m.group(1))
	return hook_name, hook_layer


def _tokens_from_offsets(prompt: str, offsets: List[Tuple[int, int]]) -> List[str]:
	parts: List[str] = []
	prev = (-1, -1)
	count = 0
	for start, end in offsets:
		if start == end:
			parts.append("<|special|>")
			continue
		chunk = prompt[start:end]
		if (start, end) == prev:
			if count == 0 and parts:
				parts[-1] = parts[-1] + f":{count}"
			count += 1
			chunk = chunk + f":{count}"
		else:
			count = 0
		parts.append(chunk)
		prev = (start, end)
	return parts


@dataclass
class TokenActivationVisualizer:
	model: HookedSAETransformer
	sae: SAE
	device: str
	hook_point: str
	stop_at_layer: int | None

	@classmethod
	def from_metadata(
		cls,
		metadata: Dict,
		device: str = "auto",
		sae_release: str | None = None,
		sae_id: str | None = None,
		model_name: str | None = None,
	) -> "TokenActivationVisualizer":
		model_name = model_name or metadata.get("model_name")
		sae_release = sae_release or metadata.get("sae_release")
		sae_id = sae_id or metadata.get("sae_id")
		if not model_name or not sae_release or not sae_id:
			raise ValueError("metadata must include model_name, sae_release, and sae_id")
		if device == "auto":
			device = (
				"cuda"
				if torch.cuda.is_available()
				else "mps" if torch.backends.mps.is_available() else "cpu"
			)

		sae_from_disk = False
		sae_disk_path = None
		if sae_release and os.path.isdir(sae_release):
			sae_disk_path = os.path.join(sae_release, sae_id)
			if os.path.isdir(sae_disk_path):
				sae_from_disk = True
			else:
				raise FileNotFoundError(f"Local SAE path not found: {sae_disk_path}")
		if sae_from_disk:
			sae = SAE.load_from_disk(sae_disk_path, device=device)
		else:
			sae = SAE.from_pretrained(sae_release, sae_id, device=device)

		model = HookedSAETransformer.from_pretrained(model_name, device=device)
		hook_name, hook_layer = _resolve_hook_info(sae)
		hook_point = hook_name + ".hook_sae_acts_post"
		stop_at_layer = hook_layer + 1 if hook_layer is not None else None
		return cls(
			model=model,
			sae=sae,
			device=device,
			hook_point=hook_point,
			stop_at_layer=stop_at_layer,
		)

	@torch.no_grad()
	def token_activations(
		self,
		prompts: List[str],
		feature_index: int,
		batch_size: int = 4,
	) -> List[List[float]]:
		results: List[List[float]] = []
		for i in range(0, len(prompts), batch_size):
			batch = prompts[i : i + batch_size]
			kwargs = {"names_filter": [self.hook_point]}
			if self.stop_at_layer is not None:
				kwargs["stop_at_layer"] = self.stop_at_layer
			_, cache = self.model.run_with_cache_with_saes(batch, saes=[self.sae], **kwargs)
			acts = cache[self.hook_point]
			selected = acts[:, :, feature_index].to(torch.float32).cpu().numpy()
			for row in selected:
				results.append(row.tolist())
			del cache, acts
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
		return results

	@torch.no_grad()
	def pooled_feature_activations(
		self,
		prompts: List[str],
		feature_index: int,
		batch_size: int = 4,
		pooling: str = "max",
	) -> List[float]:
		results: List[float] = []
		for i in range(0, len(prompts), batch_size):
			batch = prompts[i : i + batch_size]
			kwargs = {"names_filter": [self.hook_point]}
			if self.stop_at_layer is not None:
				kwargs["stop_at_layer"] = self.stop_at_layer
			_, cache = self.model.run_with_cache_with_saes(batch, saes=[self.sae], **kwargs)
			acts = cache[self.hook_point][:, :, feature_index].to(torch.float32)
			if pooling == "max":
				pooled = acts.max(dim=1).values
			elif pooling == "mean":
				pooled = acts.mean(dim=1)
			elif pooling == "last":
				pooled = acts[:, -1]
			else:
				raise ValueError(f"Unknown pooling mode: {pooling}")
			results.extend(pooled.cpu().tolist())
			del cache, acts, pooled
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
		return results

	def tokens_for_prompts(self, prompts: List[str]) -> List[List[str]]:
		encoding = self.model.tokenizer(
			prompts, return_offsets_mapping=True, add_special_tokens=True
		)
		all_offsets = encoding["offset_mapping"]
		all_tokens: List[List[str]] = []
		for prompt, offsets in zip(prompts, all_offsets):
			all_tokens.append(_tokens_from_offsets(prompt, offsets))
		return all_tokens

	def token_offsets_for_prompts(self, prompts: List[str]) -> List[List[Tuple[int, int]]]:
		encoding = self.model.tokenizer(
			prompts, return_offsets_mapping=True, add_special_tokens=True
		)
		return encoding["offset_mapping"]


def render_token_activation_plot(
	left_label: str,
	left_tokens: List[List[str]],
	left_scores: List[List[float]],
	right_label: str,
	right_tokens: List[List[str]],
	right_scores: List[List[float]],
	title: str,
	output_path: str,
	cmap: str = "Reds",
) -> None:
	def _token_rows(
		tokens_list: List[List[str]],
		scores_list: List[List[float]],
		max_chars_per_line: int,
	) -> List[List[Tuple[str, float]]]:
		rows: List[List[Tuple[str, float]]] = []
		for tokens, scores in zip(tokens_list, scores_list):
			current_row: List[Tuple[str, float]] = []
			current_len = 0
			for token, score in zip(tokens, scores):
				if token == "<|special|>":
					continue
				token_len = max(1, len(token)) + 1
				if current_row and current_len + token_len > max_chars_per_line:
					rows.append(current_row)
					current_row = []
					current_len = 0
				current_row.append((token, float(score)))
				current_len += token_len
			if current_row:
				rows.append(current_row)
			# Add a spacer row between prompts for readability.
			rows.append([])
		if rows and not rows[-1]:
			rows.pop()
		return rows

	all_scores = [s for row in left_scores + right_scores for s in row]
	max_score = max(all_scores) if all_scores else 1.0
	if max_score <= 0:
		max_score = 1.0

	norm = Normalize(vmin=0.0, vmax=max_score)
	left_rows = _token_rows(left_tokens, left_scores, max_chars_per_line=55)
	right_rows = _token_rows(right_tokens, right_scores, max_chars_per_line=55)
	n_rows = max(1, max(len(left_rows), len(right_rows)))
	fig_height = max(8.0, 1.8 + 0.24 * n_rows)

	fig = plt.figure(figsize=(14, fig_height))
	gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[24, 1], hspace=0.15)
	ax = fig.add_subplot(gs[0])
	cax = fig.add_subplot(gs[1])
	ax.axis("off")
	ax.set_title(title, fontsize=14)
	fig.patch.set_facecolor("lightgray")

	ax.text(0.25, 0.97, left_label, fontsize=12, fontweight="bold", ha="center", va="top")
	ax.text(0.75, 0.97, right_label, fontsize=12, fontweight="bold", ha="center", va="top")

	y_top = 0.93
	y_bottom = 0.03
	y_step = (y_top - y_bottom) / n_rows

	y = y_top
	for row in left_rows:
		y -= y_step
		x = 0.02
		if not row:
			continue
		for token, score in row:
			color = plt.colormaps.get_cmap(cmap)(norm(score))
			text = ax.text(
				x,
				y,
				token,
				fontsize=8,
				ha="left",
				va="center",
				bbox=dict(facecolor=color, edgecolor="none", boxstyle="round,pad=0.3"),
			)
			text_bb = text.get_window_extent().transformed(ax.transData.inverted())
			x += text_bb.width + 0.01

	y = y_top
	for row in right_rows:
		y -= y_step
		x = 0.52
		if not row:
			continue
		for token, score in row:
			color = plt.colormaps.get_cmap(cmap)(norm(score))
			text = ax.text(
				x,
				y,
				token,
				fontsize=8,
				ha="left",
				va="center",
				bbox=dict(facecolor=color, edgecolor="none", boxstyle="round,pad=0.3"),
			)
			text_bb = text.get_window_extent().transformed(ax.transData.inverted())
			x += text_bb.width + 0.01

	sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
	sm.set_array(np.array([]))
	cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
	cbar.set_label("Activation", fontsize=11)

	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	fig.subplots_adjust(top=0.96, bottom=0.06)
	plt.savefig(output_path)
	plt.close(fig)
