import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import torch
from sae_lens import SAE, HookedSAETransformer
from transformers import AutoModelForCausalLM

from prompting import make_prompt_mc, make_prompt_numeric


@dataclass(frozen=True)
class ActivationFileInfo:
	path: str
	dim: int
	prompt_type: str


@dataclass
class PromptItem:
	uid: str
	source: str
	dimension: int | None
	qtype: str | None
	question: str
	prompt_type: str


def _looks_like_activation_dir(path: str) -> bool:
	if not os.path.isdir(path):
		return False
	if os.path.exists(os.path.join(path, "metadata.json")):
		return True
	for name in os.listdir(path):
		if re.match(r"feature_activations_dim\d+_.*\.pt$", name):
			return True
	return False


def _find_activation_dirs(base_dir: str, max_depth: int = 3) -> List[str]:
	if not os.path.isdir(base_dir):
		return []
	candidates: List[str] = []
	base_depth = base_dir.rstrip(os.sep).count(os.sep)
	for root, dirs, _ in os.walk(base_dir):
		depth = root.count(os.sep) - base_depth
		if depth > max_depth:
			dirs[:] = []
			continue
		if _looks_like_activation_dir(root):
			candidates.append(root)
			dirs[:] = []
	return sorted(candidates)


def _find_model_layer_dirs(base_dir: str, model_dir: str, layer: str) -> List[str]:
	candidates: List[str] = []
	for path in _find_activation_dirs(base_dir):
		norm = os.path.normpath(path)
		parts = norm.split(os.sep)
		if len(parts) >= 2 and parts[-2] == model_dir and parts[-1] == layer:
			candidates.append(path)
	return sorted(candidates)


def _resolve_out_dir(output_dir: str, model_name: str | None, layer: str | None) -> str:
	if output_dir and (model_name is None and layer is None):
		if _looks_like_activation_dir(output_dir):
			return output_dir
		candidates = _find_activation_dirs(output_dir)
		if len(candidates) == 1:
			return candidates[0]
		if len(candidates) > 1:
			latest = max(candidates, key=lambda p: os.path.getmtime(p))
			print(
				"Multiple activation dirs found; using the most recent: "
				+ os.path.relpath(latest, output_dir)
			)
			return latest
		return output_dir

	if not output_dir:
		output_dir = "sae_activations"

	if not model_name:
		raise ValueError("model_name is required when output_dir is a base directory")
	model_dir = model_name.replace("/", "_")
	if not layer:
		raise ValueError("layer is required when output_dir is a base directory")
	primary = os.path.join(output_dir, model_dir, layer)
	if _looks_like_activation_dir(primary):
		return primary
	model_layer_candidates = _find_model_layer_dirs(output_dir, model_dir, layer)
	if len(model_layer_candidates) == 1:
		return model_layer_candidates[0]
	if len(model_layer_candidates) > 1:
		latest = max(model_layer_candidates, key=lambda p: os.path.getmtime(p))
		print(
			"Multiple activation dirs found for model/layer; using the most recent: "
			+ os.path.relpath(latest, output_dir)
		)
		return latest
	return primary


def _read_questions_augmented(csv_path: str, reasoning: bool) -> List[PromptItem]:
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
	if mode == "last":
		return acts[:, -1, :]
	if mode == "mean":
		return acts.mean(dim=1)
	if mode == "max":
		return acts.max(dim=1).values
	raise ValueError(f"Unknown pooling mode: {mode}")


@torch.no_grad()
def _compute_feature_activations(
	model: HookedSAETransformer,
	sae: SAE,
	prompts: List[str],
	batch_size: int,
	pooling: str,
) -> torch.Tensor:
	all_vecs: List[torch.Tensor] = []
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


def _resolve_layer_label(sae: SAE) -> str:
	hook_layer = getattr(sae.cfg, "hook_layer", None)
	if hook_layer is None:
		hook_name = getattr(sae.cfg, "hook_name", None)
		if hook_name is None and getattr(sae.cfg, "metadata", None) is not None:
			hook_name = getattr(sae.cfg.metadata, "hook_name", None)
		if hook_name is not None:
			m = re.search(r"blocks\.(\d+)\.", hook_name)
			if m:
				hook_layer = int(m.group(1))
	return f"layer_{hook_layer}" if hook_layer is not None else "layer_unknown"


def _resolve_device(device: str) -> str:
	if device != "auto":
		return device
	if torch.cuda.is_available():
		return "cuda"
	if torch.backends.mps.is_available():
		return "mps"
	return "cpu"


def _load_sae_and_model(
	model_name: str,
	model_name_override: str | None,
	sae_release: str,
	sae_id: str,
	device: str,
) -> Tuple[SAE, HookedSAETransformer]:
	model_name_for_load = model_name_override or model_name
	use_hf_model = model_name_override is not None and model_name_override != model_name
	hf_model = None
	if use_hf_model:
		hf_model = AutoModelForCausalLM.from_pretrained(model_name)

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

	model_kwargs = getattr(sae.cfg, "model_from_pretrained_kwargs", None) or {}
	if hf_model is not None:
		model_kwargs = {**model_kwargs, "hf_model": hf_model}
	model_loader = getattr(HookedSAETransformer, "from_pretrained_no_processing", None)
	if callable(model_loader):
		model = model_loader(model_name_for_load, device=device, **model_kwargs)
	else:
		model = HookedSAETransformer.from_pretrained(
			model_name_for_load,
			device=device,
			hf_model=hf_model,
		)
	return sae, model


def _load_metadata(out_dir: str) -> Dict:
	meta_path = os.path.join(out_dir, "metadata.json")
	if not os.path.exists(meta_path):
		return {}
	with open(meta_path, "r", encoding="utf-8") as f:
		return json.load(f)


def _parse_activation_name(name: str) -> Tuple[int, str] | None:
	m = re.match(r"feature_activations_dim(\d+)_([^.]+)\.pt$", name)
	if not m:
		return None
	dim = int(m.group(1))
	prompt_type = m.group(2)
	return dim, prompt_type


def _discover_activation_files(out_dir: str, metadata: Dict) -> List[ActivationFileInfo]:
	files = metadata.get("activation_files") or []
	candidates: List[str] = []
	if files:
		candidates = [os.path.join(out_dir, f) for f in files]
	else:
		for name in os.listdir(out_dir):
			if re.match(r"feature_activations_dim\d+_.*\.pt$", name):
				candidates.append(os.path.join(out_dir, name))
	candidates = sorted(candidates)

	infos: List[ActivationFileInfo] = []
	for path in candidates:
		name = os.path.basename(path)
		parsed = _parse_activation_name(name)
		if parsed is None:
			continue
		dim, prompt_type = parsed
		infos.append(ActivationFileInfo(path=path, dim=dim, prompt_type=prompt_type))
	return infos


def _select_prompt_types(
	files_by_dim: Dict[int, List[ActivationFileInfo]],
	requested: str,
) -> Dict[int, str]:
	if requested != "auto":
		for dim, infos in files_by_dim.items():
			if not any(info.prompt_type == requested for info in infos):
				raise ValueError(
					f"prompt_type={requested} not found for dim{dim}. "
					"Use --prompt-type auto or choose an available type."
				)
		return {dim: requested for dim in files_by_dim.keys()}

	all_types_by_dim = {
		dim: {info.prompt_type for info in infos}
		for dim, infos in files_by_dim.items()
	}
	if all("with_reasoning" in types for types in all_types_by_dim.values()):
		return {dim: "with_reasoning" for dim in files_by_dim.keys()}
	if all("without_reasoning" in types for types in all_types_by_dim.values()):
		return {dim: "without_reasoning" for dim in files_by_dim.keys()}

	selected: Dict[int, str] = {}
	for dim, types in all_types_by_dim.items():
		if len(types) == 1:
			selected[dim] = next(iter(types))
			continue
		if "with_reasoning" in types:
			selected[dim] = "with_reasoning"
		else:
			selected[dim] = sorted(types)[0]
		print(
			"Multiple prompt types found for dim{dim}; using {ptype}."
			.format(dim=dim, ptype=selected[dim])
		)
	return selected


def load_feature_activations(
	output_dir: str,
	model_name: str | None = None,
	layer: str | None = None,
	prompt_type: str = "auto",
) -> Tuple[Dict[int, torch.Tensor], Dict, Dict[int, str]]:
	"""Load cached feature activation tensors.

	Returns:
		- dict: dimension -> torch.Tensor (CPU)
		- metadata dict (if metadata.json exists)
		- dict: dimension -> prompt_type used for that dimension
	"""
	out_dir = _resolve_out_dir(output_dir, model_name, layer)
	if not os.path.isdir(out_dir):
		raise FileNotFoundError(f"Activation directory not found: {out_dir}")

	metadata = _load_metadata(out_dir)
	activation_infos = _discover_activation_files(out_dir, metadata)
	if not activation_infos:
		raise FileNotFoundError("No activation .pt files found in output directory")

	files_by_dim: Dict[int, List[ActivationFileInfo]] = {}
	for info in activation_infos:
		files_by_dim.setdefault(info.dim, []).append(info)

	prompt_type_by_dim = _select_prompt_types(files_by_dim, prompt_type)

	activations: Dict[int, torch.Tensor] = {}
	for dim, infos in files_by_dim.items():
		ptype = prompt_type_by_dim[dim]
		selected = [info for info in infos if info.prompt_type == ptype]
		if not selected:
			raise FileNotFoundError(
				f"No activation file for dim{dim} with prompt_type={ptype}"
			)
		path = selected[0].path
		activations[dim] = torch.load(path, map_location="cpu")
	return activations, metadata, prompt_type_by_dim


def ensure_feature_activations(
	output_dir: str,
	model_name: str | None,
	layer: str | None,
	prompt_type: str,
	questions_csv: str,
	numeric_csv: str,
	pooling: str,
	reasoning: str,
	batch_size: int,
	device: str,
	fallback_cpu: bool,
	sae_release: str | None,
	sae_id: str | None,
	model_name_override: str | None = None,
) -> Tuple[Dict[int, torch.Tensor], Dict, Dict[int, str]]:
	if not output_dir:
		output_dir = "sae_activations"

	if sae_release is None or sae_id is None:
		try:
			activations, metadata, prompt_type_by_dim = load_feature_activations(
				output_dir=output_dir,
				model_name=model_name,
				layer=layer,
				prompt_type=prompt_type,
			)
			return activations, metadata, prompt_type_by_dim
		except Exception as e:
			raise ValueError(
				"Missing --sae-release/--sae-id and no cached activations were usable. "
				"Provide SAE info to compute activations."
			) from e

	try:
		activations, metadata, prompt_type_by_dim = load_feature_activations(
			output_dir=output_dir,
			model_name=model_name,
			layer=layer,
			prompt_type=prompt_type,
		)
		meta_release = metadata.get("sae_release")
		meta_id = metadata.get("sae_id")
		meta_model = metadata.get("model_name")
		if meta_release and meta_id:
			if meta_release != sae_release or meta_id != sae_id:
				raise ValueError("Cached activations used different SAE settings.")
		if meta_model and model_name and meta_model != model_name:
			raise ValueError("Cached activations used a different model_name.")
		return activations, metadata, prompt_type_by_dim
	except Exception:
		pass

	if not model_name:
		raise ValueError("model_name is required to compute activations")

	device = _resolve_device(device)
	try:
		sae, model = _load_sae_and_model(
			model_name=model_name,
			model_name_override=model_name_override,
			sae_release=sae_release,
			sae_id=sae_id,
			device=device,
		)
	except RuntimeError as e:
		if fallback_cpu and "out of memory" in str(e).lower() and device != "cpu":
			device = "cpu"
			sae, model = _load_sae_and_model(
				model_name=model_name,
				model_name_override=model_name_override,
				sae_release=sae_release,
				sae_id=sae_id,
				device=device,
			)
		else:
			raise

	layer_label = _resolve_layer_label(sae)
	if layer and layer != layer_label:
		print(f"Requested layer {layer} differs from SAE layer {layer_label}.")
		layer = layer_label

	if output_dir and (model_name is None and layer is None):
		out_dir = output_dir
	else:
		model_dir = model_name.replace("/", "_")
		out_dir = os.path.join(output_dir, model_dir, layer or layer_label)
	os.makedirs(out_dir, exist_ok=True)

	reasoning_modes: List[bool] = []
	if reasoning in ("with", "both"):
		reasoning_modes.append(True)
	if reasoning in ("without", "both"):
		reasoning_modes.append(False)
	if not reasoning_modes:
		reasoning_modes = [False]

	results: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
	all_items: List[PromptItem] = []
	saved_activation_files: List[str] = []

	for reasoning_flag in reasoning_modes:
		mode_key = "with_reasoning" if reasoning_flag else "without_reasoning"
		items = _read_questions_augmented(questions_csv, reasoning=reasoning_flag)
		items.extend(_read_numeric_augmented(numeric_csv, reasoning=reasoning_flag))
		all_items.extend(items)

		by_dim: Dict[int, List[PromptItem]] = {}
		for it in items:
			if it.dimension is None:
				continue
			by_dim.setdefault(it.dimension, []).append(it)

		results[mode_key] = {}
		for dim, dim_items in sorted(by_dim.items()):
			prompts = [it.question for it in dim_items]
			acts = _compute_feature_activations(
				model=model,
				sae=sae,
				prompts=prompts,
				batch_size=batch_size,
				pooling=pooling,
			)
			activation_path = os.path.join(
				out_dir, f"feature_activations_dim{dim}_{mode_key}.pt"
			)
			torch.save(acts, activation_path)
			saved_activation_files.append(os.path.relpath(activation_path, out_dir))
			feature_mean = acts.mean(dim=0)
			vals, inds = torch.topk(feature_mean, k=min(10, feature_mean.numel()))
			top10 = [
				{"feature_index": int(idx), "mean_activation": float(val)}
				for idx, val in zip(inds.tolist(), vals.tolist())
			]
			results[mode_key][str(dim)] = top10

			summary_path = os.path.join(
				out_dir, f"top10_features_dim{dim}_{mode_key}.txt"
			)
			with open(summary_path, "w", encoding="utf-8") as f:
				f.write("rank\tfeature_index\tmean_activation\n")
				for rank, entry in enumerate(top10, start=1):
					f.write(
						f"{rank}\t{entry['feature_index']}\t{entry['mean_activation']:.6f}\n"
					)

	metadata = {
		"model_name": model_name,
		"sae_release": sae_release,
		"sae_id": sae_id,
		"pooling": pooling,
		"batch_size": batch_size,
		"reasoning": reasoning,
		"num_prompts": len(all_items),
		"top10_by_dimension": results,
		"activation_files": saved_activation_files,
		"items": [asdict(it) for it in all_items],
	}
	with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
		json.dump(metadata, f, ensure_ascii=False, indent=2)

	return load_feature_activations(
		output_dir=out_dir,
		model_name=None,
		layer=None,
		prompt_type=prompt_type,
	)
