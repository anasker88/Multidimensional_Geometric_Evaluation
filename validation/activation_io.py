import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass(frozen=True)
class ActivationFileInfo:
	path: str
	dim: int
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
	"""Load feature activation tensors saved by save_activation.py.

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
