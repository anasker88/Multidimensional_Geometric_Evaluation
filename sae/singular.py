from typing import Dict, List, Tuple

import torch


def find_singular_activations(
	activations: Dict[int, torch.Tensor],
	dim1: int,
	dim2: int,
	k: int = 10,
) -> List[Tuple[int, float]]:
	"""Find feature activations that show singular behavior between two dimensions."""
	if dim1 not in activations or dim2 not in activations:
		raise ValueError("Both dimensions must be present in the activations dictionary.")

	tensor1 = activations[dim1]
	tensor2 = activations[dim2]

	mean1 = tensor1.mean(dim=0)
	mean2 = tensor2.mean(dim=0)

	diff = mean1 - mean2
	vals, inds = torch.topk(diff, k=min(k, diff.numel()))
	singular_activations = [(int(idx), float(val)) for idx, val in zip(inds, vals)]
	return singular_activations


def _feature_entropy(tensor: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
	"""Compute per-feature entropy over samples using nonnegative activations."""
	acts = torch.relu(tensor)
	denom = acts.sum(dim=0, keepdim=True).clamp_min(eps)
	probs = acts / denom
	entropy = -(probs * (probs + eps).log()).sum(dim=0)
	return entropy


def find_singular_activations_reason_score(
	activations: Dict[int, torch.Tensor],
	dim1: int,
	dim2: int,
	k: int = 10,
	alpha: float = 0.7,
) -> List[Tuple[int, float]]:
	"""Find singular features using a ReasonScore-like criterion."""
	if dim1 not in activations or dim2 not in activations:
		raise ValueError("Both dimensions must be present in the activations dictionary.")

	tensor1 = activations[dim1]
	tensor2 = activations[dim2]

	mean1 = tensor1.mean(dim=0)
	mean2 = tensor2.mean(dim=0)

	den1 = mean1.sum().clamp_min(1e-9)
	den2 = mean2.sum().clamp_min(1e-9)

	entropy = _feature_entropy(tensor1)
	pos_term = (mean1 / den1) * (entropy.clamp_min(0.0) ** alpha)
	neg_term = mean2 / den2
	score = pos_term - neg_term

	vals, inds = torch.topk(score, k=min(k, score.numel()))
	return [(int(idx), float(val)) for idx, val in zip(inds, vals)]


def compute_reason_scores(
	activations: Dict[int, torch.Tensor],
	dim1: int,
	dim2: int,
	alpha: float = 1.0,
) -> torch.Tensor:
	if dim1 not in activations or dim2 not in activations:
		raise ValueError("Both dimensions must be present in the activations dictionary.")

	tensor1 = activations[dim1]
	tensor2 = activations[dim2]

	mean1 = tensor1.mean(dim=0)
	mean2 = tensor2.mean(dim=0)

	den1 = mean1.sum().clamp_min(1e-9)
	den2 = mean2.sum().clamp_min(1e-9)

	entropy = _feature_entropy(tensor1)

	pos_term = (mean1 / den1) * (entropy.clamp_min(0.0) ** alpha)
	neg_term = mean2 / den2
	return pos_term - neg_term
