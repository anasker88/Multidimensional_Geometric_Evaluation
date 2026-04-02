from typing import Dict, List, Tuple

import numpy as np
import torch


def _build_probe_dataset(
	activations: Dict[int, torch.Tensor],
	dim1: int,
	dim2: int,
	feature_indices: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
	if not feature_indices:
		raise ValueError("feature_indices must be non-empty for probing.")
	if dim1 not in activations or dim2 not in activations:
		raise ValueError("Both dimensions must be present in the activations dictionary.")

	feat_idx = torch.tensor(feature_indices, dtype=torch.long)
	x1 = activations[dim1].index_select(1, feat_idx).float()
	x2 = activations[dim2].index_select(1, feat_idx).float()
	y1 = torch.zeros(x1.shape[0], dtype=torch.float32)
	y2 = torch.ones(x2.shape[0], dtype=torch.float32)
	x = torch.cat([x1, x2], dim=0)
	y = torch.cat([y1, y2], dim=0)
	return x, y, x1.shape[0], x2.shape[0]


def _feature_dim(activations: Dict[int, torch.Tensor], dim1: int, dim2: int) -> int:
	if dim1 not in activations or dim2 not in activations:
		raise ValueError("Both dimensions must be present in the activations dictionary.")
	feat1 = activations[dim1].shape[1]
	feat2 = activations[dim2].shape[1]
	if feat1 != feat2:
		raise ValueError("Feature dimensions do not match between dim1 and dim2.")
	return feat1


def sample_random_features(
	activations: Dict[int, torch.Tensor],
	dim1: int,
	dim2: int,
	k: int,
	seed: int,
) -> List[int]:
	if k < 1:
		raise ValueError("k must be >= 1 for random features.")
	num_features = _feature_dim(activations, dim1, dim2)
	if k > num_features:
		raise ValueError("k exceeds the number of available features.")
	rng = np.random.default_rng(seed)
	return rng.choice(num_features, size=k, replace=False).tolist()


def _stratified_kfold_indices(
	n1: int,
	n2: int,
	k: int,
	seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
	rng = np.random.default_rng(seed)
	idx1 = rng.permutation(n1)
	idx2 = rng.permutation(n2)
	folds1 = np.array_split(idx1, k)
	folds2 = np.array_split(idx2, k)
	folds: List[Tuple[np.ndarray, np.ndarray]] = []
	for i in range(k):
		test1 = folds1[i]
		test2 = folds2[i]
		train1 = np.concatenate([f for j, f in enumerate(folds1) if j != i])
		train2 = np.concatenate([f for j, f in enumerate(folds2) if j != i])
		train_idx = np.concatenate([train1, train2 + n1])
		test_idx = np.concatenate([test1, test2 + n1])
		folds.append((train_idx, test_idx))
	return folds


def run_linear_probe(
	x: torch.Tensor,
	y: torch.Tensor,
	n1: int,
	n2: int,
	k_folds: int,
	seed: int,
	epochs: int,
	lr: float,
	weight_decay: float,
) -> List[float]:
	if min(n1, n2) < 2:
		raise ValueError("Not enough samples for probing.")
	if k_folds > min(n1, n2):
		k_folds = min(n1, n2)
		print(f"Reducing probe folds to {k_folds} due to limited samples.")
	if k_folds < 2:
		raise ValueError("Need at least 2 folds for probing.")

	accuracies: List[float] = []
	for fold_id, (train_idx, test_idx) in enumerate(
		_stratified_kfold_indices(n1, n2, k_folds, seed)
	):
		torch.manual_seed(seed + fold_id)
		x_train = x[train_idx]
		y_train = y[train_idx]
		x_test = x[test_idx]
		y_test = y[test_idx]

		mean = x_train.mean(dim=0, keepdim=True)
		std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
		x_train = (x_train - mean) / std
		x_test = (x_test - mean) / std

		model = torch.nn.Linear(x_train.shape[1], 1)
		optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
		criterion = torch.nn.BCEWithLogitsLoss()

		for _ in range(epochs):
			optimizer.zero_grad()
			logits = model(x_train).squeeze(1)
			loss = criterion(logits, y_train)
			loss.backward()
			optimizer.step()

		with torch.no_grad():
			logits = model(x_test).squeeze(1)
			preds = (torch.sigmoid(logits) >= 0.5).float()
			acc = (preds == y_test).float().mean().item()
			accuracies.append(acc)

	return accuracies


def run_probe_and_report(
	activations: Dict[int, torch.Tensor],
	dim1: int,
	dim2: int,
	feature_indices: List[int],
	label: str,
	args,
) -> None:
	try:
		x, y, n1, n2 = _build_probe_dataset(activations, dim1, dim2, feature_indices)
		accs = run_linear_probe(
			x,
			y,
			n1,
			n2,
			k_folds=args.probe_folds,
			seed=args.probe_seed,
			epochs=args.probe_epochs,
			lr=args.probe_lr,
			weight_decay=args.probe_weight_decay,
		)
		mean_acc = float(np.mean(accs))
		std_acc = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
		print(
			f"Probe accuracy ({label}): "
			f"mean={mean_acc:.4f}, std={std_acc:.4f}, folds={len(accs)}"
		)
	except ValueError as e:
		print(f"Probe skipped ({label}): {e}")


def run_random_probe_baseline_and_report(
	activations: Dict[int, torch.Tensor],
	dim1: int,
	dim2: int,
	k: int,
	args,
	n_samples: int,
	seed_offset: int = 1000,
) -> None:
	if n_samples < 1:
		raise ValueError("n_samples must be >= 1 for random baseline probing.")

	run_means: List[float] = []
	run_stds: List[float] = []
	fold_counts: List[int] = []
	for i in range(n_samples):
		seed = args.probe_seed + seed_offset + i
		features = sample_random_features(
			activations=activations,
			dim1=dim1,
			dim2=dim2,
			k=k,
			seed=seed,
		)
		x, y, n1, n2 = _build_probe_dataset(activations, dim1, dim2, features)
		accs = run_linear_probe(
			x,
			y,
			n1,
			n2,
			k_folds=args.probe_folds,
			seed=seed,
			epochs=args.probe_epochs,
			lr=args.probe_lr,
			weight_decay=args.probe_weight_decay,
		)
		run_means.append(float(np.mean(accs)))
		run_stds.append(float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0)
		fold_counts.append(len(accs))

	mean_of_means = float(np.mean(run_means))
	std_of_means = float(np.std(run_means, ddof=1)) if len(run_means) > 1 else 0.0
	avg_fold_std = float(np.mean(run_stds)) if run_stds else 0.0
	print(
		"Probe accuracy (random-baseline): "
		f"mean={mean_of_means:.4f}, std={std_of_means:.4f}, "
		f"samples={n_samples}, features={k}, avg_fold_std={avg_fold_std:.4f}, "
		f"folds_per_sample={min(fold_counts) if fold_counts else 0}"
	)
