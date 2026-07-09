"""Head-vector dimension-invariance: cross-dimension cosine vs split-half reliability.

Post-processing only (no GPU / no model): reads the Phase-3 `patch_heads_results.json`
per-pair records and quantifies, per model, how dimension-invariant the head-importance
structure is — the analysis behind §04 of the progress artifact, hardened per
REEXPERIMENT_TODO P2-3 / C5 (the artifact used a single even/odd split; this averages
many random splits and adds a Spearman-Brown-corrected reliability ceiling).

For each model:
  - head-importance vector per dimension = mean single-head denoise recovery over the
    (layer x head) window, across baseline-ok pairs of that dimension.
  - cross-dim cosine: cos(2D,3D), cos(2D,4D), cos(3D,4D).
  - split-half reliability per dimension: over --splits random halves, cosine between the
    two half-means; report mean and 2.5/97.5 percentiles. Spearman-Brown corrects the
    half-split cosine to a full-sample reliability ceiling r_full = 2r/(1+r).
  - verdict: cross-dim cosine is "at ceiling" only if it reaches the SB-corrected
    reliability within the split spread; otherwise a small systematic shift remains.

Run:
    .venv/bin/python patching/patch_reliability.py \
        --heads-glob 'results/patching/heads/*/patch_heads_results.json' \
        --out results/patching/reliability/reliability.json
"""
import argparse
import glob
import json
import math
import os
import random
from typing import Dict, List, Optional


def _flatten(mat: List[List[float]]) -> List[float]:
    return [v for row in mat for v in row]


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else float("nan")


def _mean_vec(pairs: List[dict], key: str) -> Optional[List[float]]:
    acc: Optional[List[float]] = None
    n = 0
    for p in pairs:
        eff = p.get("effects", {}).get(key)
        if eff is None:
            continue
        flat = _flatten(eff)
        if acc is None:
            acc = [0.0] * len(flat)
        for i, v in enumerate(flat):
            acc[i] += v
        n += 1
    if not n or acc is None:
        return None
    return [v / n for v in acc]


def _spearman_brown(r_half: float) -> float:
    """Full-length reliability from a split-half correlation."""
    if r_half >= 1.0:
        return 1.0
    return 2 * r_half / (1 + r_half) if (1 + r_half) != 0 else float("nan")


def split_half_reliability(pairs: List[dict], key: str, splits: int, rng: random.Random):
    """Mean split-half cosine over `splits` random halves, + percentile spread + SB ceiling."""
    idx = list(range(len(pairs)))
    if len(idx) < 4:
        return None
    vals = []
    for _ in range(splits):
        rng.shuffle(idx)
        h = len(idx) // 2
        A = [pairs[i] for i in idx[:h]]
        B = [pairs[i] for i in idx[h:2 * h]]
        va, vb = _mean_vec(A, key), _mean_vec(B, key)
        if va and vb:
            vals.append(_cosine(va, vb))
    if not vals:
        return None
    vals.sort()
    mean = sum(vals) / len(vals)
    lo = vals[max(0, int(0.025 * len(vals)))]
    hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))]
    return {"mean": mean, "lo": lo, "hi": hi, "sb_ceiling": _spearman_brown(mean), "n_splits": len(vals)}


def analyze_model(path: str, key: str, splits: int, seed: int) -> Optional[dict]:
    d = json.load(open(path))
    if key not in d.get("aggregate", {}):
        # fall back to the only available key
        keys = list(d.get("aggregate", {}).keys())
        if not keys:
            return None
        key = keys[0]
    per_pair = [p for p in d.get("per_pair", []) if p.get("baseline_ok")]
    by_dim = {dim: [p for p in per_pair if p.get("dimension") == dnum]
              for dim, dnum in (("2D", 2), ("3D", 3), ("4D", 4))}
    vecs = {dim: _mean_vec(ps, key) for dim, ps in by_dim.items()}
    rng = random.Random(seed)
    rel = {dim: split_half_reliability(ps, key, splits, rng) for dim, ps in by_dim.items()}

    def cx(d1: str, d2: str) -> Optional[float]:
        if vecs[d1] and vecs[d2]:
            return _cosine(vecs[d1], vecs[d2])
        return None

    cross = {"2D-3D": cx("2D", "3D"), "2D-4D": cx("2D", "4D"), "3D-4D": cx("3D", "4D")}
    # verdict: min cross-dim cosine vs min SB-corrected reliability ceiling
    rels = [rel[d]["sb_ceiling"] for d in rel if rel[d]]
    crosses = [c for c in cross.values() if c is not None]
    verdict = None
    if rels and crosses:
        min_ceiling = min(rels)
        min_cross = min(crosses)
        if min_ceiling < 0.7:
            verdict = "信号不足で判定不能(信頼性が低い)"
        elif min_cross >= min_ceiling - 0.01:
            verdict = "不変(信頼性天井に到達)"
        else:
            verdict = f"ほぼ不変+系統残差 ~{min_ceiling - min_cross:.3f}"
    return {
        "model": d.get("metadata", {}).get("model_name", os.path.basename(os.path.dirname(path))),
        "key": key,
        "n_by_dim": {dim: len(ps) for dim, ps in by_dim.items()},
        "cross_dim_cosine": cross,
        "split_half": rel,
        "verdict": verdict,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heads-glob", default="results/patching/heads/*/patch_heads_results.json")
    ap.add_argument("--key", default="denoise:last")
    ap.add_argument("--splits", type=int, default=200, help="random split-half repetitions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/patching/reliability/reliability.json")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.heads_glob))
    if not paths:
        raise SystemExit(f"no heads results matched {args.heads_glob}")

    out = []
    print(f"{'model':<28} {'2D-3D':>7} {'2D-4D':>7} {'3D-4D':>7} | reliability(SB) 2D/3D/4D | verdict")
    for p in paths:
        r = analyze_model(p, args.key, args.splits, args.seed)
        if r is None:
            continue
        out.append(r)
        c = r["cross_dim_cosine"]
        sh = r["split_half"]
        def fmt(x):
            return f"{x:.3f}" if isinstance(x, float) else "  -  "
        def sb(dim):
            return f"{sh[dim]['sb_ceiling']:.3f}" if sh.get(dim) else " - "
        name = r["model"].split("/")[-1]
        print(f"{name:<28} {fmt(c['2D-3D']):>7} {fmt(c['2D-4D']):>7} {fmt(c['3D-4D']):>7} | "
              f"{sb('2D')}/{sb('3D')}/{sb('4D')} | {r['verdict']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"metadata": {"heads_glob": args.heads_glob, "key": args.key,
                                "splits": args.splits, "seed": args.seed}, "models": out},
                  f, ensure_ascii=False, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
