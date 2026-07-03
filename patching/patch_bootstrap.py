"""Bootstrap 95% confidence intervals for the headline patching claims.

All statistics are recomputed from the per-pair records already stored by
patch_run.py / patch_heads.py / patch_ablate.py, so NO GPU / model re-run is
needed — bootstrap resamples the pairs (with replacement) and recomputes the
statistic B times. The pair is the resampling unit (each pair is one
independent clean/corrupted contrast).

Statistics, per model:
  Finding 1 (dimension-invariant circuit): the decision layer = argmax of the
      pooled mean denoise:last resid curve, and the edit-handoff layer = argmax
      of the pooled mean denoise:edit curve. Bootstrap gives a CI on the layer.
  Finding 4 (sparse mover heads): recovery of the #1 and #2 ranked mover heads
      (mean over pairs of that head's denoise:last effect).
  Finding 5 (necessity): mean top-5 ablation effect, mean random-5 effect, and
      the gap Δ = top5 - rand5 (the necessity signal).

Only baseline-ok pairs are used (same filter as the point estimates). Output is
a single JSON keyed by model, ready for the artifact / roadmap.

    .venv/bin/python patching/patch_bootstrap.py            # all models
    .venv/bin/python patching/patch_bootstrap.py --B 20000  # more resamples
"""
import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

RUN = "results/patching/run"
HEADS = "results/patching/heads"
ABLATE = "results/patching/ablate"

# model -> (run subdir, heads/ablate subdir or None if hybrid-attn / no per-head)
MODELS = {
    "Qwen3-8B":     ("qwen3_8b",       "qwen3_8b"),
    "Qwen3-14B":    ("qwen3_14b",      "qwen3_14b"),
    "Qwen3.5-9B":   ("qwen35_9b_full", None),
    "gemma-2-9b":   ("gemma2_9b",      "gemma2_9b"),
    "gemma-2-27b":  ("gemma2_27b",     "gemma2_9b".replace("9b", "27b")),
    "phi-4":        ("phi4",           "phi4"),
}


def _pct_ci(samples: np.ndarray) -> Dict:
    return {
        "mean": float(np.mean(samples)),
        "lo": float(np.percentile(samples, 2.5)),
        "hi": float(np.percentile(samples, 97.5)),
    }


def _boot_idx(n: int, B: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(B, n))


def _half_cross(m: np.ndarray, layers: np.ndarray, rising: bool,
                thr: float = 0.5) -> int:
    """Recovery-crossing layer — the operational landmark used in the artifact.

    rising=True  (denoise:last, the decision): first layer whose mean >= thr.
    rising=False (denoise:edit, the hand-off): first layer whose mean < thr
                 (the edit signal starts near 1.0 and drops as it is consumed).
    Default thr=0.5 (half recovery). A stricter thr (0.75) is used as a
    sensitivity check that the landmark is not an artifact of the 0.5 cutoff.
    """
    for i in range(len(m)):
        if (rising and m[i] >= thr) or (not rising and m[i] < thr):
            return int(layers[i])
    return int(layers[-1])


def _boot_layer(curves: np.ndarray, layers: List[int], boot: np.ndarray,
                rising: bool) -> Dict:
    """Bootstrap the half-recovery crossing layer. curves: [n_pairs, n_layers]."""
    lay = np.asarray(layers)
    cross = np.array([_half_cross(curves[idx].mean(0), lay, rising) for idx in boot])
    point = _half_cross(curves.mean(0), lay, rising)
    return {
        "layer": int(point),
        "lo": int(np.percentile(cross, 2.5)),
        "hi": int(np.percentile(cross, 97.5)),
        "mode_frac": float(np.mean(cross == point)),
    }


def _boot_scalar(vals: np.ndarray, boot: np.ndarray) -> Dict:
    """Bootstrap the mean of a per-pair scalar. vals: [n_pairs]."""
    return _pct_ci(vals[boot].mean(1))


def bootstrap_model(name: str, run_sub: str, ha_sub: Optional[str],
                    B: int, seed: int) -> Optional[Dict]:
    rng = np.random.default_rng(seed)
    out: Dict = {"model": name}

    # ---- Finding 1: resid decision / edit-handoff layers ----
    run_f = os.path.join(RUN, run_sub, "patch_results.json")
    if not os.path.exists(run_f):
        print(f"  ! {name}: no run file {run_f}")
        return None
    d = json.load(open(run_f))
    layers = d["metadata"]["layers"]
    ok = [p for p in d["per_pair"] if p.get("baseline_ok")]
    boot = _boot_idx(len(ok), B, rng)
    lay = np.asarray(layers)
    for eff_key, label, rising in [("denoise:last", "decision", True),
                                   ("denoise:edit", "edit_handoff", False)]:
        curves = np.array([p["effects"][eff_key] for p in ok])
        rec = {**_boot_layer(curves, layers, boot, rising), "n": len(ok)}
        # threshold-sensitivity: the landmark is an operational recovery crossing;
        # report the point-estimate crossing at a stricter threshold (0.75) so the
        # reader can see it is not an artifact of the 0.5 cutoff. For symmetric A/B
        # pairs recovery=0.5 ~ ld=0 (neutral), so 0.75 is comfortably past neutral.
        rec["layer_thr75"] = _half_cross(curves.mean(0), lay, rising, thr=0.75)
        out.setdefault("resid", {})[label] = rec

    # symmetry of the behavioral contrast: rho = -ld_corr/ld_clean per pair.
    # For rho=1 the pair is symmetric and recovery=0.5 corresponds exactly to
    # ld_patched=0 (neutral); neutral-recovery point = rho/(1+rho).
    rho = np.array([-p["ld_corr"] / p["ld_clean"] for p in ok
                    if abs(p["ld_clean"]) > 1e-6])
    out["symmetry"] = {
        "rho_median": float(np.median(rho)),
        "neutral_recovery_median": float(np.median(rho / (1 + rho))),
    }

    # ---- Finding 4 & 5: heads + ablation (std-GQA only) ----
    if ha_sub is not None:
        hf = os.path.join(HEADS, ha_sub, "patch_heads_results.json")
        af = os.path.join(ABLATE, ha_sub, "patch_ablate_results.json")
        # heads: rank heads by pooled mean denoise:last effect, CI on #1 / #2
        if os.path.exists(hf):
            dh = json.load(open(hf))
            okh = [p for p in dh["per_pair"] if p.get("baseline_ok")]
            # effect: [n_layers][n_heads] per pair -> flatten to [n_pairs, L*H]
            eff = np.array([np.asarray(p["effects"]["denoise:last"]).ravel() for p in okh])
            hl = dh["metadata"]["layers"]
            nh = dh["metadata"]["n_heads"]
            mean_eff = eff.mean(0)
            order = np.argsort(mean_eff)[::-1]
            booth = _boot_idx(len(okh), B, rng)
            heads_ci = []
            for rank in range(2):
                col = order[rank]
                L, H = hl[col // nh], col % nh
                ci = _boot_scalar(eff[:, col], booth)
                heads_ci.append({"rank": rank + 1, "layer": int(L), "head": int(H),
                                 **ci})
            out["heads"] = {"top": heads_ci, "n": len(okh)}
        # ablation: top5, rand5, gap
        if os.path.exists(af):
            da = json.load(open(af))
            oka = [p for p in da["per_pair"] if p.get("baseline_ok")]
            top5 = np.array([p["effects"]["top5"] for p in oka])
            rand5 = np.array([p["effects"]["rand5"] for p in oka])
            boota = _boot_idx(len(oka), B, rng)
            gap = top5 - rand5
            out["ablation"] = {
                "top5": _boot_scalar(top5, boota),
                "rand5": _boot_scalar(rand5, boota),
                "gap": _boot_scalar(gap, boota),
                "n": len(oka),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/patching/bootstrap_ci.json")
    args = ap.parse_args()

    res = {}
    for name, (run_sub, ha_sub) in MODELS.items():
        print(f"[{name}] bootstrapping B={args.B} ...")
        r = bootstrap_model(name, run_sub, ha_sub, args.B, args.seed)
        if r:
            res[name] = r

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"B": args.B, "seed": args.seed, "models": res},
              open(args.out, "w"), indent=2)
    print(f"\nWrote -> {args.out}\n")

    # console summary
    for name, r in res.items():
        print(f"=== {name} ===")
        rd = r["resid"]["decision"]; re_ = r["resid"]["edit_handoff"]
        sy = r["symmetry"]
        print(f"  decision L{rd['layer']} [95% CI L{rd['lo']}-L{rd['hi']}] "
              f"(thr0.75 L{rd['layer_thr75']})  "
              f"edit-handoff L{re_['layer']} [L{re_['lo']}-L{re_['hi']}] "
              f"(thr0.75 L{re_['layer_thr75']})  (n={rd['n']})")
        print(f"  symmetry: median rho={sy['rho_median']:.2f}, "
              f"neutral-recovery point={sy['neutral_recovery_median']:.2f}")
        if "heads" in r:
            for h in r["heads"]["top"]:
                print(f"  head #{h['rank']} L{h['layer']}H{h['head']}: "
                      f"{h['mean']:+.3f} [{h['lo']:+.3f},{h['hi']:+.3f}]")
        if "ablation" in r:
            a = r["ablation"]
            print(f"  ablate top5 {a['top5']['mean']:+.3f} "
                  f"[{a['top5']['lo']:+.3f},{a['top5']['hi']:+.3f}]  "
                  f"rand5 {a['rand5']['mean']:+.3f} "
                  f"[{a['rand5']['lo']:+.3f},{a['rand5']['hi']:+.3f}]  "
                  f"gap {a['gap']['mean']:+.3f} "
                  f"[{a['gap']['lo']:+.3f},{a['gap']['hi']:+.3f}]  (n={a['n']})")


if __name__ == "__main__":
    main()
