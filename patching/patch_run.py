"""Phase 1 of the activation-patching study: residual-stream (layer x position) sweep.

For each clean/corrupted pair (from patch_pairs.py), patch blocks.L.hook_resid_post
at SPECIFIC token positions and measure the change in the answer logit difference
logit(clean_ans) - logit(corr_ans) at the final position.

IMPORTANT: patching resid_post at ALL positions is degenerate — it copies the
entire residual state, so every downstream layer runs on the donor and the
normalized effect is trivially 1.0 at every layer (verified empirically). To
localize, we patch only specific positions:
  - "edit": the swapped label span (pair.edit_token_start:edit_token_end)
  - "last": the final position (where the answer is read out)
This is the causal-tracing (layer x position) decomposition.

  - denoising: base run = corrupted; inject CLEAN resid_post[L] at the positions.
       normalized effect = (patched - corr) / (clean - corr)
       0 = no recovery, 1 = clean fully restored.
  - noising:   base run = clean;     inject CORRUPTED resid_post[L] at the positions.
       normalized effect = (patched - clean) / (corr - clean)
       0 = unaffected, 1 = fully broken to corrupted.

Only pairs the model gets right in BOTH conditions are used (baseline filter:
clean logit-diff > margin and corrupted logit-diff < -margin), so each pair has
a real, separated behavioral contrast to trace.

Model is loaded once via TransformerBridge (see PATCHING_ROADMAP.md). Run with
the patching venv:

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python patching/patch_run.py \
        --pairs output/patch_pairs/qwen35_9b_aligned.json \
        --out output/patch_run/qwen35_9b
"""
import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional

import torch


def _resid_post_layers(model) -> List[int]:
    layers = []
    for h in model.hook_dict.keys():
        m = re.fullmatch(r"blocks\.(\d+)\.hook_resid_post", h)
        if m:
            layers.append(int(m.group(1)))
    return sorted(set(layers))


def _answer_token_id(tokenizer, letter: str) -> int:
    """Token id for the answer letter as it appears after 'The answer is '.

    Prefer the leading-space single-token form (' A'); fall back to the first
    sub-token of the bare letter.
    """
    ids = tokenizer.encode(" " + letter, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    ids2 = tokenizer.encode(letter, add_special_tokens=False)
    return ids2[0]


@torch.no_grad()
def _logit_diff_at_last(logits: torch.Tensor, id_pos: int, id_neg: int) -> float:
    last = logits[0, -1, :]
    return float(last[id_pos] - last[id_neg])


@torch.no_grad()
def _run_with_cache_resid(model, toks, resid_hooks: List[str]):
    """Forward pass caching resid_post at the requested layers."""
    _, cache = model.run_with_cache(
        toks, names_filter=lambda name: name in set(resid_hooks)
    )
    return cache


def _make_patch_hook(donor_act: torch.Tensor, positions: List[int]):
    pos = torch.tensor(positions, dtype=torch.long)

    def hook(act, hook):  # noqa: ARG001
        out = act.clone()
        out[:, pos, :] = donor_act[:, pos, :].to(act.dtype).to(act.device)
        return out
    return hook


def _position_sets(pair: Dict, seq_len: int, modes: List[str]) -> Dict[str, List[int]]:
    sets: Dict[str, List[int]] = {}
    for m in modes:
        if m == "edit":
            s, e = pair.get("edit_token_start"), pair.get("edit_token_end")
            if s is not None and e is not None and 0 <= s < e <= seq_len:
                sets["edit"] = list(range(s, e))
        elif m == "last":
            sets["last"] = [seq_len - 1]
        elif m == "all":
            sets["all"] = list(range(seq_len))
    return sets


@torch.no_grad()
def process_pair(
    model,
    pair: Dict,
    layers: List[int],
    directions: List[str],
    pos_modes: List[str],
    margin: float,
) -> Optional[Dict]:
    tok = model.tokenizer
    id_clean = _answer_token_id(tok, pair["clean_answer"])
    id_corr = _answer_token_id(tok, pair["corrupted_answer"])

    clean_toks = tok(pair["clean_prompt"], return_tensors="pt",
                     add_special_tokens=False)["input_ids"].to(model.cfg.device)
    corr_toks = tok(pair["corrupted_prompt"], return_tensors="pt",
                    add_special_tokens=False)["input_ids"].to(model.cfg.device)
    if clean_toks.shape[1] != corr_toks.shape[1]:
        return None  # alignment lost (should not happen for aligned pairs)
    seq_len = clean_toks.shape[1]
    pos_sets = _position_sets(pair, seq_len, pos_modes)

    resid_hooks = [f"blocks.{L}.hook_resid_post" for L in layers]
    clean_cache = _run_with_cache_resid(model, clean_toks, resid_hooks)
    corr_cache = _run_with_cache_resid(model, corr_toks, resid_hooks)

    # baseline logit diffs (metric = logit(clean_ans) - logit(corr_ans))
    ld_clean = _logit_diff_at_last(model(clean_toks), id_clean, id_corr)
    ld_corr = _logit_diff_at_last(model(corr_toks), id_clean, id_corr)

    # baseline filter: clean prefers clean_ans, corrupted prefers corr_ans
    correct = (ld_clean > margin) and (ld_corr < -margin)
    denom = ld_clean - ld_corr
    if not correct or abs(denom) < 1e-6:
        return {
            "dimension": pair["dimension"], "type_key": pair["type_key"],
            "source": pair["source"], "ld_clean": ld_clean, "ld_corr": ld_corr,
            "baseline_ok": False, "effects": {},
        }

    # effects keyed by "<direction>:<posmode>" -> per-layer normalized effect
    effects: Dict[str, List[float]] = {}
    for direction in directions:
        base_toks = corr_toks if direction == "denoise" else clean_toks
        donor_cache = clean_cache if direction == "denoise" else corr_cache
        for pmode, positions in pos_sets.items():
            eff = []
            for L in layers:
                donor = donor_cache[f"blocks.{L}.hook_resid_post"]
                logits = model.run_with_hooks(
                    base_toks,
                    fwd_hooks=[(f"blocks.{L}.hook_resid_post",
                                _make_patch_hook(donor, positions))],
                )
                ld = _logit_diff_at_last(logits, id_clean, id_corr)
                if direction == "denoise":
                    eff.append((ld - ld_corr) / denom)          # 0=corr, 1=clean
                else:
                    eff.append((ld - ld_clean) / (ld_corr - ld_clean))  # 0=clean, 1=corr
            effects[f"{direction}:{pmode}"] = eff

    return {
        "dimension": pair["dimension"], "type_key": pair["type_key"],
        "source": pair["source"], "ld_clean": ld_clean, "ld_corr": ld_corr,
        "baseline_ok": True, "effects": effects,
    }


def _curve_stats(curves: List[List[float]]) -> Dict:
    t = torch.tensor(curves)  # [n_pairs, n_layers]
    std = t.std(0).tolist() if t.shape[0] > 1 else [0.0] * t.shape[1]
    return {"n_pairs": t.shape[0], "mean": t.mean(0).tolist(), "std": std}


def _kind(r: Dict) -> str:
    return "real" if "questions_aug" in r.get("source", "") else "synthetic"


# invariance breakdowns: name -> function mapping a per-pair record to a group label
_GROUPINGS = {
    "by_dim": lambda r: f"{r['dimension']}D",
    "by_type": lambda r: f"t{r['type_key']}",
    "by_dim_type": lambda r: f"{r['dimension']}D_t{r['type_key']}",
    "by_kind": _kind,
}


def _group_curves(results: List[Dict], key: str, group_fn) -> Dict:
    by: Dict[str, List[List[float]]] = defaultdict(list)
    for r in results:
        if r.get("baseline_ok") and key in r["effects"]:
            by[group_fn(r)].append(r["effects"][key])
    return {g: _curve_stats(c) for g, c in sorted(by.items())}


def _aggregate(results: List[Dict]) -> Dict:
    """Mean normalized effect per effect-key, broken down along several axes so
    invariance can be tested: by dimension, by type, by (dimension x type), and
    real-vs-synthetic. effect-key is '<direction>:<posmode>' (e.g. 'denoise:edit').
    """
    keys = set()
    for r in results:
        if r.get("baseline_ok"):
            keys.update(r["effects"].keys())
    agg: Dict = {}
    for k in sorted(keys):
        agg[k] = {name: _group_curves(results, k, fn) for name, fn in _GROUPINGS.items()}
        pooled = [r["effects"][k] for r in results if r.get("baseline_ok") and k in r["effects"]]
        if pooled:
            agg[k]["all"] = _curve_stats(pooled)
    return agg


def _plot(agg: Dict, layers: List[int], out_dir: str) -> None:
    """One figure per effect-key; panels = invariance breakdowns (by_dim, by_type,
    by_dim_type, by_kind). Types are COMPARED, never silently pooled — the pooled
    'all' curve is reported in JSON only, as a coarse summary.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(skip plot: {e})")
        return
    panels = ["by_dim", "by_type", "by_dim_type", "by_kind"]
    os.makedirs(out_dir, exist_ok=True)
    for k in sorted(agg.keys()):
        fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.5), squeeze=False)
        for ax, panel in zip(axes[0], panels):
            groups = agg[k].get(panel, {})
            for g, stats in groups.items():
                ax.plot(layers, stats["mean"], marker="o", ms=2.5,
                        label=f"{g} (n={stats['n_pairs']})")
            ax.axhline(0, color="gray", lw=0.5)
            ax.axhline(1, color="gray", lw=0.5, ls="--")
            ax.set_title(panel)
            ax.set_xlabel("layer")
            ax.set_ylabel("normalized effect")
            ax.legend(fontsize=7)
        fig.suptitle(f"resid_post patch — {k}")
        fig.tight_layout()
        png = os.path.join(out_dir, f"curves_{k.replace(':', '_')}.png")
        fig.savefig(png, dpi=120)
        plt.close(fig)
        print(f"Wrote plot -> {png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="output/patch_pairs/qwen35_9b_aligned.json")
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--directions", default="denoise,noise")
    ap.add_argument("--positions", default="edit,last",
                    help="token positions to patch: edit (label span), last (answer pos), all "
                         "(degenerate sanity check). comma-separated.")
    ap.add_argument("--layers", default="all", help="'all' or comma-separated layer indices")
    ap.add_argument("--limit", type=int, default=0, help="process only first N aligned pairs (smoke test)")
    ap.add_argument("--margin", type=float, default=0.0, help="baseline logit-diff margin")
    ap.add_argument("--out", default="output/patch_run/qwen35_9b")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="total number of GPU shards (for multi-GPU runs)")
    ap.add_argument("--shard-id", type=int, default=0,
                    help="index of this shard [0, num-shards)")
    args = ap.parse_args()

    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    pos_modes = [p.strip() for p in args.positions.split(",") if p.strip()]

    with open(args.pairs, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = [p for p in data["pairs"] if p.get("token_aligned")]
    if args.limit > 0:
        # keep a balanced slice across dimensions for smoke tests
        per_dim: Dict[int, int] = defaultdict(int)
        sel = []
        for p in pairs:
            if per_dim[p["dimension"]] < args.limit:
                sel.append(p)
                per_dim[p["dimension"]] += 1
        pairs = sel
    if args.num_shards > 1:
        pairs = pairs[args.shard_id::args.num_shards]
    print(f"Loaded {len(pairs)} aligned pairs from {args.pairs}"
          + (f" (shard {args.shard_id}/{args.num_shards})" if args.num_shards > 1 else ""))

    print(f"Loading {args.model_name} via TransformerBridge (slow, ~8min)...")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformer_lens.model_bridge import TransformerBridge
    model = TransformerBridge.boot_transformers(args.model_name, device=args.device)
    model.eval()

    all_layers = _resid_post_layers(model)
    if args.layers == "all":
        layers = all_layers
    else:
        want = {int(x) for x in args.layers.split(",")}
        layers = [L for L in all_layers if L in want]
    print(f"resid_post layers: {len(all_layers)} available; patching {len(layers)} -> {layers}")

    results: List[Dict] = []
    n_ok = 0
    from tqdm.auto import tqdm
    for p in tqdm(pairs, desc="pairs", unit="pair"):
        r = process_pair(model, p, layers, directions, pos_modes, args.margin)
        if r is None:
            continue
        results.append(r)
        if r.get("baseline_ok"):
            n_ok += 1
    print(f"baseline-correct pairs: {n_ok}/{len(results)}")

    agg = _aggregate(results)

    os.makedirs(args.out, exist_ok=True)
    fname = (f"patch_results_shard{args.shard_id:02d}.json"
             if args.num_shards > 1 else "patch_results.json")
    out_json = os.path.join(args.out, fname)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "model_name": args.model_name, "pairs_file": args.pairs,
                "layers": layers, "directions": directions, "positions": pos_modes,
                "margin": args.margin, "n_pairs": len(results), "n_baseline_ok": n_ok,
            },
            "aggregate": agg,
            "per_pair": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"Wrote results -> {out_json}")

    # console summary: peak layer per (dimension x type) cell — the primary unit
    # is type (held fixed), compared across dimensions; pooled 'all' is coarse only.
    for k in sorted(agg.keys()):
        print(f"\n[{k}] peak layer (max mean normalized effect):")
        cells = agg[k].get("by_dim_type", {})
        for g in sorted(cells.keys()):
            m = cells[g]["mean"]
            pk = max(range(len(m)), key=lambda i: m[i])
            print(f"  {g:>8}: peak L{layers[pk]} = {m[pk]:+.3f}  (n={cells[g]['n_pairs']})")

    if args.num_shards == 1:
        _plot(agg, layers, os.path.join(args.out, "plots"))


if __name__ == "__main__":
    main()
