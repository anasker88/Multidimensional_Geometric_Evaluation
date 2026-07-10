"""Phase 2 of the activation-patching study: component-level patching.

Phase 1 (resid_post, position-resolved) localized WHEN information matters:
edit-token info is read early (handoff ~L8-12) and the answer is assembled at
the final position late (~L19+). Phase 2 asks WHICH COMPONENT carries it, by
patching the additive component outputs instead of the full residual stream:

  - attn: the per-layer attention output
  - mlp:  the per-layer MLP output

The exact hook names vary by architecture, so they are auto-detected from the model's
hook_dict (see _COMPONENT_CANDIDATES): Qwen3.5 exposes LINEAR attention as
`blocks.L.linear_attn.hook_out`, while standard-attention models (Qwen3, gemma, llama, ...)
expose `blocks.L.attn.hook_out` / `blocks.L.hook_attn_out`. A component with no matching
hook for the chosen model is skipped with a warning. So this runs across the evaluated
model families via --model-name, not only Qwen3.5.

Unlike resid_post, component outputs are incremental writes, so patching them is
informative (not the degenerate all-1.0 of full-state resid patching). We still
patch at specific positions (edit span / final token) for a (component x layer x
position) attribution.

Metric, baseline filter, normalization, directions, and invariance breakdowns
are identical to Phase 1 and reused by import from patch_run.py (which is NOT
modified). effect-key = '<component>:<direction>:<posmode>', e.g. 'attn:denoise:edit'.

Run (single GPU):
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python patching/patch_components.py \
        --pairs results/patching/pairs/qwen35_9b_aligned.json \
        --out results/patching/components/qwen35_9b

Multi-GPU (mirrors patch_run.py; merge with patch_merge.py):
    for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i .venv/bin/python \
        patching/patch_components.py --num-shards 4 --shard-id $i \
        --out results/patching/components/qwen35_9b & done
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_run import (  # reuse Phase-1 helpers unchanged
    _aggregate,
    _answer_token_id,
    _logit_diff_at_last,
    _make_patch_hook,
    _plot,
    _position_sets,
    _run_with_cache_resid,  # generic: caches any hooks matching names_filter
)

# Candidate per-layer output-hook templates per component, tried in order; the first
# whose layers exist in the model's hook_dict is used. Covers Qwen3.5 linear attention,
# TransformerBridge canonical names, and classic HookedTransformer names, so the pipeline
# works across the evaluated model families (Qwen3/3.5, gemma, llama, ...), not just Qwen3.5.
_COMPONENT_CANDIDATES = {
    "attn": ["blocks.{L}.linear_attn.hook_out", "blocks.{L}.attn.hook_out", "blocks.{L}.hook_attn_out"],
    "mlp": ["blocks.{L}.hook_mlp_out", "blocks.{L}.mlp.hook_out"],
}


def _resolve_component_hooks(model, component: str):
    """Return (hook_template, sorted_layers) for the first candidate present in the
    model's hook_dict; (None, []) if none match this architecture."""
    hooks = set(model.hook_dict.keys())
    for tmpl in _COMPONENT_CANDIDATES[component]:
        prefix, suffix = tmpl.split("{L}")
        pat = re.compile(re.escape(prefix) + r"(\d+)" + re.escape(suffix) + r"$")
        layers = sorted({int(m.group(1)) for h in hooks for m in [pat.fullmatch(h)] if m})
        if layers:
            return tmpl, layers
    return None, []


@torch.no_grad()
def process_pair(
    model,
    pair: Dict,
    components: List[str],
    hooks_by_comp: Dict[str, str],
    layers_by_comp: Dict[str, List[int]],
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
        return None
    seq_len = clean_toks.shape[1]
    pos_sets = _position_sets(pair, seq_len, pos_modes)

    hook_names = [hooks_by_comp[c].format(L=L) for c in components for L in layers_by_comp[c]]
    clean_cache = _run_with_cache_resid(model, clean_toks, hook_names)
    corr_cache = _run_with_cache_resid(model, corr_toks, hook_names)

    ld_clean = _logit_diff_at_last(model(clean_toks), id_clean, id_corr)
    ld_corr = _logit_diff_at_last(model(corr_toks), id_clean, id_corr)
    correct = (ld_clean > margin) and (ld_corr < -margin)
    denom = ld_clean - ld_corr
    base = {"dimension": pair["dimension"], "type_key": pair["type_key"],
            "clean_answer": pair["clean_answer"],
            "source": pair["source"], "ld_clean": ld_clean, "ld_corr": ld_corr}
    if not correct or abs(denom) < 1e-6:
        return {**base, "baseline_ok": False, "effects": {}}

    effects: Dict[str, List[float]] = {}
    for comp in components:
        for direction in directions:
            base_toks = corr_toks if direction == "denoise" else clean_toks
            donor_cache = clean_cache if direction == "denoise" else corr_cache
            for pmode, positions in pos_sets.items():
                eff = []
                for L in layers_by_comp[comp]:
                    hk = hooks_by_comp[comp].format(L=L)
                    donor = donor_cache[hk]
                    logits = model.run_with_hooks(
                        base_toks, fwd_hooks=[(hk, _make_patch_hook(donor, positions))])
                    ld = _logit_diff_at_last(logits, id_clean, id_corr)
                    if direction == "denoise":
                        eff.append((ld - ld_corr) / denom)
                    else:
                        eff.append((ld - ld_clean) / (ld_corr - ld_clean))
                effects[f"{comp}:{direction}:{pmode}"] = eff
    return {**base, "baseline_ok": True, "effects": effects}


@torch.no_grad()
def process_batch(model, batch, components, hooks_by_comp, layers_by_comp,
                  directions, pos_modes, margin):
    """Batched equivalent of process_pair over equal-length pairs (numerically identical).
    Component outputs are [B,L,d_model], so the resid batch-patch hook applies."""
    from patch_batch import batched_logit_diff_last, make_batched_resid_patch_hook
    clean, corr = batch["clean"], batch["corr"]
    id_clean, id_corr, pairs = batch["id_clean"], batch["id_corr"], batch["pairs"]
    B, L = clean.shape

    pos_by_mode = {m: [] for m in pos_modes}
    for p in pairs:
        sets = _position_sets(p, L, pos_modes)
        for m in pos_modes:
            pos_by_mode[m].append(sets.get(m, []))

    hook_names = [hooks_by_comp[c].format(L=Lyr) for c in components for Lyr in layers_by_comp[c]]
    clean_cache = _run_with_cache_resid(model, clean, hook_names)
    corr_cache = _run_with_cache_resid(model, corr, hook_names)
    ld_clean = batched_logit_diff_last(model(clean), id_clean, id_corr)
    ld_corr = batched_logit_diff_last(model(corr), id_clean, id_corr)
    correct = [(lc > margin) and (lr < -margin) for lc, lr in zip(ld_clean, ld_corr)]
    denom = [lc - lr for lc, lr in zip(ld_clean, ld_corr)]

    eff = [{} for _ in range(B)]
    for comp in components:
        for direction in directions:
            base_toks = corr if direction == "denoise" else clean
            donor_cache = clean_cache if direction == "denoise" else corr_cache
            for pmode in pos_modes:
                per_ex = pos_by_mode[pmode]
                curves = [[] for _ in range(B)]
                for Lyr in layers_by_comp[comp]:
                    hk = hooks_by_comp[comp].format(L=Lyr)
                    donor = donor_cache[hk]
                    logits = model.run_with_hooks(
                        base_toks, fwd_hooks=[(hk, make_batched_resid_patch_hook(donor, per_ex))])
                    ld = batched_logit_diff_last(logits, id_clean, id_corr)
                    for b in range(B):
                        if abs(denom[b]) < 1e-6:
                            curves[b].append(0.0)
                        elif direction == "denoise":
                            curves[b].append((ld[b] - ld_corr[b]) / denom[b])
                        else:
                            curves[b].append((ld[b] - ld_clean[b]) / (ld_corr[b] - ld_clean[b]))
                for b in range(B):
                    eff[b][f"{comp}:{direction}:{pmode}"] = curves[b]

    results = []
    for b in range(B):
        base = {"dimension": pairs[b]["dimension"], "type_key": pairs[b]["type_key"],
                "clean_answer": pairs[b]["clean_answer"],
                "source": pairs[b].get("source", ""), "ld_clean": ld_clean[b], "ld_corr": ld_corr[b]}
        ok = correct[b] and abs(denom[b]) >= 1e-6
        results.append({**base, "baseline_ok": ok, "effects": (eff[b] if ok else {})})
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="results/patching/pairs/qwen35_9b_aligned.json")
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                    help="model load dtype; use bfloat16 for large models on <=48GB GPUs")
    ap.add_argument("--components", default="attn,mlp")
    ap.add_argument("--directions", default="denoise,noise")
    ap.add_argument("--positions", default="edit,last")
    ap.add_argument("--layers", default="all", help="'all' or comma-separated layer indices")
    ap.add_argument("--limit", type=int, default=0, help="first N pairs per dimension (smoke test)")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--num-shards", type=int, default=1, help="total GPU shards (multi-GPU)")
    ap.add_argument("--shard-id", type=int, default=0, help="index of this shard [0, num-shards)")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="batch pairs of equal token length (1 = per-pair). Numerically identical; "
                         "~mean-group-size faster.")
    ap.add_argument("--out", default="results/patching/components/qwen35_9b")
    args = ap.parse_args()

    components = [c.strip() for c in args.components.split(",") if c.strip()]
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    pos_modes = [p.strip() for p in args.positions.split(",") if p.strip()]
    for c in components:
        if c not in _COMPONENT_CANDIDATES:
            raise ValueError(f"unknown component {c}; choose from {list(_COMPONENT_CANDIDATES)}")

    with open(args.pairs, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = [p for p in data["pairs"] if p.get("token_aligned")]
    if args.limit > 0:
        per_dim: Dict[int, int] = defaultdict(int)
        sel = []
        for p in pairs:
            if per_dim[p["dimension"]] < args.limit:
                sel.append(p); per_dim[p["dimension"]] += 1
        pairs = sel
    if args.num_shards > 1:
        pairs = pairs[args.shard_id::args.num_shards]
    print(f"Loaded {len(pairs)} aligned pairs"
          + (f" (shard {args.shard_id}/{args.num_shards})" if args.num_shards > 1 else ""))

    print(f"Loading {args.model_name} via TransformerBridge (slow, ~8min), dtype={args.dtype}...")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformer_lens.model_bridge import TransformerBridge
    model = TransformerBridge.boot_transformers(
        args.model_name, device=args.device, dtype=getattr(torch, args.dtype))
    model.eval()

    hooks_by_comp: Dict[str, str] = {}
    layers_by_comp: Dict[str, List[int]] = {}
    want = None if args.layers == "all" else {int(x) for x in args.layers.split(",")}
    for c in list(components):
        tmpl, avail = _resolve_component_hooks(model, c)
        if not avail:
            print(f"  WARNING: no '{c}' output hook for this architecture; skipping "
                  f"(tried {_COMPONENT_CANDIDATES[c]}).")
            components.remove(c)
            continue
        sel = avail if want is None else [L for L in avail if L in want]
        hooks_by_comp[c] = tmpl
        layers_by_comp[c] = sel
        print(f"  {c}: hook '{tmpl}' -> {len(avail)} layers (patching {len(sel)})")
    if not components:
        raise SystemExit("No patchable components resolved for this model architecture; aborting.")
    # layer axis for plotting/merge: union across components
    layers = sorted(set().union(*layers_by_comp.values()))

    from tqdm.auto import tqdm
    results: List[Dict] = []
    n_ok = 0
    if args.batch_size and args.batch_size > 1:
        from patch_batch import group_by_length
        batches = group_by_length(pairs, model.tokenizer, max_batch=args.batch_size, device=model.cfg.device)
        print(f"Batched: {len(pairs)} pairs -> {len(batches)} equal-length batches "
              f"(mean {len(pairs)/max(1,len(batches)):.1f}/batch)")
        for batch in tqdm(batches, desc="batches", unit="batch"):
            rs = process_batch(model, batch, components, hooks_by_comp, layers_by_comp,
                               directions, pos_modes, args.margin)
            results.extend(rs)
            n_ok += sum(int(bool(r.get("baseline_ok"))) for r in rs)
    else:
        for p in tqdm(pairs, desc="pairs", unit="pair"):
            r = process_pair(model, p, components, hooks_by_comp, layers_by_comp, directions, pos_modes, args.margin)
            if r is None:
                continue
            results.append(r)
            n_ok += int(bool(r.get("baseline_ok")))
    print(f"baseline-correct pairs: {n_ok}/{len(results)}")

    os.makedirs(args.out, exist_ok=True)
    payload_meta = {
        "model_name": args.model_name, "pairs_file": args.pairs, "layers": layers,
        "components": components, "component_hooks": hooks_by_comp,
        "directions": directions, "positions": pos_modes,
        "margin": args.margin, "n_pairs": len(results), "n_baseline_ok": n_ok,
    }
    if args.num_shards > 1:
        # write a shard file compatible with patch_merge.py
        fname = f"patch_results_shard{args.shard_id:02d}.json"
        with open(os.path.join(args.out, fname), "w", encoding="utf-8") as f:
            json.dump({"metadata": {**payload_meta, "num_shards": args.num_shards},
                       "per_pair": results}, f, ensure_ascii=False, indent=2)
        print(f"Wrote shard -> {os.path.join(args.out, fname)} (merge with patch_merge.py)")
        return

    agg = _aggregate(results)
    out_json = os.path.join(args.out, "patch_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"metadata": payload_meta, "aggregate": agg, "per_pair": results},
                  f, ensure_ascii=False, indent=2)
    print(f"Wrote results -> {out_json}")

    # console summary: peak layer per effect-key (pooled 'all' is coarse only)
    for k in sorted(agg.keys()):
        cells = agg[k].get("by_dim_type", {})
        print(f"\n[{k}] peak layer per (dim x type):")
        for g in sorted(cells.keys()):
            m = cells[g]["mean"]
            pk = max(range(len(m)), key=lambda i: m[i])
            print(f"  {g:>8}: peak L{layers[pk]} = {m[pk]:+.3f}  (n={cells[g]['n_pairs']})")

    _plot(agg, layers, os.path.join(args.out, "plots"))


if __name__ == "__main__":
    main()
