"""Phase 2 of the activation-patching study: component-level patching.

Phase 1 (resid_post, position-resolved) localized WHEN information matters:
edit-token info is read early (handoff ~L8-12) and the answer is assembled at
the final position late (~L19+). Phase 2 asks WHICH COMPONENT carries it, by
patching the additive component outputs instead of the full residual stream:

  - attn: blocks.L.linear_attn.hook_out   (Qwen3.5 uses LINEAR attention; the
          standard hook_attn_out alias is absent, so we target linear_attn.hook_out)
  - mlp:  blocks.L.hook_mlp_out

Unlike resid_post, component outputs are incremental writes, so patching them is
informative (not the degenerate all-1.0 of full-state resid patching). We still
patch at specific positions (edit span / final token) for a (component x layer x
position) attribution.

Metric, baseline filter, normalization, directions, and invariance breakdowns
are identical to Phase 1 and reused by import from patch_run.py (which is NOT
modified). effect-key = '<component>:<direction>:<posmode>', e.g. 'attn:denoise:edit'.

Run (single GPU):
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python patching/patch_components.py \
        --pairs output/patch_pairs/qwen35_9b_aligned.json \
        --out output/patch_components/qwen35_9b

Multi-GPU (mirrors patch_run.py; merge with patch_merge.py):
    for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i .venv/bin/python \
        patching/patch_components.py --num-shards 4 --shard-id $i \
        --out output/patch_components/qwen35_9b & done
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

# component name -> hook-name template (verified present in the bridge hook_dict)
_COMPONENTS = {
    "attn": "blocks.{L}.linear_attn.hook_out",
    "mlp": "blocks.{L}.hook_mlp_out",
}


def _component_layers(model, component: str) -> List[int]:
    tmpl = _COMPONENTS[component]
    prefix, suffix = tmpl.split("{L}")
    pat = re.compile(re.escape(prefix) + r"(\d+)" + re.escape(suffix) + r"$")
    layers = []
    for h in model.hook_dict.keys():
        m = pat.fullmatch(h)
        if m:
            layers.append(int(m.group(1)))
    return sorted(set(layers))


@torch.no_grad()
def process_pair(
    model,
    pair: Dict,
    components: List[str],
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

    hook_names = [_COMPONENTS[c].format(L=L) for c in components for L in layers_by_comp[c]]
    clean_cache = _run_with_cache_resid(model, clean_toks, hook_names)
    corr_cache = _run_with_cache_resid(model, corr_toks, hook_names)

    ld_clean = _logit_diff_at_last(model(clean_toks), id_clean, id_corr)
    ld_corr = _logit_diff_at_last(model(corr_toks), id_clean, id_corr)
    correct = (ld_clean > margin) and (ld_corr < -margin)
    denom = ld_clean - ld_corr
    base = {"dimension": pair["dimension"], "type_key": pair["type_key"],
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
                    hk = _COMPONENTS[comp].format(L=L)
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="output/patch_pairs/qwen35_9b_aligned.json")
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--components", default="attn,mlp")
    ap.add_argument("--directions", default="denoise,noise")
    ap.add_argument("--positions", default="edit,last")
    ap.add_argument("--layers", default="all", help="'all' or comma-separated layer indices")
    ap.add_argument("--limit", type=int, default=0, help="first N pairs per dimension (smoke test)")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--num-shards", type=int, default=1, help="total GPU shards (multi-GPU)")
    ap.add_argument("--shard-id", type=int, default=0, help="index of this shard [0, num-shards)")
    ap.add_argument("--out", default="output/patch_components/qwen35_9b")
    args = ap.parse_args()

    components = [c.strip() for c in args.components.split(",") if c.strip()]
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    pos_modes = [p.strip() for p in args.positions.split(",") if p.strip()]
    for c in components:
        if c not in _COMPONENTS:
            raise ValueError(f"unknown component {c}; choose from {list(_COMPONENTS)}")

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

    print(f"Loading {args.model_name} via TransformerBridge (slow, ~8min)...")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformer_lens.model_bridge import TransformerBridge
    model = TransformerBridge.boot_transformers(args.model_name, device=args.device)
    model.eval()

    layers_by_comp: Dict[str, List[int]] = {}
    for c in components:
        avail = _component_layers(model, c)
        if args.layers == "all":
            layers_by_comp[c] = avail
        else:
            want = {int(x) for x in args.layers.split(",")}
            layers_by_comp[c] = [L for L in avail if L in want]
        print(f"  {c}: {len(avail)} layers available -> patching {len(layers_by_comp[c])}")
    # layer axis for plotting/merge: assume both components share the same layer set
    layers = sorted(set().union(*layers_by_comp.values()))

    from tqdm.auto import tqdm
    results: List[Dict] = []
    n_ok = 0
    for p in tqdm(pairs, desc="pairs", unit="pair"):
        r = process_pair(model, p, components, layers_by_comp, directions, pos_modes, args.margin)
        if r is None:
            continue
        results.append(r)
        n_ok += int(bool(r.get("baseline_ok")))
    print(f"baseline-correct pairs: {n_ok}/{len(results)}")

    os.makedirs(args.out, exist_ok=True)
    payload_meta = {
        "model_name": args.model_name, "pairs_file": args.pairs, "layers": layers,
        "components": components, "directions": directions, "positions": pos_modes,
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
