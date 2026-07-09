"""Phase 3 of the activation-patching study: per-HEAD mover analysis.

Phase 2 found that standard-GQA models have a strong *late attention mover* at
the final position (attn·last: Qwen3-8B L24=0.59, Qwen3-14B L29=0.51,
gemma-2 L28=0.48, phi-4 L23=0.28) — attention pulls the answer to the last
token at a discrete late layer. Phase 3 decomposes that block-level attention
output into INDIVIDUAL HEADS to name *which* heads do the fetch.

Method: patch a single attention head's output `blocks.L.attn.hook_z`
(shape [batch, pos, n_heads, d_head]) at the final position, for each (layer,
head) in a focused window around the attn·last peak. denoise recovery of the
normalized answer logit-diff = that head's causal contribution.

Only standard attention exposes per-head hook_z — this is the whole reason the
standard-GQA models (Qwen3-8B/14B, gemma-2, phi-4) were added; Qwen3.5-9B uses
hybrid/linear attention and has no per-head decomposition.

Metric / baseline filter / normalization are identical to Phase 1/2 (reused
from patch_run.py). Self-contained per model (no sharding): run one model per
GPU concurrently.

Run (one model per GPU, in parallel):
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python patching/patch_heads.py \
        --model-name Qwen/Qwen3-8B --dtype bfloat16 \
        --pairs results/patching/pairs/qwen3_8b_aligned.json \
        --layers 22,23,24,25,26 --out results/patching/heads/qwen3_8b
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
    _answer_token_id,
    _logit_diff_at_last,
    _position_sets,
    _run_with_cache_resid,
)
from patch_batch import (  # length-grouped batching (numerically identical to per-pair)
    batched_logit_diff_last,
    make_batched_head_patch_hook,
    group_by_length,
)

_Z_CANDIDATES = ["blocks.{L}.attn.hook_z"]


def _resolve_z_hook(model):
    """Return (hook_template, sorted_layers) for the per-head attn output hook."""
    hooks = set(model.hook_dict.keys())
    for tmpl in _Z_CANDIDATES:
        prefix, suffix = tmpl.split("{L}")
        pat = re.compile(re.escape(prefix) + r"(\d+)" + re.escape(suffix) + r"$")
        layers = sorted({int(m.group(1)) for h in hooks for m in [pat.fullmatch(h)] if m})
        if layers:
            return tmpl, layers
    return None, []


def _make_head_patch_hook(donor_act: torch.Tensor, positions: List[int], head: int):
    pos = torch.tensor(positions, dtype=torch.long)

    def hook(act, hook):  # noqa: ARG001  act: [batch, pos, n_heads, d_head]
        out = act.clone()
        out[:, pos, head, :] = donor_act[:, pos, head, :].to(act.dtype).to(act.device)
        return out
    return hook


@torch.no_grad()
def process_pair(model, pair, z_tmpl, layers, n_heads, directions, pos_modes, margin) -> Optional[Dict]:
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

    hook_names = [z_tmpl.format(L=L) for L in layers]
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

    effects: Dict[str, List[List[float]]] = {}  # key -> [layer][head]
    for direction in directions:
        base_toks = corr_toks if direction == "denoise" else clean_toks
        donor_cache = clean_cache if direction == "denoise" else corr_cache
        for pmode, positions in pos_sets.items():
            grid = []
            for L in layers:
                hk = z_tmpl.format(L=L)
                donor = donor_cache[hk]
                row = []
                for h in range(n_heads):
                    logits = model.run_with_hooks(
                        base_toks, fwd_hooks=[(hk, _make_head_patch_hook(donor, positions, h))])
                    ld = _logit_diff_at_last(logits, id_clean, id_corr)
                    if direction == "denoise":
                        row.append((ld - ld_corr) / denom)
                    else:
                        row.append((ld - ld_clean) / (ld_corr - ld_clean))
                grid.append(row)
            effects[f"{direction}:{pmode}"] = grid
    return {**base, "baseline_ok": True, "effects": effects}


@torch.no_grad()
def process_batch(model, batch, z_tmpl, layers, n_heads, directions, pos_modes, margin) -> List[Dict]:
    """Batched equivalent of process_pair over a group of equal-length pairs.

    Numerically identical to per-pair (no padding; causal attention is batch-
    independent), but runs one forward per (layer, head) for the whole group
    instead of per pair — ~mean-group-size fewer forwards.
    """
    clean, corr = batch["clean"], batch["corr"]           # [B, L]
    id_clean, id_corr, pairs = batch["id_clean"], batch["id_corr"], batch["pairs"]
    B, L = clean.shape

    # per-example position lists for each requested mode
    pos_by_mode: Dict[str, List[List[int]]] = {m: [] for m in pos_modes}
    for p in pairs:
        sets = _position_sets(p, L, pos_modes)
        for m in pos_modes:
            pos_by_mode[m].append(sets.get(m, []))

    ld_clean = batched_logit_diff_last(model(clean), id_clean, id_corr)
    ld_corr = batched_logit_diff_last(model(corr), id_clean, id_corr)
    correct = [(lc > margin) and (lr < -margin) for lc, lr in zip(ld_clean, ld_corr)]
    denom = [lc - lr for lc, lr in zip(ld_clean, ld_corr)]

    hook_names = [z_tmpl.format(L=Lyr) for Lyr in layers]
    clean_cache = _run_with_cache_resid(model, clean, hook_names)
    corr_cache = _run_with_cache_resid(model, corr, hook_names)

    # effects[b][key] = [layer][head]
    eff: List[Dict[str, List[List[float]]]] = [{} for _ in range(B)]
    for direction in directions:
        base_toks = corr if direction == "denoise" else clean
        donor_cache = clean_cache if direction == "denoise" else corr_cache
        for pmode in pos_modes:
            per_ex = pos_by_mode[pmode]
            grids = [[] for _ in range(B)]  # per example: list of rows (per layer)
            for Lyr in layers:
                hk = z_tmpl.format(L=Lyr)
                donor = donor_cache[hk]
                rows = [[] for _ in range(B)]
                for h in range(n_heads):
                    logits = model.run_with_hooks(
                        base_toks, fwd_hooks=[(hk, make_batched_head_patch_hook(donor, per_ex, h))])
                    ld = batched_logit_diff_last(logits, id_clean, id_corr)
                    for b in range(B):
                        if abs(denom[b]) < 1e-6:
                            rows[b].append(0.0)
                        elif direction == "denoise":
                            rows[b].append((ld[b] - ld_corr[b]) / denom[b])
                        else:
                            rows[b].append((ld[b] - ld_clean[b]) / (ld_corr[b] - ld_clean[b]))
                for b in range(B):
                    grids[b].append(rows[b])
            for b in range(B):
                eff[b][f"{direction}:{pmode}"] = grids[b]

    results = []
    for b in range(B):
        base = {"dimension": pairs[b]["dimension"], "type_key": pairs[b]["type_key"],
                "source": pairs[b].get("source", ""), "ld_clean": ld_clean[b], "ld_corr": ld_corr[b]}
        if not correct[b] or abs(denom[b]) < 1e-6:
            results.append({**base, "baseline_ok": False, "effects": {}})
        else:
            results.append({**base, "baseline_ok": True, "effects": eff[b]})
    return results


def _aggregate_heads(results, layers, n_heads):
    """Mean per (layer, head) over baseline-ok pairs, plus by_dim breakdown."""
    ok = [r for r in results if r.get("baseline_ok") and r.get("effects")]
    keys = sorted({k for r in ok for k in r["effects"]})
    agg = {}
    for k in keys:
        groups = {"all": ok}
        by_dim = defaultdict(list)
        for r in ok:
            by_dim[f"{r['dimension']}D"].append(r)
        groups.update(by_dim)
        agg[k] = {}
        for gname, rows in groups.items():
            acc = [[0.0] * n_heads for _ in layers]
            n = 0
            for r in rows:
                grid = r["effects"].get(k)
                if not grid:
                    continue
                n += 1
                for i in range(len(layers)):
                    for h in range(n_heads):
                        acc[i][h] += grid[i][h]
            if n:
                acc = [[v / n for v in row] for row in acc]
            agg[k][gname] = {"n_pairs": n, "mean": acc}
    return agg


def _plot_heads(agg, layers, n_heads, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"  (skip plots: {e})")
        return
    os.makedirs(outdir, exist_ok=True)
    for k, groups in agg.items():
        m = groups.get("all", {}).get("mean")
        if not m:
            continue
        import numpy as np
        arr = np.array(m)  # [n_layers, n_heads]
        fig, ax = plt.subplots(figsize=(max(6, n_heads * 0.28), max(2.5, len(layers) * 0.5)))
        vmax = max(0.2, float(np.abs(arr).max()))
        im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_yticks(range(len(layers))); ax.set_yticklabels([f"L{L}" for L in layers])
        ax.set_xticks(range(n_heads)); ax.set_xticklabels(range(n_heads), fontsize=7)
        ax.set_xlabel("head"); ax.set_title(f"per-head recovery — {k}")
        fig.colorbar(im, ax=ax, fraction=0.025)
        fig.tight_layout()
        p = os.path.join(outdir, f"heads_{k.replace(':', '_')}.png")
        fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  wrote head heatmaps -> {outdir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="results/patching/pairs/qwen3_8b_aligned.json")
    ap.add_argument("--model-name", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--directions", default="denoise")
    ap.add_argument("--positions", default="last", help="comma-separated: last, edit")
    ap.add_argument("--layers", default="all", help="'all' or comma-separated layer indices (window around attn·last peak)")
    ap.add_argument("--limit", type=int, default=0, help="first N pairs per dimension (smoke test)")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="batch pairs of equal token length (1 = per-pair). Numerically identical; "
                         "~mean-group-size (≈8-11x) faster. Lower if OOM (esp. --layers all on 27B).")
    ap.add_argument("--out", default="results/patching/heads/qwen3_8b")
    args = ap.parse_args()

    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    pos_modes = [p.strip() for p in args.positions.split(",") if p.strip()]

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
    print(f"Loaded {len(pairs)} aligned pairs")

    print(f"Loading {args.model_name} via TransformerBridge (slow, ~8min), dtype={args.dtype}...")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformer_lens.model_bridge import TransformerBridge
    model = TransformerBridge.boot_transformers(
        args.model_name, device=args.device, dtype=getattr(torch, args.dtype))
    model.eval()

    z_tmpl, avail = _resolve_z_hook(model)
    if not avail:
        raise SystemExit(f"No per-head hook_z for this architecture (tried {_Z_CANDIDATES}); "
                         "per-head analysis needs standard attention.")
    want = None if args.layers == "all" else {int(x) for x in args.layers.split(",")}
    layers = avail if want is None else [L for L in avail if L in want]
    n_heads = int(model.cfg.n_heads)
    print(f"  hook_z '{z_tmpl}' -> {len(avail)} layers; patching {layers}; n_heads={n_heads}")

    from tqdm.auto import tqdm
    results: List[Dict] = []
    n_ok = 0
    if args.batch_size and args.batch_size > 1:
        batches = group_by_length(pairs, model.tokenizer, max_batch=args.batch_size, device=model.cfg.device)
        print(f"Batched: {len(pairs)} pairs -> {len(batches)} equal-length batches "
              f"(mean {len(pairs)/max(1,len(batches)):.1f}/batch)")
        for batch in tqdm(batches, desc="batches", unit="batch"):
            rs = process_batch(model, batch, z_tmpl, layers, n_heads, directions, pos_modes, args.margin)
            results.extend(rs)
            n_ok += sum(int(bool(r.get("baseline_ok"))) for r in rs)
    else:
        for p in tqdm(pairs, desc="pairs", unit="pair"):
            r = process_pair(model, p, z_tmpl, layers, n_heads, directions, pos_modes, args.margin)
            if r is None:
                continue
            results.append(r)
            n_ok += int(bool(r.get("baseline_ok")))
    print(f"baseline-correct pairs: {n_ok}/{len(results)}")

    os.makedirs(args.out, exist_ok=True)
    agg = _aggregate_heads(results, layers, n_heads)
    meta = {"model_name": args.model_name, "pairs_file": args.pairs, "z_hook": z_tmpl,
            "layers": layers, "n_heads": n_heads, "directions": directions,
            "positions": pos_modes, "margin": args.margin,
            "n_pairs": len(results), "n_baseline_ok": n_ok}
    out_json = os.path.join(args.out, "patch_heads_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"metadata": meta, "aggregate": agg, "per_pair": results}, f, ensure_ascii=False, indent=2)
    print(f"Wrote results -> {out_json}")

    # console: top mover heads per effect-key (pooled 'all')
    for k in sorted(agg.keys()):
        m = agg[k].get("all", {}).get("mean")
        if not m:
            continue
        flat = [(layers[i], h, m[i][h]) for i in range(len(layers)) for h in range(n_heads)]
        flat.sort(key=lambda x: x[2], reverse=True)
        print(f"\n[{k}] top mover heads (L, head, recovery):")
        for L, h, v in flat[:8]:
            print(f"  L{L} H{h}: {v:+.3f}")

    _plot_heads(agg, layers, n_heads, os.path.join(args.out, "plots"))


if __name__ == "__main__":
    main()
