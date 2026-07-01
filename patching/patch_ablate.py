"""Phase 4: causal verification of the mover heads by ablation.

Phase 3 (patch_heads.py) *named* the late attention mover heads by denoise
recovery — a correlational attribution. Phase 4 tests their *necessity*: on the
CLEAN run, zero-ablate the top-k mover heads at the final position and measure
how far the answer collapses toward the corrupted behaviour.

Metric (per pair, baseline-ok = clean→correct & corrupted→wrong):
    drop_frac = (ld_clean - ld_ablated) / (ld_clean - ld_corr)
  = fraction of the clean→corrupt logit-diff gap reproduced by the ablation.
  1.0 => ablating those heads fully reproduces corruption; 0.0 => no effect.

Mover heads are read straight from the Phase 3 results JSON (ranked by
denoise:last recovery). A random-head control (same layer window, seeded) and a
peak-layer "all heads" upper bound are reported alongside, so the mover effect
is calibrated against chance and against the whole block.

Run (one model per GPU):
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python patching/patch_ablate.py \
        --model-name Qwen/Qwen3-8B --dtype bfloat16 \
        --pairs results/patching/pairs/qwen3_8b_aligned.json \
        --heads-json results/patching/heads/qwen3_8b/patch_heads_results.json \
        --out results/patching/ablate/qwen3_8b
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_run import _answer_token_id, _logit_diff_at_last

Z_TMPL = "blocks.{L}.attn.hook_z"


def _ranked_heads(heads_json: str, key: str = "denoise:last") -> Tuple[List[Tuple[int, int]], List[int], int]:
    """Return [(layer, head), ...] ranked high->low by recovery, plus (layers, n_heads)."""
    d = json.load(open(heads_json))
    m = d["metadata"]; layers = m["layers"]; nh = m["n_heads"]
    grid = d["aggregate"][key]["all"]["mean"]
    flat = [(layers[i], h, grid[i][h]) for i in range(len(layers)) for h in range(nh)]
    flat.sort(key=lambda x: x[2], reverse=True)
    return [(L, h) for L, h, _ in flat], layers, nh


def _make_ablate_hook(heads: List[int], positions: List[int]):
    pos = torch.tensor(positions, dtype=torch.long)

    def hook(act, hook):  # noqa: ARG001  act: [batch, pos, n_heads, d_head]
        out = act.clone()
        for h in heads:
            out[:, pos, h, :] = 0.0
        return out
    return hook


def _ablate_ld(model, toks, heads: List[Tuple[int, int]], last_pos: int, id_c, id_r) -> float:
    by_layer: Dict[int, List[int]] = defaultdict(list)
    for L, h in heads:
        by_layer[L].append(h)
    fwd = [(Z_TMPL.format(L=L), _make_ablate_hook(hs, [last_pos])) for L, hs in by_layer.items()]
    logits = model.run_with_hooks(toks, fwd_hooks=fwd)
    return _logit_diff_at_last(logits, id_c, id_r)


@torch.no_grad()
def process_pair(model, pair, ranked, layers, n_heads, ks, n_rand, rng, margin) -> Optional[Dict]:
    tok = model.tokenizer
    id_c = _answer_token_id(tok, pair["clean_answer"])
    id_r = _answer_token_id(tok, pair["corrupted_answer"])
    clean = tok(pair["clean_prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.cfg.device)
    corr = tok(pair["corrupted_prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.cfg.device)
    if clean.shape[1] != corr.shape[1]:
        return None
    last = clean.shape[1] - 1
    ld_clean = _logit_diff_at_last(model(clean), id_c, id_r)
    ld_corr = _logit_diff_at_last(model(corr), id_c, id_r)
    denom = ld_clean - ld_corr
    base = {"dimension": pair["dimension"], "type_key": pair["type_key"]}
    if not (ld_clean > margin and ld_corr < -margin) or abs(denom) < 1e-6:
        return {**base, "baseline_ok": False, "effects": {}}

    eff: Dict[str, float] = {}
    # cumulative top-k mover ablation
    for k in ks:
        ld = _ablate_ld(model, clean, ranked[:k], last, id_c, id_r)
        eff[f"top{k}"] = (ld_clean - ld) / denom
    # random-head control (same window as ranked heads), averaged over n_rand draws
    pool = ranked[:]  # all (layer,head) in the scanned window
    top_set = set(ranked[:max(ks)])
    ctrl_pool = [x for x in pool if x not in top_set]
    for k in ks:
        vals = []
        for _ in range(n_rand):
            pick = rng.sample(ctrl_pool, min(k, len(ctrl_pool)))
            ld = _ablate_ld(model, clean, pick, last, id_c, id_r)
            vals.append((ld_clean - ld) / denom)
        eff[f"rand{k}"] = sum(vals) / len(vals)
    # peak-layer all-heads upper bound (all heads in the top head's layer)
    peakL = ranked[0][0]
    ld = _ablate_ld(model, clean, [(peakL, h) for h in range(n_heads)], last, id_c, id_r)
    eff[f"peakL{peakL}_all"] = (ld_clean - ld) / denom
    return {**base, "baseline_ok": True, "effects": eff}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--heads-json", required=True, help="Phase 3 patch_heads_results.json for this model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--ks", default="1,2,3,5", help="cumulative top-k head counts to ablate")
    ap.add_argument("--n-rand", type=int, default=5, help="random control draws per k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    rng = random.Random(args.seed)

    with open(args.pairs) as f:
        pairs = [p for p in json.load(f)["pairs"] if p.get("token_aligned")]
    if args.limit > 0:
        per_dim: Dict[int, int] = defaultdict(int); sel = []
        for p in pairs:
            if per_dim[p["dimension"]] < args.limit:
                sel.append(p); per_dim[p["dimension"]] += 1
        pairs = sel
    print(f"Loaded {len(pairs)} aligned pairs")

    ranked, layers, n_heads = _ranked_heads(args.heads_json)
    print(f"Top mover heads (from Phase 3): {[f'L{L}H{h}' for L,h in ranked[:5]]}")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformer_lens.model_bridge import TransformerBridge
    print(f"Loading {args.model_name} (dtype={args.dtype})...")
    model = TransformerBridge.boot_transformers(args.model_name, device=args.device, dtype=getattr(torch, args.dtype))
    model.eval()

    from tqdm.auto import tqdm
    results = []
    n_ok = 0
    for p in tqdm(pairs, desc="pairs", unit="pair"):
        r = process_pair(model, p, ranked, layers, n_heads, ks, args.n_rand, rng, args.margin)
        if r is None:
            continue
        results.append(r); n_ok += int(bool(r.get("baseline_ok")))
    print(f"baseline-correct pairs: {n_ok}/{len(results)}")

    # aggregate: mean drop_frac per key over baseline-ok, + by_dim
    ok = [r for r in results if r.get("baseline_ok")]
    keys = sorted({k for r in ok for k in r["effects"]})
    def mean_over(rows, k):
        vs = [r["effects"][k] for r in rows if k in r["effects"]]
        return sum(vs) / len(vs) if vs else None
    agg = {"all": {k: mean_over(ok, k) for k in keys}, "n_pairs": len(ok)}
    by_dim = {}
    for dim in (2, 3, 4):
        rows = [r for r in ok if r["dimension"] == dim]
        if rows:
            by_dim[f"{dim}D"] = {"n_pairs": len(rows), **{k: mean_over(rows, k) for k in keys}}
    agg["by_dim"] = by_dim

    os.makedirs(args.out, exist_ok=True)
    meta = {"model_name": args.model_name, "pairs_file": args.pairs, "heads_json": args.heads_json,
            "ranked_top": [[L, h] for L, h in ranked[:10]], "ks": ks, "n_rand": args.n_rand,
            "seed": args.seed, "n_pairs": len(results), "n_baseline_ok": n_ok}
    with open(os.path.join(args.out, "patch_ablate_results.json"), "w") as f:
        json.dump({"metadata": meta, "aggregate": agg, "per_pair": results}, f, ensure_ascii=False, indent=2)

    print("\n=== ablation drop_frac (clean run, last position) ===")
    print("  (1.0 = fully reproduces corruption; higher = more necessary)")
    for k in ks:
        mv = agg["all"].get(f"top{k}"); rd = agg["all"].get(f"rand{k}")
        print(f"  top{k} movers: {mv:+.3f}   |  random{k}: {rd:+.3f}   |  ratio {mv/rd:.1f}x" if mv and rd and abs(rd) > 1e-6 else f"  top{k}: {mv}")
    pk = [k for k in keys if k.startswith("peakL")]
    if pk:
        print(f"  {pk[0]}: {agg['all'][pk[0]]:+.3f}  (whole peak layer, upper bound)")


if __name__ == "__main__":
    main()
