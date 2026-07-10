"""Joint top-k mover denoise patch — cumulative sufficiency of the mover set.

REEXPERIMENT_TODO P2-1. Phase 3 (patch_heads.py) measures each head's *single*
denoise recovery; the progress artifact's "concentration" is built from those
single recoveries, which are NOT additive. This script measures the missing
quantity: patch the top-k mover heads' `hook_z` at the final position TOGETHER
(clean donor into the corrupted run) and read the cumulative recovery

    recovery_k = (ld_patched_topk - ld_corr) / (ld_clean - ld_corr)

so recovery_k vs k shows how much of the answer the k heads *jointly* restore
(vs the sum of singles). Mirrors patch_ablate.py but in the denoise direction
(patch-in on corrupted) instead of ablation (zero-out on clean).

Ranked heads are read from the Phase-3 results JSON (denoise:last).

Run:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python patching/patch_joint.py \
        --model-name Qwen/Qwen3-8B --dtype bfloat16 \
        --pairs results/patching/pairs/qwen3_8b_aligned.json \
        --heads-json results/patching/heads/qwen3_8b/patch_heads_results.json \
        --out results/patching/joint/qwen3_8b
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_run import _answer_token_id, _logit_diff_at_last, _run_with_cache_resid
from patch_ablate import _ranked_heads  # ranked (layer,head) by denoise:last recovery

Z_TMPL = "blocks.{L}.attn.hook_z"


def _make_multihead_patch_hook(donor_act: torch.Tensor, positions: List[int], heads: List[int]):
    pos = torch.tensor(positions, dtype=torch.long)

    def hook(act, hook):  # noqa: ARG001  act: [batch, pos, n_heads, d_head]
        out = act.clone()
        for h in heads:
            out[:, pos, h, :] = donor_act[:, pos, h, :].to(act.dtype).to(act.device)
        return out
    return hook


@torch.no_grad()
def process_pair(model, pair, ranked, layers, ks, margin) -> Optional[Dict]:
    tok = model.tokenizer
    id_c = _answer_token_id(tok, pair["clean_answer"])
    id_r = _answer_token_id(tok, pair["corrupted_answer"])
    clean = tok(pair["clean_prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.cfg.device)
    corr = tok(pair["corrupted_prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.cfg.device)
    if clean.shape[1] != corr.shape[1]:
        return None
    last = clean.shape[1] - 1
    hook_names = [Z_TMPL.format(L=L) for L in layers]
    clean_cache = _run_with_cache_resid(model, clean, hook_names)
    ld_clean = _logit_diff_at_last(model(clean), id_c, id_r)
    ld_corr = _logit_diff_at_last(model(corr), id_c, id_r)
    denom = ld_clean - ld_corr
    base = {"dimension": pair["dimension"], "type_key": pair["type_key"],
            "clean_answer": pair["clean_answer"]}
    if not (ld_clean > margin and ld_corr < -margin) or abs(denom) < 1e-6:
        return {**base, "baseline_ok": False, "effects": {}}

    eff: Dict[str, float] = {}
    for k in ks:
        by_layer: Dict[int, List[int]] = defaultdict(list)
        for L, h in ranked[:k]:
            by_layer[L].append(h)
        fwd = [(Z_TMPL.format(L=L), _make_multihead_patch_hook(clean_cache[Z_TMPL.format(L=L)], [last], hs))
               for L, hs in by_layer.items()]
        ld = _logit_diff_at_last(model.run_with_hooks(corr, fwd_hooks=fwd), id_c, id_r)
        eff[f"joint_top{k}"] = (ld - ld_corr) / denom
    return {**base, "baseline_ok": True, "effects": eff}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--heads-json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--ks", default="1,2,3,5", help="cumulative top-k head counts to patch in")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]

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
    print(f"Top mover heads: {[f'L{L}H{h}' for L, h in ranked[:max(ks)]]}")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformer_lens.model_bridge import TransformerBridge
    print(f"Loading {args.model_name} (dtype={args.dtype})...")
    model = TransformerBridge.boot_transformers(args.model_name, device=args.device, dtype=getattr(torch, args.dtype))
    model.eval()

    from tqdm.auto import tqdm
    results, n_ok = [], 0
    for p in tqdm(pairs, desc="pairs", unit="pair"):
        r = process_pair(model, p, ranked, layers, ks, args.margin)
        if r is None:
            continue
        results.append(r); n_ok += int(bool(r.get("baseline_ok")))
    print(f"baseline-correct pairs: {n_ok}/{len(results)}")

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
            "ranked_top": [[L, h] for L, h in ranked[:max(ks)]], "ks": ks,
            "n_pairs": len(results), "n_baseline_ok": n_ok}
    with open(os.path.join(args.out, "patch_joint_results.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": meta, "aggregate": agg, "per_pair": results}, f, ensure_ascii=False, indent=2)

    print("\n=== joint top-k denoise recovery (patch-in on corrupted, last position) ===")
    print("  (compare to the SUM of single-head recoveries to see non-additivity)")
    for k in ks:
        v = agg["all"].get(f"joint_top{k}")
        print(f"  joint top{k}: {v:+.3f}" if v is not None else f"  joint top{k}: n/a")


if __name__ == "__main__":
    main()
