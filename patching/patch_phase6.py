"""Phase 6: locate gemma-2-9b's BACKUP heads (the Hydra effect behind Finding 4).

Finding 4: ablating gemma-2-9b's top-5 mover heads does NOT break the answer —
other heads compensate (drop_frac ≈ 0 / negative). This backup is strongest in
the IC type (per-type gap −0.38, concentrated in box/prism). Phase 6 asks WHICH
heads do the backup.

Method (on the CLEAN run, final position, per baseline-ok pair):
  - primary set = the top-5 mover heads (from Phase 3).
  - for every candidate head h in a late-layer window, measure the MARGINAL
    necessity it gains once the primaries are gone:
        marginal(h) = drop_frac(top5 ∪ {h}) − drop_frac(top5)
    A backup head has a small standalone effect but a large marginal effect —
    it "wakes up" only when the primary movers are ablated (McGrath Hydra).
  - rank candidates by mean marginal; then measure the cumulative drop_frac of
    top5 + the N strongest backups (N = 1,2,3,5,8,12) to see how many backup
    heads are needed to finally break the answer.

Standalone drop_frac({h}) is also recorded for the ranked backups, so the
"wakes up" signature (standalone≈0, marginal large) can be shown.

Run (gemma-2-9b fits one 48GB GPU in bf16):
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python patching/patch_phase6.py \
        --model-name google/gemma-2-9b-it --dtype bfloat16 \
        --pairs results/patching/pairs/gemma2_9b_aligned.json \
        --heads-json results/patching/heads/gemma2_9b/patch_heads_results.json \
        --type-key 2 --cand-layers 26-41 \
        --out results/patching/phase6/gemma2_9b_ic
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_run import _answer_token_id, _logit_diff_at_last
from patch_ablate import _ranked_heads, _ablate_ld


@torch.no_grad()
def process_pair(model, pair, top5, cand, ks_backup, margin) -> Optional[Dict]:
    tok = model.tokenizer
    id_c = _answer_token_id(tok, pair["clean_answer"])
    id_r = _answer_token_id(tok, pair["corrupted_answer"])
    clean = tok(pair["clean_prompt"], return_tensors="pt",
                add_special_tokens=False)["input_ids"].to(model.cfg.device)
    corr = tok(pair["corrupted_prompt"], return_tensors="pt",
               add_special_tokens=False)["input_ids"].to(model.cfg.device)
    if clean.shape[1] != corr.shape[1]:
        return None
    last = clean.shape[1] - 1
    ld_clean = _logit_diff_at_last(model(clean), id_c, id_r)
    ld_corr = _logit_diff_at_last(model(corr), id_c, id_r)
    denom = ld_clean - ld_corr
    base = {"dimension": pair["dimension"], "type_key": pair["type_key"],
            "family": pair.get("family", "other")}
    if not (ld_clean > margin and ld_corr < -margin) or abs(denom) < 1e-6:
        return {**base, "baseline_ok": False}

    def drop(heads):
        ld = _ablate_ld(model, clean, heads, last, id_c, id_r)
        return (ld_clean - ld) / denom

    d_top5 = drop(top5)
    # marginal necessity of each candidate once the primaries are gone
    marg = {}
    for h in cand:
        marg[f"{h[0]}_{h[1]}"] = drop(top5 + [h]) - d_top5
    return {**base, "baseline_ok": True, "d_top5": d_top5, "marginal": marg}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--heads-json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--type-key", default="2", help="option-set type to target (2=IC)")
    ap.add_argument("--cand-layers", default="26-41", help="candidate backup layer range lo-hi")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.cand_layers.split("-"))
    ranked, layers, n_heads = _ranked_heads(args.heads_json)
    top5 = ranked[:5]
    top5_set = set(top5)
    cand = [(L, h) for L in range(lo, hi + 1) for h in range(n_heads)
            if (L, h) not in top5_set]
    print(f"top5={['L%dH%d' % (L, h) for L, h in top5]}  candidates={len(cand)} "
          f"(L{lo}-{hi} x {n_heads} heads)")

    with open(args.pairs) as f:
        pairs = [p for p in json.load(f)["pairs"]
                 if p.get("token_aligned") and str(p["type_key"]) == args.type_key]
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"{len(pairs)} aligned pairs of type {args.type_key}")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformer_lens.model_bridge import TransformerBridge
    print(f"Loading {args.model_name} ({args.dtype})...")
    model = TransformerBridge.boot_transformers(
        args.model_name, device=args.device, dtype=getattr(torch, args.dtype))
    model.eval()

    from tqdm.auto import tqdm
    rows = []
    for p in tqdm(pairs, unit="pair"):
        r = process_pair(model, p, top5, cand, None, args.margin)
        if r:
            rows.append(r)
    ok = [r for r in rows if r.get("baseline_ok")]
    print(f"baseline-ok: {len(ok)}/{len(rows)}")

    import numpy as np

    def summarize(subset: List[Dict], tag: str) -> Dict:
        if not subset:
            return {}
        d5 = float(np.mean([r["d_top5"] for r in subset]))
        # mean marginal per candidate
        marg = {k: float(np.mean([r["marginal"][k] for r in subset])) for k in cand_keys}
        ranked_bk = sorted(marg.items(), key=lambda x: x[1], reverse=True)
        backups = [{"head": k, "marginal": v} for k, v in ranked_bk[:15]]
        # winner's-curse controls for the #1 backup: (a) bootstrap CI of its
        # marginal, (b) split-half cross-fit — rank on half the pairs, evaluate the
        # winner on the other half (avg over 200 random splits), plus how often the
        # #1 backup is re-selected. If the cross-fit value ≈ the naive value and it
        # wins nearly every split, the +0.42 is not a selection artifact.
        M = np.array([[r["marginal"][k] for k in cand_keys] for r in subset])
        rng = np.random.default_rng(0)
        n = len(subset)
        top_key = ranked_bk[0][0]; top_col = cand_keys.index(top_key)
        bs = M[rng.integers(0, n, size=(10000, n)), top_col].mean(1)
        cf_vals, wins = [], 0
        for _ in range(200):
            perm = rng.permutation(n); a, b = perm[:n // 2], perm[n // 2:]
            w = int(np.argmax(M[a].mean(0)))
            cf_vals.append(float(M[b].mean(0)[w]))
            wins += int(cand_keys[w] == top_key)
        wc = {"head": top_key, "marginal": ranked_bk[0][1],
              "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
              "crossfit_mean": float(np.mean(cf_vals)), "crossfit_win_frac": wins / 200}
        print(f"  [{tag}] n={len(subset)}  drop_frac(top5)={d5:+.3f}  "
              f"top backups: {[(k, round(v,3)) for k,v in ranked_bk[:6]]}")
        print(f"    winner's-curse: {top_key} marginal {wc['marginal']:+.3f} "
              f"CI[{wc['ci'][0]:+.3f},{wc['ci'][1]:+.3f}] crossfit {wc['crossfit_mean']:+.3f} "
              f"win {wc['crossfit_win_frac']:.0%}")
        return {"n": len(subset), "drop_top5": d5, "marginal_ranked": backups,
                "top_backup": wc}

    cand_keys = [f"{L}_{h}" for L, h in cand]
    out = {
        "metadata": {"model_name": args.model_name, "type_key": args.type_key,
                     "top5": [list(x) for x in top5], "cand_layers": [lo, hi],
                     "n_heads": n_heads, "n_ok": len(ok)},
        "IC_all": summarize(ok, "IC-all"),
        "IC_box_prism": summarize([r for r in ok if r["family"] in ("box", "prism")],
                                  "IC∩box/prism"),
        "per_pair": ok,
    }

    # cumulative curve: greedy add the strongest backups (by IC∩box/prism ranking)
    # and RE-MEASURE drop_frac over those pairs for N = 1,2,3,5,8,12
    target = [r for r in ok if r["family"] in ("box", "prism")]
    if target and out["IC_box_prism"]:
        order = [tuple(int(x) for x in b["head"].split("_"))
                 for b in out["IC_box_prism"]["marginal_ranked"]]
        out["cumulative_order"] = [list(x) for x in order[:12]]
        print("Measuring cumulative drop_frac (top5 + top-N backups) on IC∩box/prism...")
        measured = _measure_cumulative(model, args.pairs, args.type_key, top5, order,
                                       [1, 2, 3, 5, 8, 12], args.margin)
        out["cumulative_measured"] = measured
        print("  cumulative:", {k: round(v, 3) for k, v in measured.items()})

    os.makedirs(args.out, exist_ok=True)
    outfile = os.path.join(args.out, "phase6_backup.json")
    with open(outfile, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote -> {outfile}")


@torch.no_grad()
def _measure_cumulative(model, pairs_file, type_key, top5, order, Ns, margin):
    """drop_frac(top5 + top-N backups), mean over baseline-ok IC∩box/prism pairs."""
    import numpy as np
    tok = model.tokenizer
    pairs = [p for p in json.load(open(pairs_file))["pairs"]
             if p.get("token_aligned") and str(p["type_key"]) == type_key
             and p.get("family") in ("box", "prism")]
    acc = defaultdict(list)
    for p in pairs:
        id_c = _answer_token_id(tok, p["clean_answer"])
        id_r = _answer_token_id(tok, p["corrupted_answer"])
        clean = tok(p["clean_prompt"], return_tensors="pt",
                    add_special_tokens=False)["input_ids"].to(model.cfg.device)
        corr = tok(p["corrupted_prompt"], return_tensors="pt",
                   add_special_tokens=False)["input_ids"].to(model.cfg.device)
        if clean.shape[1] != corr.shape[1]:
            continue
        last = clean.shape[1] - 1
        ld_clean = _logit_diff_at_last(model(clean), id_c, id_r)
        ld_corr = _logit_diff_at_last(model(corr), id_c, id_r)
        denom = ld_clean - ld_corr
        if not (ld_clean > margin and ld_corr < -margin) or abs(denom) < 1e-6:
            continue
        for N in Ns:
            heads = top5 + list(order[:N])
            ld = _ablate_ld(model, clean, heads, last, id_c, id_r)
            acc[f"top5+{N}"].append((ld_clean - ld) / denom)
        # baseline top5 for reference
        ld = _ablate_ld(model, clean, top5, last, id_c, id_r)
        acc["top5+0"].append((ld_clean - ld) / denom)
    return {k: float(np.mean(v)) for k, v in acc.items()}


if __name__ == "__main__":
    main()
