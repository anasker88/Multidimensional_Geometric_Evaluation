"""Point-1 verification: is gemma-2-27b's L29 half-recovery crossing a REAL
restoration of the clean answer, or just DESTRUCTION of the corrupted answer?

The normalized-recovery landmark uses ld = logit(A) - logit(B). ld can rise
either because logit(A) climbs (the clean answer is genuinely being written) or
because logit(B) falls (the corrupted answer is merely suppressed) — the
false-positive failure mode flagged by Heimersheim & Nanda (2024). The stored
patch results only keep ld, so we cannot tell the two apart post-hoc; this
script re-runs the denoise:last patch and logs logit(A) and logit(B)
SEPARATELY, pooled per layer.

Output (small, ~KB): per-layer mean patched logit(A) / logit(B) plus the clean
and corrupted baselines, so logit(A) and logit(B) can be plotted across layers.
If at L29 logit(A) has already risen toward its clean value, the crossing is a
real decision; if only logit(B) has dropped, the true decision is later (~L42).

Needs the model in memory once (gemma-2-27b -> A100 80GB, bf16). Reuses the
exact helpers from patch_run so the numbers are directly comparable.

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python patching/patch_verify_logits.py \
        --pairs results/patching/pairs/gemma2_27b_aligned.json \
        --model-name google/gemma-2-27b-it --dtype bfloat16 \
        --out results/patching/verify/gemma2_27b_logits.json
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import torch

from patch_run import (  # same directory
    _answer_token_id,
    _resid_post_layers,
    _make_patch_hook,
)


@torch.no_grad()
def _logits_at_last(logits: torch.Tensor, id_a: int, id_b: int):
    last = logits[0, -1, :]
    return float(last[id_a]), float(last[id_b])


@torch.no_grad()
def process_pair(model, pair: Dict, layers: List[int], margin: float):
    tok = model.tokenizer
    id_a = _answer_token_id(tok, pair["clean_answer"])
    id_b = _answer_token_id(tok, pair["corrupted_answer"])
    clean = tok(pair["clean_prompt"], return_tensors="pt",
                add_special_tokens=False)["input_ids"].to(model.cfg.device)
    corr = tok(pair["corrupted_prompt"], return_tensors="pt",
               add_special_tokens=False)["input_ids"].to(model.cfg.device)
    if clean.shape[1] != corr.shape[1]:
        return None
    seq = clean.shape[1]
    last_pos = [seq - 1]

    # cache clean resid_post at all layers (the donor for denoise)
    _, cache = model.run_with_cache(
        clean, names_filter=lambda n: n.endswith("hook_resid_post"))

    a_c, b_c = _logits_at_last(model(clean), id_a, id_b)
    a_x, b_x = _logits_at_last(model(corr), id_a, id_b)
    ld_clean, ld_corr = a_c - b_c, a_x - b_x
    if not (ld_clean > margin and ld_corr < -margin):
        return None  # same baseline filter as patch_run

    # denoise:last — inject clean resid at the final position, layer by layer
    a_by_L, b_by_L = [], []
    for L in layers:
        donor = cache[f"blocks.{L}.hook_resid_post"]
        logits = model.run_with_hooks(
            corr, fwd_hooks=[(f"blocks.{L}.hook_resid_post",
                              _make_patch_hook(donor, last_pos))])
        a, b = _logits_at_last(logits, id_a, id_b)
        a_by_L.append(a)
        b_by_L.append(b)
    return {
        "logit_a_clean": a_c, "logit_b_clean": b_c,
        "logit_a_corr": a_x, "logit_b_corr": b_x,
        "logit_a_patched": a_by_L, "logit_b_patched": b_by_L,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all baseline-ok pairs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.load(open(args.pairs))
    pairs = [p for p in data["pairs"] if p.get("token_aligned")]
    if args.limit:
        pairs = pairs[:args.limit]

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformer_lens.model_bridge import TransformerBridge
    print(f"Loading {args.model_name} ({args.dtype}) ...")
    model = TransformerBridge.boot_transformers(
        args.model_name, device=args.device, dtype=getattr(torch, args.dtype))
    model.eval()
    layers = _resid_post_layers(model)
    print(f"{len(layers)} layers; {len(pairs)} aligned pairs")

    acc = defaultdict(list)
    from tqdm.auto import tqdm
    n = 0
    for p in tqdm(pairs, unit="pair"):
        r = process_pair(model, p, layers, args.margin)
        if r is None:
            continue
        n += 1
        for k, v in r.items():
            acc[k].append(v)

    import numpy as np
    def mean(x):
        return np.asarray(x).mean(0).tolist()
    out = {
        "model_name": args.model_name, "layers": layers, "n_baseline_ok": n,
        "logit_a_clean": float(np.mean(acc["logit_a_clean"])),
        "logit_b_clean": float(np.mean(acc["logit_b_clean"])),
        "logit_a_corr": float(np.mean(acc["logit_a_corr"])),
        "logit_b_corr": float(np.mean(acc["logit_b_corr"])),
        "logit_a_patched": mean(acc["logit_a_patched"]),
        "logit_b_patched": mean(acc["logit_b_patched"]),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Wrote -> {args.out} (n={n})")


if __name__ == "__main__":
    main()
