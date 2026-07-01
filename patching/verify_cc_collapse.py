#!/usr/bin/env python
"""Finding-3 model-universality check: on true-Yes CC (type_key=3) clean prompts,
read the answer-position logits for A=Yes / B=No / C=Cannot per dimension.

Replicates the Qwen3.5-9B 4D-CC "confident No" analysis for any model so the
finding can be confirmed as universal or flagged model-specific.

Usage:
  python patching/verify_cc_collapse.py --model-name Qwen/Qwen3-8B \
      --pairs results/patching/pairs/qwen3_8b_aligned.json --dtype bfloat16
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from patching.patch_run import _answer_token_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--by-source", action="store_true",
                    help="also break the per-dimension table down by CC construction family "
                         "(source), to test whether a 4D collapse is construction-specific")
    args = ap.parse_args()

    with open(args.pairs, encoding="utf-8") as f:
        pairs = json.load(f)["pairs"]
    cc = [p for p in pairs if p["type_key"] == "3" and p["clean_answer"] == "A"]

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformer_lens.model_bridge import TransformerBridge
    print(f"Loading {args.model_name} (dtype={args.dtype})...")
    model = TransformerBridge.boot_transformers(
        args.model_name, device=args.device, dtype=getattr(torch, args.dtype))
    model.eval()
    tok = model.tokenizer
    idA, idB, idC = (_answer_token_id(tok, l) for l in ("A", "B", "C"))

    def _family(src: str) -> str:
        # synthetic_4d_t3_centroid -> centroid ; questions_augmented -> mined
        return src.split("_t3_")[-1] if "_t3_" in src else ("mined" if "augmented" in src else src)

    # compute per-example A/B/C once
    for p in cc:
        toks = tok(p["clean_prompt"], return_tensors="pt",
                   add_special_tokens=False)["input_ids"].to(model.cfg.device)
        with torch.no_grad():
            last = model(toks)[0, -1, :].float()
        p["_abc"] = (float(last[idA]), float(last[idB]), float(last[idC]))

    def _row(label, rows):
        n = len(rows)
        if not n:
            return
        mA = sum(r["_abc"][0] for r in rows) / n
        mB = sum(r["_abc"][1] for r in rows) / n
        mC = sum(r["_abc"][2] for r in rows) / n
        am = "A ✓" if mA == max(mA, mB, mC) else ("B ✗" if mB == max(mA, mB, mC) else "C")
        print(f"{label:>22} {n:>3} {mA:>7.2f} {mB:>7.2f} {mC:>8.2f} {mA-mB:>+12.2f} {am:>7}")

    print(f"\n=== {args.model_name} — true-Yes CC, answer-token logits ===")
    print(f"{'group':>22} {'n':>3} {'A=Yes':>7} {'B=No':>7} {'C=Cannot':>8} {'cleanLD(A-B)':>12} {'argmax':>7}")
    for dim in (2, 3, 4):
        sub = [p for p in cc if p["dimension"] == dim]
        _row(f"{dim}D (all)", sub)
        if args.by_source:
            for fam in sorted({_family(p["source"]) for p in sub}):
                _row(f"{dim}D · {fam}", [p for p in sub if _family(p["source"]) == fam])


if __name__ == "__main__":
    main()
