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

    print(f"\n=== {args.model_name} — true-Yes CC, answer-token logits ===")
    print(f"{'dim':>3} {'n':>3} {'A=Yes':>7} {'B=No':>7} {'C=Cannot':>8} {'cleanLD(A-B)':>12} {'argmax':>7}")
    for dim in (2, 3, 4):
        sub = [p for p in cc if p["dimension"] == dim]
        if not sub:
            continue
        sA = sB = sC = sLD = 0.0
        for p in sub:
            toks = tok(p["clean_prompt"], return_tensors="pt",
                       add_special_tokens=False)["input_ids"].to(model.cfg.device)
            with torch.no_grad():
                logits = model(toks)
            last = logits[0, -1, :].float()
            a, b, c = float(last[idA]), float(last[idB]), float(last[idC])
            sA += a; sB += b; sC += c; sLD += (a - b)
        n = len(sub)
        mA, mB, mC = sA / n, sB / n, sC / n
        am = "A ✓" if mA == max(mA, mB, mC) else ("B ✗" if mB == max(mA, mB, mC) else "C")
        print(f"{dim:>3} {n:>3} {mA:>7.2f} {mB:>7.2f} {mC:>8.2f} {sLD/n:>+12.2f} {am:>7}")


if __name__ == "__main__":
    main()
