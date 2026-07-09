"""Batching helpers for the activation-patching pipeline (REEXPERIMENT: speedup).

The Phase-1/2/3 scripts process one pair per forward pass (`process_pair` loops),
so a layer/head/position sweep costs ~N_pairs x N_locations forwards. This module
batches pairs of *identical token length* together: with equal length there is NO
padding, and standard causal attention never attends across the batch dimension,
so a batched forward is **numerically identical** to running each example alone —
no attention-mask / RoPE-position hazards. Speedup ≈ mean group size (empirically
~8-11x on the templated geometry prompts; see `group_by_length`).

Primitives:
  - answer_token_ids(tok, letters)            -> [B] answer token ids
  - batched_logit_diff_last(logits, pos, neg) -> [B] logit(pos_b) - logit(neg_b) at last token
  - make_batched_resid_patch_hook(donor, per_ex_positions)         (resid_post: [B,L,d])
  - make_batched_head_patch_hook(donor, per_ex_positions, head)    (hook_z: [B,L,H,d])
  - group_by_length(pairs, tokenizer, max_batch) -> list of batches (each = list of pairs)

Run `python patching/patch_batch.py` to execute the CPU self-test (no model needed):
it checks the batched gather / patch ops equal their per-example counterparts.
"""
from typing import Dict, List, Optional

import torch


def answer_token_ids(tokenizer, letters: List[str]) -> List[int]:
    from patch_run import _answer_token_id
    return [_answer_token_id(tokenizer, x) for x in letters]


@torch.no_grad()
def batched_logit_diff_last(logits: torch.Tensor, id_pos: List[int], id_neg: List[int]) -> List[float]:
    """Per-example (logit[pos_b] - logit[neg_b]) at the final token. logits: [B, L, V]."""
    last = logits[:, -1, :]                                    # [B, V]
    ip = torch.tensor(id_pos, device=last.device).view(-1, 1)
    ineg = torch.tensor(id_neg, device=last.device).view(-1, 1)
    pos = last.gather(1, ip).squeeze(1)
    neg = last.gather(1, ineg).squeeze(1)
    return (pos - neg).float().tolist()


def make_batched_resid_patch_hook(donor_act: torch.Tensor, per_ex_positions: List[List[int]]):
    """Patch resid_post ([B,L,d]) at per-example positions with the donor batch."""
    def hook(act, hook):  # noqa: ARG001
        out = act.clone()
        d = donor_act.to(act.dtype).to(act.device)
        for b, positions in enumerate(per_ex_positions):
            if positions:
                idx = torch.tensor(positions, dtype=torch.long, device=act.device)
                out[b, idx, :] = d[b, idx, :]
        return out
    return hook


def make_batched_head_patch_hook(donor_act: torch.Tensor, per_ex_positions: List[List[int]], head: int):
    """Patch one attention head's hook_z ([B,L,H,d]) at per-example positions."""
    def hook(act, hook):  # noqa: ARG001
        out = act.clone()
        d = donor_act.to(act.dtype).to(act.device)
        for b, positions in enumerate(per_ex_positions):
            if positions:
                idx = torch.tensor(positions, dtype=torch.long, device=act.device)
                out[b, idx, head, :] = d[b, idx, head, :]
        return out
    return hook


def make_batched_heads_patch_hook(donor_act: torch.Tensor, per_ex_positions: List[List[int]], heads: List[int]):
    """Patch several heads' hook_z ([B,L,H,d]) at per-example positions (joint patch)."""
    def hook(act, hook):  # noqa: ARG001
        out = act.clone()
        d = donor_act.to(act.dtype).to(act.device)
        for b, positions in enumerate(per_ex_positions):
            if not positions:
                continue
            idx = torch.tensor(positions, dtype=torch.long, device=act.device)
            for h in heads:
                out[b, idx, h, :] = d[b, idx, h, :]
        return out
    return hook


def group_by_length(pairs: List[Dict], tokenizer, max_batch: int = 16, device: str = "cpu"):
    """Yield batches of pairs sharing an identical (clean==corrupted) token length.

    Each batch is a dict with stacked tensors and per-example metadata:
      clean [B,L], corr [B,L], id_clean [B], id_corr [B], pairs [B], length L.
    Pairs whose clean/corrupted lengths differ are dropped (as in the per-pair path).
    Groups larger than max_batch are split into chunks so memory stays bounded.
    """
    from patch_run import _answer_token_id

    by_len: Dict[int, List[Dict]] = {}
    for p in pairs:
        c = tokenizer(p["clean_prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"]
        k = tokenizer(p["corrupted_prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"]
        if c.shape[1] != k.shape[1]:
            continue
        rec = {"pair": p, "clean": c[0], "corr": k[0],
               "id_clean": _answer_token_id(tokenizer, p["clean_answer"]),
               "id_corr": _answer_token_id(tokenizer, p["corrupted_answer"])}
        by_len.setdefault(c.shape[1], []).append(rec)

    batches = []
    for L, recs in by_len.items():
        for i in range(0, len(recs), max_batch):
            chunk = recs[i:i + max_batch]
            batches.append({
                "length": L,
                "pairs": [r["pair"] for r in chunk],
                "clean": torch.stack([r["clean"] for r in chunk]).to(device),
                "corr": torch.stack([r["corr"] for r in chunk]).to(device),
                "id_clean": [r["id_clean"] for r in chunk],
                "id_corr": [r["id_corr"] for r in chunk],
            })
    return batches


def _self_test() -> None:
    torch.manual_seed(0)
    B, L, V, H, d = 4, 7, 20, 3, 5
    # 1) batched logit-diff == per-example
    logits = torch.randn(B, L, V)
    ip = [1, 2, 3, 4]; ineg = [5, 6, 7, 8]
    got = batched_logit_diff_last(logits, ip, ineg)
    exp = [float(logits[b, -1, ip[b]] - logits[b, -1, ineg[b]]) for b in range(B)]
    assert all(abs(g - e) < 1e-5 for g, e in zip(got, exp)), (got, exp)

    # 2) batched resid patch == per-example patch
    act = torch.randn(B, L, d); donor = torch.randn(B, L, d)
    positions = [[L - 1], [0, 1], [], [2, 3, 4]]
    out = make_batched_resid_patch_hook(donor, positions)(act.clone(), None)
    exp_act = act.clone()
    for b, ps in enumerate(positions):
        for p in ps:
            exp_act[b, p, :] = donor[b, p, :]
    assert torch.allclose(out, exp_act), "resid patch mismatch"

    # 3) batched single-head patch == per-example
    actz = torch.randn(B, L, H, d); donorz = torch.randn(B, L, H, d)
    out2 = make_batched_head_patch_hook(donorz, positions, head=1)(actz.clone(), None)
    exp2 = actz.clone()
    for b, ps in enumerate(positions):
        for p in ps:
            exp2[b, p, 1, :] = donorz[b, p, 1, :]
    assert torch.allclose(out2, exp2), "head patch mismatch"

    # 4) multi-head joint patch
    out3 = make_batched_heads_patch_hook(donorz, positions, heads=[0, 2])(actz.clone(), None)
    exp3 = actz.clone()
    for b, ps in enumerate(positions):
        for p in ps:
            for h in (0, 2):
                exp3[b, p, h, :] = donorz[b, p, h, :]
    assert torch.allclose(out3, exp3), "multi-head patch mismatch"

    print("patch_batch self-test: OK (batched gather/patch == per-example)")


if __name__ == "__main__":
    _self_test()
