"""Phase 0 of the activation-patching study: build clean/corrupted pairs.

Design (see patching/PATCHING_ROADMAP.md):
  - Within a single dimension (2D / 3D / 4D), find two MC questions that
      * use the same option set (same `type` column, so A/B mean the same thing),
      * differ by exactly ONE contiguous span (a geometric label like EF / GHI / BEHL),
      * have DIFFERENT answers (the edit flips A<->B).
  - Render both with rotation=0 and the chosen prompt format, apply the chat
    template, then tokenize with the target model's tokenizer to verify the two
    prompts are token-aligned (same length, one contiguous differing block).
    Only token-aligned pairs are usable for activation patching.

Output: a JSON file with the clean/corrupted prompts, answer letters, the edit
token span [start, end), and provenance, ready for Phase 1 (patch_run.py).

Example:
    python patching/patch_pairs.py \
        --model-name Qwen/Qwen3.5-9B \
        --dims 2,3,4 --prompt-type simple_prompt \
        --out results/patching/pairs/qwen35_9b.json
"""
import argparse
import itertools
import json
import os
import re
import string
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv

from common.prompting import apply_chat_template, make_prompt_mc, resolve_prompt_key

# A geometric reference label is a short run of uppercase letters: a line (EF),
# plane (GHI) or 3D-hyperplane (BEHL). The corrupted variant swaps this label.
_LABEL_RE = re.compile(r"^[A-Z]{2,5}$")


@dataclass
class QRow:
    dimension: int
    type_key: str       # option-set selector ("1" parallel/perp, etc.)
    answer: str         # canonical answer letter at rotation 0 (A/B/...)
    question: str       # raw question text (the dimension-specific column)
    row_index: int      # 0-based data row (excludes header)
    aug_source_id: str


@dataclass
class Pair:
    dimension: int
    type_key: str
    clean_answer: str
    corrupted_answer: str
    clean_question: str
    corrupted_question: str
    clean_edit: str             # the swapped label in clean (e.g. "EF")
    corrupted_edit: str         # the swapped label in corrupted (e.g. "BE")
    clean_prompt: str           # chat-formatted, ready to tokenize/feed
    corrupted_prompt: str
    edit_token_start: Optional[int]   # token span [start,end) that differs
    edit_token_end: Optional[int]
    n_tokens: Optional[int]           # total token length (clean == corrupted)
    token_aligned: Optional[bool]     # None if tokenizer unavailable
    clean_row_index: int
    corrupted_row_index: int
    source: str = "questions_augmented"   # or "synthetic_2d"


def _read_rows(csv_path: str, dims: List[int]) -> List[QRow]:
    rows: List[QRow] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            type_key = (row.get("type") or "1").strip() or "1"
            answer = (row.get("answer") or "").strip().upper()
            if not answer:
                continue
            for dim in dims:
                col = f"{dim}D"
                q = (row.get(col) or "").strip()
                if not q or q == "-":
                    continue
                rows.append(
                    QRow(
                        dimension=dim,
                        type_key=type_key,
                        answer=answer,
                        question=q,
                        row_index=i,
                        aug_source_id=(row.get("aug_source_id") or "").strip(),
                    )
                )
    return rows


def _single_span_edit(a: str, b: str) -> Optional[Tuple[str, str]]:
    """If a and b differ by exactly one contiguous label span, return it.

    The differing substrings on both sides must be pure uppercase labels and
    must be cleanly bounded (the surrounding common chars are not uppercase), so
    we never cut a label in half.
    """
    if a == b:
        return None
    n_a, n_b = len(a), len(b)
    # longest common prefix
    p = 0
    while p < n_a and p < n_b and a[p] == b[p]:
        p += 1
    # longest common suffix (not overlapping the prefix)
    s = 0
    while s < (n_a - p) and s < (n_b - p) and a[n_a - 1 - s] == b[n_b - 1 - s]:
        s += 1
    edit_a = a[p:n_a - s]
    edit_b = b[p:n_b - s]
    if not edit_a or not edit_b:
        return None
    if not _LABEL_RE.match(edit_a) or not _LABEL_RE.match(edit_b):
        return None
    # clean boundaries: char immediately before/after the edit must not be an
    # uppercase letter (else the label extends into the "common" region).
    if p > 0 and a[p - 1].isupper():
        return None
    if s > 0 and a[n_a - s].isupper():
        return None
    return edit_a, edit_b


def _token_align(
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    max_edit_tokens: int,
) -> Optional[Tuple[int, int, int]]:
    """Return (edit_start, edit_end, n_tokens) if the two prompts are aligned.

    Aligned = identical token length and a single contiguous differing block no
    longer than max_edit_tokens. Returns None otherwise.
    """
    tc = tokenizer(clean_prompt, add_special_tokens=False)["input_ids"]
    tk = tokenizer(corrupted_prompt, add_special_tokens=False)["input_ids"]
    if len(tc) != len(tk):
        return None
    n = len(tc)
    tp = 0
    while tp < n and tc[tp] == tk[tp]:
        tp += 1
    if tp == n:
        return None  # identical -> not a real pair
    ts = 0
    while ts < (n - tp) and tc[n - 1 - ts] == tk[n - 1 - ts]:
        ts += 1
    start, end = tp, n - ts
    if (end - start) > max_edit_tokens:
        return None
    return start, end, n


def build_pairs(
    rows: List[QRow],
    prompt_key: str,
    model_name: str,
    tokenizer,
    max_edit_tokens: int,
) -> List[Pair]:
    # group by (dimension, type) so options and answer semantics match within a pair
    groups: Dict[Tuple[int, str], List[QRow]] = {}
    for r in rows:
        groups.setdefault((r.dimension, r.type_key), []).append(r)

    pairs: List[Pair] = []
    seen_prompt_pairs = set()
    for (dim, type_key), grp in groups.items():
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                a, b = grp[i], grp[j]
                if a.answer == b.answer:
                    continue
                edit = _single_span_edit(a.question, b.question)
                if edit is None:
                    continue
                edit_a, edit_b = edit
                # canonical direction: clean = lexicographically smaller answer
                if a.answer <= b.answer:
                    clean, corr = a, b
                    clean_edit, corr_edit = edit_a, edit_b
                else:
                    clean, corr = b, a
                    clean_edit, corr_edit = edit_b, edit_a

                clean_prompt = apply_chat_template(
                    make_prompt_mc(clean.question, type_key, reasoning=prompt_key, rotation=0),
                    model_name,
                )
                corr_prompt = apply_chat_template(
                    make_prompt_mc(corr.question, type_key, reasoning=prompt_key, rotation=0),
                    model_name,
                )
                key = (clean_prompt, corr_prompt)
                if key in seen_prompt_pairs:
                    continue
                seen_prompt_pairs.add(key)

                edit_start = edit_end = n_tokens = None
                token_aligned: Optional[bool] = None
                if tokenizer is not None:
                    aligned = _token_align(tokenizer, clean_prompt, corr_prompt, max_edit_tokens)
                    token_aligned = aligned is not None
                    if aligned is not None:
                        edit_start, edit_end, n_tokens = aligned

                pairs.append(
                    Pair(
                        dimension=dim,
                        type_key=type_key,
                        clean_answer=clean.answer,
                        corrupted_answer=corr.answer,
                        clean_question=clean.question,
                        corrupted_question=corr.question,
                        clean_edit=clean_edit,
                        corrupted_edit=corr_edit,
                        clean_prompt=clean_prompt,
                        corrupted_prompt=corr_prompt,
                        edit_token_start=edit_start,
                        edit_token_end=edit_end,
                        n_tokens=n_tokens,
                        token_aligned=token_aligned,
                        clean_row_index=clean.row_index,
                        corrupted_row_index=corr.row_index,
                    )
                )
    return pairs


# (prefix, #vertices) per dimension, matching the benchmark template exactly:
#   2D "In rectangle ABEF, ..."           3D "In rectangular solid ABEF-GHIJ, ..."
#   4D "In tesseract (ABEF-GHIJ)-(KLMN-OPQR), ..."
_SYNTH_DIM = {
    2: ("rectangle ", 4),
    3: ("rectangular solid ", 8),
    4: ("tesseract ", 16),
}


def _synth_figure(dim: int, active: Tuple[str, ...], ctx: Tuple[str, ...]) -> str:
    """Build the figure label; `active` (w,x,y,z) is the first quad used by the
    two compared lines, `ctx` fills the remaining vertices (context only)."""
    w, x, y, z = active
    if dim == 2:
        return f"{w}{x}{y}{z}"
    if dim == 3:
        return f"{w}{x}{y}{z}-{ctx[0]}{ctx[1]}{ctx[2]}{ctx[3]}"
    c = ctx
    return f"({w}{x}{y}{z}-{c[0]}{c[1]}{c[2]}{c[3]})-({c[4]}{c[5]}{c[6]}{c[7]}-{c[8]}{c[9]}{c[10]}{c[11]})"


def build_synthetic_pairs(
    dim: int,
    n: int,
    prompt_key: str,
    model_name: str,
    tokenizer,
    max_edit_tokens: int,
    exclude_keys: set,
    type_key: str = "1",
    letters: str = string.ascii_uppercase,
) -> List[Pair]:
    """Generate up to `n` token-aligned minimal pairs matching the benchmark
    template for `dim`, on a cyclic quad with vertices w,x,y,z. Two compared
    lines: WX (fixed) and the second line = OPPOSITE side YZ or ADJACENT side XY.
    The clean variant is always answer A; only the geometry->answer mapping flips
    between option sets:
        type1 (Parallel/Perpendicular): clean A = WX vs YZ (parallel, opposite);
                                        corrupt B = WX vs XY (perpendicular, adjacent)
        type2 (Intersecting/Not):       clean A = WX vs XY (intersecting, adjacent);
                                        corrupt B = WX vs YZ (not intersecting, opposite)
    Only the second line label is edited. The active quad varies fastest
    (edit-label diversity). Pairs whose two labels tokenize to different lengths,
    or duplicate `exclude_keys`, are skipped.
    """
    if tokenizer is None or n <= 0 or dim not in _SYNTH_DIM:
        return []
    if type_key not in ("1", "2"):
        raise ValueError(f"synthetic generation supports type 1/2, got {type_key}")
    prefix, n_vertices = _SYNTH_DIM[dim]
    tmpl = ("In {prefix}{fig}, {wx}=5. "
            "What is the relationship between line {wx} and line {second}?")
    pairs: List[Pair] = []
    seen = set(exclude_keys)
    # active quad in the LAST 4 positions -> it varies fastest across permutations
    for perm in itertools.permutations(letters, n_vertices):
        if len(pairs) >= n:
            break
        active = perm[-4:]
        ctx = perm[:n_vertices - 4]
        w, x, y, z = active
        fig, wx = _synth_figure(dim, active, ctx), f"{w}{x}"
        opposite, adjacent = f"{y}{z}", f"{x}{y}"
        # clean is answer A in both types; the geometry assigned to A flips by type
        clean_edit, corr_edit = (opposite, adjacent) if type_key == "1" else (adjacent, opposite)
        clean_q = tmpl.format(prefix=prefix, fig=fig, wx=wx, second=clean_edit)
        corr_q = tmpl.format(prefix=prefix, fig=fig, wx=wx, second=corr_edit)
        clean_prompt = apply_chat_template(
            make_prompt_mc(clean_q, type_key, reasoning=prompt_key, rotation=0), model_name)
        corr_prompt = apply_chat_template(
            make_prompt_mc(corr_q, type_key, reasoning=prompt_key, rotation=0), model_name)
        key = (clean_prompt, corr_prompt)
        if key in seen:
            continue
        aligned = _token_align(tokenizer, clean_prompt, corr_prompt, max_edit_tokens)
        if aligned is None:
            continue
        seen.add(key)
        es, ee, nt = aligned
        pairs.append(Pair(
            dimension=dim, type_key=type_key, clean_answer="A", corrupted_answer="B",
            clean_question=clean_q, corrupted_question=corr_q,
            clean_edit=clean_edit, corrupted_edit=corr_edit,
            clean_prompt=clean_prompt, corrupted_prompt=corr_prompt,
            edit_token_start=es, edit_token_end=ee, n_tokens=nt,
            token_aligned=True, clean_row_index=-1, corrupted_row_index=-1,
            source=f"synthetic_{dim}d_t{type_key}",
        ))
    return pairs


# Number of distinct vertex letters per dimension for the type3 construction:
#   2D triangle(3) + constructed point; 3D tetra(4) + pt; 4D 4-simplex(5) + pt.
_TYPE3_DIM = {2: 4, 3: 5, 4: 6}


def _type3_questions(dim: int, vs: Tuple[str, ...]) -> Tuple[str, str, str, str]:
    """Return (clean_q, corr_q, clean_edit, corr_edit) for the collinear/coplanar
    construction. The constructed point lies in the flat spanned by the queried
    original vertices in `clean` (answer Yes/A) but not in `corr` (answer No/B);
    only the construction's argument label is edited.
    """
    if dim == 2:
        a, b, c, e = vs
        q = "Are points {a}, {c}, and {e} collinear?".format(a=a, c=c, e=e)
        base = "In triangle {a}{b}{c}, point {e} is the midpoint of ".format(a=a, b=b, c=c, e=e)
        clean_edit, corr_edit = f"{a}{c}", f"{b}{c}"
        return base + clean_edit + ". " + q, base + corr_edit + ". " + q, clean_edit, corr_edit
    if dim == 3:
        a, b, c, d, e = vs
        q = "Are points {a}, {c}, {d} and {e} coplanar?".format(a=a, c=c, d=d, e=e)
        base = "In triangular pyramid {a}{b}{c}{d}, point {e} is the innercenter of ".format(
            a=a, b=b, c=c, d=d, e=e)
        clean_edit, corr_edit = f"{a}{c}{d}", f"{b}{c}{d}"
        return base + clean_edit + ". " + q, base + corr_edit + ". " + q, clean_edit, corr_edit
    a, b, c, d, e, f = vs
    q = "Are points {a}, {c}, {d}, {e} and {f} cohyperplanar?".format(a=a, c=c, d=d, e=e, f=f)
    base = "In 4-simplex {a}{b}{c}{d}{e}, point {f} is the innercenter of ".format(
        a=a, b=b, c=c, d=d, e=e, f=f)
    clean_edit, corr_edit = f"{a}{c}{d}{e}", f"{b}{c}{d}{e}"
    return base + clean_edit + ". " + q, base + corr_edit + ". " + q, clean_edit, corr_edit


def build_synthetic_type3_pairs(
    dim: int,
    n: int,
    prompt_key: str,
    model_name: str,
    tokenizer,
    max_edit_tokens: int,
    exclude_keys: set,
    letters: str = string.ascii_uppercase,
) -> List[Pair]:
    """Generate up to `n` token-aligned type3 (Yes/No) minimal pairs.

    clean (Yes/A): the constructed point (midpoint in 2D, innercenter in 3D/4D)
    lies in the flat spanned by the queried vertices -> collinear/coplanar/
    cohyperplanar. corrupt (No/B): swap one vertex of the construction's argument
    so the point leaves that flat. Only that argument label is edited.
    """
    if tokenizer is None or n <= 0 or dim not in _TYPE3_DIM:
        return []
    n_letters = _TYPE3_DIM[dim]
    pairs: List[Pair] = []
    seen = set(exclude_keys)
    for vs in itertools.permutations(letters, n_letters):
        if len(pairs) >= n:
            break
        clean_q, corr_q, clean_edit, corr_edit = _type3_questions(dim, vs)
        clean_prompt = apply_chat_template(
            make_prompt_mc(clean_q, "3", reasoning=prompt_key, rotation=0), model_name)
        corr_prompt = apply_chat_template(
            make_prompt_mc(corr_q, "3", reasoning=prompt_key, rotation=0), model_name)
        key = (clean_prompt, corr_prompt)
        if key in seen:
            continue
        aligned = _token_align(tokenizer, clean_prompt, corr_prompt, max_edit_tokens)
        if aligned is None:
            continue
        seen.add(key)
        es, ee, nt = aligned
        pairs.append(Pair(
            dimension=dim, type_key="3", clean_answer="A", corrupted_answer="B",
            clean_question=clean_q, corrupted_question=corr_q,
            clean_edit=clean_edit, corrupted_edit=corr_edit,
            clean_prompt=clean_prompt, corrupted_prompt=corr_prompt,
            edit_token_start=es, edit_token_end=ee, n_tokens=nt,
            token_aligned=True, clean_row_index=-1, corrupted_row_index=-1,
            source=f"synthetic_{dim}d_t3",
        ))
    return pairs


def _load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:  # offline / unsupported -> emit char-level pairs only
        print(
            f"WARNING: could not load tokenizer for {model_name}: {e}\n"
            "         Emitting char-level pairs WITHOUT token-alignment "
            "verification (token_aligned=null). Re-run with the tokenizer "
            "available before Phase 1.",
            file=sys.stderr,
        )
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-9B", help="HF model id (for tokenizer + chat template)")
    ap.add_argument("--questions-csv", default="data/questions_augmented.csv")
    ap.add_argument("--dims", default="2,3,4", help="comma-separated dimensions")
    ap.add_argument("--prompt-type", default="simple_prompt",
                    help="simple_prompt / without_reasoning / with_reasoning")
    ap.add_argument("--max-edit-tokens", type=int, default=6,
                    help="reject pairs whose differing token span exceeds this")
    ap.add_argument("--balance", type=int, default=0,
                    help="top up EACH (dimension x type) cell with token-aligned synthetic "
                         "pairs (benchmark template) to reach this many aligned pairs, "
                         "compensating for sparse coincidental matches in the data")
    ap.add_argument("--balance-types", default="1,2,3",
                    help="option-set types to balance with synthetic pairs: 1=par/perp, "
                         "2=intersect/not, 3=collinear/coplanar (Yes/No).")
    ap.add_argument("--out", default="results/patching/pairs/pairs.json")
    ap.add_argument("--aligned-only", action="store_true",
                    help="write only token-aligned pairs (recommended for Phase 1)")
    args = ap.parse_args()

    dims = [int(d.strip()) for d in args.dims.split(",") if d.strip()]
    prompt_key = resolve_prompt_key(args.prompt_type)

    rows = _read_rows(args.questions_csv, dims)
    print(f"Loaded {len(rows)} dimension-rows from {args.questions_csv} (dims={dims})")

    tokenizer = _load_tokenizer(args.model_name)
    pairs = build_pairs(rows, prompt_key, args.model_name, tokenizer, args.max_edit_tokens)

    if args.balance > 0:
        existing_keys = {(p.clean_prompt, p.corrupted_prompt) for p in pairs}
        balance_types = [t.strip() for t in args.balance_types.split(",") if t.strip()]
        for type_key in balance_types:
            for dim in (2, 3, 4):
                have = sum(1 for p in pairs
                           if p.dimension == dim and p.type_key == type_key and p.token_aligned)
                need = args.balance - have
                if need <= 0:
                    continue
                if type_key == "3":
                    syn = build_synthetic_type3_pairs(
                        dim, need, prompt_key, args.model_name, tokenizer,
                        args.max_edit_tokens, existing_keys)
                else:
                    syn = build_synthetic_pairs(
                        dim, need, prompt_key, args.model_name, tokenizer,
                        args.max_edit_tokens, existing_keys, type_key=type_key)
                existing_keys.update((p.clean_prompt, p.corrupted_prompt) for p in syn)
                print(f"dim{dim} type{type_key}: had {have} aligned, "
                      f"generated {len(syn)} synthetic (target {args.balance})")
                pairs.extend(syn)

    if args.aligned_only and tokenizer is not None:
        pairs = [p for p in pairs if p.token_aligned]

    # ---- summary ----
    by_dim: Dict[int, Dict[str, int]] = {}
    for p in pairs:
        d = by_dim.setdefault(p.dimension, {"total": 0, "aligned": 0})
        d["total"] += 1
        if p.token_aligned:
            d["aligned"] += 1

    print("\n=== pair counts by dimension ===")
    print(f"{'dim':>4} {'pairs':>8} {'aligned':>8}")
    for dim in sorted(by_dim):
        c = by_dim[dim]
        print(f"{dim:>4} {c['total']:>8} {c['aligned']:>8}")
    print(f"{'ALL':>4} {len(pairs):>8} {sum(c['aligned'] for c in by_dim.values()):>8}")
    src = {}
    for p in pairs:
        src[p.source] = src.get(p.source, 0) + 1
    print("by source:", src)
    print("\n=== aligned pairs by (dimension x type) ===")
    cell = {}
    for p in pairs:
        if p.token_aligned:
            cell[(p.dimension, p.type_key)] = cell.get((p.dimension, p.type_key), 0) + 1
    types = sorted({p.type_key for p in pairs})
    print(f"{'dim':>4} " + " ".join(f"type{t:>2}" for t in types))
    for dim in sorted({p.dimension for p in pairs}):
        print(f"{dim:>4} " + " ".join(f"{cell.get((dim, t), 0):>6}" for t in types))

    aligned_examples = [p for p in pairs if p.token_aligned][:3]
    if aligned_examples:
        print("\n=== example aligned pairs ===")
        for p in aligned_examples:
            print(f"  [{p.dimension}D type{p.type_key}] {p.clean_edit}({p.clean_answer}) -> "
                  f"{p.corrupted_edit}({p.corrupted_answer})  "
                  f"edit_tokens=[{p.edit_token_start},{p.edit_token_end}) n={p.n_tokens}")
            print(f"      clean: {p.clean_question}")
            print(f"      corr : {p.corrupted_question}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    payload = {
        "metadata": {
            "model_name": args.model_name,
            "questions_csv": args.questions_csv,
            "dims": dims,
            "prompt_type": prompt_key,
            "max_edit_tokens": args.max_edit_tokens,
            "tokenizer_available": tokenizer is not None,
            "aligned_only": bool(args.aligned_only and tokenizer is not None),
            "num_pairs": len(pairs),
            "num_aligned": sum(c["aligned"] for c in by_dim.values()),
        },
        "pairs": [asdict(p) for p in pairs],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(pairs)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
