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

from common.prompting import (
    apply_chat_template,
    make_prompt_mc,
    resolve_prompt_key,
    remap_answer_for_rotation,
    remap_answer_for_perm,
    _choice_count_for_type,
)

# A geometric reference label is a short run of uppercase letters: a single
# vertex (E), a line (EF), plane (GHI) or 3D-hyperplane (BEHL). The corrupted
# variant swaps this label. Single-letter labels (1) enable query-vertex swaps
# (e.g. "...A, C, and E collinear" -> "...A, C, and B collinear"), which is how
# CC (type3) minimal pairs are extracted from the dataset; the boundary checks
# in _single_span_edit ensure we only match a cleanly-delimited label.
_LABEL_RE = re.compile(r"^[A-Z]{1,5}$")


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
    family: str = "other"                 # construction family (for balancing/stratified analysis)
    rotation: int = 0                     # option-list left-rotation applied (0 = canonical, clean=A)


# Construction families, matched against the clean question text. Ordered: the
# first matching pattern wins. Used to balance/stratify the box-dominated pool
# (a tesseract has many edge/face queries, so box vastly outnumbers the rest).
_FAMILY_PATTERNS = [
    ("box", r"tesseract|rectangular solid|rectangle"),
    ("prism", r"triangular prism|tetrahedral prism"),
    ("circle/sphere", r"circle|sphere"),
    ("regular-tetra", r"regular tetrahedron"),
    ("simplex/tetra", r"5-cell|simplex|tetrahedron|triangular pyramid|pyramid"),
    ("equilateral/isosceles", r"equilateral|isosceles|AB ?= ?AC"),
    ("perp-bisector", r"perpendicular bisector|perpendicular to"),
    ("altitude", r"altitude"),
    ("midpoint", r"midpoint"),
    ("centroid/center", r"centroid|innercenter|circumcenter|orthocenter|reflection"),
    ("intersect", r"intersect"),
    ("same-space", r"same \d|same plane|same space"),
    ("tangent", r"tangent"),
    ("diameter/chord", r"diameter|chord"),
]


def _classify_family(question: str) -> str:
    # transitivity family: label-agnostic (augmentation relabels the line names),
    # detected by its two "parallel to line" clauses plus a "perpendicular to line"
    if question.count("is parallel to line") >= 2 and "is perpendicular to line" in question:
        return "transitivity"
    for name, pat in _FAMILY_PATTERNS:
        if re.search(pat, question, re.I):
            return name
    return "other"


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
    rotations: Optional[List[int]] = None,
    swap_ab: bool = False,
) -> List[Pair]:
    """Build clean/corrupted minimal pairs.

    `rotations` = option-list left-rotations to emit per pair (default [0], the
    canonical clean=A layout). Passing e.g. [0, 1] emits each pair cyclically
    rotated so the clean answer lands on a non-A letter.

    `swap_ab` (REEXPERIMENT_TODO P1-1, preferred over rotation): emit each pair
    TWICE — canonical (clean=A, corrupted=B) and with options A/B TRANSPOSED
    (clean=B, corrupted=A), C/D untouched. This directly counterbalances the
    answer LETTER against the geometric relation with a clean 2-subset (correct=A
    vs correct=B) design, without the cyclic sweep's A->C/D remapping. The A/B
    transposition shifts only the option block identically in clean and corrupted
    prompts, so question-edit token alignment is preserved; the stored
    clean_answer/corrupted_answer letters flip via remap_answer_for_perm.
    """
    if rotations is None:
        rotations = [0]
    # Each variant is (kind, param): ("rot", k) cyclic-rotates by k; ("swap", None)
    # transposes options A/B. swap_ab overrides rotations with [canonical, A/B-swap].
    if swap_ab:
        variants: List[Tuple[str, Optional[int]]] = [("rot", 0), ("swap", None)]
    else:
        variants = [("rot", r) for r in rotations]
    # group by (dimension, type) so options and answer semantics match within a pair
    groups: Dict[Tuple[int, str], List[QRow]] = {}
    for r in rows:
        groups.setdefault((r.dimension, r.type_key), []).append(r)

    pairs: List[Pair] = []
    seen_prompt_pairs = set()
    for (dim, type_key), grp in groups.items():
        nch = _choice_count_for_type(type_key)
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

                for vkind, vparam in variants:
                    # render kwargs + answer-letter remap depend on the variant
                    if vkind == "swap":
                        if nch < 2:
                            continue  # need >=2 options to transpose A/B
                        perm = [1, 0] + list(range(2, nch))
                        mc_kw = {"perm": perm}
                        remap = lambda ans: remap_answer_for_perm(ans, perm) if nch else ans
                        rot_field = 0
                    else:  # ("rot", k)
                        rot = vparam or 0
                        if rot != 0 and nch <= 1:  # rotation that moves no option
                            continue
                        mc_kw = {"rotation": rot}
                        remap = lambda ans, _r=rot: remap_answer_for_rotation(ans, _r, nch) if nch else ans
                        rot_field = rot
                    clean_prompt = apply_chat_template(
                        make_prompt_mc(clean.question, type_key, reasoning=prompt_key, **mc_kw),
                        model_name,
                    )
                    corr_prompt = apply_chat_template(
                        make_prompt_mc(corr.question, type_key, reasoning=prompt_key, **mc_kw),
                        model_name,
                    )
                    key = (clean_prompt, corr_prompt)
                    if key in seen_prompt_pairs:
                        continue
                    seen_prompt_pairs.add(key)

                    # answer letters after the reorder (canonical letters are A/B at rot 0)
                    clean_letter = remap(clean.answer)
                    corr_letter = remap(corr.answer)

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
                            clean_answer=clean_letter,
                            corrupted_answer=corr_letter,
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
                            family=_classify_family(clean.question),
                            rotation=rot_field,
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
            family=_classify_family(clean_q),
        ))
    return pairs


# Number of distinct vertex letters per dimension for the type3 construction:
#   2D triangle(3) + constructed point; 3D tetra(4) + pt; 4D 4-simplex(5) + pt.
_TYPE3_DIM = {2: 4, 3: 5, 4: 6}


# Alternative CC constructions per dimension. Every listed center lies in the
# affine hull (flat) of the vertices in its construction argument, so the ground
# truth (collinear/coplanar/cohyperplanar) is preserved — only the phrasing of
# the construction changes. This lets us test whether the 4D-CC collapse is
# specific to a single construction (innercenter) or general.
#   - 2D (segment): midpoint, reflection — both put the point on line AC.
#   - 3D (triangle face) / 4D (tetra cell): centroid / circumcenter / incenter
#     (=innercenter) are always in the face's flat; orthocenter is in-plane for a
#     TRIANGLE (3D) but not guaranteed for a tetrahedron (4D) → 3D-only.
_TYPE3_CENTERS = {
    2: ["midpoint", "reflection"],
    3: ["innercenter", "centroid", "circumcenter", "orthocenter"],
    4: ["innercenter", "centroid", "circumcenter"],
}


def _type3_questions(dim: int, vs: Tuple[str, ...], center: str) -> Tuple[str, str, str, str]:
    """Return (clean_q, corr_q, clean_edit, corr_edit) for a collinear/coplanar
    construction. The constructed point lies in the flat spanned by the queried
    original vertices in `clean` (answer Yes/A) but not in `corr` (answer No/B);
    only the construction's argument (first vertex) is edited. `center` selects
    the construction family (see _TYPE3_CENTERS)."""
    if dim == 2:
        a, b, c, e = vs
        q = "Are points {a}, {c}, and {e} collinear?".format(a=a, c=c, e=e)
        if center == "reflection":
            base = "In triangle {a}{b}{c}, point {e} is the reflection of ".format(a=a, b=b, c=c, e=e)
            clean_edit, corr_edit = f"{a} across {c}", f"{b} across {c}"
        else:  # midpoint
            base = "In triangle {a}{b}{c}, point {e} is the midpoint of ".format(a=a, b=b, c=c, e=e)
            clean_edit, corr_edit = f"{a}{c}", f"{b}{c}"
        return base + clean_edit + ". " + q, base + corr_edit + ". " + q, clean_edit, corr_edit
    if dim == 3:
        a, b, c, d, e = vs
        q = "Are points {a}, {c}, {d} and {e} coplanar?".format(a=a, c=c, d=d, e=e)
        base = "In triangular pyramid {a}{b}{c}{d}, point {e} is the {center} of ".format(
            a=a, b=b, c=c, d=d, e=e, center=center)
        clean_edit, corr_edit = f"{a}{c}{d}", f"{b}{c}{d}"
        return base + clean_edit + ". " + q, base + corr_edit + ". " + q, clean_edit, corr_edit
    a, b, c, d, e, f = vs
    q = "Are points {a}, {c}, {d}, {e} and {f} cohyperplanar?".format(a=a, c=c, d=d, e=e, f=f)
    base = "In 4-simplex {a}{b}{c}{d}{e}, point {f} is the {center} of ".format(
        a=a, b=b, c=c, d=d, e=e, f=f, center=center)
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
    constructions: Optional[List[str]] = None,
) -> List[Pair]:
    """Generate up to `n` token-aligned type3 (Yes/No) minimal pairs.

    clean (Yes/A): the constructed point lies in the flat spanned by the queried
    vertices -> collinear/coplanar/cohyperplanar. corrupt (No/B): swap one vertex
    of the construction's argument so the point leaves that flat.

    `constructions` selects the CC construction families (see _TYPE3_CENTERS);
    default = the first (original: midpoint / innercenter). The quota `n` is split
    round-robin across the requested constructions so the pool is balanced, and
    each pair's `source` records its construction (synthetic_{d}d_t3_{center}).
    """
    if tokenizer is None or n <= 0 or dim not in _TYPE3_DIM:
        return []
    centers = constructions or [_TYPE3_CENTERS[dim][0]]
    centers = [c for c in centers if c in _TYPE3_CENTERS[dim]]
    if not centers:
        return []
    n_letters = _TYPE3_DIM[dim]
    per = {c: 0 for c in centers}
    cap = -(-n // len(centers))  # ceil: per-construction quota for balance
    pairs: List[Pair] = []
    seen = set(exclude_keys)
    for vs in itertools.permutations(letters, n_letters):
        if len(pairs) >= n:
            break
        for center in centers:
            if len(pairs) >= n or per[center] >= cap:
                continue
            clean_q, corr_q, clean_edit, corr_edit = _type3_questions(dim, vs, center)
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
            per[center] += 1
            pairs.append(Pair(
                dimension=dim, type_key="3", clean_answer="A", corrupted_answer="B",
                clean_question=clean_q, corrupted_question=corr_q,
                clean_edit=clean_edit, corrupted_edit=corr_edit,
                clean_prompt=clean_prompt, corrupted_prompt=corr_prompt,
                edit_token_start=es, edit_token_end=ee, n_tokens=nt,
                token_aligned=True, clean_row_index=-1, corrupted_row_index=-1,
                source=f"synthetic_{dim}d_t3_{center}",
                family=_classify_family(clean_q),
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
    ap.add_argument("--type3-constructions", default="",
                    help="comma-separated CC construction families to diversify type3 over "
                         "(2D: midpoint,reflection; 3D: innercenter,centroid,circumcenter,orthocenter; "
                         "4D: innercenter,centroid,circumcenter). Empty = original single "
                         "(midpoint/innercenter). The per-cell quota is split across them.")
    ap.add_argument("--out", default="results/patching/pairs/pairs.json")
    ap.add_argument("--aligned-only", action="store_true",
                    help="write only token-aligned pairs (recommended for Phase 1)")
    ap.add_argument("--swap-ab", action="store_true",
                    help="P1-1 (preferred): emit each pair canonical (clean=A,corr=B) AND with "
                         "options A/B transposed (clean=B,corr=A). Clean 2-subset counterbalance of "
                         "the answer LETTER vs the relation; C/D untouched (no cyclic A->C/D remap).")
    ap.add_argument("--counterbalance", action="store_true",
                    help="[legacy] emit each pair at rotation 0 AND rotation 1 (cyclic; clean A->C/D). "
                         "Prefer --swap-ab for a clean correct=A / correct=B design.")
    ap.add_argument("--rotations", default="",
                    help="explicit comma-separated rotations to emit per pair (overrides "
                         "--counterbalance), e.g. '0,1,2,3' for a full cyclic sweep")
    args = ap.parse_args()

    dims = [int(d.strip()) for d in args.dims.split(",") if d.strip()]
    prompt_key = resolve_prompt_key(args.prompt_type)
    if args.rotations.strip():
        rotations = [int(x) for x in args.rotations.split(",") if x.strip() != ""]
    elif args.counterbalance:
        rotations = [0, 1]
    else:
        rotations = [0]

    rows = _read_rows(args.questions_csv, dims)
    print(f"Loaded {len(rows)} dimension-rows from {args.questions_csv} (dims={dims})")
    if args.swap_ab:
        print("Variants per pair: canonical (clean=A) + A/B swap (clean=B)  [--swap-ab]")
    else:
        print(f"Rotations per pair: {rotations}"
              + ("  (counterbalanced A/B)" if rotations != [0] else ""))

    tokenizer = _load_tokenizer(args.model_name)
    pairs = build_pairs(rows, prompt_key, args.model_name, tokenizer, args.max_edit_tokens,
                        rotations, swap_ab=args.swap_ab)

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
                    t3c = [c.strip() for c in args.type3_constructions.split(",") if c.strip()]
                    syn = build_synthetic_type3_pairs(
                        dim, need, prompt_key, args.model_name, tokenizer,
                        args.max_edit_tokens, existing_keys, constructions=t3c or None)
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
