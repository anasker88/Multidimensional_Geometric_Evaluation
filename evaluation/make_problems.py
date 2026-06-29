import argparse
import csv
import io
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from openai import AzureOpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


prompt="""
**Role & Goal**

Generate geometry problems as **dimensionally aligned triplets** (2D, 3D, 4D). Each row asks the same relationship in each dimension. Output must be CSV with header:

    2D,3D,4D,answer,type

Row format:
* **2D**: one 2D statement
* **3D**: corresponding 3D analogue
* **4D**: corresponding 4D analogue
* **answer**: one letter in {A, B, C, D}
* **type**: one integer in {1, 2, 3}

If a dimension is intentionally unavailable, use a single hyphen - (minimize this).

***

### Dimensional Lifting (Keep the Structure Aligned)

For each row, 2D/3D/4D must be isomorphic: same relationship lifted by dimension.

*   **Subspaces / Incidence**
    *   2D **line** → 3D **plane** → 4D **3D hyperplane**
    *   “collinear” → “coplanar” → “cohyperplanar”
    *   “intersect at a point” (2D) → “intersect in a line” (3D) → “intersect in a plane” (4D)

*   **Polytopes & canonical figures**
    *   **Triangle** → **Tetrahedron (triangular pyramid)** → **5-cell (4-simplex)**
    *   **Rectangle / rectangular** → **rectangular solid / cuboid** → **tesseract / 4D orthotope**
    *   **Circle** → **sphere** → **3-sphere (hypersphere)**
    *   Chord / tangent / diameter / altitude / midpoint / centroid / incenter generalize naturally
    *   altitude: line ⟂ side → segment ⟂ base plane → segment ⟂ base 3D hyperplane
    *   tangent: tangent line ↔ tangent plane ↔ tangent 3D hyperplane

*   **Labeling conventions (use these to make relationships determinate, not ambiguous):**
    *   **Rectangle ABCD**: AB ∥ CD, BC ∥ AD; diagonals AC and BD bisect each other
    *   **Rectangular solid ABCD-EFGH**: bottom ABCD, top EFGH, vertical AE/BF/CG/DH; opposite faces parallel; corresponding edges (e.g., AB and EF) parallel
    *   **Tesseract (ABCD-EFGH)-(IJKL-MNOP)**: two parallel 3D cubes with corresponding vertices connected; corresponding edges/faces/hyperfaces are parallel in 4D
    *   **Regular tetrahedron ABCD**; **5-cell ABCDE**

Use uppercase point labels (A, B, C, ...) and keep labels aligned across 2D/3D/4D within each row.

***

### Problem Families and Answer Key

Each row belongs to exactly one family (`type`), and `answer` must follow the family key.

**Family 1 (Parallel Perpendicular Classification): Relationship classification with this answer mapping**

*   **A: Parallel**
*   **B: Perpendicular**
*   **C: Intersecting but not perpendicular**
*   **D: Cannot be inferred**
    (Examples: skew lines in 3D; a line and a plane with no determinate relation from the given constraints.)

**Family 2 (Intersection Classification): Same styles as Family 1, but the choices are different**

*   **A: Intersecting but not perpendicular**
*   **B: Not intersecting**
*   **C: Cannot be inferred**

**Family 3 (Co-subspace Classification): Incidence / membership (Yes-No style)**

*   **A: Yes**
*   **B: No**
*   **C: Cannot be inferred**

> **Important:** The 2D/3D/4D statements in a row must imply the same answer letter under the selected family key.

***

### Style, Wording, and Diversity Constraints

*   Keep each statement concise, unambiguous, and single-target (one relationship per row).
*   Preserve geometric precision, but vary wording across rows. Avoid repeated openings.
*   Use a mix of equivalent question styles, for example:
    *   **“What is the relationship between … ?”**
    *   **“How are … related?”**
    *   **“Classify the relationship between … .”**
    *   **“Do points … lie on the same line/plane/3D hyperplane?”**
    *   **“Determine whether … intersect.”**
*   Use varied connective language across rows (e.g., “given that,” “suppose,” “in this figure,” “under the standard labeling”).
*   Keep object names and dimensional terms exact when they carry mathematical meaning: **line**, **plane**, **3D hyperplane**, **triangle**, **tetrahedron**, **5-cell (4-simplex)**, **circle**, **sphere**, **3-sphere**, **rectangle**, **rectangular solid**, **tesseract**.
*   Do not change the answer schema, family mappings, or CSV format to achieve diversity.
*   When using canonical solids (e.g., rectangular solid ABCD-EFGH), rely on the standard labeling above so relations remain determinate.

### Diversity Targets (Important)

*   Across the full output, diversify both **vocabulary** and **syntax** while keeping the same mathematical intent.
*   Avoid near-duplicate templates; if two rows share a relation archetype, vary figure context, symbol choices, and sentence structure.
*   Prefer fresh formulations over trivial paraphrases.
*   Maintain isomorphic lifting (2D/3D/4D) even when wording differs.

***

### CSV Output Rules

*   First line (once):
    2D,3D,4D,answer,type
*   Then one row per triplet; each statement in **double quotes**.
*   No explanations, notes, or extra columns: **CSV only**.

***

### Self-Checks (apply to every row)

1.  Dimensional consistency: 3D/4D are valid lifts of 2D.
2.  Determinacy: relation is fixed by the described setup unless intentionally "cannot be inferred."
3.  Answer agreement: same answer letter across 2D/3D/4D under the selected type.
4.  Terminology: use "3D hyperplane" (4D codimension-1) and "3-sphere".
5.  No contradictions or degenerate setups unless intentional.

***

### Mini Examples (follow exactly the format and mappings above)

    2D,3D,4D,answer,type
    "In rectangle ABCD, what is the relationship between line AB and line CD?","In rectangular solid ABCD-EFGH, what is the relationship between line AB and line CD?","In tesseract (ABCD-EFGH)-(IJKL-MNOP), what is the relationship between line AB and line CD?",A,1
    "In triangle ABC, BE is the altitude to AC. What is the relationship between line BE and line AC?","In triangular pyramid ABCD, BE is the altitude to plane ACD. What is the relationship between line BE and plane ACD?","In 5-cell ABCDE, BF is the altitude to 3D hyperplane ACDE. What is the relationship between line BF and 3D hyperplane ACDE?",A,2
    "Line AB is tangent to circle O at point C. What is the relationship between line AB and line OC?","Plane ABC is tangent to sphere O at point D. What is the relationship between plane ABC and line OD?","3D hyperplane ABCD is tangent to 3-sphere O at point E. What is the relationship between 3D hyperplane ABCD and line OE?",B,2
    "Points A, B, and C form a triangle. Are points A, B, and C collinear?","Points A, B, C, and D form a triangular pyramid. Are points A, B, C, and D coplanar?","Points A, B, C, D, and E form a 4-simplex. Are points A, B, C, D, and E cohyperplanar?",B,3

*(In the first example, opposite edges in a rectangle/rectangular solid/tesseract are parallel → A for type 1.
In the second example, “altitude” is perpendicular to the base (line/plane/3D hyperplane). Family 2 maps perpendicular to **A**.)*

***

**Produce only the CSV per the rules above.**
"""


EXPECTED_HEADER = ["2D", "3D", "4D", "answer", "type"]


def _build_client() -> AzureOpenAI:
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "")

    missing: List[str] = []
    if not api_key:
        missing.append("AZURE_OPENAI_API_KEY")
    if not endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not api_version:
        missing.append("AZURE_OPENAI_API_VERSION")

    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required environment variables: {joined}")

    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    # Handle model outputs wrapped in ```csv ... ``` or ``` ... ```.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def _build_user_prompt(
    rows: int,
    type_mix: str,
    shapes_mix: str,
    allow_missing: bool,
    random_seed: int,
    banned_triplets: Optional[List[Tuple[str, str, str]]] = None,
) -> str:
    controls = (
        "\n\nGeneration controls for this run:\n"
        f"- rows: {rows}\n"
        f"- type_mix: {type_mix}\n"
        f"- shapes_mix: {shapes_mix}\n"
        f"- allow_missing: {str(allow_missing).lower()}\n"
        f"- random_seed: {random_seed}\n"
    )

    anti_dup = (
        "\nHard anti-duplication constraints for this run:\n"
        "- Every output row must be novel within this batch.\n"
        "- Do not repeat the same (2D,3D,4D) triplet wording.\n"
        "- Do not reuse common templates more than once in this batch (e.g., AB vs CD in rectangle, AB vs BC in rectangle, altitude to base).\n"
        "- Vary figure families and relation archetypes across rows.\n"
        "- If uncertain, prefer creating a fresh structure over rephrasing a known template.\n"
    )

    banned_block = ""
    if banned_triplets:
        lines = ["\nPreviously used triplets. Do NOT generate any of the following again:\n"]
        for i, (q2, q3, q4) in enumerate(banned_triplets, start=1):
            lines.append(f"{i}. 2D: {q2}")
            lines.append(f"   3D: {q3}")
            lines.append(f"   4D: {q4}")
        banned_block = "\n".join(lines) + "\n"

    return prompt + controls + anti_dup + banned_block


def _validate_csv(csv_text: str) -> Tuple[bool, str]:
    try:
        reader = csv.reader(io.StringIO(csv_text))
        rows = list(reader)
    except Exception as exc:
        return False, f"CSV parse error: {exc}"

    if not rows:
        return False, "No CSV rows returned"
    if rows[0] != EXPECTED_HEADER:
        return False, f"Invalid header: expected {EXPECTED_HEADER}, got {rows[0]}"
    if len(rows) < 2:
        return False, "No data rows returned"

    for idx, row in enumerate(rows[1:], start=2):
        if len(row) != 5:
            return False, f"Row {idx} must have 5 columns, got {len(row)}"

        answer = row[3].strip()
        q_type = row[4].strip()
        if answer not in {"A", "B", "C", "D"}:
            return False, f"Row {idx} has invalid answer '{answer}'"
        if q_type not in {"1", "2", "3"}:
            return False, f"Row {idx} has invalid type '{q_type}'"

    return True, "ok"


def _parse_csv_rows(csv_text: str) -> List[List[str]]:
    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)
    return rows[1:]


def _build_csv_text_from_rows(data_rows: List[List[str]]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(EXPECTED_HEADER)
    writer.writerows(data_rows)
    return out.getvalue().strip("\n")


def _triplet_key(row: List[str]) -> Tuple[str, str, str]:
    return (row[0].strip(), row[1].strip(), row[2].strip())


def _build_validation_report(
    data_rows: List[List[str]],
    max_duplicate_rows: int,
    min_count_per_type: int,
) -> Tuple[bool, str]:
    total_rows = len(data_rows)
    type_counter: Counter[str] = Counter(row[4].strip() for row in data_rows)

    # Duplicates are checked on the full triplet (2D,3D,4D) to avoid repeated problems.
    triplets = [(row[0].strip(), row[1].strip(), row[2].strip()) for row in data_rows]
    dup_counter: Counter[Tuple[str, str, str]] = Counter(triplets)
    duplicate_rows = sum(count - 1 for count in dup_counter.values() if count > 1)

    type_ok = all(type_counter.get(t, 0) >= min_count_per_type for t in ("1", "2", "3"))
    dup_ok = duplicate_rows <= max_duplicate_rows
    is_ok = type_ok and dup_ok

    report_lines: List[str] = [
        "Validation summary:",
        f"- total_rows: {total_rows}",
        f"- type_counts: type1={type_counter.get('1', 0)}, type2={type_counter.get('2', 0)}, type3={type_counter.get('3', 0)}",
        f"- duplicate_rows(full_triplet): {duplicate_rows}",
        f"- thresholds: min_count_per_type>={min_count_per_type}, max_duplicate_rows<={max_duplicate_rows}",
    ]

    if not type_ok:
        report_lines.append("- status: FAIL (type distribution below threshold)")
    if not dup_ok:
        report_lines.append("- status: FAIL (too many duplicates)")
    if is_ok:
        report_lines.append("- status: PASS")

    return is_ok, "\n".join(report_lines)


def generate_csv_text(
    model: str,
    rows: int,
    type_mix: str,
    shapes_mix: str,
    allow_missing: bool,
    random_seed: int,
    max_retries: int,
    banned_triplets: Optional[List[Tuple[str, str, str]]] = None,
) -> str:
    client = _build_client()
    user_prompt = _build_user_prompt(
        rows=rows,
        type_mix=type_mix,
        shapes_mix=shapes_mix,
        allow_missing=allow_missing,
        random_seed=random_seed,
        banned_triplets=banned_triplets,
    )

    last_error = "unknown error"
    for attempt in range(1, max_retries + 1):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Follow the user instructions exactly and output only valid CSV.",
                },
                {"role": "user", "content": user_prompt},
            ],
        )

        content = ""
        try:
            content = response.choices[0].message.content or ""
        except Exception:
            content = str(response)

        csv_text = _strip_code_fence(content)
        ok, msg = _validate_csv(csv_text)
        if ok:
            return csv_text
        last_error = f"attempt {attempt}: {msg}"

    raise RuntimeError(f"Failed to get valid CSV from model after {max_retries} tries: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate multidimensional geometry triplet problems as CSV using GPT-5.4 (Azure OpenAI)."
    )
    parser.add_argument("--model", default="gpt-5.4", help="Model deployment/model name (default: gpt-5.4)")
    parser.add_argument("--rows", type=int, default=20, help="Total number of rows to generate")
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=20,
        help="Rows requested per API call (default: 20)",
    )
    parser.add_argument("--type-mix", default="roughly_even", help="Type balance instruction")
    parser.add_argument(
        "--shapes-mix",
        default="rectangle,triangle,circle",
        help="Shape variety instruction",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow '-' in a dimension column",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed instruction")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for valid CSV")
    parser.add_argument(
        "--output",
        default="master/data/questions_generated.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--max-duplicate-rows",
        type=int,
        default=0,
        help="Maximum allowed number of duplicated triplet rows",
    )
    parser.add_argument(
        "--min-count-per-type",
        type=int,
        default=1,
        help="Minimum required count for each type (1,2,3)",
    )
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Exit with error when validation fails",
    )
    parser.add_argument(
        "--allow-duplicate-triplets",
        action="store_true",
        help="Allow duplicated (2D,3D,4D) triplets in final output",
    )
    parser.add_argument(
        "--tail-retry-margin",
        type=int,
        default=20,
        help="Extra rows to request when remaining rows are within one batch",
    )
    args = parser.parse_args()

    if args.rows <= 0:
        raise ValueError("--rows must be > 0")
    if args.batch_rows <= 0:
        raise ValueError("--batch-rows must be > 0")
    if args.tail_retry_margin < 0:
        raise ValueError("--tail-retry-margin must be >= 0")

    data_rows: List[List[str]] = []
    seen_triplets: set[Tuple[str, str, str]] = set()
    seen_triplets_order: List[Tuple[str, str, str]] = []
    call_idx = 0
    planned_calls = max(1, math.ceil(args.rows / args.batch_rows))
    max_calls = planned_calls * 10

    while len(data_rows) < args.rows:
        remaining = args.rows - len(data_rows)
        request_rows = args.batch_rows
        if remaining <= args.batch_rows:
            # Near the end, request extra rows and trim overflow to reduce retries.
            request_rows = args.batch_rows + args.tail_retry_margin
        batch_seed = args.seed + call_idx
        call_idx += 1

        print(
            f"Generating batch {call_idx}/{planned_calls} with seed={batch_seed}, requested_rows={request_rows}"
        )

        # Feed a recent subset of seen triplets back to the model as explicit negatives.
        recent_banned_triplets = seen_triplets_order[-40:]
        batch_csv_text = generate_csv_text(
            model=args.model,
            rows=request_rows,
            type_mix=args.type_mix,
            shapes_mix=args.shapes_mix,
            allow_missing=args.allow_missing,
            random_seed=batch_seed,
            max_retries=args.max_retries,
            banned_triplets=recent_banned_triplets,
        )

        batch_rows = _parse_csv_rows(batch_csv_text)
        if not batch_rows:
            raise RuntimeError("Model returned no data rows for a batch")

        accepted = 0
        skipped_duplicates = 0
        for row in batch_rows:
            if len(data_rows) >= args.rows:
                break
            if args.allow_duplicate_triplets:
                data_rows.append(row)
                accepted += 1
                continue

            key = _triplet_key(row)
            if key in seen_triplets:
                skipped_duplicates += 1
                continue

            seen_triplets.add(key)
            seen_triplets_order.append(key)
            data_rows.append(row)
            accepted += 1

        print(
            f"Batch result: accepted={accepted}, skipped_duplicates={skipped_duplicates}, collected_total={len(data_rows)}/{args.rows}"
        )

        # Prevent runaway loops when model repeatedly under-produces.
        if call_idx >= max_calls and len(data_rows) < args.rows:
            raise RuntimeError(
                f"Unable to reach target rows={args.rows} within {max_calls} calls; collected only {len(data_rows)} rows"
            )

    data_rows = data_rows[: args.rows]
    csv_text = _build_csv_text_from_rows(data_rows)

    output_path = Path(args.output)
    if output_path.suffix.lower() != ".csv":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path / f"questions_generated_{ts}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(csv_text + "\n", encoding="utf-8")

    passed, report = _build_validation_report(
        data_rows=data_rows,
        max_duplicate_rows=args.max_duplicate_rows,
        min_count_per_type=args.min_count_per_type,
    )

    print(f"Saved generated CSV: {output_path}")
    print(report)
    if args.strict_validation and not passed:
        raise SystemExit("Validation failed in strict mode")


if __name__ == "__main__":
    main()
