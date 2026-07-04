#!/usr/bin/env python
"""Cost-efficient correction of an eval run after the 3-sphere->4-ball dataset fix.

Only the numeric family carries the changed questions ("hyper-volume of a
3-sphere" -> "... 4-ball"), and they live in BOTH `numeric` and `numeric_mc`
(4D). So instead of re-running the whole ~9k-question benchmark, this re-runs
ONLY `numeric,numeric_mc` at dim 4 for each model (via evaluate.py's own
machinery, so prompting/scoring/rotation are identical), then splices the
corrected `numeric` / `numeric_mc` rows into each model's existing results.text
and recomputes the 4D Overall row. The geometry types (1/2/3) are untouched.

Two phases (run either or both):
  --run-eval   re-evaluate numeric,numeric_mc @dim4 -> results/eval/<run>-numfix/<model>/
  --merge      splice corrected numeric rows into results/eval/<run>/<model>/results.text

Local models run via vLLM (needs GPU / A100); gpt-5 via Azure (reasoning_effort).
Big models get tensor-parallel from TP_MAP. numeric_mc is ALWAYS included.

Examples
--------
  # local models: re-eval numeric subset then merge, in one go
  python scripts/reeval_changed.py --run results/eval/final_20260701 --run-eval --merge
  # a subset
  python scripts/reeval_changed.py --run ... --models Qwen_Qwen3.5-122B-A10B --run-eval --merge
  # gpt-5 (API): re-eval via Azure (medium) then merge
  python scripts/reeval_changed.py --run ... --models gpt-5 --run-eval --merge

Follow-up: regenerate confusion matrices / summary with the existing tools.
"""
import argparse, csv, glob, os, re, subprocess, sys
csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TYPES = "numeric,numeric_mc"          # BOTH numeric and numeric_mc are re-run
DIM = "4"                             # the 4-ball questions are 4D only
# tensor-parallel per large local model (A100 80GB); default 1
TP_MAP = {"Qwen_Qwen3.5-27B": 2, "Qwen_Qwen3.5-35B-A3B": 2, "Qwen_Qwen3.5-122B-A10B": 4,
          "Qwen_Qwen3-30B-A3B": 2, "Qwen_Qwen3-32B": 2, "Qwen_Qwen3-Next-80B-A3B-Instruct": 4,
          "google_gemma-4-26B-A4B-it": 2, "google_gemma-4-31B-it": 2,
          "google_gemma-2-27b-it": 2}
API_EFFORT = {"gpt-5": "medium", "gpt-5-minimal": "minimal"}
NUMTYPES = ("numeric", "numeric_mc")


def dir_to_model(safe):
    if safe in API_EFFORT or safe.startswith("gpt-"):
        return ("gpt-5", True)                # azure deployment name
    return (safe.replace("_", "/", 1), False)


def run_eval(run, safe, py):
    model_arg, is_api = dir_to_model(safe)
    numfix_ts = os.path.basename(run) + "-numfix"
    cmd = [py, "-m", "evaluation.evaluate", "--models", model_arg,
           "--dims", DIM, "--types", TYPES, "--timestamp", numfix_ts,
           "--prompt-type", "simple_prompt"]
    if is_api:
        cmd += ["--reasoning-effort", API_EFFORT.get(safe, "medium")]
    else:
        cmd += ["--tensor-parallel-size", str(TP_MAP.get(safe, 1)), "--dtype", "bfloat16"]
    # API deployment 'gpt-5' writes to dir 'gpt-5'; for gpt-5-minimal we relocate afterwards
    print("  $", " ".join(cmd))
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    produced = os.path.join(ROOT, "results/eval", numfix_ts, model_arg.replace("/", "_"))
    want = os.path.join(ROOT, "results/eval", numfix_ts, safe)
    if produced != want and os.path.isdir(produced):     # gpt-5 -> gpt-5-minimal
        os.replace(produced, want)


def numeric_counts(pq_path):
    """{type: (n, correct, empty)} from a dim_4_per_question.csv (numeric types)."""
    out = {}
    if not os.path.exists(pq_path):
        return out
    rows = list(csv.DictReader(open(pq_path, encoding="utf-8")))
    for tk in NUMTYPES:
        tr = [r for r in rows if r["type"] == tk]
        if not tr:
            continue
        n = len(tr)
        c = sum(1 for r in tr if r["is_correct"] in ("1", "True", "true"))
        e = sum(1 for r in tr if not (r.get("predicted_normalized") or r.get("predicted_raw")))
        out[tk] = (n, c, e)
    return out


def merge(run, safe):
    """Splice corrected numeric/numeric_mc rows into the 4D block of results.text."""
    numfix = run + "-numfix"
    new = numeric_counts(os.path.join(numfix, safe, "dim_4_per_question.csv"))
    if not new:
        print(f"  [merge] {safe}: no corrected numeric data in {numfix} (run --run-eval first)"); return
    rt_path = os.path.join(run, safe, "results.text")
    if not os.path.exists(rt_path):
        print(f"  [merge] {safe}: no results.text to update"); return
    lines = open(rt_path, encoding="utf-8").read().splitlines()
    # locate the 4D block: header line containing '·   4D'
    blk = next((i for i, l in enumerate(lines) if re.search(r"·\s*4D\s*$", l)), None)
    if blk is None:
        print(f"  [merge] {safe}: no 4D block"); return
    # parse rows in the 4D block: " label  N  Correct  Acc%  Empty% ..."
    row_re = re.compile(r"^\s*(PPC \(1\)|IC \(2\)|CC \(3\)|numeric|numeric_mc|Overall)\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%")
    counts = {}   # label -> (n, correct, empty_pct) for all rows in this block
    idx = {}
    end = None
    for i in range(blk, min(blk + 40, len(lines))):
        m = row_re.match(lines[i])
        if m:
            counts[m.group(1)] = [int(m.group(2)), int(m.group(3)), float(m.group(5))]
            idx[m.group(1)] = i
        if m and m.group(1) == "Overall":
            end = i; break
    if "numeric" not in idx and "numeric_mc" not in idx:
        print(f"  [merge] {safe}: 4D block has no numeric rows"); return
    # apply new numeric/numeric_mc counts
    label_of = {"numeric": "numeric", "numeric_mc": "numeric_mc"}
    for tk, (n, c, e) in new.items():
        lab = label_of[tk]
        if lab in idx:
            emp = 100.0 * e / n if n else 0.0
            counts[lab] = [n, c, emp]
            lines[idx[lab]] = f" {lab:<12} {n:>6} {c:>8} {100*c/n:>7.2f}% {emp:>6.1f}%   {'-':>5}   {'-':>5}   {'-':>5}"
    # recompute Overall from all non-Overall rows in the block
    tot_n = sum(v[0] for k, v in counts.items() if k != "Overall")
    tot_c = sum(v[1] for k, v in counts.items() if k != "Overall")
    tot_e = sum(v[0] * v[2] / 100.0 for k, v in counts.items() if k != "Overall")
    if "Overall" in idx:
        lines[idx["Overall"]] = f" {'Overall':<12} {tot_n:>6} {tot_c:>8} {100*tot_c/tot_n:>7.2f}% {100*tot_e/tot_n:>6.1f}%   {'-':>5}"
    open(rt_path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"  [merge] {safe}: 4D numeric/numeric_mc updated, Overall -> {100*tot_c/tot_n:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/eval/final_20260701")
    ap.add_argument("--models", nargs="*", default=None, help="model dir names (default: all)")
    ap.add_argument("--run-eval", action="store_true", help="re-evaluate numeric,numeric_mc @dim4")
    ap.add_argument("--merge", action="store_true", help="splice corrected numeric rows into results.text")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    run = args.run.rstrip("/")
    safes = args.models or sorted(os.path.basename(d) for d in glob.glob(os.path.join(run, "*")) if os.path.isdir(d))
    if not (args.run_eval or args.merge):
        print("Nothing to do: pass --run-eval and/or --merge"); return
    for safe in safes:
        print(f"[{safe}]")
        try:
            if args.run_eval:
                run_eval(run, safe, args.python)
            if args.merge:
                merge(run, safe)
        except subprocess.CalledProcessError as e:
            print(f"  [error] {safe}: eval failed ({e}); skipping")
    print("\nDone. Optional: python scripts/rebuild_semantic_results.py ; regenerate summary.")


if __name__ == "__main__":
    main()
