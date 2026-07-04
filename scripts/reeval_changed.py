#!/usr/bin/env python
"""Cost-efficient correction of an eval run after the 3-sphere->4-ball dataset fix.

Background
----------
"hyper-volume of a 3-sphere with radius R" was ambiguous: mathematically a
3-sphere is S^3 (surface, 3-volume 2*pi^2*R^3), while the intended answer is the
enclosed 4-ball (pi^2/2 * R^4). Frontier models (GPT-5) consistently gave the
S^3 reading and were marked wrong. The dataset now says "hyper-volume of a
4-ball ..." (surface questions, phrased "hyper-surface volume of a 3-sphere",
are unchanged).

This re-scores an existing run for that fix by re-running ONLY the affected
rows (the handful of numeric AND numeric_mc questions per model whose saved
question still says the old phrase) — reusing each row's saved, already
chat-templated `input` with just the phrase substituted, so the prompt is
identical except the wording. It updates dim_*_per_question.csv,
dim_*_{correct,incorrect}.csv and results.text (Conf preserved from the
per_question `confidence` column; unaffected rows are untouched).

Requires the per_question CSVs to be present (they are saved on the eval /
A100 machine even though not committed). Both `numeric` and `numeric_mc` are
included. Local models run via vLLM (GPU); gpt-5[-minimal] via Azure.

Usage
-----
  python scripts/reeval_changed.py --run results/eval/final_20260701
  python scripts/reeval_changed.py --run ... --models Qwen_Qwen3.5-122B-A10B --tp 4
  python scripts/reeval_changed.py --run ... --models gpt-5 gpt-5-minimal
Follow-up: python scripts/rebuild_semantic_results.py ; regenerate summary.
"""
import argparse, csv, glob, os, statistics, sys
csv.field_size_limit(10 ** 7)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OLD_PHRASE = "hyper-volume of a 3-sphere"
NEW_PHRASE = "hyper-volume of a 4-ball"
NUMTYPES = ("numeric", "numeric_mc")
TYPE_LABEL = [("1", "PPC (1)"), ("2", "IC (2)"), ("3", "CC (3)"),
              ("numeric", "numeric"), ("numeric_mc", "numeric_mc")]
TP_MAP = {"Qwen_Qwen3.5-27B": 2, "Qwen_Qwen3.5-35B-A3B": 2, "Qwen_Qwen3.5-122B-A10B": 4,
          "Qwen_Qwen3-30B-A3B": 2, "Qwen_Qwen3-32B": 2, "Qwen_Qwen3-Next-80B-A3B-Instruct": 4,
          "google_gemma-4-26B-A4B-it": 2, "google_gemma-4-31B-it": 2, "google_gemma-2-27b-it": 2}
API_EFFORT = {"gpt-5": "medium", "gpt-5-minimal": "minimal"}
TRUE = ("1", "True", "true")


def dir_to_model(safe):
    if safe in API_EFFORT or safe.startswith("gpt-"):
        return "gpt-5", True
    return safe.replace("_", "/", 1), False


def score_row(type_key, response, gt_raw, extract_answer, extract_numeric, norm_label):
    import re
    if type_key == "numeric":
        pred = extract_numeric(response)
        ok = False
        if pred is not None:
            try:
                ok = int(pred) == int(re.sub(r"[^0-9\-]", "", str(gt_raw)))
            except Exception:
                ok = False
        return (pred or ""), (pred or ""), ok
    pred = extract_answer(response)
    true_label = norm_label(gt_raw)
    plabel = ""
    if pred is not None:
        a = pred.strip().upper()
        if true_label.isdigit():
            i = ord(a) - ord("A"); plabel = str(i + 1) if 0 <= i < 26 else a
        else:
            plabel = a
    return (pred or ""), plabel, (plabel != "" and true_label == plabel)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def _cf(x):
    try:
        return float(x)
    except Exception:
        return None


def regen_results_text(model_disp, run, safe):
    lines = []
    for dim in ("2", "3", "4"):
        p = os.path.join(run, safe, f"dim_{dim}_per_question.csv")
        if not os.path.exists(p):
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        if not rows:
            continue
        lines += ["", "=" * 72, f" {model_disp}   ·   {dim}D", "-" * 72,
                  f" {'Type':<12} {'N':>6} {'Correct':>8} {'Acc':>8} {'Empty':>7} {'Conf':>7} {'Conf✓':>7} {'Conf✗':>7}",
                  "-" * 72]
        tn = tc = 0.0; te = 0.0; allc = []
        for tk, label in TYPE_LABEL:
            tr = [r for r in rows if r["type"] == tk]
            if not tr:
                continue
            n = len(tr)
            c = sum(1 for r in tr if r["is_correct"] in TRUE)
            e = sum(1 for r in tr if not (r.get("predicted_normalized") or r.get("predicted_raw")))
            cf = [_cf(r.get("confidence")) for r in tr]
            cok = [_cf(r["confidence"]) for r in tr if r["is_correct"] in TRUE]
            cno = [_cf(r["confidence"]) for r in tr if r["is_correct"] not in TRUE]
            mc, mok, mno = _mean(cf), _mean(cok), _mean(cno)
            allc += [v for v in cf if v is not None]
            tn += n; tc += c; te += e
            fmt = lambda v: f"{v:.3f}" if v is not None else "-"
            lines.append(f" {label:<12} {n:>6} {c:>8} {100*c/n:>7.2f}% {100*e/n:>6.1f}% {fmt(mc):>7} {fmt(mok):>7} {fmt(mno):>7}")
        oc = _mean(allc)
        lines += ["-" * 72,
                  f" {'Overall':<12} {int(tn):>6} {int(tc):>8} {100*tc/tn:>7.2f}% {100*te/tn:>6.1f}% {(f'{oc:.3f}' if oc is not None else '-'):>7}",
                  "-" * 72,
                  " Acc = accuracy   Empty = share with no parseable answer",
                  " Conf = mean P(model's chosen answer token); Conf✓/Conf✗ = on correct/incorrect",
                  " (blank/'-' = unavailable for this backend)", "=" * 72]
    open(os.path.join(run, safe, "results.text"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def rewrite_splits(run, safe, dim, rows):
    d = os.path.join(run, safe); fields = list(rows[0].keys())
    for name, want in [("per_question", None), ("correct", True), ("incorrect", False)]:
        sel = rows if want is None else [r for r in rows if (r["is_correct"] in TRUE) == want]
        with open(os.path.join(d, f"dim_{dim}_{name}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/eval/final_20260701")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--tp", type=int, default=None, help="override tensor_parallel_size")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--rebuild", action="store_true",
                    help="after updating, run rebuild_semantic_results.py (confusion matrices + semantic_accuracy.json)")
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from evaluation.evaluate import (chat, extract_answer, extract_numeric,
                                     _normalize_label, _answer_confidence)
    from common.prompting import set_reasoning_effort

    run = args.run.rstrip("/")
    safes = args.models or sorted(os.path.basename(d) for d in glob.glob(os.path.join(run, "*")) if os.path.isdir(d))
    for safe in safes:
        model_name, is_api = dir_to_model(safe)
        affected = {}
        for dim in ("2", "3", "4"):
            p = os.path.join(run, safe, f"dim_{dim}_per_question.csv")
            if not os.path.exists(p):
                continue
            rows = list(csv.DictReader(open(p, encoding="utf-8")))
            idx = [i for i, r in enumerate(rows)
                   if r["type"] in NUMTYPES and OLD_PHRASE in r.get("question", "")]
            if idx:
                affected[dim] = (rows, idx)
        naff = sum(len(v[1]) for v in affected.values())
        if not naff:
            print(f"[skip] {safe}: no affected rows (per_question present? "
                  f"{os.path.exists(os.path.join(run, safe, 'dim_4_per_question.csv'))})")
            continue
        print(f"[{safe}] model={model_name} api={is_api} affected={naff} (numeric+numeric_mc)")

        llm = None
        if is_api:
            set_reasoning_effort(API_EFFORT.get(safe, "medium"))
        else:
            from vllm import LLM
            tp = args.tp or TP_MAP.get(safe, 1)
            for dt in ([args.dtype, "float32"] if args.dtype != "float32" else ["float32"]):
                try:
                    llm = LLM(model=model_name, dtype=dt, gpu_memory_utilization=args.gpu_mem,
                              tensor_parallel_size=tp, disable_log_stats=True)
                    break
                except Exception as e:
                    print(f"   vLLM dtype={dt} tp={tp} failed: {str(e)[:100]}")
            if llm is None:
                print(f"[warn] {safe}: could not load; skip"); continue

        for dim, (rows, idx) in affected.items():
            prompts = [rows[i]["input"].replace(OLD_PHRASE, NEW_PHRASE) for i in idx]
            if is_api:
                resp, lps = chat(prompts, model_name=model_name)
            else:
                resp, lps = chat(prompts, model=llm, do_sample=False,
                                 max_new_tokens=args.max_new_tokens, repetition_penalty=1.0)
            for j, i in enumerate(idx):
                r = rows[i]
                r["question"] = r["question"].replace(OLD_PHRASE, NEW_PHRASE)
                r["input"] = prompts[j]
                r["response"] = resp[j]
                praw, pnorm, ok = score_row(r["type"], resp[j], r["ground_truth_raw"],
                                            extract_answer, extract_numeric, _normalize_label)
                r["predicted_raw"] = praw; r["predicted_normalized"] = pnorm
                r["is_correct"] = "1" if ok else "0"
                conf = _answer_confidence(lps[j] if lps else None,
                                          praw if r["type"] == "numeric" else pnorm,
                                          is_numeric=(r["type"] == "numeric"))
                r["confidence"] = "" if conf is None else f"{conf:.6f}"
            rewrite_splits(run, safe, dim, rows)
        regen_results_text(model_name, run, safe)
        print(f"[done] {safe}: per_question / correct / incorrect / results.text updated")
        del llm

    print("\nUpdated per model: dim_4_per_question.csv, dim_4_correct/incorrect.csv, results.text.")
    print("Unaffected (numeric has no confusion matrix; types 1/2/3 only): confusion_matrix_*.png, semantic_accuracy.json.")
    if args.rebuild:
        import subprocess
        print("Running rebuild_semantic_results.py ...")
        subprocess.run([sys.executable, "scripts/rebuild_semantic_results.py"],
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    else:
        print("Optional: python scripts/rebuild_semantic_results.py   (or pass --rebuild)")
    print("STILL TO DO (run-level aggregate): regenerate summary.md — every model's numeric acc changed.")


if __name__ == "__main__":
    main()
