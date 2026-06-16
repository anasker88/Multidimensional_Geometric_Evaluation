#!/usr/bin/env python3
"""
Rebuild confusion matrices and per-class accuracy in the CANONICAL (semantic)
frame for already-completed runs, without re-running any model.

Background
----------
`cli/evaluate.py` rotates the answer choices through every cyclic position and
remaps the gold letter accordingly. The per-question CSV therefore stores
`ground_truth_normalized` / `predicted_normalized` in the *presented-slot*
frame (the letter as shown in that rotated variant), NOT the original meaning.
Slicing accuracy by that column re-exposes positional (primacy-aversion) bias.

This script joins each row back to its canonical answer (via the question text)
to recover the rotation, maps both the gold answer and the prediction back to
the canonical pre-rotation frame, and then:
  * regenerates confusion_matrix_{dim}d_type_{t}.png in the semantic frame
  * computes accuracy grouped by canonical meaning (semantic) and, for
    comparison, by presented slot (positional)

It mirrors the canonical-frame logic now built into cli/evaluate.py, so the
artifacts match what a fresh run would produce.

Usage:
    python scripts/rebuild_semantic_results.py            # all runs under results/
    python scripts/rebuild_semantic_results.py --no-plots # numbers only
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from prompting import _choice_count_for_type, _parse_option_choices, options  # noqa: E402

csv.field_size_limit(10 ** 7)

DATA_FILES = ["data/questions_augmented.csv", "data/questions.csv"]


def build_canon_lookup():
    """(type, question_text) -> canonical answer letter."""
    canon = {}
    for path in DATA_FILES:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                t = (r.get("type") or "").strip()
                ans = (r.get("answer") or "").strip().upper()
                for d in ("2D", "3D", "4D"):
                    txt = (r.get(d) or "").strip()
                    if txt and txt != "-":
                        canon.setdefault((t, txt), ans)
    return canon


def meanings_for(type_key):
    try:
        idx = max(0, min(int(type_key) - 1, len(options) - 1))
        return _parse_option_choices(options[idx])
    except Exception:
        return []


def letter_idx(s):
    s = (s or "").strip().upper()
    if len(s) == 1 and s.isalpha():
        return ord(s) - ord("A")
    return None


def plot_cm(cm, label_meanings, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.nan_to_num(cm / row_sums)
    pred_labels = label_meanings + ["Other / no answer"]
    plt.figure(figsize=(8, 6))
    plt.imshow(norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlabel("Predicted (canonical meaning)")
    plt.ylabel("True (canonical meaning)")
    plt.xticks(np.arange(len(pred_labels)), pred_labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(label_meanings)), label_meanings)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def process_csv(csv_path, canon, make_plots):
    out_dir = os.path.dirname(csv_path)
    base = os.path.basename(csv_path)            # dim_2_per_question.csv
    dim = base.split("_")[1]
    model = os.path.relpath(out_dir, "results")

    # type -> {"sem": {canon_letter: [correct, total]},
    #          "slot": {slot_letter: [correct, total]},
    #          "cm": np.array}
    per_type = {}
    unmatched = 0
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = (r.get("type") or "").strip()
            if t == "numeric" or not t:
                continue
            n = _choice_count_for_type(t)
            if n <= 0:
                continue
            q = (r.get("question") or "").strip()
            c = canon.get((t, q))
            if c is None:
                unmatched += 1
                continue
            c_idx = letter_idx(c)
            gt_slot_idx = letter_idx(r.get("ground_truth_normalized"))
            if c_idx is None or gt_slot_idx is None:
                continue
            rot = (c_idx - gt_slot_idx) % n            # recovered rotation
            pred_slot_idx = letter_idx(r.get("predicted_normalized"))
            pred_canon_idx = None
            if pred_slot_idx is not None and 0 <= pred_slot_idx < n:
                pred_canon_idx = (pred_slot_idx + rot) % n

            d = per_type.setdefault(
                t, {"sem": {}, "slot": {}, "cm": np.zeros((n, n + 1), dtype=int), "n": n}
            )
            is_corr = int(r.get("is_correct", "0"))

            # semantic-frame accuracy (group by canonical meaning)
            cl = chr(ord("A") + c_idx)
            s = d["sem"].setdefault(cl, [0, 0]); s[1] += 1; s[0] += is_corr
            # presented-slot accuracy (group by shown letter of the gold answer)
            sl = chr(ord("A") + gt_slot_idx)
            ss = d["slot"].setdefault(sl, [0, 0]); ss[1] += 1; ss[0] += is_corr
            # canonical confusion matrix
            if pred_canon_idx is None:
                d["cm"][c_idx, -1] += 1
            else:
                d["cm"][c_idx, pred_canon_idx] += 1

    results = {}
    for t, d in sorted(per_type.items()):
        n = d["n"]
        labels = [chr(ord("A") + i) for i in range(n)]
        meanings = meanings_for(t)
        label_meanings = [f"{labels[i]}. {meanings[i]}" if i < len(meanings) else labels[i]
                          for i in range(n)]
        # sanity: diagonal == total correct
        diag = int(np.trace(d["cm"][:, :n]))
        tot_correct = sum(v[0] for v in d["sem"].values())
        assert diag == tot_correct, f"frame mismatch in {csv_path} type {t}: {diag} != {tot_correct}"

        if make_plots:
            out_png = os.path.join(out_dir, f"confusion_matrix_{dim}d_type_{t}.png")
            plot_cm(d["cm"], label_meanings,
                    f"Confusion Matrix (semantic) - {model} - {dim}D - type {t}", out_png)

        results[t] = {
            "n_choices": n,
            "semantic": {meanings[i] if i < len(meanings) else labels[i]:
                         round(d["sem"].get(labels[i], [0, 0])[0] / d["sem"][labels[i]][1], 4)
                         for i in range(n) if labels[i] in d["sem"]},
            "slot": {labels[i]: round(d["slot"][labels[i]][0] / d["slot"][labels[i]][1], 4)
                     for i in range(n) if labels[i] in d["slot"]},
        }
    return model, dim, results, unmatched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plots", action="store_true", help="skip regenerating PNGs")
    args = ap.parse_args()

    canon = build_canon_lookup()
    csvs = sorted(glob.glob("results/**/dim_*_per_question.csv", recursive=True))
    if not csvs:
        print("No per-question CSVs found under results/.")
        return

    agg = {}
    total_unmatched = 0
    for p in csvs:
        model, dim, res, unmatched = process_csv(p, canon, not args.no_plots)
        total_unmatched += unmatched
        agg.setdefault(model, {})[f"dim_{dim}"] = res
        tag = "(plots updated)" if not args.no_plots else ""
        print(f"[{model}] dim {dim}: {len(res)} types, unmatched={unmatched} {tag}")

    out_json = "results/semantic_accuracy.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_json}  (total unmatched rows: {total_unmatched})")


if __name__ == "__main__":
    main()
