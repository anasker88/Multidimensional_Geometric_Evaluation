import csv
import os
import re
from collections import OrderedDict
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from common.prompting import (
    apply_chat_template,
    make_prompt_mc_variants,
    make_prompt_numeric,
    remap_answer_for_rotation,
)


_TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def _normalize_label(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", (s or "")).upper()


def _extract_answer(response: str) -> Optional[str]:
    if not response:
        return None

    tag_matches = re.findall(
        r"<answer>\s*([^<\n]+?)\s*</answer>",
        response,
        re.IGNORECASE,
    )
    if tag_matches:
        val = tag_matches[-1].strip()
        m = re.match(r"^([A-E])$", val, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    last_box = None
    for m in re.finditer(r"\\boxed\{([^}]*)\}", response, re.IGNORECASE):
        last_box = m.group(1)
    if last_box is not None:
        m2 = re.search(r"[A-Z]", last_box)
        if m2:
            return m2.group(0).upper()
        m3 = re.search(r"[A-Za-z]", last_box)
        if m3:
            return m3.group(0).upper()

    simple_match = re.search(r"Assistant:\s*The answer is\s*([A-E])\b", response, re.IGNORECASE)
    if simple_match:
        return simple_match.group(1).upper()

    answer_is_match = re.search(r"answer\s+is\s*\*{0,2}\s*\(?([A-E])\)?", response, re.IGNORECASE)
    if answer_is_match:
        return answer_is_match.group(1).upper()

    choice_match = re.search(r"\b([A-E])[.:](?:\s|$)", response, re.MULTILINE)
    if choice_match:
        return choice_match.group(1).upper()

    md_choice_match = re.search(r"\*{1,2}\s*\(?([A-E])\)?", response, re.IGNORECASE)
    if md_choice_match:
        return md_choice_match.group(1).upper()

    simple_token = re.search(r"^\s*([A-E])\s*$", response, re.IGNORECASE | re.MULTILINE)
    if simple_token:
        return simple_token.group(1).upper()
    return None


def _extract_numeric(response: str) -> Optional[str]:
    if not response:
        return None

    tag_matches = re.findall(
        r"<answer>\s*([^<\n]+?)\s*</answer>",
        response,
        re.IGNORECASE,
    )
    if tag_matches:
        txt = tag_matches[-1]
        m = re.search(r"(-?\d+)", txt)
        if m:
            return m.group(1)

    last_box = None
    for m in re.finditer(r"\\boxed\{([^}]*)\}", response, re.IGNORECASE):
        last_box = m.group(1)
    if last_box is not None:
        m = re.search(r"(-?\d+)", last_box)
        if m:
            return m.group(1)

    simple_match = re.search(r"Assistant:\s*The answer is\s*(-?\d+)\b", response, re.IGNORECASE)
    if simple_match:
        return simple_match.group(1)

    number_match = re.search(r"\b(-?\d+)\b", response)
    if number_match:
        return number_match.group(1)

    simple_token = re.match(r"^\s*(-?\d+)\s*$", response, re.MULTILINE)
    if simple_token:
        return simple_token.group(1)

    for line in reversed([ln.strip() for ln in response.splitlines() if ln.strip()]):
        m = re.search(r"(-?\d+)", line)
        if m:
            return m.group(1)
    return None


def _build_eval_samples(
    dim: int,
    reasoning: bool | str,
    questions_csv_path: str,
    numeric_csv_path: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    with open(questions_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            col_key = f"{dim}D"
            if col_key not in row:
                continue
            question = (row.get(col_key) or "").strip()
            if not question or question == "-":
                continue
            type_key = (row.get("type") or "unknown").strip() or "unknown"
            answer = (row.get("answer") or "").strip()
            if not answer:
                continue
            variants = make_prompt_mc_variants(question, type_key, reasoning=reasoning)
            for variant in variants:
                remapped = remap_answer_for_rotation(
                    answer,
                    int(variant["rotation"]),
                    int(variant["num_choices"]),
                )
                rows.append(
                    {
                        "task_type": "mc",
                        "type": type_key,
                        "source_file": questions_csv_path,
                        "question": question,
                        "prompt": str(variant["prompt"]),
                        "answer": remapped,
                    }
                )

    if os.path.exists(numeric_csv_path):
        with open(numeric_csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    row_dim = int((row.get("dimension") or "").strip())
                except Exception:
                    continue
                if row_dim != dim:
                    continue
                question = (row.get("question") or "").strip()
                answer = (row.get("answer") or "").strip()
                if not question or not answer:
                    continue
                rows.append(
                    {
                        "task_type": "numeric",
                        "type": "numeric",
                        "source_file": numeric_csv_path,
                        "question": question,
                        "prompt": make_prompt_numeric(question, reasoning=reasoning),
                        "answer": answer,
                    }
                )

    return rows


def _write_confusion_matrix_png(
    confusion_matrix: np.ndarray,
    labels: List[str],
    title: str,
    output_path: str,
) -> None:
    plt.figure(figsize=(8, 6))
    row_sums = np.sum(confusion_matrix, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = confusion_matrix / row_sums
        cm_norm = np.nan_to_num(cm_norm)
    pred_labels = labels + ["Other"]
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(np.arange(len(pred_labels)), pred_labels, rotation=45)
    plt.yticks(np.arange(len(labels)), labels, rotation=45)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def run_ablation_eval_with_report(
    response_fn: Callable[[List[str]], List[str]],
    dims: List[int],
    prompt_type: str,
    questions_csv_path: str,
    numeric_csv_path: str,
    output_dir: str,
    generation: Dict,
    model_name: str = "",
) -> Dict:
    prompt_key = prompt_type
    os.makedirs(output_dir, exist_ok=True)

    fieldnames = [
        "dimension",
        "type",
        "source_file",
        "question",
        "ground_truth_raw",
        "ground_truth_normalized",
        "predicted_raw",
        "predicted_normalized",
        "is_correct",
        "input",
        "response",
    ]

    total = 0
    total_correct = 0
    per_dim_payload: List[Dict] = []

    summary_path = os.path.join(output_dir, "results.text")
    with open(summary_path, "w", encoding="utf-8") as rf:
        rf.write("")

    for dim in tqdm(
        dims,
        desc="Ablation eval dims",
        unit="dim",
        bar_format=_TQDM_BAR_FORMAT,
    ):
        samples = _build_eval_samples(
            dim=dim,
            reasoning=prompt_key,
            questions_csv_path=questions_csv_path,
            numeric_csv_path=numeric_csv_path,
        )
        prompts = [row["prompt"] for row in samples]
        formatted_prompts = [apply_chat_template(p, model_name) for p in prompts]
        responses = response_fn(formatted_prompts) if formatted_prompts else []

        per_question_records: List[Dict[str, str]] = []
        stats_by_type: Dict[str, Dict[str, int]] = {}
        confusion_by_type: Dict[str, Dict] = {}

        for sample, formatted_prompt, response in zip(samples, formatted_prompts, responses):
            t = sample["type"]
            stats_by_type.setdefault(t, {"correct": 0, "total": 0})
            stats_by_type[t]["total"] += 1

            true_raw = sample["answer"]
            true_label = _normalize_label(true_raw)
            predicted_raw = ""
            predicted_normalized = ""
            is_correct = False

            if sample["task_type"] == "numeric":
                pred = _extract_numeric(response)
                predicted_raw = pred or ""
                predicted_normalized = pred or ""
                if pred is not None:
                    try:
                        truth = int(re.sub(r"[^0-9\-]", "", str(true_raw)))
                        is_correct = int(pred) == truth
                    except Exception:
                        is_correct = False
            else:
                pred = _extract_answer(response)
                predicted_raw = pred or ""
                pred_label = None
                if pred is not None:
                    pred_alpha = pred.strip().upper()
                    if true_label.isdigit():
                        idx = ord(pred_alpha) - ord("A")
                        pred_label = str(idx + 1) if 0 <= idx < 26 else pred_alpha
                    else:
                        pred_label = pred_alpha
                predicted_normalized = pred_label or ""
                is_correct = pred_label is not None and pred_label == true_label

                if t not in confusion_by_type:
                    confusion_by_type[t] = {
                        "labels_order": OrderedDict(),
                        "pairs": [],
                    }
                if true_label:
                    confusion_by_type[t]["labels_order"].setdefault(true_label, None)
                confusion_by_type[t]["pairs"].append((true_label, predicted_normalized))

            if is_correct:
                stats_by_type[t]["correct"] += 1

            per_question_records.append(
                {
                    "dimension": str(dim),
                    "type": t,
                    "source_file": sample["source_file"],
                    "question": sample["question"],
                    "ground_truth_raw": str(true_raw),
                    "ground_truth_normalized": true_label,
                    "predicted_raw": predicted_raw,
                    "predicted_normalized": predicted_normalized,
                    "is_correct": "1" if is_correct else "0",
                    "input": formatted_prompt,
                    "response": response,
                }
            )

        dim_total = len(per_question_records)
        dim_correct = sum(1 for r in per_question_records if r["is_correct"] == "1")
        total += dim_total
        total_correct += dim_correct

        dim_dir = os.path.join(output_dir, f"dim_{dim}")
        os.makedirs(dim_dir, exist_ok=True)

        per_question_path = os.path.join(dim_dir, "per_question.csv")
        correct_path = os.path.join(dim_dir, "correct.csv")
        incorrect_path = os.path.join(dim_dir, "incorrect.csv")

        with open(per_question_path, "w", encoding="utf-8", newline="") as pf:
            writer = csv.DictWriter(pf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_question_records)

        with open(correct_path, "w", encoding="utf-8", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([r for r in per_question_records if r["is_correct"] == "1"])

        with open(incorrect_path, "w", encoding="utf-8", newline="") as inf:
            writer = csv.DictWriter(inf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([r for r in per_question_records if r["is_correct"] == "0"])

        per_type_payload: List[Dict] = []
        for type_key in sorted(stats_by_type.keys(), key=lambda x: (x == "unknown", x)):
            c = stats_by_type[type_key]["correct"]
            n = stats_by_type[type_key]["total"]
            acc = float(c / n) if n else 0.0
            per_type_payload.append(
                {
                    "type": type_key,
                    "num_correct": c,
                    "num_questions": n,
                    "accuracy": acc,
                }
            )

            if type_key != "numeric" and type_key in confusion_by_type:
                labels = list(confusion_by_type[type_key]["labels_order"].keys())
                if labels:
                    labels = sorted(labels)
                    cm = np.zeros((len(labels), len(labels) + 1), dtype=int)
                    for true_label, pred_label in confusion_by_type[type_key]["pairs"]:
                        if true_label not in labels:
                            continue
                        t_idx = labels.index(true_label)
                        if pred_label in labels:
                            p_idx = labels.index(pred_label)
                            cm[t_idx, p_idx] += 1
                        else:
                            cm[t_idx, -1] += 1
                    cm_path = os.path.join(dim_dir, f"confusion_matrix_{dim}d_type_{type_key}.png")
                    _write_confusion_matrix_png(
                        confusion_matrix=cm,
                        labels=labels,
                        title=f"Confusion Matrix - dim{dim} - type {type_key}",
                        output_path=cm_path,
                    )

        with open(summary_path, "a", encoding="utf-8") as rf:
            rf.write("\n")
            rf.write("========================================\n")
            rf.write(f"Dimension: {dim}D\n")
            rf.write("----------------------------------------\n")
            rf.write(f"{'Type':<12}{'Correct':>8}{'Total':>8}{'Accuracy':>12}\n")
            rf.write("----------------------------------------\n")
            for entry in per_type_payload:
                rf.write(
                    f"{entry['type']:<12}{entry['num_correct']:8d}{entry['num_questions']:8d}{entry['accuracy']:12.2%}\n"
                )
            rf.write("----------------------------------------\n")
            rf.write(f"{'Overall':<12}{dim_correct:8d}{dim_total:8d}{(dim_correct / dim_total if dim_total else 0.0):12.2%}\n")
            rf.write("========================================\n")

        per_dim_payload.append(
            {
                "dimension": dim,
                "num_questions": dim_total,
                "num_correct": dim_correct,
                "accuracy": float(dim_correct / dim_total) if dim_total else 0.0,
                "per_type": per_type_payload,
                "artifacts": {
                    "per_question_csv": per_question_path,
                    "correct_csv": correct_path,
                    "incorrect_csv": incorrect_path,
                    "dim_dir": dim_dir,
                },
            }
        )

    return {
        "prompt_type": prompt_type,
        "generation": generation,
        "overall": {
            "num_questions": total,
            "num_correct": total_correct,
            "accuracy": float(total_correct / total) if total else 0.0,
        },
        "per_dimension": per_dim_payload,
        "artifacts": {
            "summary_text": summary_path,
            "output_dir": output_dir,
        },
    }
