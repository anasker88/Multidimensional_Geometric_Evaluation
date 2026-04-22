import csv
import os
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from openai import AzureOpenAI
from collections import OrderedDict
from vllm import LLM, SamplingParams
import argparse
import sys

from prompting import make_prompt_mc, make_prompt_numeric


# -------------------------------------------------------------
# Model client
# -------------------------------------------------------------
def _build_azure_client() -> AzureOpenAI:
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION","")
    return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)


@torch.no_grad()
def chat(
    prompts: List[str],
    model_name: Optional[str] = None,
    model=None,
    tokenizer=None,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    do_sample: bool = True,
    top_k: int = 0,
    top_p: float = 0.9,
    temperature: float = 0.1,
    repetition_penalty: float = 1.1,
) -> List[str]:
    responses: List[str] = []

    if model is None:
        client = _build_azure_client()
        for prompt in prompts:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            # AzureOpenAI client may return different shapes depending on SDK/version:
            # - an object with .choices[0].message.content
            # - a dict-like with ['choices'][0]['message']['content']
            # - sometimes a plain string
            content = None
            try:
                # Try attribute-style access first
                content = response.choices[0].message.content.strip()
            except Exception:
                try:
                    # Try dict-like access
                    content = response["choices"][0]["message"]["content"].strip()
                except Exception:
                    # Fallback to str() of response
                    try:
                        content = str(response).strip()
                    except Exception:
                        content = ""
            responses.append(content)
        return responses

    # Local model path: run in batches to avoid huge memory usage
    if len(prompts) == 0:
        return responses

    # determine device for inputs; handle sharded/multi-device models safely
    try:
        first_param = next(model.parameters())
        device = first_param.device
    except Exception:
        device = getattr(model, "device", torch.device("cpu"))

    batch_num = (len(prompts) + batch_size - 1) // batch_size

    for i in range(batch_num):
        batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
        # Handle vllm LLM instances separately (they expect raw prompts)
        if isinstance(model, LLM):
            try:
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_new_tokens,
                )
                # vLLM accepts a sequence of prompt strings and returns a list of RequestOutput
                outputs = model.generate(batch_prompts, sampling_params=sampling_params, use_tqdm=False)
            except RuntimeError as e:
                err = str(e).lower()
                if "out of memory" in err or "cuda out of memory" in err:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    try:
                        sampling_params = SamplingParams(
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            repetition_penalty=repetition_penalty,
                            max_tokens=max(32, max_new_tokens // 4),
                        )
                        outputs = model.generate(batch_prompts, sampling_params=sampling_params, use_tqdm=False)
                    except Exception:
                        outputs = None
                else:
                    raise

            if outputs is None:
                batch_responses = ["" for _ in batch_prompts]
            else:
                batch_responses = []
                for prompt, req_out in zip(batch_prompts, outputs):
                    # concatenate all completion outputs for this request
                    try:
                        text = "".join([c.text for c in req_out.outputs])
                    except Exception:
                        # best-effort fallback
                        text = getattr(req_out, "prompt", "") or ""
                    # strip prompt prefix if echoed
                    if prompt and text.startswith(prompt):
                        text = text[len(prompt) :].strip()
                    batch_responses.append(text.strip())
        else:
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_k=top_k if top_k > 0 else None,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                )
            except RuntimeError as e:
                err = str(e).lower()
                if "out of memory" in err or "cuda out of memory" in err:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    try:
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max(32, max_new_tokens // 4),
                            do_sample=do_sample,
                            top_k=top_k if top_k > 0 else None,
                            top_p=top_p,
                            temperature=temperature,
                            repetition_penalty=repetition_penalty,
                            use_cache=False,
                        )
                    except Exception:
                        outputs = None
                else:
                    raise
            if outputs is None:
                batch_responses = ["" for _ in batch_prompts]
            else:
                batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                # remove the input part from the response (best-effort)
                batch_responses = [
                    resp[len(prompt) :].strip() if resp.startswith(prompt) else resp.strip()
                    for resp, prompt in zip(batch_responses, batch_prompts)
                ]
        responses.extend(batch_responses)
    return responses


# -------------------------------------------------------------
# Extractors
# -------------------------------------------------------------
def extract_answer(response: str) -> Optional[str]:
    """Extract the model's answer letter (A-E) from the response, preferring the last match.
    Prioritizes the last `<final_answer>...</final_answer>`; if none,
    falls back to the last `\boxed{...}` match. Returns the uppercase letter A-E or None.
    """
    if not response:
        return None

    tag_matches = re.findall(r"<final_answer>\s*([^<\n]+?)\s*</final_answer>", response, re.IGNORECASE)
    if tag_matches:
        val = tag_matches[-1].strip()
        m = re.match(r"^([A-E])$", val, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Fallback: last \boxed{...}
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
    return None


def extract_numeric(response: str) -> Optional[str]:
    """Extract a numeric answer from response.
    Priority:
    1. Last <final_answer>...<final_answer> content: return first integer found.
    2. Last \boxed{...} content: return first integer found.
    3. Last non-empty line: extract first integer token.
    Returns the integer as a string, or None if not found.
    """
    if not response:
        return None

    tag_matches = re.findall(r"<final_answer>\s*([^<\n]+?)\s*</final_answer>", response, re.IGNORECASE)
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

    for line in reversed([ln.strip() for ln in response.splitlines() if ln.strip()]):
        m = re.search(r"(-?\d+)", line)
        if m:
            return m.group(1)
    return None


# -------------------------------------------------------------
# Core evaluation
# -------------------------------------------------------------
def _normalize_label(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", (s or "")).upper()


# Global timestamp for all runs in this session (can be overridden by CLI)
_RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
# Base results directory (can be overridden by CLI)
RESULTS_ROOT = "results"


def _ensure_dirs(model_name: str) -> Tuple[str, str]:
    model_name = model_name.replace("/", "_")
    output_dir = os.path.join(RESULTS_ROOT, _RUN_TIMESTAMP, model_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, os.path.join(output_dir, "results.text")


def evaluate(
    dim: int,
    model_name: Optional[str] = None,
    model=None,
    tokenizer=None,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    do_sample: bool = True,
    top_k: int = 0,
    top_p: float = 0.9,
    temperature: float = 0.1,
    repetition_penalty: float = 1.1,
    reasoning: bool = True,
    questions_csv_path: str = "data/questions_augmented.csv",
    numeric_csv_path: str = "data/numeric_augmented.csv",
):
    """Evaluate model by `type` groups (multiple-choice) and numeric, save per-type materials.
    Returns the overall accuracy across all types (including numeric).
    """
    output_dir, result_path = _ensure_dirs(model_name or "model")
    csv_path = questions_csv_path
    # csv_path = "data/questions.csv"
    log_path = os.path.join(output_dir, f"dim_{dim}.log")

    # Build per-type dataset from questions.csv
    per_type_data: Dict[str, Dict[str, List[str]]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            col_key = f"{dim}D"
            if col_key not in row:
                continue
            if not row[col_key] or row[col_key].strip() == "-":
                continue
            question = row[col_key]
            raw_answer = (row.get("answer") or "").strip()
            type_key = (row.get("type") or "unknown").strip() or "unknown"
            per_type_data.setdefault(type_key, {"prompts": [], "answers": [], "rows": []})
            per_type_data[type_key]["prompts"].append(make_prompt_mc(question, type_key, reasoning=reasoning))
            per_type_data[type_key]["answers"].append(raw_answer)
            per_type_data[type_key]["rows"].append(row)

    # add numeric from data/numeric.csv as its own type: "numeric"
    numeric_path = numeric_csv_path
    # numeric_path = "data/numeric.csv"
    if os.path.exists(numeric_path):
        with open(numeric_path, "r", encoding="utf-8") as nf:
            nreader = csv.DictReader(nf)
            for row in nreader:
                try:
                    rdim = int(row.get("dimension", ""))
                except Exception:
                    continue
                if rdim != dim:
                    continue
                q = (row.get("question") or "").strip()
                ans = (row.get("answer") or "").strip()
                if not q or not ans:
                    continue
                per_type_data.setdefault("numeric", {"prompts": [], "answers": [], "rows": []})
                per_type_data["numeric"]["prompts"].append(make_prompt_numeric(q, reasoning=reasoning))
                per_type_data["numeric"]["answers"].append(ans)
                per_type_data["numeric"]["rows"].append(row)

    total = 0
    total_correct = 0
    per_type_stats: List[Tuple[str, int, int, float]] = []
    per_question_records: List[Dict[str, str]] = []

    with open(log_path, "w", encoding="utf-8") as log_file:
        # sort types, keeping "unknown" last
        for type_key in sorted(per_type_data.keys(), key=lambda x: (x == "unknown", x)):
            prompts = per_type_data[type_key]["prompts"]
            answers = per_type_data[type_key]["answers"]
            n = len(prompts)
            if n == 0:
                continue

            # derive labels for MC confusion matrix
            labels: List[str] = []
            if type_key != "numeric":
                seen = OrderedDict()
                for a in answers:
                    norm = _normalize_label(a)
                    if norm:
                        seen.setdefault(norm, None)
                labels = sorted(list(seen.keys())) or ["A", "B", "C"]
                confusion_matrix = np.zeros((len(labels), len(labels) + 1), dtype=int)
            else:
                confusion_matrix = None  # not used for numeric

            # Query model for this type
            responses = chat(
                prompts,
                model_name=model_name,
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )

            correct = 0
            rows = per_type_data[type_key]["rows"]
            for prompt, response, true_raw, row in zip(prompts, responses, answers, rows):
                log_file.write(f"Type: {type_key}\n")
                log_file.write(f"Question: {prompt}\n")
                log_file.write(f"Response: {response}\n")

                true_label = _normalize_label(true_raw)
                pred_label: Optional[str] = None
                predicted_raw = ""
                predicted_normalized = ""
                is_correct = False

                if type_key == "numeric":
                    predicted_num = extract_numeric(response)
                    predicted_raw = predicted_num or ""
                    predicted_normalized = predicted_num or ""
                    log_file.write(f"Predicted (raw numeric): {predicted_num}\n")
                    log_file.write(f"Correct (raw numeric): {true_raw}\n\n")
                    if predicted_num is not None:
                        try:
                            if int(predicted_num) == int(re.sub(r"[^0-9\-]", "", str(true_raw))):
                                correct += 1
                                is_correct = True
                        except Exception:
                            pass
                else:
                    predicted = extract_answer(response)
                    predicted_raw = predicted or ""
                    log_file.write(f"Predicted (raw): {predicted}\n")
                    log_file.write(f"Correct (raw): {true_raw}\n\n")
                    if predicted is not None:
                        pred_alpha = predicted.strip().upper()
                        if true_label.isdigit():
                            # map A->1, B->2, ...
                            idx = ord(pred_alpha) - ord("A")
                            if 0 <= idx < 26:
                                pred_label = str(idx + 1)
                            else:
                                pred_label = pred_alpha
                        else:
                            pred_label = pred_alpha

                    predicted_normalized = pred_label or ""
                    if pred_label is not None and true_label == pred_label:
                        correct += 1
                        is_correct = True

                    # fill confusion matrix for MC types
                    if confusion_matrix is not None:
                        if true_label in labels:
                            t_idx = labels.index(true_label)
                            if pred_label in labels:
                                p_idx = labels.index(pred_label)
                                confusion_matrix[t_idx, p_idx] += 1
                            else:
                                confusion_matrix[t_idx, -1] += 1

                if type_key == "numeric":
                    question_text = (row.get("question") or "").strip()
                    source_file = numeric_path
                else:
                    question_text = (row.get(f"{dim}D") or "").strip()
                    source_file = csv_path

                per_question_records.append(
                    {
                        "dimension": str(dim),
                        "type": type_key,
                        "source_file": source_file,
                        "question": question_text,
                        "ground_truth_raw": str(true_raw),
                        "ground_truth_normalized": true_label,
                        "predicted_raw": predicted_raw,
                        "predicted_normalized": predicted_normalized,
                        "is_correct": "1" if is_correct else "0",
                        "response": response,
                    }
                )

            # Collect per-type stats
            accuracy = correct / n if n > 0 else 0
            total += n
            total_correct += correct
            per_type_stats.append((type_key, correct, n, accuracy))

            # Save per-type confusion matrix plot (MC only)
            if confusion_matrix is not None:
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
                plt.title(f"Confusion Matrix - {model_name} - {dim}D - type {type_key}")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"confusion_matrix_{dim}d_type_{type_key}.png"))
                plt.close()

    # overall accuracy across types (including numeric)
    overall_accuracy = total_correct / total if total > 0 else 0

    # Save per-question judgments for post-analysis.
    per_question_path = os.path.join(output_dir, f"dim_{dim}_per_question.csv")
    correct_path = os.path.join(output_dir, f"dim_{dim}_correct.csv")
    incorrect_path = os.path.join(output_dir, f"dim_{dim}_incorrect.csv")
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
        "response",
    ]

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

    # Pretty-print results for this dimension into results.text
    with open(result_path, "a", encoding="utf-8") as result_file:
        result_file.write("\n")
        result_file.write("========================================\n")
        result_file.write(f"Model: {model_name} \nDimension: {dim}D\n")
        result_file.write("----------------------------------------\n")
        result_file.write(f"{'Type':<12}{'Correct':>8}{'Total':>8}{'Accuracy':>12}\n")
        result_file.write("----------------------------------------\n")
        for tkey, c, n, acc in per_type_stats:
            result_file.write(f"{tkey:<12}{c:8d}{n:8d}{acc:12.2%}\n")
        result_file.write("----------------------------------------\n")
        result_file.write(f"{'Overall':<12}{total_correct:8d}{total:8d}{overall_accuracy:12.2%}\n")
        result_file.write("========================================\n")

    return overall_accuracy



# -------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on geometry dataset using vLLM or Azure OpenAI.")
    parser.add_argument("--models", type=str, help="Comma-separated model names to evaluate (default: built-in list)")
    parser.add_argument("--dims", type=str, default="2", help="Comma-separated dimensions to evaluate (default: 2)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for local generation")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens to generate per prompt")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (do_sample=False)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p for sampling")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k for sampling (0 means disabled)")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--dtype", type=str, default="float16", help="Preferred dtype for vLLM local models: float16|bfloat16|float32")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7, help="GPU memory utilization fraction for vLLM")
    parser.add_argument("--no-reasoning", action="store_true", help="Disable chain-of-thought reasoning prompts")
    parser.add_argument("--results-root", type=str, default=RESULTS_ROOT, help="Base directory to write results into")
    parser.add_argument("--timestamp", type=str, default=None, help="Timestamp string to use for this run (default: now)")
    parser.add_argument(
        "--questions-csv",
        type=str,
        default="data/questions_augmented.csv",
        help="Path to multiple-choice questions CSV",
    )
    parser.add_argument(
        "--numeric-csv",
        type=str,
        default="data/numeric_augmented.csv",
        help="Path to numeric questions CSV",
    )
    args = parser.parse_args()

    # default model list (kept for backwards compatibility)
    default_models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-Math-7B-Instruct",
        "google/gemma-2-9b-it",
        "meta-llama/Llama-3.1-8B",
        "gpt-4o",
        "gpt-4o-mini",
        "o1",
    ]

    # configure globals from args
    if args.timestamp:
        try:
            # simple validation: must be digits and underscores
            _ = str(args.timestamp)
            globals()["_RUN_TIMESTAMP"] = args.timestamp
        except Exception:
            print("Invalid timestamp provided; using default.", file=sys.stderr)
    globals()["RESULTS_ROOT"] = args.results_root

    if args.models:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        model_names = default_models

    dims = [int(d.strip()) for d in args.dims.split(",") if d.strip()]
    reasoning = not args.no_reasoning

    for model_name in model_names:
        output_dir, result_path = _ensure_dirs(model_name)
        # Truncate results file at the start of this model's run so re-runs overwrite previous results
        with open(result_path, "w", encoding="utf-8") as _rf:
            _rf.write("")

        for dim in dims:
            print(f"Evaluating {model_name} for {dim} dimensions...")
            model = None
            # Use Azure for hosted models, otherwise try to load local vLLM
            if model_name not in ["gpt-4o", "gpt-4o-mini", "o1", "gpt-5"]:
                preferred_dtype = getattr(args, "dtype", "float16")
                gpu_mem_util = getattr(args, "gpu_memory_utilization", 0.7)
                # Build a list of dtype candidates to try; prefer user choice but
                # fall back to bfloat16/float32 if the model complains about float16.
                tried = []
                if preferred_dtype == "float16":
                    dtype_candidates = ["float16", "bfloat16", "float32"]
                elif preferred_dtype == "bfloat16":
                    dtype_candidates = ["bfloat16", "float32"]
                else:
                    dtype_candidates = [preferred_dtype, "float32"]

                last_exc = None
                for dt in dtype_candidates:
                    try:
                        model = LLM(model=model_name, dtype=dt, gpu_memory_utilization=gpu_mem_util, disable_log_stats=True)
                        break
                    except Exception as e:
                        last_exc = e
                        msg = str(e).lower()
                        # If the model specifically rejects float16, continue to next candidate
                        if "does not support float16" in msg or "does not support bfloat16" in msg or "numerical instability" in msg:
                            continue
                        # otherwise, stop trying and surface the error
                        model = None
                        break
                if model is None and last_exc is not None:
                    print(f"Warning: could not initialize vLLM for {model_name}: {last_exc}", file=sys.stderr)

            evaluate(
                dim,
                model_name=model_name,
                model=model,
                tokenizer=None,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                do_sample=not args.greedy,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                reasoning=reasoning,
                questions_csv_path=args.questions_csv,
                numeric_csv_path=args.numeric_csv,
            )

            # free GPU/CPU resources
            try:
                del model
            except Exception:
                pass
            import gc

            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
