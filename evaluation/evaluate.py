import csv
import os
import re
import math
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from openai import AzureOpenAI
from vllm import LLM, SamplingParams
import argparse
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from common.prompting import (
    apply_chat_template as _apply_chat_template,
    set_reasoning_effort as _set_reasoning_effort,
    make_prompt_mc_variants,
    make_prompt_numeric,
    make_prompt_numeric_mc_variants,
    remap_answer_for_rotation,
    _choice_count_for_type,
    _parse_option_choices,
    options as _option_specs,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def _resolve_data_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


# -------------------------------------------------------------
# Model client
# -------------------------------------------------------------
def _build_azure_client() -> AzureOpenAI:
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "")
    return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)


def _vllm_token_logprobs(req_out) -> Optional[List[Tuple[str, float]]]:
    """Per-generated-token (decoded_token, logprob) for a vLLM RequestOutput.

    Returns the logprob of each *sampled* token (not the top-k alternatives),
    so downstream code can read the probability the model assigned to whichever
    token it emitted as its answer. None if logprobs are unavailable.
    """
    try:
        out = req_out.outputs[0]
        lps = out.logprobs
        tids = out.token_ids
        if not lps or tids is None:
            return None
        pairs: List[Tuple[str, float]] = []
        for pos, tid in enumerate(tids):
            entry = lps[pos].get(tid) if pos < len(lps) else None
            if entry is None:
                continue
            tok = getattr(entry, "decoded_token", None)
            pairs.append((tok if tok is not None else "", float(entry.logprob)))
        return pairs or None
    except Exception:
        return None


# Number of top logprobs requested from vLLM (we only need the sampled token's
# logprob; a small top-k is kept cheap and enables richer offline analysis).
_LOGPROBS_TOPK = 20


@torch.no_grad()
def chat(
    prompts: List[str],
    model_name: Optional[str] = None,
    model=None,
    tokenizer=None,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    top_k: int = 0,
    top_p: float = 0.9,
    temperature: float = 0.1,
    repetition_penalty: float = 1.0,
    desc: str = "",
) -> Tuple[List[str], List[Optional[List[Tuple[str, float]]]]]:
    """Returns (responses, token_logprobs). token_logprobs[i] is the per-token
    (decoded_token, logprob) list for response i under the vLLM path, or None
    (HF / Azure paths, or on failure)."""
    responses: List[str] = []
    token_lps: List[Optional[List[Tuple[str, float]]]] = []

    def _strip_prompt_echo(prompt: str, text: str) -> str:
        if prompt and text.startswith(prompt):
            text = text[len(prompt):]
        return text.strip()

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
            content = None
            try:
                content = response.choices[0].message.content.strip()
            except Exception:
                try:
                    content = response["choices"][0]["message"]["content"].strip()
                except Exception:
                    try:
                        content = str(response).strip()
                    except Exception:
                        content = ""
            responses.append(content)
            token_lps.append(None)  # logprobs not available via Azure chat API
        return responses, token_lps

    if len(prompts) == 0:
        return responses, token_lps

    # Determine device for HF models; vLLM handles its own device management.
    try:
        first_param = next(model.parameters())
        device = first_param.device
    except Exception:
        device = getattr(model, "device", torch.device("cpu"))

    # vLLM: top_k=0 means "sample from 0 tokens" and causes empty output.
    # Use -1 to disable top-k filtering instead.
    vllm_top_k = top_k if top_k > 0 else -1

    # Greedy decoding: vLLM (unlike HF) ignores do_sample and selects greedily
    # only when temperature < ~1e-5, so force temperature=0 and neutralize
    # top_p/top_k. raw_logprobs (logprobs_mode default) are computed from the
    # pre-temperature logits, so per-token confidence is unaffected by this.
    if not do_sample:
        temperature = 0.0
        top_p = 1.0
        vllm_top_k = -1

    batch_num = (len(prompts) + batch_size - 1) // batch_size

    for i in tqdm(range(batch_num), desc=desc or "batches", unit="batch", leave=False):
        batch_prompts = prompts[i * batch_size: (i + 1) * batch_size]
        batch_lps: List[Optional[List[Tuple[str, float]]]] = [None] * len(batch_prompts)

        if isinstance(model, LLM):
            try:
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=vllm_top_k,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_new_tokens,
                    min_tokens=1,
                    logprobs=_LOGPROBS_TOPK,
                )
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
                            top_k=vllm_top_k,
                            repetition_penalty=repetition_penalty,
                            max_tokens=max(32, max_new_tokens // 4),
                            min_tokens=1,  # Fix: was missing in OOM fallback
                            logprobs=_LOGPROBS_TOPK,
                        )
                        outputs = model.generate(batch_prompts, sampling_params=sampling_params, use_tqdm=False)
                    except Exception:
                        outputs = None
                else:
                    raise

            if outputs is None:
                batch_responses = ["" for _ in batch_prompts]
                allow_retry = False
            else:
                batch_responses = []
                lps_list: List[Optional[List[Tuple[str, float]]]] = []
                for req_out in outputs:
                    try:
                        text = "".join([c.text for c in req_out.outputs])
                    except Exception:
                        text = getattr(req_out, "prompt", "") or ""
                    # vLLM returns only the generated text (not the prompt),
                    # so _strip_prompt_echo is not needed here.
                    batch_responses.append(text.strip())
                    lps_list.append(_vllm_token_logprobs(req_out))
                batch_lps = lps_list
                allow_retry = True
        else:
            # Strip <think> from assistant prefix so decoder-only models
            # generate direct answers rather than extended reasoning chains.
            clean_batch = [p.replace("\n<think>\n", "\n") for p in batch_prompts]
            inputs = tokenizer(clean_batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=1,
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
                allow_retry = False
            else:
                input_length = inputs["input_ids"].shape[1]
                new_tokens = outputs[:, input_length:]
                batch_responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                batch_responses = [r.strip() for r in batch_responses]
                allow_retry = True

        if allow_retry:
            empty_idxs = [idx for idx, text in enumerate(batch_responses) if not text.strip()]
            if empty_idxs:
                retry_prompts_raw = [batch_prompts[idx] for idx in empty_idxs]
                if isinstance(model, LLM):
                    retry_params = SamplingParams(
                        temperature=0.0,
                        top_p=1.0,
                        top_k=-1,  # Fix: use -1 (disabled) instead of 0
                        repetition_penalty=repetition_penalty,
                        max_tokens=max(8, min(64, max_new_tokens)),
                        min_tokens=1,
                        logprobs=_LOGPROBS_TOPK,
                    )
                    try:
                        retry_outputs = model.generate(retry_prompts_raw, sampling_params=retry_params, use_tqdm=False)
                    except Exception:
                        retry_outputs = None
                    if retry_outputs is not None:
                        for rel_idx, req_out in enumerate(retry_outputs):
                            try:
                                text = "".join([c.text for c in req_out.outputs])
                            except Exception:
                                text = getattr(req_out, "prompt", "") or ""
                            cleaned = text.strip()
                            if cleaned:
                                batch_responses[empty_idxs[rel_idx]] = cleaned
                                batch_lps[empty_idxs[rel_idx]] = _vllm_token_logprobs(req_out)
                else:
                    retry_inputs = tokenizer(retry_prompts_raw, return_tensors="pt", padding=True, truncation=True)
                    retry_inputs = {k: v.to(device) for k, v in retry_inputs.items()}
                    try:
                        retry_outputs = model.generate(
                            **retry_inputs,
                            max_new_tokens=max(8, min(64, max_new_tokens)),
                            min_new_tokens=1,
                            do_sample=False,
                            top_k=None,
                            top_p=1.0,
                            temperature=0.0,
                            repetition_penalty=repetition_penalty,
                        )
                    except Exception:
                        retry_outputs = None
                    if retry_outputs is not None:
                        retry_texts = tokenizer.batch_decode(retry_outputs, skip_special_tokens=True)
                        for rel_idx, (prompt, text) in enumerate(zip(retry_prompts_raw, retry_texts)):
                            cleaned = _strip_prompt_echo(prompt, text)
                            if cleaned:
                                batch_responses[empty_idxs[rel_idx]] = cleaned
        responses.extend(batch_responses)
        token_lps.extend(batch_lps)
    return responses, token_lps


# -------------------------------------------------------------
# Extractors
# -------------------------------------------------------------
def extract_answer(response: str) -> Optional[str]:
    """Extract the model's answer letter (A-E) from the response, preferring the last match.
    Prioritizes the last `<answer>...</answer>`; if none,
    falls back to the last `\\boxed{...}` match. Returns the uppercase letter A-E or None.
    """
    if not response:
        return None

    # gpt-oss harmony format: the committed answer lives in the 'final' channel,
    # which decodes (special tokens stripped) as '...analysis...assistantfinal<answer>'.
    # Parse only the final-channel tail so the analysis text is ignored.
    harmony = re.search(r"assistantfinal\s*(.+)$", response, re.IGNORECASE | re.DOTALL)
    if harmony:
        response = harmony.group(1).strip()

    tag_matches = re.findall(r"<answer>\s*([^<\n]+?)\s*</answer>", response, re.IGNORECASE)
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

    # Quoted letter: “B” or 'B' at the start of the response (e.g. Qwen3 style)
    quoted_match = re.match(r'[“””\']\s*([A-E])\s*[“””\']', response.strip(), re.IGNORECASE)
    if quoted_match:
        return quoted_match.group(1).upper()

    # Letter in parentheses: (B)
    paren_match = re.search(r'\(([A-E])\)', response, re.IGNORECASE)
    if paren_match:
        return paren_match.group(1).upper()

    # Letter at start of response followed by space/period (e.g. “A Perpendicular.”)
    start_match = re.match(r'^([A-E])[\s.,]', response.strip(), re.IGNORECASE)
    if start_match:
        return start_match.group(1).upper()

    simple_token = re.search(r"^\s*([A-E])\s*$", response, re.IGNORECASE | re.MULTILINE)
    if simple_token:
        return simple_token.group(1).upper()
    return None


def extract_numeric(response: str) -> Optional[str]:
    """Extract a numeric answer from response.
    Priority:
    1. Last <answer>...<answer> content: return first integer found.
    2. Last \\boxed{...} content: return first integer found.
    3. Last non-empty line: extract first integer token.
    Returns the integer as a string, or None if not found.
    """
    if not response:
        return None

    # gpt-oss harmony: prefer the 'final' channel tail (after 'assistantfinal').
    harmony = re.search(r"assistantfinal\s*(.+)$", response, re.IGNORECASE | re.DOTALL)
    if harmony:
        response = harmony.group(1).strip()

    tag_matches = re.findall(r"<answer>\s*([^<\n]+?)\s*</answer>", response, re.IGNORECASE)
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


# -------------------------------------------------------------
# Core evaluation
# -------------------------------------------------------------
def _normalize_label(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", (s or "")).upper()


def _answer_confidence(
    token_lps: Optional[List[Tuple[str, float]]],
    predicted: Optional[str],
    is_numeric: bool,
) -> Optional[float]:
    """Probability the model assigned to the answer token it emitted.

    - multiple choice (letter): the first generated token whose first alpha char
      is the predicted letter → exp(logprob).
    - numeric (free-form): the first generated token that starts the predicted
      number (first token containing a digit) → exp(logprob). This is the
      first-token probability, not the joint probability of the whole number.
    Returns None when logprobs or a matching token are unavailable.
    """
    if not token_lps or not predicted:
        return None
    if is_numeric:
        for tok, lp in token_lps:
            if any(ch.isdigit() for ch in (tok or "")):
                return math.exp(lp)
        return None
    letter = predicted.strip().upper()[:1]
    if not letter.isalpha():
        return None
    for tok, lp in token_lps:
        alphas = [c for c in (tok or "").upper() if c.isalpha()]
        if alphas and alphas[0] == letter:
            return math.exp(lp)
    return None


# Global timestamp for all runs in this session (can be overridden by CLI)
_RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
# Base results directory (can be overridden by CLI)
RESULTS_ROOT = "results/eval"


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
    do_sample: bool = False,
    top_k: int = 0,
    top_p: float = 0.9,
    temperature: float = 0.1,
    repetition_penalty: float = 1.0,
    reasoning: bool = True,
    prompt_type: Optional[str] = None,
    questions_csv_path: str = "data/questions_augmented.csv",
    numeric_csv_path: str = "data/numeric_augmented.csv",
):
    """Evaluate model by `type` groups (multiple-choice) and numeric, save per-type materials.
    Returns the overall accuracy across all types (including numeric).
    """
    output_dir, result_path = _ensure_dirs(model_name or "model")
    prompt_key = prompt_type or ("with_reasoning" if reasoning else "without_reasoning")
    csv_path = _resolve_data_path(questions_csv_path)
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
            per_type_data.setdefault(
                type_key,
                {"prompts": [], "answers": [], "rows": [], "rotations": [], "num_choices": [], "canon_answers": []},
            )
            variants = make_prompt_mc_variants(question, type_key, reasoning=prompt_key)
            for variant in variants:
                remapped = remap_answer_for_rotation(
                    raw_answer,
                    int(variant["rotation"]),
                    int(variant["num_choices"]),
                )
                per_type_data[type_key]["prompts"].append(str(variant["prompt"]))
                per_type_data[type_key]["answers"].append(remapped)
                per_type_data[type_key]["rows"].append(row)
                # Keep rotation / choice count / canonical answer so the confusion
                # matrix can be tallied in the canonical (pre-rotation, semantic)
                # frame rather than the presented-slot frame.
                per_type_data[type_key]["rotations"].append(int(variant["rotation"]))
                per_type_data[type_key]["num_choices"].append(int(variant["num_choices"]))
                per_type_data[type_key]["canon_answers"].append(raw_answer)

    # add numeric from data/numeric.csv as its own type: "numeric"
    numeric_path = _resolve_data_path(numeric_csv_path)
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
                per_type_data.setdefault(
                    "numeric",
                    {"prompts": [], "answers": [], "rows": [], "rotations": [], "num_choices": [], "canon_answers": []},
                )
                per_type_data["numeric"]["prompts"].append(make_prompt_numeric(q, reasoning=prompt_key))
                per_type_data["numeric"]["answers"].append(ans)
                per_type_data["numeric"]["rows"].append(row)
                per_type_data["numeric"]["rotations"].append(0)
                per_type_data["numeric"]["num_choices"].append(0)
                per_type_data["numeric"]["canon_answers"].append(ans)

                # Also build a multiple-choice version of the same numeric
                # question: correct value + 3 distractors, presented in all
                # cyclic rotations. The canonical correct slot is "A" (the
                # value sits at index 0 before rotation). Skipped silently if
                # distractors can't be built (non-integer answer).
                mc_variants = make_prompt_numeric_mc_variants(q, ans, reasoning=prompt_key)
                if mc_variants:
                    per_type_data.setdefault(
                        "numeric_mc",
                        {"prompts": [], "answers": [], "rows": [], "rotations": [], "num_choices": [], "canon_answers": []},
                    )
                    for v in mc_variants:
                        rot = int(v["rotation"])
                        nch = int(v["num_choices"])
                        remapped = remap_answer_for_rotation("A", rot, nch)
                        per_type_data["numeric_mc"]["prompts"].append(str(v["prompt"]))
                        per_type_data["numeric_mc"]["answers"].append(remapped)
                        per_type_data["numeric_mc"]["rows"].append(row)
                        per_type_data["numeric_mc"]["rotations"].append(rot)
                        per_type_data["numeric_mc"]["num_choices"].append(nch)
                        per_type_data["numeric_mc"]["canon_answers"].append("A")

    total = 0
    total_correct = 0
    total_empty = 0
    overall_conf_sum = 0.0
    overall_conf_n = 0
    per_type_stats: List[Tuple[str, int, int, float]] = []
    per_question_records: List[Dict[str, str]] = []

    type_keys = sorted(per_type_data.keys(), key=lambda x: (x == "unknown", x))
    with open(log_path, "w", encoding="utf-8") as log_file:
        for type_key in tqdm(type_keys, desc=f"dim{dim}", unit="type"):
            prompts = per_type_data[type_key]["prompts"]
            answers = per_type_data[type_key]["answers"]
            n = len(prompts)
            if n == 0:
                continue

            labels: List[str] = []
            label_meanings: List[str] = []
            if type_key not in ("numeric", "numeric_mc"):
                # Canonical (pre-rotation) label set for this type. The confusion
                # matrix is built in this semantic frame, so a cell (i, j) means
                # "true meaning i was predicted as meaning j" regardless of which
                # letter slot each meaning occupied in a given rotated variant.
                nch = _choice_count_for_type(type_key)
                if nch <= 0:
                    nch = 3
                labels = [chr(ord("A") + i) for i in range(nch)]
                try:
                    idx_opt = max(0, min(int(type_key) - 1, len(_option_specs) - 1))
                    meanings = _parse_option_choices(_option_specs[idx_opt])
                except Exception:
                    meanings = []
                label_meanings = [
                    f"{labels[i]}. {meanings[i]}" if i < len(meanings) else labels[i]
                    for i in range(nch)
                ]
                confusion_matrix = np.zeros((len(labels), len(labels) + 1), dtype=int)
            else:
                confusion_matrix = None

            formatted_prompts = [_apply_chat_template(p, model_name) for p in prompts]
            responses, response_logprobs = chat(
                formatted_prompts,
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
                desc=f"{dim}D/{type_key}",
            )

            correct = 0
            empty_n = 0             # predictions with no parseable answer
            conf_sum = 0.0          # mean confidence over questions with a logprob
            conf_n = 0
            conf_correct_sum = 0.0
            conf_correct_n = 0
            conf_incorrect_sum = 0.0
            conf_incorrect_n = 0
            rows = per_type_data[type_key]["rows"]
            rotations = per_type_data[type_key]["rotations"]
            nchoices_list = per_type_data[type_key]["num_choices"]
            canon_answers = per_type_data[type_key]["canon_answers"]
            for prompt, formatted_prompt, response, token_lps, true_raw, row, rot, nch_row, canon_ans in zip(
                prompts, formatted_prompts, responses, response_logprobs, answers, rows, rotations, nchoices_list, canon_answers
            ):
                log_file.write(f"Type: {type_key}\n")
                log_file.write(f"Question: {prompt}\n")
                log_file.write(f"Response: {response}\n")

                true_label = _normalize_label(true_raw)
                pred_label: Optional[str] = None
                predicted_raw = ""
                predicted_normalized = ""
                is_correct = False
                is_empty = False
                confidence: Optional[float] = None

                if type_key == "numeric":
                    predicted_num = extract_numeric(response)
                    predicted_raw = predicted_num or ""
                    predicted_normalized = predicted_num or ""
                    is_empty = predicted_num is None
                    confidence = _answer_confidence(token_lps, predicted_num, is_numeric=True)
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
                    is_empty = predicted is None
                    confidence = _answer_confidence(token_lps, predicted, is_numeric=False)
                    log_file.write(f"Predicted (raw): {predicted}\n")
                    log_file.write(f"Correct (raw): {true_raw}\n\n")
                    if predicted is not None:
                        pred_alpha = predicted.strip().upper()
                        if true_label.isdigit():
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

                    if confusion_matrix is not None and nch_row > 0:
                        # Map both the true answer and the prediction back to the
                        # canonical (pre-rotation) frame so the matrix reflects
                        # semantic confusion, not letter-slot position.
                        true_canon = _normalize_label(canon_ans)
                        pred_canon: Optional[str] = None
                        if pred_label is not None and len(pred_label) == 1 and pred_label.isalpha():
                            pidx = ord(pred_label) - ord("A")
                            if 0 <= pidx < nch_row:
                                pred_canon = remap_answer_for_rotation(
                                    pred_label, (nch_row - rot) % nch_row, nch_row
                                )
                        if true_canon in labels:
                            t_idx = labels.index(true_canon)
                            if pred_canon in labels:
                                p_idx = labels.index(pred_canon)
                                confusion_matrix[t_idx, p_idx] += 1
                            else:
                                confusion_matrix[t_idx, -1] += 1

                if type_key in ("numeric", "numeric_mc"):
                    question_text = (row.get("question") or "").strip()
                    source_file = numeric_path
                else:
                    question_text = (row.get(f"{dim}D") or "").strip()
                    source_file = csv_path

                if is_empty:
                    empty_n += 1
                if confidence is not None:
                    conf_sum += confidence
                    conf_n += 1
                    if is_correct:
                        conf_correct_sum += confidence
                        conf_correct_n += 1
                    else:
                        conf_incorrect_sum += confidence
                        conf_incorrect_n += 1

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
                        "confidence": f"{confidence:.6f}" if confidence is not None else "",
                        "input": formatted_prompt,
                        "response": response,
                    }
                )

            accuracy = correct / n if n > 0 else 0
            conf_mean = conf_sum / conf_n if conf_n > 0 else None
            conf_correct_mean = conf_correct_sum / conf_correct_n if conf_correct_n > 0 else None
            conf_incorrect_mean = conf_incorrect_sum / conf_incorrect_n if conf_incorrect_n > 0 else None
            total += n
            total_correct += correct
            total_empty += empty_n
            overall_conf_sum += conf_sum
            overall_conf_n += conf_n
            per_type_stats.append(
                {
                    "type": type_key,
                    "n": n,
                    "correct": correct,
                    "acc": accuracy,
                    "empty": empty_n / n if n > 0 else 0.0,
                    "conf": conf_mean,
                    "conf_ok": conf_correct_mean,
                    "conf_no": conf_incorrect_mean,
                }
            )

            if confusion_matrix is not None:
                plt.figure(figsize=(8, 6))
                row_sums = np.sum(confusion_matrix, axis=1, keepdims=True)
                with np.errstate(invalid="ignore", divide="ignore"):
                    cm_norm = confusion_matrix / row_sums
                    cm_norm = np.nan_to_num(cm_norm)
                pred_labels = label_meanings + ["Other / no answer"]
                plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
                plt.xlabel("Predicted (canonical meaning)")
                plt.ylabel("True (canonical meaning)")
                plt.xticks(np.arange(len(pred_labels)), pred_labels, rotation=45, ha="right")
                plt.yticks(np.arange(len(label_meanings)), label_meanings)
                plt.title(f"Confusion Matrix (semantic) - {model_name} - {dim}D - type {type_key}")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"confusion_matrix_{dim}d_type_{type_key}.png"))
                plt.close()

    overall_accuracy = total_correct / total if total > 0 else 0

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
        "confidence",
        "input",
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

    type_label = {"1": "PPC (1)", "2": "IC (2)", "3": "CC (3)"}
    overall_conf = overall_conf_sum / overall_conf_n if overall_conf_n > 0 else None
    overall_empty = total_empty / total if total > 0 else 0.0
    SEP, SUB = "=" * 72, "-" * 72

    def _cf(v):
        return f"{v:.3f}" if v is not None else "    -"

    with open(result_path, "a", encoding="utf-8") as result_file:
        result_file.write("\n" + SEP + "\n")
        result_file.write(f" {model_name}   ·   {dim}D\n")
        result_file.write(SUB + "\n")
        result_file.write(
            f" {'Type':<11}{'N':>6}{'Correct':>9}{'Acc':>9}{'Empty':>8}"
            f"{'Conf':>8}{'Conf✓':>8}{'Conf✗':>8}\n"
        )
        result_file.write(SUB + "\n")
        for s in per_type_stats:
            result_file.write(
                f" {type_label.get(s['type'], s['type']):<11}{s['n']:>6}{s['correct']:>9}"
                f"{s['acc']:>9.2%}{s['empty']:>8.1%}"
                f"{_cf(s['conf']):>8}{_cf(s['conf_ok']):>8}{_cf(s['conf_no']):>8}\n"
            )
        result_file.write(SUB + "\n")
        result_file.write(
            f" {'Overall':<11}{total:>6}{total_correct:>9}{overall_accuracy:>9.2%}"
            f"{overall_empty:>8.1%}{_cf(overall_conf):>8}\n"
        )
        result_file.write(SUB + "\n")
        result_file.write(
            " Acc = accuracy   Empty = share with no parseable answer\n"
            " Conf = mean P(model's chosen answer token); Conf✓/Conf✗ = on correct/incorrect\n"
            " (vLLM logprobs at the run's sampling temperature; blank = unavailable)\n"
        )
        result_file.write(SEP + "\n")

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
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=True, help="Greedy decoding (default). Use --no-greedy to sample at --temperature.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p for sampling")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k for sampling (0 means disabled)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty (default 1.0; 1.1 caused MC primacy aversion / A-slot underestimate — see summary)")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Preferred dtype for vLLM local models: float16|bfloat16|float32")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7, help="GPU memory utilization fraction for vLLM")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs for tensor parallelism in vLLM")
    parser.add_argument("--max-model-len", type=int, default=None, help="Override max model sequence length for vLLM (reduces KV cache usage)")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile/CUDA graphs in vLLM (workaround for unstable MXFP4/MoE kernels)")
    parser.add_argument("--no-reasoning", action="store_true", help="Disable chain-of-thought reasoning prompts")
    parser.add_argument(
        "--prompt-type",
        type=str,
        default=None,
        choices=["with_reasoning", "without_reasoning", "simple_prompt", "simple_prompt_strict"],
        help="Override prompt style (default: based on --no-reasoning)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help="Reasoning-effort hint for chat templates that support it (e.g. gpt-oss harmony)",
    )
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

    default_models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-Math-7B-Instruct",
        "google/gemma-2-9b-it",
        "meta-llama/Llama-3.1-8B",
        "gpt-4o",
        "gpt-4o-mini",
        "o1",
    ]

    if args.timestamp:
        try:
            _ = str(args.timestamp)
            globals()["_RUN_TIMESTAMP"] = args.timestamp
        except Exception:
            print("Invalid timestamp provided; using default.", file=sys.stderr)
    globals()["RESULTS_ROOT"] = args.results_root
    _set_reasoning_effort(getattr(args, "reasoning_effort", None))

    if args.models:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        model_names = default_models

    dims = [int(d.strip()) for d in args.dims.split(",") if d.strip()]
    reasoning = not args.no_reasoning
    prompt_type = args.prompt_type

    for model_name in model_names:
        output_dir, result_path = _ensure_dirs(model_name)
        with open(result_path, "w", encoding="utf-8") as _rf:
            _rf.write("")

        for dim in tqdm(dims, desc=model_name.split("/")[-1], unit="dim"):
            print(f"Evaluating {model_name} for {dim} dimensions...")
            model = None
            tokenizer_obj = None
            if model_name not in ["gpt-4o", "gpt-4o-mini", "o1", "gpt-5"]:
                preferred_dtype = getattr(args, "dtype", "bfloat16")
                gpu_mem_util = getattr(args, "gpu_memory_utilization", 0.7)
                # Prefer bfloat16 by default; Gemma-2 is numerically unstable with float16.
                if preferred_dtype == "float16":
                    dtype_candidates = ["float16", "bfloat16", "float32"]
                elif preferred_dtype == "bfloat16":
                    dtype_candidates = ["bfloat16", "float32"]
                else:
                    dtype_candidates = [preferred_dtype, "float32"]

                last_exc = None
                for dt in dtype_candidates:
                    try:
                        tensor_parallel_size = getattr(args, "tensor_parallel_size", 1)
                        max_model_len = getattr(args, "max_model_len", None)
                        llm_kwargs = dict(dtype=dt, gpu_memory_utilization=gpu_mem_util, disable_log_stats=True, tensor_parallel_size=tensor_parallel_size)
                        if max_model_len is not None:
                            llm_kwargs["max_model_len"] = max_model_len
                        if getattr(args, "enforce_eager", False):
                            llm_kwargs["enforce_eager"] = True
                        model = LLM(model=model_name, **llm_kwargs)
                        break
                    except Exception as e:
                        last_exc = e
                        msg = str(e).lower()
                        if "does not support float16" in msg or "does not support bfloat16" in msg or "numerical instability" in msg:
                            continue
                        model = None
                        break
                if model is None and last_exc is not None:
                    print(f"Warning: could not initialize vLLM for {model_name}: {last_exc}", file=sys.stderr)
                    # Fallback to HuggingFace Transformers
                    try:
                        from transformers import AutoModelForCausalLM, AutoTokenizer
                        print(f"Trying HuggingFace Transformers fallback for {model_name}...", file=sys.stderr)
                        tokenizer_obj = AutoTokenizer.from_pretrained(model_name)
                        tokenizer_obj.padding_side = "left"
                        if tokenizer_obj.pad_token_id is None:
                            tokenizer_obj.pad_token_id = tokenizer_obj.eos_token_id
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name, torch_dtype=torch.bfloat16, device_map="auto"
                        )
                        print(f"HuggingFace Transformers fallback succeeded for {model_name}", file=sys.stderr)
                    except Exception as hf_exc:
                        print(f"Warning: HuggingFace fallback also failed for {model_name}: {hf_exc}", file=sys.stderr)
                        model = None
                        tokenizer_obj = None

                # If both vLLM and HF failed for a local model, skip rather than
                # falling through to the Azure API path with a wrong deployment name.
                if model is None:
                    print(f"Skipping {model_name} dim {dim}: no local model available.", file=sys.stderr)
                    continue

            evaluate(
                dim,
                model_name=model_name,
                model=model,
                tokenizer=tokenizer_obj,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                do_sample=not args.greedy,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                reasoning=reasoning,
                prompt_type=prompt_type,
                questions_csv_path=args.questions_csv,
                numeric_csv_path=args.numeric_csv,
            )

            try:
                del model
                del tokenizer_obj
            except Exception:
                pass
            import gc
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
