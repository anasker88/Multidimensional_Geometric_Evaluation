import re
import sys
from typing import Dict, Optional

# -------------------------------------------------------------
# Prompt definitions (aligned with evaluate.py)
# -------------------------------------------------------------
prelude_explanations = {
    "with_reasoning": """You are evaluating a multiple-choice geometry question.
You may think step-by-step and output your full reasoning.
After your reasoning, you MUST output the final answer in the format:
<answer>A</answer>
where A is one of {choices}.
Nothing else should appear inside <answer> tags.
Question:
""",
    "without_reasoning": """You are evaluating a multiple-choice geometry question.
You MUST output the final answer in the format:
<answer>A</answer>
where A is one of {choices}.
Nothing else should appear inside <answer> tags.
Question:
""",
    "simple_prompt": """System: You are a helpful assistant for solving geometry problems.
User: Answer the following geometry question by selecting one option. Begin your response with the option letter only (A, B, C, or D).
""",
    "simple_prompt_strict": """System: You are a geometry problem solver. You MUST always select one of the listed answer choices — never say the answer is unavailable or not among the options. If unsure, pick the closest option.
User: Answer the geometry question below by choosing the best option from those listed. Output ONLY the single letter of your choice — no explanation, no punctuation, nothing else.
""",
}

postlude_explanations = {
    "with_reasoning": """
Explain your reasoning, then output the answer tag on the last line.
""",
    "without_reasoning": """
Output the answer tag on the last line.
""",
    "simple_prompt": """
Assistant: The answer is
""",
    "simple_prompt_strict": """
Assistant: The answer is
""",
}

options = [
    "\nAnswer choices: A. Parallel B. Perpendicular C. Neither parallel nor perpendicular D. Cannot be inferred",
    "\nAnswer choices: A. Intersecting B. Not intersecting C. Cannot be inferred",
    "\nAnswer choices: A. Yes B. No C. Cannot be inferred",
]

numeric_preludes = {
    "with_reasoning": """You are evaluating a numeric geometry question.
You may think step-by-step and output your full reasoning.
After your reasoning, you MUST output the final answer in the format:
<answer>...</answer>
where ... is the numeric answer.
If the answer is a multiple of π or π^2, output the numeric multiplier (e.g. 12 for 12π).
Do not include units or explanatory text inside the tags.
Question:
""",
    "without_reasoning": """You are evaluating a numeric geometry question.
You MUST output the final answer in the format:
<answer>...</answer>
where ... is the numeric answer.
If the answer is a multiple of π or π^2, output the numeric multiplier (e.g. 12 for 12π).
Do not include units or explanatory text inside the tags.
Question:
""",
    "simple_prompt": """System: You are a helpful assistant for solving geometry problems.
User: Answer the following geometry question with a numeric answer. Begin your response with the number only.
If the answer is a multiple of π or π^2, output the numeric multiplier (e.g. 12 for 12π).
""",
    "simple_prompt_strict": """System: You are a geometry problem solver. Output only a number.
User: Answer the geometry question below with a number only. Begin your response with the number only.
If the answer is a multiple of π or π^2, output the numeric multiplier (e.g. 12 for 12π).
""",
}


def resolve_prompt_key(prompt_type_or_reasoning: bool | str) -> str:
    """Normalize prompt type input into a known prompt key."""
    if isinstance(prompt_type_or_reasoning, str):
        return prompt_type_or_reasoning
    return "with_reasoning" if prompt_type_or_reasoning else "without_reasoning"


def _parse_option_choices(opt: str) -> list[str]:
    """Extract choice texts from an options string (e.g., 'A. Foo B. Bar')."""
    matches = list(re.finditer(r"\b([A-E])\.\s*", opt))
    if not matches:
        return []
    choices: list[str] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(opt)
        text = opt[start:end].strip()
        if text:
            choices.append(text)
    return choices


def _format_options(choices: list[str]) -> str:
    """Format choices into the canonical prompt options string."""
    parts = []
    for idx, choice in enumerate(choices):
        letter = chr(ord("A") + idx)
        parts.append(f"{letter}. {choice}")
    return "\nAnswer choices: " + " ".join(parts)


def _rotate_list(items: list[str], shift: int) -> list[str]:
    if not items:
        return items
    shift = shift % len(items)
    if shift == 0:
        return items
    return items[shift:] + items[:shift]


def _choice_count_for_type(type_key: str) -> int:
    try:
        idx = int(type_key) - 1
    except Exception:
        idx = 0
    idx = max(0, min(idx, len(options) - 1))
    choices = _parse_option_choices(options[idx])
    return len(choices) if choices else 0


def remap_answer_for_rotation(answer: str, rotation: int, num_choices: int) -> str:
    """Remap an answer letter after a left rotation of the options list."""
    if not answer or num_choices <= 0:
        return answer
    letter = answer.strip().upper()
    if not letter.isalpha():
        return answer
    idx = ord(letter) - ord("A")
    if idx < 0 or idx >= num_choices:
        return answer
    new_idx = (idx - (rotation % num_choices)) % num_choices
    return chr(ord("A") + new_idx)


def derive_choices_text(opt: str) -> str:
    """Derive a compact choices list like 'A, B, C, D' from an options string."""
    letters = re.findall(r"\b([A-E])\.", opt)
    return ", ".join(letters) if letters else "A, B, C, D, E"


def remap_answer_for_perm(answer: str, perm: list[int]) -> str:
    """Remap an answer letter after reordering the options by `perm`.

    `perm` is the new-position -> old-index map used by make_prompt_mc (the new
    options list is [base[perm[0]], base[perm[1]], ...]). An answer at old index
    `o` therefore lands at new position `perm.index(o)`. E.g. perm=[1,0,2,3]
    (swap A/B) maps A->B and B->A, C/D unchanged."""
    if not answer or not perm:
        return answer
    letter = answer.strip().upper()
    if not letter.isalpha():
        return answer
    idx = ord(letter) - ord("A")
    if idx not in perm:
        return answer
    return chr(ord("A") + perm.index(idx))


def make_prompt_mc(
    question: str,
    type_key: str,
    reasoning: bool | str = True,
    rotation: int = 0,
    perm: list[int] | None = None,
) -> str:
    """Build a multiple-choice prompt for the given type.

    `perm` (optional) is an explicit new-position -> old-index reordering of the
    options (e.g. [1,0,2,3] transposes A/B); it takes precedence over the cyclic
    `rotation`. Used by patch_pairs --swap-ab to counterbalance the answer LETTER
    against the relation without a full cyclic sweep."""
    key = resolve_prompt_key(reasoning)
    try:
        idx = int(type_key) - 1
    except Exception:
        idx = 0
    idx = max(0, min(idx, len(options) - 1))
    opt = options[idx]
    base_choices = _parse_option_choices(opt)
    if base_choices:
        if perm is not None:
            if sorted(perm) != list(range(len(base_choices))):
                raise ValueError(f"perm {perm} is not a permutation of {len(base_choices)} choices")
            reordered = [base_choices[i] for i in perm]
        else:
            reordered = _rotate_list(base_choices, rotation)
        opt = _format_options(reordered)
        choices = ", ".join([chr(ord("A") + i) for i in range(len(reordered))])
    else:
        choices = derive_choices_text(opt)
    return prelude_explanations[key].format(choices=choices) + question + opt + postlude_explanations[key]


def make_prompt_mc_variants(
    question: str,
    type_key: str,
    reasoning: bool | str = True,
) -> list[dict[str, int | str]]:
    """Build all cyclic permutations of a multiple-choice prompt."""
    num_choices = _choice_count_for_type(type_key)
    if num_choices <= 0:
        return [
            {
                "prompt": make_prompt_mc(question, type_key, reasoning=reasoning),
                "rotation": 0,
                "num_choices": 0,
            }
        ]

    variants: list[dict[str, int | str]] = []
    for rotation in range(num_choices):
        variants.append(
            {
                "prompt": make_prompt_mc(
                    question,
                    type_key,
                    reasoning=reasoning,
                    rotation=rotation,
                ),
                "rotation": rotation,
                "num_choices": num_choices,
            }
        )
    return variants


def make_prompt_numeric(question: str, reasoning: bool | str = True) -> str:
    """Build a numeric prompt."""
    key = resolve_prompt_key(reasoning)
    prompt = numeric_preludes[key] + "\n" + question
    if key == "simple_prompt":
        prompt += postlude_explanations[key]
    return prompt


# -------------------------------------------------------------
# Numeric multiple-choice: solve a numeric question by SELECTING from 4 options
# (the correct value + 3 distractors), rotated like the regular MC questions.
# This complements (does not replace) the free-form numeric format above.
# -------------------------------------------------------------
NUMERIC_MC_NUM_CHOICES = 4


# Vertices / edges / 2-faces of each figure used in the count questions.
_POLYTOPE_COUNTS: dict[str, tuple[int, int, int]] = {
    "triangular pyramid": (4, 6, 4),
    "triangle": (3, 3, 1),
    "square": (4, 4, 1),
    "cube": (8, 12, 6),
    "4-simplex": (5, 10, 10),
    "tesseract": (16, 32, 24),
}
_COUNT_POOL = sorted({v for tpl in _POLYTOPE_COUNTS.values() for v in tpl})


def _distractor_numbers(question: str) -> list[int]:
    """Integer parameters of a numeric question (strips π / dimension tokens)."""
    s = question.lower().replace("2-dimensional", "")
    s = s.replace("\\pi^2", "").replace("\\pi", "").replace("pi^2", "").replace("pi", "")
    s = re.sub(r"\b\d+-(simplex|sphere|parallelotope|cell|ball)\b", "", s)
    s = re.sub(r"\b4d\b", "", s)
    return [int(x) for x in re.findall(r"\d+", s)]


def _magnitude_fallback(correct: int, n: int) -> list[int]:
    """Magnitude-aware perturbations, used only to top up to `n` distractors."""
    step = max(1, round(abs(correct) * 0.1))
    cands = []
    for d in (step, 2 * step, 1, 2, 3):
        cands += [correct + d, correct - d]
    cands += [2 * correct, correct // 2 if correct >= 2 else None]
    out, seen = [], {correct}
    for v in cands:
        if v is None or v <= 0 or v in seen:
            continue
        seen.add(v)
        out.append(v)
        if len(out) == n:
            break
    return out


def make_numeric_distractors(question: str, answer: str, n: int = 3) -> list[str]:
    """Build `n` *error-mode* distractors for a numeric geometry question.

    Distractors are the values a solver would get by a characteristic mistake
    for that exact question family (dropped/confused coefficient, wrong exponent,
    surface↔volume confusion, sum-instead-of-hypotenuse, V/E/F confusion, …),
    ordered most-plausible-first. Falls back to magnitude-aware perturbations
    only to guarantee `n` results. Returns [] if the answer is not an integer.
    """
    try:
        C = int(str(answer).strip())
    except (TypeError, ValueError):
        return []

    cands: list[int] = []

    def add(*vals):
        for v in vals:
            if v is None:
                continue
            if isinstance(v, float):
                if abs(v - round(v)) > 1e-9:
                    continue
                v = int(round(v))
            if isinstance(v, int) and v > 0:
                cands.append(v)

    s = question.lower()
    try:
        m = _distractor_numbers(question)
        if any(k in s for k in ("number of", "how many", "vertex count", "edge count", "face count")):
            for fig, (V, E, F) in _POLYTOPE_COUNTS.items():
                if fig in s:
                    add(V, E, F)  # V/E/F confusion of the same figure
                    break
            for c in sorted(_COUNT_POOL, key=lambda c: abs(c - C)):
                add(c)  # nearby geometrically-meaningful counts
        # --- 4D (hyper-) families first (avoid 'volume of a' / 'area of a' shadowing) ---
        elif "hyper-surface volume of a regular 4-simplex" in s:
            cv = m[0]; add(4 * cv, 6 * cv, 10 * cv, 3 * cv)            # wrong #cells
        elif "hyper-surface volume of a tesseract" in s:
            sd = m[0]; add(sd ** 4, 24 * sd * sd, 6 * sd * sd, sd ** 3)  # hypervol / wrong count
        elif "hyper-surface volume of a 3-sphere" in s:
            r = m[0]; add((r ** 4) / 2, r ** 3, 4 * r ** 3, r ** 4)    # hypervol mult / wrong coeff
        elif "hyper-surface volume of a rectangular 4-parallelotope" in s:
            l, w, h, d = m[:4]
            add(l * w * h + l * w * d + l * h * d + w * h * d, l * w * h * d, 2 * l * w * h)  # forgot x2 / hypervol
        elif "hyper-volume of a tesseract" in s:
            sd = m[0]; add(8 * sd ** 3, sd ** 3, 4 * sd ** 3, sd * sd)  # hypersurface / 3D vol
        elif "hyper-volume of a rectangular 4-parallelotope" in s:
            l, w, h, d = m[:4]
            add(2 * (l * w * h + l * w * d + l * h * d + w * h * d), l * w * h, 2 * l * w * h * d)
        elif "hyper-volume of a 4-ball" in s:
            r = m[0]; add(2 * r ** 3, r ** 4, 4 * r ** 3, r ** 3)       # hypersurface mult / wrong coeff
        elif "4-simplex" in s and "hyper-volume" in s:                  # (1/4)·base·height
            if "legs" in s:
                base = (m[0] * m[1] / 2) * m[2] / 3; depth = m[3]
            else:
                base, depth = m[0], m[1]
            add(base * depth, base * depth / 3, base * depth / 2, base * depth * 2)  # dropped/confused 1/4
        # --- circle / diagonal ---
        elif "circumference of a circle" in s or "perimeter of a circle" in s:
            r = m[0]; add(r * r, 4 * r, r, 3 * r)                       # area mult / wrong factor
        elif "hypotenuse" in s or "diagonal of a rectangle" in s:
            a, b = m[0], m[1]; add(a + b, abs(a - b), a * a + b * b, max(a, b))  # sum / no sqrt
        elif "space diagonal" in s or "diagonal of a rectangular 4-parallelotope" in s:
            add(sum(m), sum(x * x for x in m), max(m))                  # sum / no sqrt
        # --- perimeter ---
        elif "perimeter of a equilateral triangle" in s:
            sd = m[0]; add(sd * sd, 4 * sd, 2 * sd, 6 * sd)
        elif "perimeter of a square" in s:
            sd = m[0]; add(sd * sd, 2 * sd, 6 * sd, 3 * sd)             # area / wrong #sides
        elif "perimeter of a rectangle" in s:
            l, w = m[0], m[1]; add(l * w, l + w, 2 * l * w, 4 * max(l, w))  # area / half
        # --- surface area ---
        elif "surface area of a regular triangular pyramid" in s:
            fa = m[0]; add(3 * fa, 6 * fa, 2 * fa, fa)                  # wrong #faces
        elif "surface area of a cube" in s:
            sd = m[0]; add(sd ** 3, sd * sd, 4 * sd * sd, 12 * sd * sd)  # volume / one face / wrong coeff
        elif "surface area of a sphere" in s:
            r = m[0]; add(r * r, 2 * r * r, 3 * r * r, 8 * r * r)        # wrong coeff
        elif "surface area of a rectangular prism" in s:
            l, w, h = m[0], m[1], m[2]
            add(l * w + l * h + w * h, l * w * h, 2 * (l + w + h), l * w)  # forgot x2 / volume
        # --- area ---
        elif "area of a" in s and "triangle" in s:
            b, h = m[0], m[1]; add(b * h, b + h, (b + h) / 2, 2 * b * h)  # forgot 1/2 / sum
        elif "area of a square" in s:
            sd = m[0]; add(4 * sd, sd ** 3, 2 * sd * sd, 2 * sd)        # perimeter / volume
        elif "area of a rectangle" in s:
            l, w = m[0], m[1]; add(2 * (l + w), l + w, 2 * l * w, l * l)  # perimeter
        elif "area of a circle" in s:
            r = m[0]; add(2 * r, 4 * r * r, r, 3 * r * r)               # circumference mult / wrong coeff
        # --- volume ---
        elif "volume of a cube" in s:
            sd = m[0]; add(6 * sd * sd, sd * sd, 3 * sd ** 3, 2 * sd ** 3)  # surface / one face
        elif "volume of a rectangular prism" in s:
            l, w, h = m[0], m[1], m[2]
            add(2 * (l * w + l * h + w * h), l * w, l + w + h, 2 * l * w * h)  # surface
        elif "volume of a sphere" in s:
            r = m[0]; add(4 * r * r, 4 * r ** 3, (r ** 3) / 3, 2 * r ** 3)  # surface mult / wrong coeff
        elif "pyramid" in s and "volume of a" in s:                    # (1/3)·base·height
            if "legs" in s:
                base = m[0] * m[1] / 2; h = m[2]
            else:
                base, h = m[0], m[1]
            add(base * h, base * h / 2, base * h / 4, base * h * 2 / 3)  # dropped/confused 1/3
    except Exception:
        pass

    out: list[int] = []
    seen = {C}
    for v in cands:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
        if len(out) == n:
            break
    if len(out) < n:
        for v in _magnitude_fallback(C, n):
            if v not in seen:
                seen.add(v)
                out.append(v)
                if len(out) == n:
                    break
    return [str(v) for v in out[:n]]


def make_prompt_numeric_mc(
    question: str,
    base_choices: list[str],
    reasoning: bool | str = True,
    rotation: int = 0,
) -> str:
    """Build a multiple-choice prompt for a numeric question.

    `base_choices` is the canonical order (correct value at index 0). The list
    is left-rotated by `rotation` for presentation, mirroring make_prompt_mc, so
    remap_answer_for_rotation("A", rotation, len) gives the correct letter.
    """
    key = resolve_prompt_key(reasoning)
    rotated = _rotate_list(list(base_choices), rotation)
    opt = _format_options(rotated)
    choices = ", ".join(chr(ord("A") + i) for i in range(len(rotated)))
    return prelude_explanations[key].format(choices=choices) + question + opt + postlude_explanations[key]


def make_prompt_numeric_mc_variants(
    question: str,
    answer: str,
    reasoning: bool | str = True,
) -> list[dict[str, int | str]]:
    """Build all cyclic-rotation variants of a numeric multiple-choice prompt.

    The correct value sits at canonical index 0 (letter "A"); evaluate.py remaps
    it per rotation with remap_answer_for_rotation. Returns [] when distractors
    cannot be built (non-integer answer) so the caller can skip that question.
    """
    distractors = make_numeric_distractors(question, answer, n=NUMERIC_MC_NUM_CHOICES - 1)
    if len(distractors) < NUMERIC_MC_NUM_CHOICES - 1:
        return []
    base_choices = [str(answer).strip()] + distractors  # correct at index 0
    nch = len(base_choices)
    variants: list[dict[str, int | str]] = []
    for rotation in range(nch):
        variants.append(
            {
                "prompt": make_prompt_numeric_mc(question, base_choices, reasoning, rotation),
                "rotation": rotation,
                "num_choices": nch,
            }
        )
    return variants


# -------------------------------------------------------------
# Chat template helpers (shared between evaluation/evaluate.py and sae/ablation_eval.py)
# -------------------------------------------------------------

# Models that require a chat template applied before generation.
# vLLM / HookedTransformer do NOT auto-apply the tokenizer's chat template to
# raw string prompts, so we must do it ourselves for instruction-tuned models.
_CHAT_TEMPLATE_MODELS: tuple[str, ...] = (
    "gemma",
    "llama",
    "qwen",
    "mistral",
    "phi",
    "falcon",
    "gpt-oss",
    "gpt_oss",
)

# Optional reasoning-effort hint passed to chat templates that support it
# (e.g. gpt-oss harmony format: reasoning_effort="low"|"medium"|"high").
# Set via set_reasoning_effort() from the CLI; None means "do not pass".
_REASONING_EFFORT: Optional[str] = None


def set_reasoning_effort(effort: Optional[str]) -> None:
    """Set the global reasoning-effort hint for chat-template rendering."""
    global _REASONING_EFFORT
    _REASONING_EFFORT = effort or None


# When True, render the chat template in thinking/CoT mode (Qwen3 `enable_thinking`,
# Gemma-4 `thinking`) and keep the <think> tag so the model produces a reasoning trace
# before its answer. Default False = single-pass (thinking disabled). Set from the CLI.
_ENABLE_THINKING: bool = False


def set_enable_thinking(enabled: bool) -> None:
    """Toggle CoT/thinking mode for chat-template rendering (default: disabled)."""
    global _ENABLE_THINKING
    _ENABLE_THINKING = bool(enabled)


def get_enable_thinking() -> bool:
    """Whether CoT/thinking mode is enabled for chat-template rendering."""
    return _ENABLE_THINKING


def _is_harmony_model(model_name: str) -> bool:
    """gpt-oss uses the harmony format; manual assistant prefill corrupts it."""
    lower = (model_name or "").lower()
    return "gpt-oss" in lower or "gpt_oss" in lower


def _template_kwargs_variants() -> list[dict]:
    """Candidate apply_chat_template kwargs, most-preferred first.

    When a reasoning effort is configured, try variants that pass it so
    templates supporting it (gpt-oss) honor it; always fall back to plain.
    """
    # CoT mode: try both family conventions (Qwen3 `enable_thinking`, Gemma-4 `thinking`).
    if _ENABLE_THINKING:
        base = [{"enable_thinking": True}, {"thinking": True}, {}]
    else:
        base = [{"enable_thinking": False}, {}]
    if _REASONING_EFFORT:
        re_kw = {"reasoning_effort": _REASONING_EFFORT}
        return [re_kw, {**base[0], **re_kw}] + base
    return base

_tokenizer_cache: Dict[str, object] = {}


def _get_tokenizer(model_name: str) -> Optional[object]:
    """Load (and cache) the HF tokenizer for a model."""
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]
    try:
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(model_name)
        _tokenizer_cache[model_name] = tok
        return tok
    except Exception as e:
        print(f"Warning: could not load tokenizer for {model_name}: {e}", file=sys.stderr)
        return None


def needs_chat_template(model_name: str) -> bool:
    """Return True if this model name looks like an instruction-tuned model."""
    lower = (model_name or "").lower()
    return any(kw in lower for kw in _CHAT_TEMPLATE_MODELS)


_SYSTEM_MESSAGE = (
    "Always begin your response with your final answer: "
    "a single letter (e.g. A) for multiple-choice questions, "
    "or a number for numeric questions. "
    "You may explain your reasoning after the answer."
)

# Matches the completion-style suffix appended by simple_prompt.
# MC prompts use "The answer is option"; numeric prompts use "The answer is".
_SIMPLE_PROMPT_SUFFIX_RE = re.compile(
    r"\nAssistant:\s*(The answer is(?:\s+option)?)\s*$", re.IGNORECASE
)

# Parses "System: <sys>\nUser: <user>" structure embedded in simple_prompt.
_SIMPLE_PROMPT_ROLES_RE = re.compile(
    r"^System:\s*(.*?)\nUser:\s*(.*)",
    re.DOTALL,
)


def apply_chat_template(prompt: str, model_name: str) -> str:
    """Wrap a raw prompt string in the model's chat template.

    For simple_prompt format (ends with "Assistant: The answer is"), the
    embedded System:/User: markers are parsed into proper chat roles, and
    "The answer is" is injected as an assistant prefill.
    For models that don't support the system role (e.g. Gemma), the system
    message is prepended to the user message instead.

    For with_reasoning / without_reasoning prompts, no system message is added.

    Falls back to the raw prompt if the model is not in the known list,
    the tokenizer cannot be loaded, or the tokenizer has no chat_template.
    """
    if not needs_chat_template(model_name):
        return prompt

    tok = _get_tokenizer(model_name)
    if tok is None:
        return prompt

    m = _SIMPLE_PROMPT_SUFFIX_RE.search(prompt)
    if m:
        user_content = prompt[: m.start()].strip()
        prefill = m.group(1)  # "The answer is"

        # Parse embedded System:/User: roles from simple_prompt format.
        sm = _SIMPLE_PROMPT_ROLES_RE.match(user_content)
        if sm:
            sys_msg = sm.group(1).strip()
            user_msg = sm.group(2).strip()
        else:
            sys_msg = None
            user_msg = user_content

        # gpt-oss harmony format already opens the assistant turn via
        # add_generation_prompt; appending a raw "The answer is" prefill would
        # corrupt the channel structure, so suppress it for harmony models.
        effective_prefill = "" if _is_harmony_model(model_name) else prefill

        # Try with explicit system role.
        if sys_msg:
            for kwargs in _template_kwargs_variants():
                try:
                    base = tok.apply_chat_template(
                        [
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        **kwargs,
                    )
                    # Single-pass: strip the dangling <think> so the model answers directly.
                    # CoT mode keeps it so the model produces a reasoning trace first.
                    if not _ENABLE_THINKING:
                        base = base.replace("<think>\n", "").replace("<think>", "")
                    return base + effective_prefill
                except Exception:
                    pass

        # Fallback: prepend system to user for models that don't support
        # the system role (e.g. Gemma raises an exception for it).
        combined = (sys_msg + "\n\n" + user_msg) if sys_msg else user_msg
        for kwargs in _template_kwargs_variants():
            try:
                base = tok.apply_chat_template(
                    [{"role": "user", "content": combined}],
                    tokenize=False,
                    add_generation_prompt=True,
                    **kwargs,
                )
                if not _ENABLE_THINKING:
                    base = base.replace("<think>\n", "").replace("<think>", "")
                return base + effective_prefill
            except Exception as e:
                print(
                    f"Warning: apply_chat_template failed for {model_name}: {e}",
                    file=sys.stderr,
                )
        return prompt

    # with_reasoning / without_reasoning: no system message
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        print(
            f"Warning: apply_chat_template failed for {model_name}: {e}",
            file=sys.stderr,
        )
        return prompt
