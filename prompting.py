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
User: Answer the following geometry question by selecting one option.
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
User: Answer the following geometry question with a numeric answer.
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


def make_prompt_mc(
    question: str,
    type_key: str,
    reasoning: bool | str = True,
    rotation: int = 0,
) -> str:
    """Build a multiple-choice prompt for the given type."""
    key = resolve_prompt_key(reasoning)
    try:
        idx = int(type_key) - 1
    except Exception:
        idx = 0
    idx = max(0, min(idx, len(options) - 1))
    opt = options[idx]
    base_choices = _parse_option_choices(opt)
    if base_choices:
        rotated_choices = _rotate_list(base_choices, rotation)
        opt = _format_options(rotated_choices)
        choices = ", ".join([chr(ord("A") + i) for i in range(len(rotated_choices))])
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
# Chat template helpers (shared between cli/evaluate.py and ablation_eval.py)
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
)

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

# Matches the completion-style suffix appended by simple_prompt:
#   "\nAssistant: The answer is\n"
_SIMPLE_PROMPT_SUFFIX_RE = re.compile(
    r"\nAssistant:\s*(The answer is)\s*$", re.IGNORECASE
)


def apply_chat_template(prompt: str, model_name: str) -> str:
    """Wrap a raw prompt string in the model's chat template.

    For simple_prompt format (ends with "Assistant: The answer is"), the suffix
    is extracted and injected as an assistant prefill so the model completes it
    directly — preserving the completion-style behavior without a chat template.

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
        # simple_prompt: inject "The answer is" as assistant prefill.
        # We apply the template with add_generation_prompt=True (which appends the
        # assistant turn opening token), then manually append the prefill text so
        # the model continues within the assistant turn — no closing token is added.
        user_content = prompt[: m.start()].strip()
        prefill = m.group(1)  # "The answer is"
        try:
            base = tok.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return base + prefill
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
