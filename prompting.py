import re

# -------------------------------------------------------------
# Prompt definitions (aligned with evaluate.py)
# -------------------------------------------------------------
prelude_explanations = {
    "with_reasoning": """
You are evaluating a multiple-choice geometry question.
You may think step-by-step and output your full reasoning.
After your reasoning, you MUST output the final answer in the format:
<final_answer>A</final_answer>
where A is one of {choices}.
Nothing else should appear inside <final_answer> tags.
Question:
""",
    "without_reasoning": """
You are evaluating a multiple-choice geometry question.
You MUST output the final answer in the format:
<final_answer>A</final_answer>
where A is one of {choices}.
Nothing else should appear inside <final_answer> tags.
Question:
""",
}

postlude_explanations = {
    "with_reasoning": """
Explain your reasoning, then output the answer tag on the last line.
""",
    "without_reasoning": """
Output the answer tag on the last line.
""",
}

options = [
    "\nAnswer choices: A. Parallel B. Perpendicular C. Neither parallel nor perpendicular D. Cannot be inferred",
    "\nAnswer choices: A. Intersecting B. Not intersecting C. Cannot be inferred",
    "\nAnswer choices: A. Yes B. No C. Cannot be inferred",
]

numeric_preludes = {
    "with_reasoning": """
You are evaluating a numeric geometry question.
You may think step-by-step and output your full reasoning.
After your reasoning, you MUST output the final answer in the format:
<final_answer>...</final_answer>
where ... is the numeric answer.
If the answer is a multiple of π or π^2, output the numeric multiplier (e.g. 12 for 12π).
Do not include units or explanatory text inside the tags.
Question:
""",
    "without_reasoning": """
You are evaluating a numeric geometry question.
You MUST output the final answer in the format:
<final_answer>...</final_answer>
where ... is the numeric answer.
If the answer is a multiple of π or π^2, output the numeric multiplier (e.g. 12 for 12π).
Do not include units or explanatory text inside the tags.
Question:
""",
}


def derive_choices_text(opt: str) -> str:
    """Derive a compact choices list like 'A, B, C, D' from an options string."""
    letters = re.findall(r"\b([A-E])\.", opt)
    return ", ".join(letters) if letters else "A, B, C, D, E"


def make_prompt_mc(question: str, type_key: str, reasoning: bool = True) -> str:
    """Build a multiple-choice prompt for the given type."""
    key = "with_reasoning" if reasoning else "without_reasoning"
    try:
        idx = int(type_key) - 1
    except Exception:
        idx = 0
    idx = max(0, min(idx, len(options) - 1))
    opt = options[idx]
    choices = derive_choices_text(opt)
    return prelude_explanations[key].format(choices=choices) + question + opt + postlude_explanations[key]


def make_prompt_numeric(question: str, reasoning: bool = True) -> str:
    """Build a numeric prompt."""
    key = "with_reasoning" if reasoning else "without_reasoning"
    return numeric_preludes[key] + "\n" + question
