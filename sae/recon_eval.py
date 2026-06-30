"""
Evaluates the impact of SAE reconstruction error by comparing two conditions
using the same TransformerLens backend and identical generation parameters:

  - no_sae:   HookedSAETransformer WITHOUT model.saes() context (clean activations)
  - with_sae: HookedSAETransformer WITH  model.saes() context, no feature ablation

The delta between these two conditions isolates SAE reconstruction error,
independent of inference-backend differences (vLLM vs TransformerLens).
"""
import argparse
import contextlib
import json
import os
import sys
from datetime import datetime
from typing import List

import torch
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sae.ablation_eval import run_ablation_eval_with_report
from sae.activation_io import _load_sae_and_model

_TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def _batched_generate(
    model,
    sae,
    prompts: List[str],
    use_sae: bool,
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    top_k: int,
    top_p: float,
    temperature: float,
    freq_penalty: float,
) -> List[str]:
    """Generate responses using TransformerLens.

    use_sae=False: clean activations (no SAE context at all)
    use_sae=True:  SAE reconstruction applied, but no features ablated
    """
    responses: List[str] = []
    effective_top_k = top_k if top_k and top_k > 0 else None

    bar = tqdm(
        total=len(prompts),
        desc=f"{'with_sae' if use_sae else 'no_sae '} generation",
        unit="prompt",
        bar_format=_TQDM_BAR_FORMAT,
    )

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        batch_outputs: List[str] = []

        sae_ctx = (
            model.saes(saes=[sae], reset_saes_end=True)
            if use_sae
            else contextlib.nullcontext()
        )
        with sae_ctx:
            for prompt in batch:
                n_input = model.to_tokens(prompt).shape[1]
                generated_ids = model.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_k=effective_top_k,
                    top_p=top_p,
                    temperature=temperature,
                    freq_penalty=freq_penalty,
                    stop_at_eos=True,
                    return_type="tokens",
                    verbose=False,
                )
                new_token_ids = generated_ids[0, n_input:]
                text = model.tokenizer.decode(new_token_ids, skip_special_tokens=True)
                batch_outputs.append(text.strip())
                bar.update(1)

        responses.extend(batch_outputs)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    bar.close()
    return responses


def _print_comparison(no_sae_stats: dict, with_sae_stats: dict) -> None:
    header = f"{'Dimension':<12} {'Condition':<12} {'Type':<10} {'Correct':>8} {'Total':>8} {'Accuracy':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for dim_no, dim_with in zip(
        no_sae_stats["per_dimension"], with_sae_stats["per_dimension"]
    ):
        d = dim_no["dim"]
        for type_label in sorted(
            set(list(dim_no["per_type"].keys()) + list(dim_with["per_type"].keys()))
        ):
            for cond_label, dim_stats in [("no_sae", dim_no), ("with_sae", dim_with)]:
                t = dim_stats["per_type"].get(type_label, {"correct": 0, "total": 0})
                acc = t["correct"] / t["total"] if t["total"] else 0.0
                print(
                    f"{d!s:<12} {cond_label:<12} {type_label:<10}"
                    f" {t['correct']:>8} {t['total']:>8} {acc * 100:>9.2f}%"
                )
        for cond_label, dim_stats in [("no_sae", dim_no), ("with_sae", dim_with)]:
            ov = dim_stats["overall"]
            acc = ov["accuracy"]
            print(
                f"{d!s:<12} {cond_label:<12} {'overall':<10}"
                f" {ov['num_correct']:>8} {ov['num_questions']:>8} {acc * 100:>9.2f}%"
            )
        print("-" * len(header))

    for cond_label, stats in [("no_sae", no_sae_stats), ("with_sae", with_sae_stats)]:
        ov = stats["overall"]
        acc = ov["accuracy"]
        print(
            f"{'ALL':<12} {cond_label:<12} {'overall':<10}"
            f" {ov['num_correct']:>8} {ov['num_questions']:>8} {acc * 100:>9.2f}%"
        )

    delta = with_sae_stats["overall"]["accuracy"] - no_sae_stats["overall"]["accuracy"]
    print("=" * len(header))
    print(f"\nDelta (with_sae - no_sae): {delta * 100:+.2f}pp  (SAE reconstruction effect)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare TransformerLens no-SAE vs SAE-reconstructed accuracy to isolate reconstruction error.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", required=True, help="HuggingFace model ID")
    parser.add_argument("--sae-release", required=True, help="SAE release name or local path")
    parser.add_argument("--sae-id", required=True, help="SAE id within the release")
    parser.add_argument(
        "--dims",
        default="3,4",
        help="Comma-separated evaluation dimensions",
    )
    parser.add_argument(
        "--prompt-type",
        default="simple_prompt",
        help="Prompt type (simple_prompt / with_reasoning / without_reasoning)",
    )
    parser.add_argument(
        "--questions-csv",
        default="data/questions_augmented.csv",
        help="Path to MC questions CSV",
    )
    parser.add_argument(
        "--numeric-csv",
        default="data/numeric_augmented.csv",
        help="Path to numeric questions CSV",
    )
    parser.add_argument("--output-dir", default=None, help="Directory to save results (auto-named if omitted)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--freq-penalty", type=float, default=0.0)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (do_sample=False)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default=None,
        help="Model dtype override (e.g. bfloat16). If omitted, uses dtype from SAE config.",
    )
    args = parser.parse_args()

    dims = [int(d.strip()) for d in args.dims.split(",") if d.strip()]

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = args.model_name.replace("/", "_")
        args.output_dir = os.path.join(
            "results", "sae", "recon_eval", f"{ts}_{model_slug}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model '{args.model_name}' and SAE '{args.sae_id}' ...")
    sae, model = _load_sae_and_model(
        model_name=args.model_name,
        model_name_override=None,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        device=args.device,
        dtype=args.dtype,
    )

    do_sample = not args.greedy
    generation_cfg = {
        "do_sample": do_sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "freq_penalty": args.freq_penalty,
        "max_new_tokens": args.max_new_tokens,
    }

    run_conditions_path = os.path.join(args.output_dir, "run_conditions.json")
    with open(run_conditions_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": args.model_name,
                "sae_release": args.sae_release,
                "sae_id": args.sae_id,
                "dims": dims,
                "prompt_type": args.prompt_type,
                "generation": generation_cfg,
            },
            f,
            indent=2,
        )

    def _make_response_fn(use_sae: bool):
        return lambda prompts: _batched_generate(
            model=model,
            sae=sae,
            prompts=prompts,
            use_sae=use_sae,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            freq_penalty=args.freq_penalty,
        )

    print("\n--- Condition 1: no_sae (clean TransformerLens activations) ---")
    no_sae_stats = run_ablation_eval_with_report(
        response_fn=_make_response_fn(use_sae=False),
        dims=dims,
        prompt_type=args.prompt_type,
        questions_csv_path=args.questions_csv,
        numeric_csv_path=args.numeric_csv,
        output_dir=os.path.join(args.output_dir, "no_sae"),
        generation=generation_cfg,
        model_name=args.model_name,
    )

    print("\n--- Condition 2: with_sae (SAE reconstruction, no feature ablation) ---")
    with_sae_stats = run_ablation_eval_with_report(
        response_fn=_make_response_fn(use_sae=True),
        dims=dims,
        prompt_type=args.prompt_type,
        questions_csv_path=args.questions_csv,
        numeric_csv_path=args.numeric_csv,
        output_dir=os.path.join(args.output_dir, "with_sae"),
        generation=generation_cfg,
        model_name=args.model_name,
    )

    _print_comparison(no_sae_stats, with_sae_stats)

    comparison = {
        "no_sae": no_sae_stats,
        "with_sae": with_sae_stats,
        "delta_accuracy": with_sae_stats["overall"]["accuracy"] - no_sae_stats["overall"]["accuracy"],
    }
    comparison_path = os.path.join(args.output_dir, "comparison.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
