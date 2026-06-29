#!/usr/bin/env python3
"""
Test baseline accuracy without ablation:
Compare raw model vs SAE-augmented model on full test set.
"""
import csv
import os
import sys
import json
import torch

sys.path.insert(0, os.getcwd())

from common.prompting import make_prompt_mc

def load_questions_csv(csv_path, max_samples=None):
    """Load all questions from CSV."""
    rows = []
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Try 4D first, then 3D, then 2D
            for dim_key in ['4D', '3D', '2D']:
                q = (row.get(dim_key) or '').strip()
                if q and q != '-':
                    answer = (row.get('answer') or '').strip()
                    rows.append({
                        'dimension': dim_key,
                        'type': (row.get('type') or '1').strip() or '1',
                        'question': q,
                        'expected_answer': answer,
                    })
                    break
            if max_samples and len(rows) >= max_samples:
                break
    return rows

def evaluate_raw_model(prompts, correct_answers):
    """Evaluate raw vLLM model."""
    print(f"\n{'='*60}")
    print(f"RAW MODEL BASELINE EVALUATION")
    print(f"{'='*60}")

    try:
        from vllm import LLM, SamplingParams
        import re

        print(f"Loading vLLM model...")
        llm = LLM(
            model='google/gemma-2-9b-it',
            dtype='bfloat16',
            gpu_memory_utilization=0.7,
            disable_log_stats=True,
        )
        print(f"✓ vLLM model loaded")

        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=256,
        )

        print(f"Generating {len(prompts)} prompts...")
        outputs_obj = llm.generate(prompts, sampling_params, use_tqdm=True)

        correct = 0
        for i, output in enumerate(outputs_obj):
            text = output.outputs[0].text
            match = re.search(r'<answer>([A-D])</answer>', text)
            answer = match.group(1) if match else 'ERROR'

            if answer == correct_answers[i]:
                correct += 1

        accuracy = 100 * correct / len(prompts)
        print(f"\n✓ Raw model: {correct}/{len(prompts)} correct ({accuracy:.2f}%)")

        del llm
        torch.cuda.empty_cache()

        return accuracy
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_sae_model(prompts, correct_answers):
    """Evaluate SAE-augmented model."""
    print(f"\n{'='*60}")
    print(f"SAE MODEL BASELINE EVALUATION")
    print(f"{'='*60}")

    try:
        from sae_lens import HookedSAETransformer, SAE
        import re

        print(f"Loading SAE model...")
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"  GPU Memory before: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

        sae_model = HookedSAETransformer.from_pretrained(
            'google/gemma-2-9b-it',
            device=device,
            dtype=torch.bfloat16,
        )
        sae = SAE.from_pretrained(
            'gemma-scope-9b-it-res-canonical',
            'layer_20/width_16k/canonical',
            device=device,
        )
        print(f"✓ SAE model loaded")
        print(f"  GPU Memory after: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

        print(f"\nGenerating {len(prompts)} prompts...")
        correct = 0
        for i, prompt in enumerate(prompts):
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(prompts)}...")

            try:
                output = sae_model.generate(
                    prompt,
                    max_new_tokens=256,
                    temperature=0.1,
                    top_p=0.9,
                )

                match = re.search(r'<answer>([A-D])</answer>', output)
                answer = match.group(1) if match else 'ERROR'

                if answer == correct_answers[i]:
                    correct += 1
            except Exception as e:
                print(f"    Error on prompt {i+1}: {str(e)[:50]}")

        accuracy = 100 * correct / len(prompts)
        print(f"\n✓ SAE model: {correct}/{len(prompts)} correct ({accuracy:.2f}%)")

        del sae_model, sae
        torch.cuda.empty_cache()

        return accuracy
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main evaluation."""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Load sample questions (use 100 samples for reasonable eval time)
    csv_path = os.path.join('data', 'questions_augmented.csv')
    samples = load_questions_csv(csv_path, max_samples=100)

    print(f"\nLoaded {len(samples)} sample questions")

    # Create prompts and extract correct answers
    prompts = [make_prompt_mc(s['question'], s['type'], reasoning=False) for s in samples]
    correct_answers = [s['expected_answer'] for s in samples]

    print(f"Dimension distribution:")
    dims = {}
    for s in samples:
        dims[s['dimension']] = dims.get(s['dimension'], 0) + 1
    for dim, count in sorted(dims.items()):
        print(f"  {dim}: {count}")

    print(f"Expected answers distribution:")
    ans_dist = {}
    for ans in correct_answers:
        ans_dist[ans] = ans_dist.get(ans, 0) + 1
    for ans in ['A', 'B', 'C', 'D']:
        count = ans_dist.get(ans, 0)
        pct = 100 * count / len(correct_answers) if len(correct_answers) > 0 else 0
        print(f"  {ans}: {count} ({pct:.1f}%)")

    # Test raw model
    print(f"\n\n{'#'*60}")
    print(f"# PHASE 1: RAW MODEL BASELINE")
    print(f"{'#'*60}")
    raw_accuracy = evaluate_raw_model(prompts, correct_answers)

    if raw_accuracy is None:
        print("✗ Raw model failed, skipping SAE test")
        return

    # Test SAE model
    print(f"\n\n{'#'*60}")
    print(f"# PHASE 2: SAE MODEL BASELINE")
    print(f"{'#'*60}")
    sae_accuracy = evaluate_sae_model(prompts, correct_answers)

    if sae_accuracy is None:
        print("✗ SAE model failed")
        return

    # Compare
    print(f"\n\n{'#'*60}")
    print(f"# BASELINE COMPARISON (no ablation)")
    print(f"{'#'*60}")
    print(f"\nRaw Model Accuracy:  {raw_accuracy:.2f}%")
    print(f"SAE Model Accuracy:  {sae_accuracy:.2f}%")

    if raw_accuracy >= sae_accuracy:
        diff = raw_accuracy - sae_accuracy
        print(f"\nSAE model is {diff:.2f}% lower than raw model")
    else:
        diff = sae_accuracy - raw_accuracy
        print(f"\nSAE model is {diff:.2f}% higher than raw model")

    report = {
        'num_samples': len(samples),
        'raw_accuracy': raw_accuracy,
        'sae_accuracy': sae_accuracy,
        'difference': sae_accuracy - raw_accuracy,
    }

    report_path = '/tmp/baseline_accuracy_comparison.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Report saved to {report_path}")

if __name__ == '__main__':
    main()
