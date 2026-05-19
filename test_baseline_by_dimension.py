#!/usr/bin/env python3
"""
Detailed baseline accuracy comparison:
- Break down by dimension (2D, 3D, 4D)
- Break down by question type (multiple-choice vs numeric)
"""
import csv
import os
import sys
import json
import torch
from collections import defaultdict

sys.path.insert(0, os.getcwd())

from prompting import make_prompt_mc

def load_all_data(questions_csv, numeric_csv, max_per_type=None):
    """Load questions and numeric data separately."""
    mc_data = []
    numeric_data = []

    # Load multiple-choice questions
    with open(questions_csv, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for dim_key in ['2D', '3D', '4D']:
                q = (row.get(dim_key) or '').strip()
                if q and q != '-':
                    answer = (row.get('answer') or '').strip()
                    mc_data.append({
                        'type': 'mc',
                        'dimension': dim_key,
                        'question_type': (row.get('type') or '1').strip() or '1',
                        'question': q,
                        'expected_answer': answer,
                    })
                    break
            if max_per_type and len(mc_data) >= max_per_type:
                break

    # Load numeric questions
    if os.path.exists(numeric_csv):
        with open(numeric_csv, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for dim_key in ['2D', '3D', '4D']:
                    q = (row.get(dim_key) or '').strip()
                    if q and q != '-':
                        answer = (row.get('answer') or '').strip()
                        numeric_data.append({
                            'type': 'numeric',
                            'dimension': dim_key,
                            'question_type': 'numeric',
                            'question': q,
                            'expected_answer': answer,
                        })
                        break
                if max_per_type and len(numeric_data) >= max_per_type:
                    break

    return mc_data, numeric_data

def evaluate_raw_model(data_list, batch_size=8):
    """Evaluate raw vLLM model."""
    from vllm import LLM, SamplingParams
    import re

    print("Loading vLLM model...")
    llm = LLM(
        model='google/gemma-2-9b-it',
        dtype='bfloat16',
        gpu_memory_utilization=0.7,
        disable_log_stats=True,
    )
    print("✓ vLLM loaded")

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=256,
    )

    results = defaultdict(lambda: {'correct': 0, 'total': 0})

    # Create prompts
    prompts = []
    for item in data_list:
        if item['type'] == 'mc':
            prompt = make_prompt_mc(item['question'], item['question_type'], reasoning=False)
        else:
            # Numeric - use same format as mc for now
            prompt = f"Answer: {item['question']}\n<answer>"
        prompts.append(prompt)

    print(f"Generating {len(prompts)} prompts...")
    outputs_obj = llm.generate(prompts, sampling_params, use_tqdm=True)

    for i, output in enumerate(outputs_obj):
        text = output.outputs[0].text

        # Extract answer
        if data_list[i]['type'] == 'mc':
            match = re.search(r'<answer>([A-D])</answer>', text)
            answer = match.group(1) if match else 'ERROR'
        else:
            # For numeric, try to extract number
            match = re.search(r'<answer>(.*?)</answer>', text)
            answer = match.group(1).strip() if match else 'ERROR'

        # Record result
        key = f"{data_list[i]['dimension']}"
        if answer == data_list[i]['expected_answer']:
            results[key]['correct'] += 1
        results[key]['total'] += 1

    del llm
    torch.cuda.empty_cache()

    return results

def evaluate_sae_model(data_list):
    """Evaluate SAE-augmented model."""
    from sae_lens import HookedSAETransformer, SAE
    import re

    print("Loading SAE model...")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
    print("✓ SAE loaded")

    results = defaultdict(lambda: {'correct': 0, 'total': 0})

    print(f"Generating {len(data_list)} prompts...")
    for i, item in enumerate(data_list):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(data_list)}...")

        if item['type'] == 'mc':
            prompt = make_prompt_mc(item['question'], item['question_type'], reasoning=False)
        else:
            prompt = f"Answer: {item['question']}\n<answer>"

        try:
            output = sae_model.generate(
                prompt,
                max_new_tokens=256,
                temperature=0.1,
                top_p=0.9,
            )

            if item['type'] == 'mc':
                match = re.search(r'<answer>([A-D])</answer>', output)
                answer = match.group(1) if match else 'ERROR'
            else:
                match = re.search(r'<answer>(.*?)</answer>', output)
                answer = match.group(1).strip() if match else 'ERROR'

            key = f"{item['dimension']}"
            if answer == item['expected_answer']:
                results[key]['correct'] += 1
            results[key]['total'] += 1
        except Exception as e:
            print(f"    Error: {str(e)[:50]}")

    del sae_model, sae
    torch.cuda.empty_cache()

    return results

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Load data
    print("Loading data...")
    questions_csv = os.path.join('data', 'questions_augmented.csv')
    numeric_csv = os.path.join('data', 'numeric_augmented.csv')

    mc_data, numeric_data = load_all_data(questions_csv, numeric_csv, max_per_type=30)

    print(f"\nData Distribution:")
    print(f"  Multiple-choice: {len(mc_data)}")
    for dim in ['2D', '3D', '4D']:
        count = sum(1 for d in mc_data if d['dimension'] == dim)
        print(f"    {dim}: {count}")

    print(f"  Numeric: {len(numeric_data)}")
    for dim in ['2D', '3D', '4D']:
        count = sum(1 for d in numeric_data if d['dimension'] == dim)
        print(f"    {dim}: {count}")

    all_data = mc_data + numeric_data
    print(f"  Total: {len(all_data)}")

    # Test raw model
    print(f"\n\n{'#'*60}")
    print(f"# RAW MODEL")
    print(f"{'#'*60}")
    raw_results = evaluate_raw_model(all_data)

    # Test SAE model
    print(f"\n\n{'#'*60}")
    print(f"# SAE MODEL")
    print(f"{'#'*60}")
    sae_results = evaluate_sae_model(all_data)

    # Compare
    print(f"\n\n{'#'*60}")
    print(f"# RESULTS BY DIMENSION")
    print(f"{'#'*60}\n")

    print(f"{'Dimension':<12} {'Raw %':<12} {'SAE %':<12} {'Diff %':<12}")
    print(f"{'-'*48}")

    report = {'raw_results': {}, 'sae_results': {}, 'comparison': {}}

    for dim in ['2D', '3D', '4D']:
        raw_acc = 100 * raw_results[dim]['correct'] / raw_results[dim]['total'] if raw_results[dim]['total'] > 0 else 0
        sae_acc = 100 * sae_results[dim]['correct'] / sae_results[dim]['total'] if sae_results[dim]['total'] > 0 else 0
        diff = sae_acc - raw_acc

        print(f"{dim:<12} {raw_acc:>10.2f}% {sae_acc:>10.2f}% {diff:>+10.2f}%")

        report['raw_results'][dim] = {
            'correct': raw_results[dim]['correct'],
            'total': raw_results[dim]['total'],
            'accuracy': raw_acc,
        }
        report['sae_results'][dim] = {
            'correct': sae_results[dim]['correct'],
            'total': sae_results[dim]['total'],
            'accuracy': sae_acc,
        }
        report['comparison'][dim] = diff

    # Overall
    raw_total = sum(r['correct'] for r in raw_results.values())
    raw_all = sum(r['total'] for r in raw_results.values())
    sae_total = sum(r['correct'] for r in sae_results.values())
    sae_all = sum(r['total'] for r in sae_results.values())

    raw_overall = 100 * raw_total / raw_all if raw_all > 0 else 0
    sae_overall = 100 * sae_total / sae_all if sae_all > 0 else 0

    print(f"{'-'*48}")
    print(f"{'Overall':<12} {raw_overall:>10.2f}% {sae_overall:>10.2f}% {sae_overall - raw_overall:>+10.2f}%")

    report['overall_raw'] = raw_overall
    report['overall_sae'] = sae_overall

    # Save
    report_path = '/tmp/baseline_by_dimension.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Report saved to {report_path}")

if __name__ == '__main__':
    main()
