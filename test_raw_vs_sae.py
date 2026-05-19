#!/usr/bin/env python3
"""
Test script: Compare raw model vs SAE-augmented model outputs on same prompts.
Runs sequentially to avoid GPU memory conflicts.
"""
import csv
import os
import sys
import json
import torch

sys.path.insert(0, os.getcwd())

from prompting import make_prompt_mc
from cli.evaluate import chat

def load_sample_questions(csv_path, num_samples=8):
    """Load sample questions from CSV."""
    rows = []
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Try 4D first, then 3D, then 2D
            for dim_key in ['4D', '3D', '2D']:
                q = (row.get(dim_key) or '').strip()
                if q and q != '-':
                    rows.append({
                        'dimension': dim_key,
                        'type': (row.get('type') or '1').strip() or '1',
                        'question': q,
                    })
                    break
            if len(rows) >= num_samples:
                break
    return rows

def generate_with_raw_model(prompts, model_name='google/gemma-2-9b-it'):
    """Generate outputs using raw vLLM model."""
    print(f"\n{'='*60}")
    print(f"RAW MODEL: {model_name}")
    print(f"{'='*60}")

    outputs = []
    try:
        from vllm import LLM, SamplingParams
        import re

        print("Loading vLLM model...")
        print(f"  GPU Memory before: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

        llm = LLM(
            model=model_name,
            dtype='bfloat16',
            gpu_memory_utilization=0.7,
            disable_log_stats=True,
        )
        print(f"✓ vLLM model loaded")
        print(f"  GPU Memory after: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=256,
        )

        print(f"\nGenerating {len(prompts)} prompts with raw model...")
        outputs_obj = llm.generate(prompts, sampling_params, use_tqdm=False)

        # Extract answer from tags
        for i, output in enumerate(outputs_obj):
            text = output.outputs[0].text
            match = re.search(r'<answer>([A-D])</answer>', text)
            answer = match.group(1) if match else 'ERROR'
            outputs.append({'raw_output': text, 'answer': answer})
            print(f"Prompt {i+1}: {answer}")
            print(f"  Output snippet: {text[-150:]}...")

        # Clean up
        del llm
        torch.cuda.empty_cache()

        return outputs
    except Exception as e:
        print(f"ERROR in raw model: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_with_sae_model(prompts, model_name='google/gemma-2-9b-it'):
    """Generate outputs using SAE-augmented model."""
    print(f"\n{'='*60}")
    print(f"SAE MODEL: {model_name} + gemma-scope-9b-it-res-canonical")
    print(f"{'='*60}")

    outputs = []
    try:
        # Import SAE dependencies
        from sae_lens import HookedSAETransformer, SAE
        import re

        print("Loading SAE model...")
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"  Device: {device}")
        print(f"  GPU Memory before: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

        # Load SAE model
        sae_model = HookedSAETransformer.from_pretrained(
            model_name,
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

        # Generate with SAE
        print(f"\nGenerating {len(prompts)} prompts with SAE model...")
        for i, prompt in enumerate(prompts):
            try:
                # Use generate for SAE model with SAE active
                output = sae_model.generate(
                    prompt,
                    max_new_tokens=256,
                    temperature=0.1,
                    top_p=0.9,
                )

                # Extract answer
                match = re.search(r'<answer>([A-D])</answer>', output)
                answer = match.group(1) if match else 'ERROR'
                outputs.append({'sae_output': output, 'answer': answer})
                print(f"Prompt {i+1}: {answer}")
                print(f"  Output snippet: {output[-150:]}...")

            except Exception as e:
                print(f"  Prompt {i+1}: ERROR - {type(e).__name__}: {str(e)[:100]}")
                outputs.append({'sae_output': '', 'answer': 'ERROR'})

        # Clean up
        del sae_model, sae
        torch.cuda.empty_cache()

        return outputs
    except Exception as e:
        print(f"ERROR in SAE model: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main comparison test."""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Load sample questions
    csv_path = os.path.join('data', 'questions_augmented.csv')
    samples = load_sample_questions(csv_path, num_samples=8)

    print(f"\nLoaded {len(samples)} sample questions")
    for i, s in enumerate(samples, 1):
        print(f"  {i}. [{s['dimension']} Type {s['type']}] {s['question'][:60]}...")

    # Create prompts
    prompts = [make_prompt_mc(s['question'], s['type'], reasoning=False) for s in samples]

    # Test raw model
    print(f"\n\n{'#'*60}")
    print(f"# PHASE 1: RAW MODEL")
    print(f"{'#'*60}")
    raw_results = generate_with_raw_model(prompts)

    if raw_results is None:
        print("✗ Raw model failed, skipping SAE test")
        return

    print(f"\n✓ Raw model completed: {len(raw_results)} outputs")
    raw_answers = [r['answer'] for r in raw_results]
    print(f"  Answers: {', '.join(raw_answers)}")

    # Test SAE model
    print(f"\n\n{'#'*60}")
    print(f"# PHASE 2: SAE MODEL")
    print(f"{'#'*60}")
    sae_results = generate_with_sae_model(prompts)

    if sae_results is None:
        print("✗ SAE model failed")
        return

    print(f"\n✓ SAE model completed: {len(sae_results)} outputs")
    sae_answers = [r['answer'] for r in sae_results]
    print(f"  Answers: {', '.join(sae_answers)}")

    # Compare results
    print(f"\n\n{'#'*60}")
    print(f"# COMPARISON")
    print(f"{'#'*60}")

    agreement = sum(1 for r, s in zip(raw_answers, sae_answers) if r == s)
    print(f"\nAnswer Agreement: {agreement}/{len(raw_answers)} ({100*agreement/len(raw_answers):.1f}%)")
    print(f"\nDetailed Comparison:")
    print(f"{'Idx':<4} {'Raw':<6} {'SAE':<6} {'Match':<8}")
    print(f"{'-'*24}")
    for i, (r, s) in enumerate(zip(raw_answers, sae_answers), 1):
        match = '✓ YES' if r == s else '✗ NO'
        print(f"{i:<4} {r:<6} {s:<6} {match:<8}")

    # Save detailed report
    report = {
        'num_samples': len(samples),
        'raw_answers': raw_answers,
        'sae_answers': sae_answers,
        'agreement': agreement,
        'agreement_pct': 100*agreement/len(raw_answers),
        'samples': [
            {
                'dimension': s['dimension'],
                'type': s['type'],
                'question': s['question'][:100],
                'raw_answer': r['answer'],
                'sae_answer': s_res['answer'],
            }
            for s, r, s_res in zip(samples, raw_results, sae_results)
        ]
    }

    report_path = '/tmp/raw_vs_sae_comparison.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Report saved to {report_path}")

if __name__ == '__main__':
    main()
