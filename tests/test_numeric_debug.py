#!/usr/bin/env python3
"""
Test script: Compare raw model vs SAE-augmented model outputs on numeric questions.
Focus on debugging numeric generation failures.
"""
import csv
import os
import sys
import json
import torch

sys.path.insert(0, os.getcwd())

from prompting import make_prompt_numeric

def load_numeric_questions(csv_path, num_samples=4, filter_phrase=None):
    """Load numeric questions from CSV. Optionally filter by phrase."""
    rows = []
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get('question') or '').strip()
            dim = (row.get('dimension') or '3').strip()
            answer = (row.get('answer') or '?').strip()
            if q:
                # Filter if specified
                if filter_phrase and filter_phrase.lower() not in q.lower():
                    continue
                rows.append({
                    'dimension': dim,
                    'question': q,
                    'answer': answer,
                })
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
            max_tokens=128,
        )

        print(f"\nGenerating {len(prompts)} prompts with raw model...")
        outputs_obj = llm.generate(prompts, sampling_params, use_tqdm=False)

        # Extract answer from tags
        for i, output in enumerate(outputs_obj):
            text = output.outputs[0].text
            match = re.search(r'<answer>(\d+)</answer>', text)
            answer = match.group(1) if match else 'ERROR'
            outputs.append({'raw_output': text, 'answer': answer})
            print(f"Prompt {i+1}: {answer}")
            print(f"  Output snippet: {text[-120:]}...")

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
        print(f"\nGenerating {len(prompts)} prompts with SAE model (with SAE enabled)...")
        for i, prompt in enumerate(prompts):
            try:
                # Use generate for SAE model with SAE active
                print(f"\n  Prompt {i+1}:")
                print(f"    Calling model.generate() with SAE enabled...")
                output = sae_model.generate(
                    prompt,
                    max_new_tokens=128,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    stop_at_eos=True,
                    return_type='str',
                    verbose=False,
                )

                # Extract answer
                match = re.search(r'<answer>(\d+)</answer>', output)
                answer = match.group(1) if match else 'ERROR'
                outputs.append({'sae_output': output, 'answer': answer})
                print(f"    Answer: {answer}")
                print(f"    Output snippet: {output[-120:]}...")

            except Exception as e:
                print(f"    ERROR: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                outputs.append({'sae_output': f'EXCEPTION: {str(e)}', 'answer': 'ERROR'})

        # Clean up
        del sae_model, sae
        torch.cuda.empty_cache()

        return outputs
    except Exception as e:
        print(f"ERROR in SAE model init: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main comparison test."""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Load numeric questions
    csv_path = os.path.join('data', 'numeric_augmented.csv')

    # Test specific questions including the one that fails
    print("\n" + "="*60)
    print("TESTING: Finding cube-related questions...")
    print("="*60)

    all_samples = load_numeric_questions(csv_path, num_samples=10)
    print(f"First 10 questions from numeric_augmented.csv:")
    for i, s in enumerate(all_samples, 1):
        print(f"  {i}. {s['question']}")

    # Filter to focus on specific questions
    cube_samples = load_numeric_questions(csv_path, num_samples=100, filter_phrase='cube')
    print(f"\n\nFound {len(cube_samples)} 'cube' related questions:")
    for s in cube_samples[:5]:
        print(f"  - {s['question']} (ans: {s['answer']})")

    # Use cube-related questions if available, otherwise use first 4
    if len(cube_samples) >= 1:
        samples = cube_samples[:4]
    else:
        samples = all_samples[:4]

    print(f"\n\nUsing {len(samples)} sample questions for comparison:")
    for i, s in enumerate(samples, 1):
        print(f"  {i}. [Dim {s['dimension']}] {s['question']} (expected: {s['answer']})")

    # Create prompts
    prompts = [make_prompt_numeric(s['question'], reasoning=False) for s in samples]

    print(f"\nSample prompt (first 200 chars):")
    print(prompts[0][:200] + "...")

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
    print(f"{'Idx':<4} {'Question':<50} {'Raw':<8} {'SAE':<8} {'Match':<8}")
    print(f"{'-'*78}")
    for i, (sample, r, s) in enumerate(zip(samples, raw_answers, sae_answers), 1):
        match = '✓ YES' if r == s else '✗ NO'
        q_short = sample['question'][:45]
        print(f"{i:<4} {q_short:<50} {r:<8} {s:<8} {match:<8}")

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
                'question': s['question'],
                'expected_answer': s['answer'],
                'raw_answer': r['answer'],
                'sae_answer': s_res['answer'],
                'raw_output_snippet': r['raw_output'][-200:] if 'raw_output' in r else '',
                'sae_output_snippet': s_res['sae_output'][-200:] if 'sae_output' in s_res else '',
            }
            for s, r, s_res in zip(samples, raw_results, sae_results)
        ]
    }

    report_path = '/tmp/raw_vs_sae_numeric.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Report saved to {report_path}")

if __name__ == '__main__':
    main()
