# Multidimensional Geometry Evaluation Project

Evaluate LLM performance on geometry questions across 2D, 3D, and 4D shapes, and
investigate *how* models compute (and fail at) dimensional geometric reasoning via
activation patching.

> **Current focus:** model evaluation + activation patching. SAE-based analysis
> (`sae/`) is earlier auxiliary work — kept for reference but **not part of the
> current experiment**; see [the de-emphasized section below](#validating-sae-activations-legacy).

## Contents

- [Environment](#environment)
- [Repository Structure](#repository-structure)
- [Run Scripts](#run-scripts)
- [Usage](#usage) — [running evaluations](#running-evaluations) · [output format](#output-format) · [augmenting](#augmenting-questions) · [generating questions](#generating-new-questions-with-gpt)
- [Activation Patching](#activation-patching-mechanistic-interpretability)
- [Validating SAE Activations](#validating-sae-activations-legacy) — *legacy, out of current scope*
- [File Purposes & Gitignore Rationale](#file-purposes--gitignore-rationale)

## Environment

A single virtualenv **`.venv`** runs every experiment — model evaluation (vLLM), activation
patching (TransformerLens), and the legacy SAE code. Verified packages (Python 3.10.12):
vllm 0.14.0 · transformer-lens 3.3.0 · sae-lens 6.44.2 · torch 2.9.1 (CUDA) · transformers 5.10.2.
Patching loads Qwen3.5-9B via TransformerLens's `TransformerBridge` (Qwen3.5 text-only bridge
support landed in TL 3.2). Current working box: 4 × NVIDIA RTX A6000 48GB.

> The recorded eval sweep under `results/` was produced earlier on a 4 × A100 80GB host
> (vllm 0.23 / torch 2.11 / Python 3.12). The code is hardware-independent — any CUDA box with
> vLLM — but 70B-class models in `run_all.sh` need ≥80GB-class GPUs (`tensor-parallel-size 2`).

**Credentials:** Hugging Face token read at runtime from `~/.cache/huggingface/token`
(exported as `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`); never hardcoded. Azure OpenAI via env vars (see [Usage](#usage)).

**Evaluation runtime config** (`scripts/run_all.sh`): dims `2,3,4` · `dtype=bfloat16` ·
`max-new-tokens=16` · `temperature=0.1`, `top-p=0.9` · prompt `simple_prompt`
(Llama-3.3-70B also `simple_prompt_strict`). GPU scheduling: ≤32B → `tensor-parallel-size 1`
(two models in parallel, one per GPU); 70B-class + `Qwen3.5-35B-A3B` → `tensor-parallel-size 2`,
sequential. 70B caches (~135 GB each) are deleted after use. Results → `results/<timestamp>/<model_name>/`.

## Repository Structure

Source code is grouped **by experiment**; each package holds the runner(s) and the
library code for that experiment. Files shared across experiments live in `common/`.

### Git-Tracked Source

#### Root Level
- **`requirements.txt`**, **`requirements-min.txt`** — Python dependencies. Run scripts as `python <package>/<module>.py …` from the repo root (each script adds the repo root to `sys.path`).

#### Shared (`common/`)
- **`prompting.py`** — Shared core used by every experiment. Builds multiple-choice / numeric prompts, applies chat templates, and provides the answer-rotation mechanism (`make_prompt_mc`, `make_prompt_mc_variants`, `make_prompt_numeric`, `make_prompt_numeric_mc_variants`, `remap_answer_for_rotation`).

#### Model Evaluation (`evaluation/`)
- **`evaluate.py`** — Main evaluation driver (vLLM local models + Azure OpenAI hosted models).
- **`make_problems.py`** — GPT-based generation of new 2D/3D/4D question triplets.

#### Activation Patching (`patching/`)
Dimensional-geometry activation-patching pipeline (runs under `.venv`). Stages:
- **`patch_pairs.py`** — Phase 0: build token-aligned clean/corrupted minimal pairs (mined from the benchmark + synthetic, balanced over dimension × type).
- **`patch_run.py`** — Phase 1: residual-stream (layer × position) patching sweep with denoise/noise and invariance breakdowns; supports `--num-shards`/`--shard-id` for multi-GPU.
- **`patch_components.py`** — Phase 2: attention (linear-attention) / MLP component-level patching.
- **`patch_merge.py`** — Merge multi-GPU shard outputs.
- **`PATCHING_ROADMAP.md`** — Design decisions, methodology, and findings.

#### SAE / Interpretability (`sae/`) — *legacy, out of current scope*
- SAE-activation validation/probing and reconstruction-error evaluation: **`validate.py`**, **`recon_eval.py`** (runners) plus **`ablation_eval.py`**, **`activation_io.py`**, **`probe.py`**, **`singular.py`**, **`visualize.py`** (helpers). Retained for reference; see [Validating SAE Activations (legacy)](#validating-sae-activations-legacy).

#### Data (`data/`)
- **`questions.csv`** — Multiple-choice geometry questions. Columns: `2D`, `3D`, `4D`, `answer`, `type`.
  - `2D/3D/4D`: question text per dimension; `answer`: A–D; `type`: category 1/2/3:
    - **Type 1 — PPC (Parallel / Perpendicular):** A=Parallel, B=Perpendicular, C=Neither, D=Cannot be inferred.
    - **Type 2 — IC (Intersection):** A=Intersecting, B=Not intersecting, C=Cannot be inferred.
    - **Type 3 — CC (Collinearity / coplanarity, Yes/No):** A=Yes, B=No, C=Cannot be inferred.
- **`numeric.csv`** — Integer-answer questions (`question`, `dimension`, `answer`). Scored both free-form (`numeric`) and as 4-option multiple choice (`numeric_mc` — correct value + 3 error-mode distractors in all 4 cyclic rotations).
- **`questions_augmented.csv`** / **`numeric_augmented.csv`** — Vertex-label-permuted variants (extra columns `aug_source_id`, `aug_map`). These feed the patching pair construction.
- **`questions_generated_2000_gpt54.csv`** — GPT-generated question set.

#### Scripts (`scripts/`)
- **`augment_questions.py`** / **`augment_numeric.py`** — Dataset augmentation → `data/*_augmented.csv`.
- **`run_all.sh`** — The evaluation runner: full multi-model sweep in one memory-/disk-aware pass. See [Run Scripts](#run-scripts).
- **`patch_launch_multigpu.sh`** — Multi-GPU launcher for Phase 1 patching (splits pairs across GPUs, then merges).
- **`gen_summary.py`** / **`gen_bias.py`** — Build the cross-model `summary.md` and the position-bias analysis from completed runs.
- **`rebuild_semantic_results.py`** — Rebuild canonical-frame confusion matrices / per-class accuracy without re-running models.
- **`local/`** — Gitignored scratch dir for ad-hoc per-experiment shell scripts (not tracked).

#### Tests (`tests/`)
- **`test_baseline_accuracy.py`**, **`test_baseline_by_dimension.py`**, **`test_numeric_debug.py`**, **`test_raw_vs_sae.py`** — Baseline / debugging scripts. They add the CWD to `sys.path`, so **run them from the repository root**.

### Gitignored Directories
- **`results/`** — Evaluation outputs (`results/<timestamp>/<model_name>/`).
- **`output/`** — Generated artifacts: `output/patch_pairs/`, `output/patch_run/`, `output/patch_components/` (patching); `output/validate/`, `output/recon_eval/` (legacy SAE).
- **`logs/`**, **`past_results/`**, **`__pycache__/`**, **`.venv/`**, **`sae_activations/`**, **`analysis/`** — logs, archives, build artifacts, virtualenv, generated activations.

---

## Run Scripts

`scripts/run_all.sh` activates `.venv`, reads the Hugging Face token, schedules GPU placement, and runs the full model sweep, logging to `logs/`.

**Hugging Face token** is never hardcoded:

```bash
mkdir -p ~/.cache/huggingface
printf '%s' "<your_hf_token>" > ~/.cache/huggingface/token
chmod 600 ~/.cache/huggingface/token
```

**Prompt types:** the sweep uses `simple_prompt`, except Llama-3.3-70B-Instruct which uses `simple_prompt_strict`. The model list, per-model tensor-parallel size, and prompt types are encoded in `run_all.sh`.

```bash
# Full sweep (auto-timestamps results/<TS>/; foreground):
bash scripts/run_all.sh

# Detached so it survives SSH disconnect, explicit timestamp:
RUN_TS=$(date -u +%Y%m%d_%H%M%S) setsid nohup bash scripts/run_all.sh >/dev/null 2>&1 &
# progress: tail -f logs/run_master_<TS>.log
```

Per-model / per-experiment runners live untracked in `scripts/local/`; for one-off single-model runs, call `python evaluation/evaluate.py` directly (see [Usage](#usage)).

---

## Usage

### Running Evaluations

From the repository root:

```bash
# For hosted Azure models, export credentials first:
export AZURE_OPENAI_API_KEY="<your_api_key>"
export AZURE_OPENAI_ENDPOINT="<your_endpoint>"
export AZURE_OPENAI_API_VERSION="<your_api_version>"

# Defaults (built-in model list, 2D):
python evaluation/evaluate.py

# A single local vLLM model for 2D and 3D:
python evaluation/evaluate.py --models Qwen/Qwen3-32B --dims 2,3

# A 70B-class model with explicit prompt type and tensor parallelism:
python evaluation/evaluate.py --models meta-llama/Llama-3.3-70B-Instruct --dims 2,3,4 \
  --prompt-type simple_prompt_strict --max-new-tokens 16 --tensor-parallel-size 2
```

- Local models in `--models` (e.g. `Qwen/…`) load via `vllm.LLM`.
- Hosted/OpenAI-style models (e.g. `gpt-4o`, `gpt-5`) go through the Azure OpenAI client when no local vLLM model matches.

Key CLI options:
- `--models`, `--dims`, `--batch-size`, `--max-new-tokens`
- `--prompt-type` (`with_reasoning`, `without_reasoning`, `simple_prompt`, `simple_prompt_strict`); `--no-reasoning`
- decoding: `--greedy` / `--temperature` / `--top-p` / `--top-k` / `--repetition-penalty`
- vLLM runtime: `--dtype`, `--gpu-memory-utilization`, `--tensor-parallel-size`, `--max-model-len`
- `--results-root`, `--timestamp`

### Output Format

```
results/{timestamp}/{model_name}/
├── results.text                       # Summary stats (per type, incl. numeric & numeric_mc)
├── dim_{2,3,4}.log
├── dim_{2,3,4}_per_question.csv        # / _correct.csv / _incorrect.csv
└── confusion_matrix_{2,3,4}d_type_{1,2,3}.png   # PPC/IC/CC confusion matrices
```

Confusion matrices are emitted for the multiple-choice types only (PPC/IC/CC); `numeric` and `numeric_mc` are reported as accuracy rows in `results.text`. The cross-model summary is built with `python scripts/gen_summary.py` (→ `results/<base>/summary.md`) and position-bias analysis with `python scripts/gen_bias.py`.

### Augmenting Questions

```bash
python scripts/augment_questions.py     # -> data/questions_augmented.csv
python scripts/augment_numeric.py       # -> data/numeric_augmented.csv
```

### Generating New Questions with GPT

`evaluation/make_problems.py` generates new 2D/3D/4D aligned triplet questions and saves them as CSV.

```bash
export AZURE_OPENAI_API_KEY="<your_api_key>"
export AZURE_OPENAI_ENDPOINT="<your_endpoint>"
export AZURE_OPENAI_API_VERSION="<your_api_version>"

python evaluation/make_problems.py --model gpt-5 --rows 60 --batch-rows 20 --seed 42 \
  --output data/questions_generated.csv
```

Useful options: `--rows`, `--batch-rows`, `--type-mix`, `--shapes-mix`, `--seed`, `--allow-missing`, `--max-retries`, `--allow-duplicate-triplets`, `--tail-retry-margin`, `--max-duplicate-rows`, `--min-count-per-type`, `--strict-validation`. The script prints a validation summary (type distribution, duplicate count, pass/fail) after generation.

---

## Activation Patching (Mechanistic Interpretability)

Causal localization of dimensional geometric reasoning in **Qwen3.5-9B**, via residual-stream
and component activation patching. Runs under the single **`.venv`** (transformer-lens 3.3.0).
Full design and findings: [`patching/PATCHING_ROADMAP.md`](patching/PATCHING_ROADMAP.md).

The pipeline is **model-general** — every stage takes `--model-name` and runs on the evaluated
model families that `TransformerBridge` can load (Qwen3/3.5, gemma, llama, …); Qwen3.5-9B is the
studied default. Phase 2 auto-detects per-architecture attention/MLP hooks. See [`patching/README.md`](patching/README.md).

**Method.** Clean/corrupted minimal pairs differ by exactly one token (a geometric label) that
flips the answer (A↔B). Patch one layer at one token position, read the normalized recovery of the
answer logit difference `logit(A) − logit(B)`. Only pairs the model answers correctly in *both*
conditions are traced.

```bash
# Phase 0 — build token-aligned pairs (dimension × type1/2/3, balanced):
.venv/bin/python patching/patch_pairs.py \
  --balance 40 --balance-types 1,2,3 --aligned-only \
  --out output/patch_pairs/qwen35_9b_aligned.json

# Phase 1 — residual-stream (layer × position) sweep, multi-GPU then merge:
bash scripts/patch_launch_multigpu.sh output/patch_pairs/qwen35_9b_aligned.json output/patch_run/qwen35_9b 4
#   (single GPU: .venv/bin/python patching/patch_run.py --pairs … --out … --positions edit,last)

# Phase 2 — attention (linear-attn) / MLP component patching:
.venv/bin/python patching/patch_components.py \
  --pairs output/patch_pairs/qwen35_9b_aligned.json --out output/patch_components/qwen35_9b
```

Outputs: `output/patch_*/…/patch_results.json` + per-effect-key plots under `plots/`. Aggregates are
broken down by dimension, type, dimension×type, and real-vs-synthetic so circuit invariance can be tested.

**Key findings (Phase 0–1).**
- **Dimension-invariant circuit:** on problems the model solves, edit-token information is read in early
  layers (handoff ~L8–12) and the answer is assembled at the final position from ~L19 — the same in 2D/3D/4D.
- **4D difficulty is a failure *rate*, not circuit relocation:** baseline-correct pairs fall from 120/120 (2D)
  to 51/123 (4D); the 4D cases that *do* solve use the same-layer circuit.
- **4D coplanarity collapses to a confident wrong "No":** for CC (Yes/No), the true-Yes logit margin degrades
  monotonically (2D +1.86 → 3D +1.42 → 4D −0.81); "Cannot" stays lowest. The aggregate 64% 4D-CC accuracy
  hides this Yes-recall collapse — controlled pairs expose it.

---

## Validating SAE Activations (legacy)

> Earlier auxiliary work, **not part of the current experiment**. Retained for reference; runs under `.venv`.

`sae/validate.py` loads cached feature-activation tensors (or generates them on demand under `sae_activations/`)
and runs probing / singular-direction analysis; `sae/recon_eval.py` measures SAE reconstruction-error impact.

```bash
python sae/validate.py --output-dir sae_activations --model-name google/gemma-2-9b --layer layer_20 \
  --sae-release gemma-scope-9b-pt-mlp-canonical --sae-id layer_20/width_16k/canonical \
  --singular-dims 4,3 --singular-method both --topk 10

python sae/recon_eval.py --model-name <hf-model> --sae-release <release> --sae-id <id> --dims 3,4
```

Outputs go to `output/validate/` and `output/recon_eval/`. See module docstrings / `--help` for the full option set.

---

## File Purposes & Gitignore Rationale

| File/Dir | Git-Tracked | Reason |
|----------|:-----------:|--------|
| `common/`, `evaluation/`, `patching/` | ✓ | Core source (current experiments); versioned for reproducibility |
| `sae/` | ✓ | Legacy SAE source; versioned but out of current scope |
| `data/*.csv` | ✓ | Source datasets |
| `scripts/*.py`, `scripts/*.sh` | ✓ | Data generation, evaluation + patching runners, analysis |
| `tests/*.py` | ✓ | Baseline / debugging scripts |
| `results/`, `past_results/` | ✗ | Large generated eval outputs / archives; local-only |
| `output/` | ✗ | Generated patching + SAE artifacts; local-only |
| `logs/` | ✗ | Runtime logs |
| `__pycache__/`, `.venv/`, `sae_activations/`, `analysis/` | ✗ | Build artifacts / virtualenv / generated data |
