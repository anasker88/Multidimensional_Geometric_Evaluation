# patching/ — activation patching

Causal localization of dimensional geometric reasoning, via residual-stream and
component activation patching. **Design, methodology, and findings (studied on
Qwen3.5-9B): [`PATCHING_ROADMAP.md`](PATCHING_ROADMAP.md).**

**Model-general.** Every stage takes `--model-name`; the default is `Qwen/Qwen3.5-9B`
but the pipeline is built to run on the evaluated model families (Qwen3/3.5, gemma,
llama, …) that TransformerLens's `TransformerBridge` can load:
- Phase 0/1 are architecture-agnostic (tokenizer + `hook_resid_post`, present for all).
- Phase 2 auto-detects each component's output hook (Qwen3.5 linear attention
  `linear_attn.hook_out` vs standard `attn.hook_out`/`hook_attn_out`); a component with
  no matching hook is skipped with a warning.

Pipeline (run from the repo root, single `.venv`):

- **`patch_pairs.py`** — Phase 0: build token-aligned clean/corrupted minimal pairs
  (mined + synthetic, balanced over dimension × type). Re-run per model (token alignment is tokenizer-specific).
- **`patch_run.py`** — Phase 1: residual-stream (layer × position) sweep; `--num-shards`/`--shard-id` for multi-GPU.
- **`patch_components.py`** — Phase 2: attention / MLP component patching (auto-detected hooks).
- **`patch_merge.py`** — merge multi-GPU shard outputs.

Quickstart (swap `--model-name` for other models):

```bash
python patching/patch_pairs.py --model-name Qwen/Qwen3.5-9B \
  --balance 40 --balance-types 1,2,3 --aligned-only \
  --out output/patch_pairs/qwen35_9b_aligned.json
bash scripts/patch_launch_multigpu.sh output/patch_pairs/qwen35_9b_aligned.json output/patch_run/qwen35_9b 4
python patching/patch_components.py --model-name Qwen/Qwen3.5-9B \
  --pairs output/patch_pairs/qwen35_9b_aligned.json --out output/patch_components/qwen35_9b
```
