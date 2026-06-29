# evaluation/ — model evaluation

Benchmark LLM accuracy on 2D/3D/4D geometry (PPC / IC / CC multiple-choice + numeric).

- **`evaluate.py`** — main driver (vLLM local models + Azure OpenAI hosted models).
- **`make_problems.py`** — GPT-based generation of new 2D/3D/4D question triplets.

Run from the repo root, e.g.:

```bash
python evaluation/evaluate.py --models Qwen/Qwen3-32B --dims 2,3
```

The full multi-model sweep, output format, and all CLI options are documented in the
root [`README.md`](../README.md).
