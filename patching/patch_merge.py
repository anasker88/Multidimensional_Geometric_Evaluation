"""Merge per-shard JSON outputs from patch_run.py --num-shards N.

Usage:
    .venv/bin/python patching/patch_merge.py \
        --num-shards 4 \
        --in-dir results/patching/run/qwen35_9b
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_run import _aggregate, _plot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--in-dir", required=True, help="directory containing shard JSON files")
    ap.add_argument("--out-dir", default=None,
                    help="output directory for merged results (default: same as --in-dir)")
    args = ap.parse_args()

    out_dir = args.out_dir or args.in_dir
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    meta = None
    layers = None

    for shard_id in range(args.num_shards):
        path = os.path.join(args.in_dir, f"patch_results_shard{shard_id:02d}.json")
        print(f"Loading shard {shard_id}: {path}")
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if meta is None:
            meta = dict(d["metadata"])
            layers = meta["layers"]
        all_results.extend(d["per_pair"])

    n_ok = sum(1 for r in all_results if r.get("baseline_ok"))
    meta["n_pairs"] = len(all_results)
    meta["n_baseline_ok"] = n_ok
    meta["num_shards"] = args.num_shards
    print(f"Merged {len(all_results)} pairs total, {n_ok} baseline-correct")

    agg = _aggregate(all_results)

    out_json = os.path.join(out_dir, "patch_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"metadata": meta, "aggregate": agg, "per_pair": all_results},
                  f, ensure_ascii=False, indent=2)
    print(f"Wrote merged results -> {out_json}")

    # console summary (same as patch_run.py)
    for k in sorted(agg.keys()):
        print(f"\n[{k}] peak layer (max mean normalized effect):")
        cells = agg[k].get("by_dim_type", {})
        for g in sorted(cells.keys()):
            m = cells[g]["mean"]
            pk = max(range(len(m)), key=lambda i: m[i])
            print(f"  {g:>8}: peak L{layers[pk]} = {m[pk]:+.3f}  (n={cells[g]['n_pairs']})")

    _plot(agg, layers, os.path.join(out_dir, "plots"))


if __name__ == "__main__":
    main()
