#!/usr/bin/env python
"""C1 fix (REEXPERIMENT_TODO P1-2): re-run mean/resample ablation on the SWAP pairs.

The original ablate_{mean,resample}_ms ran on the ORIGINAL aligned pairs (all clean=A),
so the mean/resample donor bank carried an 'A-direction' write; substituting the mover
then re-injected the answer(A) signal -> drop_frac ~0/negative (a C4 artifact, NOT
'mover unnecessary'). On the A/B-balanced swap pairs the donor bank is answer-neutral,
so this closes the necessity check. zero-on-swap already exists (ablate_swap/<m>).

10 jobs (5 models x {mean, resample}); heads from heads_swap (swap-identified movers),
--rand-seeds 20 to match the original _ms multi-seed random control. 4-GPU pool,
resume-aware (skips a job whose output json already exists).
"""
import json, os, subprocess, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = ".venv/bin/python"; GPUS = [0, 1, 2, 3]
LOGDIR = "logs/reexp"; os.makedirs(LOGDIR, exist_ok=True)
MODELS = {"qwen3_8b": "Qwen/Qwen3-8B", "qwen3_14b": "Qwen/Qwen3-14B",
          "gemma2_9b": "google/gemma-2-9b-it", "phi4": "microsoft/phi-4",
          "gemma2_27b": "google/gemma-2-27b-it"}
PRIO = {"gemma2_27b": 90, "phi4": 80, "qwen3_14b": 70, "gemma2_9b": 60, "qwen3_8b": 55}

JOBS = {}
for s, hf in MODELS.items():
    for mode in ("mean", "resample"):
        out = f"results/patching/ablate_swap_{mode}/{s}"
        JOBS[f"{mode}_{s}"] = {
            "cmd": (f"{PY} patching/patch_ablate.py --model-name {hf} "
                    f"--pairs results/patching/pairs/{s}_swap.json "
                    f"--heads-json results/patching/heads_swap/{s}/patch_heads_results.json "
                    f"--dtype bfloat16 --ablation {mode} --rand-seeds 20 --out {out}"),
            "out": f"{out}/patch_ablate_results.json", "prio": PRIO[s]}

def log(m):
    line = f"[{time.strftime('%F %T', time.gmtime())} UTC] {m}"
    print(line, flush=True); open(f"{LOGDIR}/c1.log", "a").write(line + "\n")

done, running, free = set(), {}, set(GPUS)
for j in JOBS:  # resume
    if os.path.exists(os.path.join(ROOT, JOBS[j]["out"])):
        done.add(j)
log(f"=== C1 swap-ablation START: {len(JOBS)} jobs ({len(done)} already done) on GPUs {GPUS} ===")
while len(done) < len(JOBS):
    active = {j for j, *_ in running.values()}
    ready = sorted([j for j in JOBS if j not in done and j not in active],
                   key=lambda j: -JOBS[j]["prio"])
    while free and ready:
        j = ready.pop(0); gpu = min(free); free.discard(gpu)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1",
                   VLLM_WORKER_MULTIPROC_METHOD="spawn")
        lf = open(f"{LOGDIR}/c1_{j}.log", "w")
        p = subprocess.Popen(JOBS[j]["cmd"], shell=True, stdout=lf, stderr=subprocess.STDOUT, env=env)
        running[gpu] = (j, p, time.time()); log(f">>> launch {j} on GPU {gpu} (prio {JOBS[j]['prio']})")
    time.sleep(10)
    for gpu, (j, p, t0) in list(running.items()):
        rc = p.poll()
        if rc is None:
            continue
        dt = int(time.time() - t0); del running[gpu]; free.add(gpu)
        if rc == 0 and os.path.exists(os.path.join(ROOT, JOBS[j]["out"])):
            done.add(j); log(f"<<< DONE {j} (GPU {gpu}, {dt//60}m{dt%60}s)")
        else:
            done.add(j); log(f"!!! FAIL {j} (GPU {gpu}, rc={rc}, {dt//60}m) -- see {LOGDIR}/c1_{j}.log")
log("=== C1_SWAP_ABLATION COMPLETE ===")
