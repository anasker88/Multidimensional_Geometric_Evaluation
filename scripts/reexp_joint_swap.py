#!/usr/bin/env python
"""P2-1 sufficiency fix: re-run the joint top-k denoise patch-in on the SWAP pairs.

The original joint/<m> ran on the ORIGINAL aligned pairs (all clean=A), so the
'patch in the clean (top-k) activations' donor is answer-biased toward A; a head that
merely writes letter-A would then inflate the joint recovery. On the A/B-balanced swap
pairs the population is answer-neutral, so joint recovery reflects relation-carrying
(true sufficiency), mirroring the C1 fix on the ablation (necessity) side.

5 jobs (one per standard-GQA model), heads from heads_swap (swap-identified movers).
Fast (~6-8 min/job). 4-GPU pool, resume-aware (skips a job whose output json exists).
"""
import os, subprocess, time

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
    out = f"results/patching/joint_swap/{s}"
    JOBS[s] = {"cmd": (f"{PY} patching/patch_joint.py --model-name {hf} "
                       f"--pairs results/patching/pairs/{s}_swap.json "
                       f"--heads-json results/patching/heads_swap/{s}/patch_heads_results.json "
                       f"--dtype bfloat16 --ks 1,2,3,5 --out {out}"),
               "out": f"{out}/patch_joint_results.json", "prio": PRIO[s]}

def log(m):
    line = f"[{time.strftime('%F %T', time.gmtime())} UTC] {m}"
    print(line, flush=True); open(f"{LOGDIR}/joint_swap.log", "a").write(line + "\n")

done, running, free = set(), {}, set(GPUS)
for j in JOBS:  # resume
    if os.path.exists(os.path.join(ROOT, JOBS[j]["out"])):
        done.add(j)
log(f"=== JOINT_SWAP START: {len(JOBS)} jobs ({len(done)} already done) on GPUs {GPUS} ===")
while len(done) < len(JOBS):
    active = {j for j, *_ in running.values()}
    ready = sorted([j for j in JOBS if j not in done and j not in active], key=lambda j: -JOBS[j]["prio"])
    while free and ready:
        j = ready.pop(0); gpu = min(free); free.discard(gpu)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1",
                   VLLM_WORKER_MULTIPROC_METHOD="spawn")
        lf = open(f"{LOGDIR}/joint_swap_{j}.log", "w")
        p = subprocess.Popen(JOBS[j]["cmd"], shell=True, stdout=lf, stderr=subprocess.STDOUT, env=env)
        running[gpu] = (j, p, time.time()); log(f">>> launch {j} on GPU {gpu} (prio {JOBS[j]['prio']})")
    time.sleep(5)
    for gpu, (j, p, t0) in list(running.items()):
        rc = p.poll()
        if rc is None:
            continue
        dt = int(time.time() - t0); del running[gpu]; free.add(gpu)
        ok = rc == 0 and os.path.exists(os.path.join(ROOT, JOBS[j]["out"]))
        done.add(j); log(f"{'<<< DONE' if ok else '!!! FAIL'} {j} (GPU {gpu}, {dt//60}m{dt%60}s)"
                         + ("" if ok else f", rc={rc} see {LOGDIR}/joint_swap_{j}.log"))
log("=== JOINT_SWAP COMPLETE ===")
