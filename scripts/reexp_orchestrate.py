#!/usr/bin/env python
"""REEXPERIMENT_TODO runnable items — dependency-aware 4-GPU scheduler.

Scope (per user): P1-1 (counterbalance, 5 models), P1-2 (mean/resample ablation, 5),
P2-1 (joint top-k, 5), P2-2 (random multi-seed, folded via --rand-seeds 20), and
P1-3 (all-positions/all-heads sweep) on the REPRESENTATIVE Qwen3-8B only.
(P2-3 already done; P1-4/P2-4/P3-* are design-only / not implemented.)

Scheduling: a pool of 4 A100-80GB GPUs. Each patching process loads ONE model on
ONE GPU (TransformerBridge, no TP). Ready jobs (all deps done) are launched on free
GPUs, highest priority first (longest / critical-path / 27B first). The P1-3 8B chain
holds a single GPU for hours while the other 3 GPUs churn the many short jobs.

Dependency graph (the point the user flagged):
  cheap: abl_{zero,mean,resample}_<m>, joint_<m>  -- reuse EXISTING heads-json + aligned pairs; no deps
  P1-1 : pairs generated up front (CPU). Then per model:
         cb_run_<m>            (leaf: resid sweep, nothing depends on it)
         cb_comp_<m> -> cb_heads_<m> (window from cb_comp's attn.last peak) -> cb_abl_<m>
  P1-3 : p13_run_8b -> p13_heads_8b  (chained onto one GPU)
"""
import json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = ".venv/bin/python"
GPUS = [0, 1, 2, 3]
LOGDIR = "logs/reexp"
os.makedirs(LOGDIR, exist_ok=True)

# safe -> (HF name, existing aligned pairs, existing Phase-3 heads-json)
MODELS = {
    "qwen3_8b":   ("Qwen/Qwen3-8B",        "results/patching/pairs/qwen3_8b_aligned.json"),
    "qwen3_14b":  ("Qwen/Qwen3-14B",       "results/patching/pairs/qwen3_14b_aligned.json"),
    "gemma2_9b":  ("google/gemma-2-9b-it", "results/patching/pairs/gemma2_9b_aligned.json"),
    "phi4":       ("microsoft/phi-4",      "results/patching/pairs/phi4_aligned.json"),
    "gemma2_27b": ("google/gemma-2-27b-it","results/patching/pairs/gemma2_27b_aligned.json"),
}
def HEADS(s): return f"results/patching/heads/{s}/patch_heads_results.json"
def BP(s):    return f"results/patching/pairs/{s}_swap.json"   # P1-1 A/B-swap counterbalanced pairs

# job id -> {"cmd": callable(gpu)->shell str, "deps": [...], "prio": int}
JOBS = {}
def add(jid, cmd, deps=(), prio=50):
    JOBS[jid] = {"cmd": (cmd if callable(cmd) else (lambda g, c=cmd: c)), "deps": list(deps), "prio": prio}

def big(s): return 90 if s == "gemma2_27b" else (70 if s == "qwen3_14b" else 55)

# ---- cheap: P1-2 (mean/resample) + P2-2 (zero, all with --rand-seeds 20) + P2-1 (joint) ----
for s, (hf, pairs) in MODELS.items():
    for mode in ("zero", "mean", "resample"):
        add(f"abl_{mode}_{s}",
            f"{PY} patching/patch_ablate.py --model-name {hf} --pairs {pairs} "
            f"--heads-json {HEADS(s)} --dtype bfloat16 --ablation {mode} --rand-seeds 20 "
            f"--out results/patching/ablate_{mode}_ms/{s}", prio=big(s))
    add(f"joint_{s}",
        f"{PY} patching/patch_joint.py --model-name {hf} --pairs {pairs} "
        f"--heads-json {HEADS(s)} --dtype bfloat16 --ks 1,2,3,5 "
        f"--out results/patching/joint/{s}", prio=big(s))

# ---- P1-1 counterbalance: full pipeline on balanced pairs ----
def mk_sw_heads(s, hf):
    def _c(gpu, s=s, hf=hf):
        cmp_json = f"results/patching/components_swap/{s}/patch_results.json"
        d = json.load(open(cmp_json))
        m = d["aggregate"]["attn:denoise:last"]["all"]["mean"]; L = d["metadata"]["layers"]
        pk = max(range(len(m)), key=lambda i: m[i]); p = L[pk]
        win = ",".join(str(x) for x in L if p - 2 <= x <= p + 2)
        return (f"{PY} patching/patch_heads.py --model-name {hf} --pairs {BP(s)} --dtype bfloat16 "
                f"--layers {win} --batch-size 8 --out results/patching/heads_swap/{s}")
    return _c
for s, (hf, _) in MODELS.items():
    add(f"sw_run_{s}",
        f"{PY} patching/patch_run.py --model-name {hf} --pairs {BP(s)} --dtype bfloat16 "
        f"--positions edit,last --batch-size 8 --out results/patching/run_swap/{s}", prio=big(s) - 15)
    add(f"sw_comp_{s}",
        f"{PY} patching/patch_components.py --model-name {hf} --pairs {BP(s)} --dtype bfloat16 "
        f"--positions edit,last --batch-size 8 --out results/patching/components_swap/{s}", prio=big(s) + 5)
    add(f"sw_heads_{s}", mk_sw_heads(s, hf), deps=[f"sw_comp_{s}"], prio=big(s) + 4)
    add(f"sw_abl_{s}",
        f"{PY} patching/patch_ablate.py --model-name {hf} --pairs {BP(s)} "
        f"--heads-json results/patching/heads_swap/{s}/patch_heads_results.json --dtype bfloat16 "
        f"--out results/patching/ablate_swap/{s}", deps=[f"sw_heads_{s}"], prio=big(s) + 3)

# ---- P1-3 Qwen3-8B: all-positions run -> all-layers heads (chained, one GPU) ----
add("p13_run_8b",
    f"{PY} patching/patch_run.py --model-name Qwen/Qwen3-8B --pairs {MODELS['qwen3_8b'][1]} "
    f"--dtype bfloat16 --positions all --batch-size 8 --out results/patching/run_allpos/qwen3_8b", prio=100)
add("p13_heads_8b",
    f"{PY} patching/patch_heads.py --model-name Qwen/Qwen3-8B --pairs {MODELS['qwen3_8b'][1]} "
    f"--dtype bfloat16 --layers all --batch-size 8 --out results/patching/heads_alllayers/qwen3_8b",
    deps=["p13_run_8b"], prio=99)

# ---------------- scheduler ----------------
def log(msg):
    line = f"[{time.strftime('%F %T', time.gmtime())} UTC] {msg}"
    print(line, flush=True)
    open(os.path.join(LOGDIR, "orchestrate.log"), "a").write(line + "\n")

def OUTPUT(jid):
    """Marker file that exists iff the job already completed (for resume)."""
    for m in MODELS:  # model keys are unique jid suffixes (e.g. qwen3_8b)
        if jid.endswith("_" + m):
            pre = jid[: -(len(m) + 1)]
            return {
                "abl_zero":     f"results/patching/ablate_zero_ms/{m}/patch_ablate_results.json",
                "abl_mean":     f"results/patching/ablate_mean_ms/{m}/patch_ablate_results.json",
                "abl_resample": f"results/patching/ablate_resample_ms/{m}/patch_ablate_results.json",
                "joint":        f"results/patching/joint/{m}/patch_joint_results.json",
                "sw_run":       f"results/patching/run_swap/{m}/patch_results.json",
                "sw_comp":      f"results/patching/components_swap/{m}/patch_results.json",
                "sw_heads":     f"results/patching/heads_swap/{m}/patch_heads_results.json",
                "sw_abl":       f"results/patching/ablate_swap/{m}/patch_ablate_results.json",
            }.get(pre)
    return {"p13_run_8b":  "results/patching/run_allpos/qwen3_8b/patch_results.json",
            "p13_heads_8b": "results/patching/heads_alllayers/qwen3_8b/patch_heads_results.json"}.get(jid)

done, failed, running = set(), set(), {}   # running: gpu -> (jid, Popen, t0)
free = set(GPUS)
# resume: skip jobs whose output already exists (survives VM auto-shutdown restarts)
for j in list(JOBS):
    mk = OUTPUT(j)
    if mk and os.path.exists(os.path.join(ROOT, mk)):
        done.add(j)
_resumed = len(done)
log(f"=== reexp orchestrate START: {len(JOBS)} jobs on GPUs {GPUS} "
    f"(resume: {_resumed} already done, {len(JOBS) - _resumed} to run) ===")
while len(done) + len(failed) < len(JOBS):
    # launch ready jobs onto free GPUs, highest priority first
    active = {jid for jid, *_ in running.values()}
    ready = [j for j in JOBS if j not in done and j not in failed and j not in active
             and all(d in done for d in JOBS[j]["deps"])]
    # if any dep failed, mark job failed (skip)
    for j in list(JOBS):
        if j not in done and j not in failed and j not in active and any(d in failed for d in JOBS[j]["deps"]):
            failed.add(j); log(f"SKIP {j} (dependency failed)")
    ready.sort(key=lambda j: -JOBS[j]["prio"])
    while free and ready:
        j = ready.pop(0)
        gpu = min(free); free.discard(gpu)
        try:
            cmd = JOBS[j]["cmd"](gpu)
        except Exception as e:
            failed.add(j); free.add(gpu); log(f"BUILD-FAIL {j}: {e}"); continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   HF_HUB_OFFLINE="1", VLLM_WORKER_MULTIPROC_METHOD="spawn")
        lf = open(os.path.join(LOGDIR, f"{j}.log"), "w")
        p = subprocess.Popen(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT, env=env)
        running[gpu] = (j, p, time.time())
        log(f">>> launch {j} on GPU {gpu} (prio {JOBS[j]['prio']})")
    # poll
    time.sleep(10)
    for gpu, (jid, p, t0) in list(running.items()):
        rc = p.poll()
        if rc is None:
            continue
        dt = int(time.time() - t0)
        del running[gpu]; free.add(gpu)
        if rc == 0:
            done.add(jid); log(f"<<< DONE  {jid} (GPU {gpu}, {dt//60}m{dt%60}s)")
        else:
            failed.add(jid); log(f"!!! FAIL  {jid} (GPU {gpu}, rc={rc}, {dt//60}m, see {LOGDIR}/{jid}.log)")

log(f"=== reexp orchestrate COMPLETE: {len(done)} done, {len(failed)} failed ===")
if failed:
    log("failed/skipped: " + ", ".join(sorted(failed)))
print("REEXP_ORCHESTRATE_DONE_MARKER")
