#!/usr/bin/env bash
# Wait for the re-experiment orchestrator to finish, push all result files + the
# code changes to remote, then deallocate this Azure VM. Runs detached.
#
# Safety: the VM is only deallocated AFTER a successful push (so result files are
# on remote first). If any push fails, we STOP and leave the VM running so the
# problem can be investigated (Azure deallocate preserves the disk anyway, but we
# prefer results on remote before releasing). Writes status markers to logs/reexp/.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
ORCH_LOG="logs/reexp/orchestrate.log"
STATUS="logs/reexp/finish_watcher.log"
VM_RG="mynlp-azure"; VM_NAME="azure-abe"
ts(){ date -u '+%F %T UTC'; }
say(){ echo "[$(ts)] $*" | tee -a "$STATUS"; }

say "watcher started; waiting for orchestrate COMPLETE"
# NOTE: the orchestrator logs '=== reexp orchestrate COMPLETE ...' to ORCH_LOG via log();
# the bare REEXP_ORCHESTRATE_DONE_MARKER print goes to nohup.out, so match COMPLETE here.
while ! grep -q "orchestrate COMPLETE" "$ORCH_LOG" 2>/dev/null; do sleep 60; done
say "orchestrator finished. summary line:"
grep -E "COMPLETE:|failed/skipped" "$ORCH_LOG" | tail -2 | tee -a "$STATUS"

# --- clean up superseded rotation pairs (uncommitted, replaced by *_swap.json) ---
rm -f results/patching/pairs/*_balanced.json

CODE_FILES="common/prompting.py patching/patch_pairs.py patching/patch_run.py \
patching/patch_components.py patching/patch_heads.py patching/patch_ablate.py \
patching/patch_joint.py scripts/reexp_orchestrate.py scripts/reexp_finish_watcher.sh"

push_ok=1
# 1) results submodule: all new patching artifacts + swap pairs
( cd results && git add -A patching/ \
  && git commit -q -m "REEXPERIMENT_TODO re-runs: P1-1 A/B-swap pipeline, P1-2 mean/resample + P2-2 multiseed ablation, P2-1 joint, P1-3 Qwen3-8B all-pos/all-heads

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" \
  && git push origin main ) >> "$STATUS" 2>&1 || push_ok=0
say "results submodule push: $([ $push_ok = 1 ] && echo OK || echo FAILED)"

# 2) parent: code changes (swap impl, per_pair clean_answer, orchestrator) + results pointer
if [ "$push_ok" = 1 ]; then
  git add $CODE_FILES results >> "$STATUS" 2>&1
  git commit -q -m "patching: --swap-ab counterbalance (P1-1) + clean_answer in per_pair + reexp orchestrator; bump results

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" >> "$STATUS" 2>&1
  git push origin main >> "$STATUS" 2>&1 || push_ok=0
  say "parent repo push: $([ $push_ok = 1 ] && echo OK || echo FAILED)"
fi

if [ "$push_ok" != 1 ]; then
  say "PUSH FAILED -> NOT deallocating. VM left running for investigation."
  echo "PUSH_FAILED" > logs/reexp/FINISH_STATUS
  exit 1
fi

echo "SAFE_TO_DEALLOCATE" > logs/reexp/FINISH_STATUS
say "all pushed. deallocating VM $VM_RG/$VM_NAME ..."
az vm deallocate -g "$VM_RG" -n "$VM_NAME" >> "$STATUS" 2>&1 \
  && say "deallocate command returned OK" \
  || say "deallocate FAILED (check az auth / RBAC); VM still running"
