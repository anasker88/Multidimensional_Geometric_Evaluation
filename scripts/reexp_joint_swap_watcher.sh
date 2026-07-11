#!/usr/bin/env bash
# Wait for the joint-swap run to END (clean completion OR launcher death), best-effort
# push, then UNCONDITIONALLY deallocate the VM for cost savings. Same robustness as the
# C1 watcher: marker-or-launcher-exit wait, push never blocks deallocate, disk preserved
# on deallocate, DevTestLab 11:00 JST schedule as ultimate backstop.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
JLOG="logs/reexp/joint_swap.log"; ST="logs/reexp/joint_swap_watcher.log"
LPAT='[r]eexp_joint_swap.py'   # bracket trick: won't match this grep's own argv
ts(){ date -u '+%F %T UTC'; }
say(){ echo "[$(ts)] $*" | tee -a "$ST"; }

say "watcher started; waiting for JOINT_SWAP COMPLETE or launcher exit"
reason=""
while true; do
  if grep -q "JOINT_SWAP COMPLETE" "$JLOG" 2>/dev/null; then reason="clean-complete"; break; fi
  if ! ps -eo cmd | grep -q "$LPAT"; then reason="launcher-exited-without-marker"; break; fi
  sleep 20
done
say "wait ended: reason=$reason"
grep -E "<<< DONE|!!! FAIL" "$JLOG" 2>/dev/null | tee -a "$ST" || true

push="none"
if ( cd results && git add -A patching/ \
     && git commit -q -m "P2-1 sufficiency fix: joint top-k denoise on SWAP pairs (answer-neutral population)

Re-runs the joint top-k patch-in on the A/B-balanced swap pairs + heads_swap movers, so
the sufficiency measure reflects relation-carrying rather than letter-A writing (the
all-clean=A confound, mirroring the C1 necessity fix). Output: joint_swap/<model>.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" \
     && git push origin main ) >> "$ST" 2>&1; then
  if git add results scripts/reexp_joint_swap.py scripts/reexp_joint_swap_watcher.sh >> "$ST" 2>&1 \
     && git commit -q -m "Bump results: P2-1 joint on swap pairs + launcher/watcher

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" >> "$ST" 2>&1 \
     && git push origin main >> "$ST" 2>&1; then push="ok"; else push="parent-failed"; fi
else
  push="submodule-failed-or-nothing"
fi
say "push result: $push"

echo "DEALLOCATING (reason=$reason, push=$push)" > logs/reexp/JOINT_SWAP_FINISH_STATUS
say "deallocating VM mynlp-azure/azure-abe (reason=$reason, push=$push) ..."
if az vm deallocate -g mynlp-azure -n azure-abe >> "$ST" 2>&1; then
  echo "DEALLOCATED (reason=$reason, push=$push)" > logs/reexp/JOINT_SWAP_FINISH_STATUS
  say "deallocate OK (VM powering off)"
else
  echo "DEALLOCATE_FAILED (reason=$reason, push=$push)" > logs/reexp/JOINT_SWAP_FINISH_STATUS
  say "deallocate FAILED (az auth/RBAC?); VM still running — 11:00 JST schedule is backstop"
fi
