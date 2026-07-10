#!/usr/bin/env bash
# Wait for the C1 swap-ablation run to END — by clean completion OR by the launcher
# process dying (crash / interruption) — then best-effort push and UNCONDITIONALLY
# deallocate the VM for cost savings.
#
# Robustness (addressing "does it deallocate even on error?"):
#  * The wait loop exits on EITHER the completion marker in c1.log OR the launcher
#    process disappearing without it. So a launcher crash still leads to deallocate.
#  * Push is best-effort and NEVER blocks deallocate. Azure deallocate preserves the
#    OS disk, so on-disk results survive even if the push failed (re-push after a
#    restart). Cost is the priority here, per user.
#  * If this watcher itself is killed (e.g. VM force-shutdown), the DevTestLab
#    08:00 JST schedule deallocates the VM as the ultimate backstop.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
C1LOG="logs/reexp/c1.log"; ST="logs/reexp/c1_push_watcher.log"
LPAT='[r]eexp_c1_swap_ablation.py'   # bracket trick: won't match this grep's own argv
ts(){ date -u '+%F %T UTC'; }
say(){ echo "[$(ts)] $*" | tee -a "$ST"; }

say "watcher started; waiting for completion marker OR launcher exit"
reason=""
while true; do
  if grep -q "C1_SWAP_ABLATION COMPLETE" "$C1LOG" 2>/dev/null; then reason="clean-complete"; break; fi
  if ! ps -eo cmd | grep -q "$LPAT"; then reason="launcher-exited-without-marker"; break; fi
  sleep 30
done
say "wait ended: reason=$reason"
grep -E "<<< DONE|!!! FAIL" "$C1LOG" 2>/dev/null | tee -a "$ST" || true

# ---- best-effort push (never aborts the deallocate below) ----
push="none"
if ( cd results && git add -A patching/ \
     && git commit -q -m "P1-2/C1 fix: mean/resample ablation on SWAP pairs (answer-neutral donor bank)

Re-runs ablate_{mean,resample} on the A/B-balanced swap pairs + heads_swap movers so the
donor bank is answer-neutral, fixing the all-clean=A C4 confound that made the original
ablate_{mean,resample}_ms drops spuriously ~0/negative. zero-on-swap = ablate_swap/<m>.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" \
     && git push origin main ) >> "$ST" 2>&1; then
  if git add results scripts/reexp_c1_swap_ablation.py scripts/reexp_c1_push_watcher.sh >> "$ST" 2>&1 \
     && git commit -q -m "Bump results: P1-2/C1 mean/resample on swap pairs + C1 launcher/watcher

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" >> "$ST" 2>&1 \
     && git push origin main >> "$ST" 2>&1; then push="ok"; else push="parent-failed"; fi
else
  push="submodule-failed-or-nothing"
fi
say "push result: $push"

# ---- deallocate UNCONDITIONALLY (cost priority; disk is preserved) ----
echo "DEALLOCATING (reason=$reason, push=$push)" > logs/reexp/C1_FINISH_STATUS
say "deallocating VM mynlp-azure/azure-abe for cost savings (reason=$reason, push=$push) ..."
if az vm deallocate -g mynlp-azure -n azure-abe >> "$ST" 2>&1; then
  echo "DEALLOCATED (reason=$reason, push=$push)" > logs/reexp/C1_FINISH_STATUS
  say "deallocate OK (VM powering off)"
else
  echo "DEALLOCATE_FAILED (reason=$reason, push=$push)" > logs/reexp/C1_FINISH_STATUS
  say "deallocate FAILED (az auth/RBAC?); VM still running — 08:00 JST schedule is backstop"
fi
