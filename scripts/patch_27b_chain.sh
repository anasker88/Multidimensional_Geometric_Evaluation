#!/usr/bin/env bash
# Watcher: wait for Phase1+2 to finish (P12_DONE_MARKER + merged jsons), then run Phase 3+4.
# Detached so the pipeline reaches Phase 4 without a blocking foreground poll.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
ts(){ date -u '+%F %T UTC'; }
P12LOG="logs/patch_27b_p12.log"
CMP="results/patching/components/gemma2_27b/patch_results.json"
RUN="results/patching/run/gemma2_27b/patch_results.json"

echo "[$(ts)] chain: waiting for P12_DONE_MARKER..."
while ! grep -q "P12_DONE_MARKER" "$P12LOG" 2>/dev/null; do
  if grep -q "ABORT" "$P12LOG" 2>/dev/null; then echo "[$(ts)] chain: P12 ABORTED, stopping."; exit 1; fi
  sleep 30
done
if [ ! -f "$CMP" ] || [ ! -f "$RUN" ]; then
  echo "[$(ts)] chain: merged json missing ($CMP / $RUN), stopping."; exit 1
fi
echo "[$(ts)] chain: P12 complete, launching Phase 3+4."
bash scripts/patch_27b_p34.sh
echo "[$(ts)] chain: finished."
