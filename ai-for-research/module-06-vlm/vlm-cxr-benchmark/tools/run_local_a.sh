#!/bin/bash
set -u

PACK_ROOT=/home/puem/Desktop/HPCP/ai-for-research/module-06-vlm/vlm-cxr-benchmark
RUN_ROOT=runs/pilot-matrix-20260923/local/A
MANIFEST=runs/pilot-matrix-20260923/manifests/performance-1000/blind_manifest.csv
cd "$PACK_ROOT" || exit 1
mkdir -p "$RUN_ROOT" logs

{
  printf 'host=%s\n' "$(hostname)"
  printf 'start_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  uname -a
  nvidia-smi -q
} > "$RUN_ROOT/environment-start.txt"

nvidia-smi dmon -s pucm -d 1 -o DT > "$RUN_ROOT/nvidia-smi-dmon.log" 2>&1 &
MONITOR_PID=$!
overall_rc=0
for profile in 2b-bf16 4b-bf16 8b-nf4; do
  printf '%s start %s\n' "$profile" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUN_ROOT/controller.log"
  uv run --frozen --extra local-blackwell python tools/run_repeated_inference.py \
    --config "configs/experiments/a-$profile-batch1.json" \
    --blind-manifest "$MANIFEST" \
    --output-root "$RUN_ROOT/$profile" \
    --repeats 5 >> "$RUN_ROOT/controller.log" 2>&1
  rc=$?
  printf '%s end %s exit_code=%s\n' "$profile" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" | tee -a "$RUN_ROOT/controller.log"
  if [[ $rc -ne 0 ]]; then
    overall_rc=$rc
  fi
done
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
{
  printf 'end_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'exit_code=%s\n' "$overall_rc"
  nvidia-smi -q
} > "$RUN_ROOT/environment-end.txt"
exit "$overall_rc"
