#!/bin/bash
set -euo pipefail

PACK_ROOT=/lustrefs/disk/project/lt200203-aimedi/puem/tmp/vlm-cxr-benchmark
RUN_ROOT=runs/pilot-matrix-20260923
SUBMISSIONS="$RUN_ROOT/submissions.tsv"
MANIFEST="$RUN_ROOT/manifests/performance-1000/blind_manifest.csv"
cd "$PACK_ROOT"

for batch in 1 2 4 8; do
  config="configs/experiments/c1-4b-bf16-batch${batch}.json"
  output="$RUN_ROOT/lanta/C1-selected-4b-bf16/batch${batch}"
  job=$(sbatch --parsable -J "vlm-c1-4b-b${batch}" \
    --export="ALL,CONFIG=$config,MANIFEST=$MANIFEST,OUTPUT_ROOT=$output,REPEATS=5" \
    slurm/repeated-development.sbatch)
  printf 'C1-selected-4b-bf16\t%s\t%s\t%s\t%s\n' \
    "$config" "$job" "$output" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$SUBMISSIONS"
done
