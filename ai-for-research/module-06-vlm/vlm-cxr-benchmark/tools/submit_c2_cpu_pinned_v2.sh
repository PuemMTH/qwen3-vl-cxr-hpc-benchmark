#!/bin/bash
set -euo pipefail

PACK_ROOT=/lustrefs/disk/project/lt200203-aimedi/puem/tmp/vlm-cxr-benchmark
RUN_ROOT=runs/pilot-matrix-20260923
SUBMISSIONS="$RUN_ROOT/submissions.tsv"
MANIFEST="$RUN_ROOT/manifests/scaling-2000-selected-4b-bf16/blind_manifest.csv"
CONFIG=configs/experiments/c2-4b-bf16-batch8.json
cd "$PACK_ROOT"

for gpus in 1 2 4; do
  case "$gpus" in
    1) cpus=16; memory=124000M ;;
    2) cpus=32; memory=248000M ;;
    4) cpus=64; memory=496000M ;;
  esac
  output="$RUN_ROOT/lanta/C2-selected-4b-bf16-cpu-pinned-v2/gpu${gpus}"
  job=$(sbatch --parsable -J "vlm-c2v2-g${gpus}" \
    --gres="gpu:${gpus}" --cpus-per-task="$cpus" --mem="$memory" \
    --export="ALL,CONFIG=$CONFIG,MANIFEST=$MANIFEST,OUTPUT_ROOT=$output,REPEATS=5,REPLICA_GPUS=$gpus" \
    slurm/replica-scaling.sbatch)
  printf 'C2-selected-4b-bf16-cpu-pinned-v2\t%s\t%s\t%s\t%s\n' \
    "$CONFIG" "$job" "$output" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$SUBMISSIONS"
done
