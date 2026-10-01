#!/bin/bash
set -euo pipefail

PACK_ROOT=/lustrefs/disk/project/lt200203-aimedi/puem/tmp/vlm-cxr-benchmark
RUN_ROOT=runs/pilot-matrix-20260923
SUBMISSIONS="$RUN_ROOT/submissions.tsv"
cd "$PACK_ROOT"
mkdir -p "$RUN_ROOT" logs
if [[ ! -f "$SUBMISSIONS" ]]; then
  printf 'experiment\tconfiguration\tjob_id\toutput_root\tsubmitted_utc\n' > "$SUBMISSIONS"
fi

submit_repeated() {
  local experiment=$1
  local config=$2
  local manifest=$3
  local output=$4
  local repeats=$5
  local name=$6
  local job
  job=$(sbatch --parsable -J "$name" \
    --export="ALL,CONFIG=$config,MANIFEST=$manifest,OUTPUT_ROOT=$output,REPEATS=$repeats" \
    slurm/repeated-development.sbatch)
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$experiment" "$config" "$job" "$output" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$SUBMISSIONS"
}

PERF_MANIFEST="$RUN_ROOT/manifests/performance-1000/blind_manifest.csv"
QUALITY_MANIFEST="$RUN_ROOT/manifests/quality-development-600/blind_manifest.csv"

for profile in 2b-bf16 4b-bf16 8b-nf4; do
  submit_repeated A "configs/experiments/a-$profile-batch1.json" "$PERF_MANIFEST" \
    "$RUN_ROOT/lanta/A/$profile" 5 "vlm-a-${profile}"
done

for profile in 2b-bf16 4b-bf16 8b-nf4 8b-bf16; do
  submit_repeated B-development "configs/experiments/b-development-$profile-batch1.json" \
    "$QUALITY_MANIFEST" "$RUN_ROOT/lanta/B-development/$profile" 1 "vlm-b-${profile}"
done

for batch in 1 2 4 8; do
  submit_repeated C1 "configs/experiments/c1-8b-bf16-batch${batch}.json" "$PERF_MANIFEST" \
    "$RUN_ROOT/lanta/C1/batch${batch}" 5 "vlm-c1-b${batch}"
done
