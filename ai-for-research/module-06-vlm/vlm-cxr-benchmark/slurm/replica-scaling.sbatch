#!/bin/bash -l
#SBATCH -A lt200203
#SBATCH -p gpu-devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=124000M
#SBATCH -t 02:00:00
#SBATCH -J vlm-c2
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

: "${CONFIG:?submit with --export=ALL,CONFIG=...}"
: "${MANIFEST:?submit with --export=ALL,MANIFEST=...}"
: "${OUTPUT_ROOT:?submit with --export=ALL,OUTPUT_ROOT=...}"
: "${REPEATS:?submit with --export=ALL,REPEATS=...}"
: "${REPLICA_GPUS:?submit with --export=ALL,REPLICA_GPUS=...}"

PACK_ROOT=/lustrefs/disk/project/lt200203-aimedi/puem/tmp/vlm-cxr-benchmark
ENV_ROOT=/project/lt200203-aimedi/puem/envs/env_vlm-cxr-benchmark
export HF_HOME=/lustrefs/disk/project/lt200203-aimedi/puem/tmp/hf-home-vlm
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

module purge
module load Mamba FFmpeg cuda
cd "$PACK_ROOT"
mkdir -p "$OUTPUT_ROOT" logs

{
  printf 'job_id=%s\n' "$SLURM_JOB_ID"
  printf 'job_name=%s\n' "$SLURM_JOB_NAME"
  printf 'node=%s\n' "$(hostname)"
  printf 'start_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'replica_gpus=%s\n' "$REPLICA_GPUS"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
  env | LC_ALL=C sort
  nvidia-smi -q
} > "$OUTPUT_ROOT/slurm-environment-start.txt"

nvidia-smi dmon -s pucm -d 1 -o DT > "$OUTPUT_ROOT/nvidia-smi-dmon.log" 2>&1 &
MONITOR_PID=$!
cleanup() {
  rc=$?
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
  {
    printf 'end_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'exit_code=%s\n' "$rc"
    nvidia-smi -q
  } > "$OUTPUT_ROOT/slurm-environment-end.txt"
  exit "$rc"
}
trap cleanup EXIT INT TERM

"$ENV_ROOT/bin/python" tools/run_replica_scaling.py \
  --config "$CONFIG" \
  --blind-manifest "$MANIFEST" \
  --output-root "$OUTPUT_ROOT" \
  --gpus "$REPLICA_GPUS" \
  --repeats "$REPEATS"
