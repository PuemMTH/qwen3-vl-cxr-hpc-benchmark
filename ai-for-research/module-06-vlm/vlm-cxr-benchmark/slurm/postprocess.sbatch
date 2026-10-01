#!/bin/bash -l
#SBATCH -p compute-devel
#SBATCH -N 1 -c 4
#SBATCH --ntasks-per-node=1
#SBATCH -t 0:30:00
#SBATCH -A lt200203
#SBATCH -J vlm-cxr-benchmark-postprocess
#SBATCH --output=/lustrefs/disk/project/lt200203-aimedi/puem/tmp/vlm-cxr-benchmark-postprocess-%j.out
#SBATCH --error=/lustrefs/disk/project/lt200203-aimedi/puem/tmp/vlm-cxr-benchmark-postprocess-%j.err

set -euo pipefail

cd "$(dirname "$0")/.."
module purge && module load Mamba FFmpeg cuda
export UV_PROJECT_ENVIRONMENT="/project/lt200203-aimedi/puem/envs/env_vlm-cxr-benchmark"
/home/teiamarj/.local/bin/uv run --offline --frozen --no-sync python run_stage.py postprocess --input outputs/intermediate --output outputs/final --config configs/development.json --save-json logs/postprocess-${SLURM_JOB_ID}.json
