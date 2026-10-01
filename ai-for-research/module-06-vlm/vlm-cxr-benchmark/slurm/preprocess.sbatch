#!/bin/bash -l
#SBATCH -p compute-devel
#SBATCH -N 1 -c 4
#SBATCH --ntasks-per-node=1
#SBATCH -t 0:10:00
#SBATCH -A lt200203
#SBATCH -J vlm-cxr-benchmark-preprocess
#SBATCH --output=/lustrefs/disk/project/lt200203-aimedi/puem/tmp/vlm-cxr-benchmark-preprocess-%j.out
#SBATCH --error=/lustrefs/disk/project/lt200203-aimedi/puem/tmp/vlm-cxr-benchmark-preprocess-%j.err

set -euo pipefail

cd "$(dirname "$0")/.."
module purge && module load Mamba FFmpeg cuda
export UV_PROJECT_ENVIRONMENT="/project/lt200203-aimedi/puem/envs/env_vlm-cxr-benchmark"
/home/teiamarj/.local/bin/uv run --offline --frozen --no-sync python run_stage.py preprocess --input data/dataset --output outputs/preprocess --config configs/development.json --save-json logs/preprocess-${SLURM_JOB_ID}.json
