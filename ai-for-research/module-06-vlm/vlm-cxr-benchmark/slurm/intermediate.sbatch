#!/bin/bash -l
#SBATCH -p gpu-devel
#SBATCH -N 1 -c 16
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=124000M
#SBATCH -t 0:30:00
#SBATCH -A lt200203
#SBATCH -J vlm-cxr-benchmark-intermediate
#SBATCH --output=/lustrefs/disk/project/lt200203-aimedi/puem/tmp/vlm-cxr-benchmark-intermediate-%j.out
#SBATCH --error=/lustrefs/disk/project/lt200203-aimedi/puem/tmp/vlm-cxr-benchmark-intermediate-%j.err

set -euo pipefail

cd "$(dirname "$0")/.."
module purge && module load Mamba FFmpeg cuda
export UV_PROJECT_ENVIRONMENT="/project/lt200203-aimedi/puem/envs/env_vlm-cxr-benchmark"
/home/teiamarj/.local/bin/uv run --offline --frozen --no-sync python run_stage.py intermediate --input outputs/preprocess --output outputs/intermediate --config configs/development.json --save-json logs/intermediate-${SLURM_JOB_ID}.json
