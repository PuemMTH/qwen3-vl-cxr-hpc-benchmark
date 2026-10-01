#!/bin/bash -l
#SBATCH -p gpu-devel
#SBATCH -N 1 -c 32
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=248000M
#SBATCH -t 1:00:00
#SBATCH -A lt200203
#SBATCH -J vlm-30b-dev
#SBATCH --output=/lustrefs/disk/project/lt200203-aimedi/puem/tmp/vlm-cxr-benchmark/logs/vlm-30b-dev-%j.out
#SBATCH --error=/lustrefs/disk/project/lt200203-aimedi/puem/tmp/vlm-cxr-benchmark/logs/vlm-30b-dev-%j.err

set -euo pipefail

PACK=/lustrefs/disk/project/lt200203-aimedi/puem/tmp/vlm-cxr-benchmark
cd "$PACK"
module purge && module load Mamba FFmpeg cuda
export UV_PROJECT_ENVIRONMENT=/project/lt200203-aimedi/puem/envs/env_vlm-cxr-benchmark
export HF_HOME=/lustrefs/disk/project/lt200203-aimedi/puem/tmp/hf-home-vlm
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
/home/teiamarj/.local/bin/uv run --offline --frozen --no-sync python run_stage.py preprocess --input data/dataset --output outputs/large-30b/preprocess --config configs/large-30b-development.json --save-json logs/large-30b-preprocess-${SLURM_JOB_ID}.json
/home/teiamarj/.local/bin/uv run --offline --frozen --no-sync python run_stage.py intermediate --input outputs/large-30b/preprocess --output outputs/large-30b/intermediate --config configs/large-30b-development.json --save-json logs/large-30b-intermediate-${SLURM_JOB_ID}.json
/home/teiamarj/.local/bin/uv run --offline --frozen --no-sync python run_stage.py postprocess --input outputs/large-30b/intermediate --output outputs/large-30b/final --config configs/large-30b-development.json --save-json logs/large-30b-postprocess-${SLURM_JOB_ID}.json
