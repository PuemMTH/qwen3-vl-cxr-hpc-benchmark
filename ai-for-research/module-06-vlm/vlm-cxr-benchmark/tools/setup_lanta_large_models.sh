#!/bin/bash -l
set -euo pipefail

PACK=/lustrefs/disk/project/lt200203-aimedi/puem/tmp/vlm-cxr-benchmark
export UV_PROJECT_ENVIRONMENT=/project/lt200203-aimedi/puem/envs/env_vlm-cxr-benchmark
export HF_HOME=/lustrefs/disk/project/lt200203-aimedi/puem/tmp/hf-home-vlm
export HF_XET_HIGH_PERFORMANCE=1

cd "$PACK"
module purge && module load Mamba FFmpeg cuda
/home/teiamarj/.local/bin/uv sync --frozen --extra inference --python 3.12
/home/teiamarj/.local/bin/uv run --frozen --extra inference python tools/prepare_models.py \
  --download \
  --profile 30b-a3b-bf16 \
  --profile 32b-bf16 \
  --output artifacts/model_cache_inventory.lanta-large.json
sha256sum artifacts/model_cache_inventory.lanta-large.json
