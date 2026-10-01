#!/bin/bash
set -euo pipefail

PACK_ROOT=/home/puem/Desktop/HPCP/ai-for-research/module-06-vlm/vlm-cxr-benchmark
RUN_ROOT="$PACK_ROOT/runs/pilot-matrix-20260923/local/A"
REMOTE=/lustrefs/disk/project/lt200203-aimedi/puem/tmp/hf-home-vlm/hub/models--Qwen--Qwen3-VL-32B-Instruct/snapshots/0cfaf48183f594c314753d30a4c4974bc75f3ccb
LOCAL=/home/puem/.cache/huggingface/manual/Qwen3-VL-32B-Instruct-0cfaf48183f594c314753d30a4c4974bc75f3ccb
cd "$PACK_ROOT"

while true; do
  if python - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
profiles = ('2b-bf16', '4b-bf16', '8b-nf4')
paths = [root / profile / 'state.json' for profile in profiles]
raise SystemExit(0 if all(path.is_file() and json.loads(path.read_text()).get('status') == 'complete' for path in paths) else 1)
PY
  then
    break
  fi
  printf 'waiting_for_local_a %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep 60
done

printf 'staging_32b_start %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
mkdir -p "$LOCAL"
rsync -aL --exclude='*.safetensors' "tl:${REMOTE}/" "$LOCAL/"
ssh tl "find ${REMOTE} -maxdepth 1 -type l -name '*.safetensors' -printf '%f\\n'" \
  | xargs -P 14 -I{} scp -q "tl:${REMOTE}/{}" "$LOCAL/{}"
sha256sum "$LOCAL"/*.safetensors > artifacts/local-32b-weight-checksums.sha256

uv run --frozen --extra local-blackwell python tools/probe_local_capacity.py \
  --profile 32b-bf16 --model-path "$LOCAL" \
  --output artifacts/local-capacity-32b.json \
  > logs/local-capacity-32b.log 2>&1
uv run --frozen --extra local-blackwell python tools/probe_local_capacity.py \
  --profile 32b-bf16 --skip-allocator-warmup --model-path "$LOCAL" \
  --output artifacts/local-capacity-32b-no-allocator-warmup.json \
  > logs/local-capacity-32b-no-allocator-warmup.log 2>&1

sha256sum \
  artifacts/local-capacity-32b.json \
  artifacts/local-capacity-32b-no-allocator-warmup.json \
  artifacts/local-32b-weight-checksums.sha256 \
  logs/local-capacity-32b.log \
  logs/local-capacity-32b-no-allocator-warmup.log \
  > artifacts/local-32b-capacity-evidence.sha256
printf 'complete %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
