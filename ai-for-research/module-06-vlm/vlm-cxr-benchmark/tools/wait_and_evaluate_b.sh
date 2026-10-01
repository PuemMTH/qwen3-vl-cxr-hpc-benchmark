#!/bin/bash
set -u

PACK_ROOT=/lustrefs/disk/project/lt200203-aimedi/puem/tmp/vlm-cxr-benchmark
PYTHON=/project/lt200203-aimedi/puem/envs/env_vlm-cxr-benchmark/bin/python
cd "$PACK_ROOT" || exit 1
while true; do
  if "$PYTHON" - <<'PY'
import json
from pathlib import Path
root = Path('runs/pilot-matrix-20260923/lanta/B-development')
profiles = ('2b-bf16', '4b-bf16', '8b-nf4', '8b-bf16')
paths = [root / profile / 'state.json' for profile in profiles]
raise SystemExit(0 if all(path.is_file() and json.loads(path.read_text()).get('status') == 'complete' for path in paths) else 1)
PY
  then
    "$PYTHON" tools/evaluate_development_matrix.py \
      --run-root runs/pilot-matrix-20260923/lanta/B-development \
      --labels runs/pilot-matrix-20260923/manifests/quality-development-600/labels.csv
    exit $?
  fi
  printf 'waiting_for_b %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep 30
done
