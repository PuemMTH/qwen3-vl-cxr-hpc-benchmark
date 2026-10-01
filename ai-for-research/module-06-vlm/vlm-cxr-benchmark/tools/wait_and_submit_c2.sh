#!/bin/bash
set -u

PACK_ROOT=/lustrefs/disk/project/lt200203-aimedi/puem/tmp/vlm-cxr-benchmark
PYTHON=/project/lt200203-aimedi/puem/envs/env_vlm-cxr-benchmark/bin/python
cd "$PACK_ROOT" || exit 1
while true; do
  if "$PYTHON" - <<'PY'
import json
from pathlib import Path
root = Path('runs/pilot-matrix-20260923/lanta/C1-selected-4b-bf16')
paths = [root / f'batch{batch}' / 'state.json' for batch in (1, 2, 4, 8)]
raise SystemExit(0 if all(path.is_file() and json.loads(path.read_text()).get('status') == 'complete' for path in paths) else 1)
PY
  then
    "$PYTHON" tools/submit_c2_after_c1.py
    exit $?
  fi
  printf 'waiting_for_c1 %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep 60
done
