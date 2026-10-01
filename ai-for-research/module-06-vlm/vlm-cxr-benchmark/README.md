# VLM CXR benchmark pack

Reproducible Qwen3-VL inference and patient-level evaluation for the official
Mendeley Data V3 CXR subset. The pack separates blind inference from labels and
keeps `UNCERTAIN`/failure rows in every metric denominator.

## Prepared models

| Profile | Model | Revision | Precision | Intended use |
|---|---|---|---|---|
| `2b-bf16` | Qwen3-VL-2B-Instruct | `89644892…203` | BF16 | shared local/A100 hardware test |
| `4b-bf16` | Qwen3-VL-4B-Instruct | `ebb281ec…b17` | BF16 | shared local/A100 hardware test |
| `8b-nf4` | Qwen3-VL-8B-Instruct | `0c351dd0…f3b` | NF4 | shared local/A100 comparator |
| `8b-bf16` | Qwen3-VL-8B-Instruct | `0c351dd0…f3b` | BF16 | A100 capacity/quality candidate |
| `30b-a3b-bf16` | Qwen3-VL-30B-A3B-Instruct | `9c4b90e1…b86c` | BF16 | two-A100 capacity proof |
| `32b-bf16` | Qwen3-VL-32B-Instruct | `0cfaf481…3ccb` | BF16 | two-A100 dense-model capacity proof |

Exact IDs, revisions and licenses are in `configs/models.lock.json`.

Verify already-cached snapshots without network access:

```bash
uv run --with huggingface-hub python tools/prepare_models.py
```

Add `--download` only on a login/staging machine when model download is intended.
Compute jobs use `local_files_only=true` and must not download weights.

## Environment

Core code/tests, without GPU inference dependencies:

```bash
uv sync --extra dev
uv run --extra dev pytest
```

LANTA A100 environment (CUDA 12.6 wheels):

```bash
uv sync --extra inference --extra dev
```

Local RTX 5070/Blackwell environment (CUDA 13.0 wheels):

```bash
uv sync --extra local-blackwell --extra dev
```

The two GPU extras intentionally conflict, so uv refuses to install both at once.
`inference` locks PyTorch 2.7.1+cu126 for the validated LANTA driver boundary;
`local-blackwell` locks PyTorch 2.14.0+cu130 with `sm_120` support. `uv.lock` is
the shared execution lock file for both profiles.

## Data

Stage canonical V3 data at `data/dataset/`. The expected structure is documented
in `data/README.md`. The inference stage reads only `blind_manifest.csv`; labels
remain in the preprocess output and are joined only during postprocess.

## Three-stage commands

CPU/mock end-to-end validation:

```bash
uv run python run_stage.py preprocess \
  --input data/dataset --output outputs/preprocess --config configs/mock.json
uv run python run_stage.py intermediate \
  --input outputs/preprocess --output outputs/intermediate --config configs/mock.json
uv run python run_stage.py postprocess \
  --input outputs/intermediate --output outputs/final --config configs/mock.json
```

Real 2B BF16 development smoke replaces `configs/mock.json` with
`configs/development.json` and requires the GPU extra for the target machine.
`configs/local-smoke.json` is a two-image feasibility check only and must not be
reported as quality evidence.

The verified local smoke used `local-blackwell`, two V3 development images and
Qwen3-VL-2B BF16. It produced two strict-schema predictions, completed both
evaluation levels, and recorded a 4,347,611,136-byte peak PyTorch allocation.
Its two-image accuracy is scientifically uninterpretable.

`configs/confirmatory.template.json` is intentionally locked. Preprocess refuses
the V3 test split until `dataset.confirmatory_unlocked=true` is deliberately set
in a frozen config after the readiness gates are closed.

## Evaluation outputs

- `joined_predictions.csv`: one row per image, including every failure
- `patient_predictions.csv`: strict-majority patient aggregation
- `metrics.json`: primary patient metrics, bootstrap CIs, secondary image metrics
- `intermediate/summary.json`: separated model-load, batch-latency,
  amortized-sec/image, throughput and peak-VRAM scopes
- `evaluation_report.md`: readable summary
- `artifact_checksums.sha256`: output hashes

Metric definitions are frozen in `METRICS.md`. Optional paired comparison is
enabled by setting `evaluation.comparator_predictions` to another prediction JSONL
on the identical patient set.

## LANTA

- Remote pack: `/lustrefs/disk/project/lt200203-aimedi/puem/tmp/vlm-cxr-benchmark`
- Remote env: `/project/lt200203-aimedi/puem/envs/env_vlm-cxr-benchmark`
- Account: `lt200203`
- Smoke inference: `gpu-devel`
- Pre/post processing: `compute-devel`

Create/sync the environment on the login node before submitting jobs:

```bash
module purge && module load Mamba FFmpeg cuda
export UV_PROJECT_ENVIRONMENT=/project/lt200203-aimedi/puem/envs/env_vlm-cxr-benchmark
/home/teiamarj/.local/bin/uv sync --frozen --extra inference
```

Then submit the dependency chain described in `slurm/submit-order.md`. The SLURM
scripts run offline with `--no-sync`; they do not install packages on compute nodes.

## Executed development matrix

The complete Development/Pilot run is archived under
`runs/pilot-matrix-20260923/`. It contains immutable manifests, five-round local
and A100 timing states, the 600-image Development quality evaluation, batch
scaling, CPU-pinned 1/2/4-GPU replica scaling, SLURM accounting, raw logs and
checksums. `EXPERIMENT_RESULTS.md` in that directory is the human-readable
result index. The official V3 confirmatory Test split was not accessed.

This package is research evaluation infrastructure, not a clinical diagnostic tool.
