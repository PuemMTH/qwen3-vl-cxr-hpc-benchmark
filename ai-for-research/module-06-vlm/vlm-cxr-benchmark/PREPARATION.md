# Model, code and evaluation preparation checklist

**Prepared:** 2026-09-23  
**Scope:** readiness/development only; confirmatory V3 test data remains locked

## Models

- [x] Pin official Qwen3-VL 2B/4B/8B Instruct model IDs and immutable revisions.
- [x] Record Apache-2.0 model-license metadata.
- [x] Define `2b-bf16`, `4b-bf16`, `8b-nf4`, `8b-bf16`,
  `30b-a3b-bf16` and `32b-bf16` profiles with immutable revisions.
- [x] Verify all three exact snapshots already exist in the local Hugging Face cache.
- [x] Record snapshot file counts/bytes in `artifacts/model_cache_inventory.local.json`.
- [ ] Select the final best-local comparator from a frozen patient-grouped
  development/validation split; owner/statistician decision still required.

## Runtime and code

- [x] Lock Python dependencies in `uv.lock`.
- [x] Separate incompatible accelerator profiles: `inference` for LANTA
  A100/cu126 and `local-blackwell` for RTX 5070/cu130 (`sm_120`).
- [x] Implement deterministic V3 preprocessing with manifest/image SHA-256 checks.
- [x] Keep labels out of the inference manifest and join them only for evaluation.
- [x] Implement Qwen3-VL BF16/NF4 loading from pinned offline snapshots.
- [x] Implement strict JSON parsing; invalid/missing/OOM/runtime outputs become
  `UNCERTAIN` and remain one row per requested image.
- [x] Keep the confirmatory config locked with `confirmatory_unlocked=false`.
- [x] Prepare CPU preprocess/postprocess and one-GPU `gpu-devel` SLURM scripts.
- [x] Run and record the current pack on one LANTA A100.
- [x] Add and validate disjoint replica-per-GPU sharding for Experiment C2
  (2/4 GPUs), including disjoint 16-core CPU affinity and one inter-op thread.

## Quality metrics

- [x] Primary patient-level balanced accuracy.
- [x] Accuracy, macro-F1, class precision/recall/F1, uncertainty rate and coverage.
- [x] Strict-majority aggregation across all images for each patient.
- [x] Patient-stratified percentile bootstrap confidence intervals.
- [x] Paired patient-bootstrap confidence intervals for candidate-minus-comparator
  metric differences.
- [x] Exact paired two-sided McNemar test on identical patient sets.
- [x] Secondary image-level metrics.
- [x] Failure/parse accounting and artifact checksums.
- [ ] Freeze sample size, multiplicity, alpha and comparison margins in a
  statistician-approved SAP before confirmatory access.

## Systems metrics

- [x] Separate model-loading time from measured inference.
- [x] Record raw batch generation and pipeline times.
- [x] Report batch-latency and amortized-sec/image p50/p95 separately.
- [x] Report generation/pipeline images per hour.
- [x] Record peak PyTorch allocated/reserved VRAM and environment metadata.
- [x] Collect matched five-round RTX/A100 evidence and compute the frozen
  paired-round bootstrap speedup CI for 2B BF16, 4B BF16 and 8B NF4.
- [x] Collect SLURM submit/start/end/exit records and allocated GPU-hours;
  observed total including Set D/diagnostics/reruns = 9.3200 GPU-hours.
- [x] Run C1 batch 1/2/4/8 and C2 one/two/four-GPU studies. Final CPU-pinned
  C2 achieved 99.53% efficiency on two GPUs and 97.86% on four GPUs.

## Verification evidence

- [x] Automated suite: 9 tests passed.
- [x] Python compilation and all SLURM scripts passed syntax checks.
- [x] `uv lock --check` passed with 89 resolved packages.
- [x] Execution checksum manifest SHA-256:
  `bced27c6fe894d3254aaf2fd1885d384d7eb8e628ef9358411bf79fd3aa575c4`.
- [x] Local model inventory SHA-256:
  `ae1827dea86fba77a462988c73c731f29c87488d362427ccf2b196ba4c46e618`.
- [x] Real local GPU smoke passed end to end on two V3 Development images:
  Qwen3-VL-2B BF16, PyTorch 2.14.0+cu130, RTX 5070 `sm_120`, two strict-schema
  outputs, evaluation artifacts and checksums.
- [x] Smoke peak allocation/reservation: 4,347,611,136 / 4,397,727,744 bytes.
- [x] Smoke outputs: `outputs/local-smoke-blackwell-v2/`.
- [x] Complete Development/Pilot matrix, readable report, states and raw logs:
  `runs/pilot-matrix-20260923/EXPERIMENT_RESULTS.md`.
- [x] Verify 1,199 evidence files against
  `runs/pilot-matrix-20260923/experiment-evidence.sha256`; manifest SHA-256
  `bce7df5296a584f1445d2ecae9498b1aa2ec44287fb0a0b5e44fcdfb9f12a9c9`.
- [x] Preserve actual local 30B/32B BF16 OOM traces from complete pinned
  snapshots and successful two-A100 runs (`6293741`, `6293742`).

> Do not interpret the two-image smoke accuracy or throughput as a scientific
> result; it is only a functional pipeline check.

The first local attempt with the LANTA cu126 environment correctly exposed an
architecture mismatch (`sm_120` unsupported). The accelerator extras were then
split and the full smoke rerun successfully. No confirmatory Test predictions were
generated during preparation.
