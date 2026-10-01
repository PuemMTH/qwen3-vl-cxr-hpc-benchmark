# BLOCK-04 — LANTA Live Resource Audit

**Initial audit time:** 2026-09-22 23:44 ICT  
**Execution follow-up:** 2026-09-23  
**Mode:** initial read-only audit followed by Development/Pilot workload execution  
**User/account:** `teiamarj` / `lt200203`

## Verified live

Commands were executed through the `tl` login path in a login shell.

| Item | Live evidence |
|---|---|
| Cluster | SLURM cluster `lanta` |
| Development partition | `gpu-devel`: nodes `lanta-g-[175-176]`, max walltime 2:00:00 |
| Production partition | `gpu`: nodes `lanta-g-[001-174]`, max walltime 5-00:00:00 |
| GPU inventory | every listed GPU node advertises `gpu:a100:4` |
| GPU model/VRAM | current jobs recorded NVIDIA A100-SXM4-40GB, 42,409,000,960 bytes/GPU; driver 565.57.01 |
| Same-node topology | 4 A100 GPUs/node; scheduler configuration supports requests for 1, 2 or 4 GPUs on one node |
| CPU/node | 64 physical/effective CPUs |
| Memory/node | 514,400 MB configured; example idle node `lanta-g-175` had 442,654 MB free at audit time |
| Default allocation/GPU | 16 CPU and 124,000 MB RAM per GPU in both target partitions |
| Account access | partition `AllowAccounts=ALL`; association `teiamarj|lt200203|lanta` exists with QOS `lt200203` |
| Project storage | `/lustrefs` 55 TB total, 37 TB used, 19 TB available |
| Project storage quota | `/project/lt200203-aimedi`: 36.84 TB used of 55 TB |
| Current user jobs | none after the experiment matrix completed |
| Runtime | Python 3.11.10; PyTorch 2.7.1+cu126; CUDA runtime 12.6; Transformers 5.17.0; Accelerate 1.15.0; bitsandbytes 0.50.2 |
| Last captured quota before Set D | 4,366.41/115,056.00 shared SHr (3.80%); warning below 10% |
| Actual experiment use | 9.3200 allocated GPU-hours including Set D, diagnostics and repeated C2 jobs |

## Current-workload evidence

- One-GPU model load, inference, strict parser and artifact writing completed in
  jobs `6293861`-`6293867`; all expected predictions were preserved.
- Large-model two-GPU capacity runs completed in jobs `6293741` (30B MoE) and
  `6293742` (32B dense). Both jobs completed with exit code 0.
- Full 600-image Development Set D completed twice/model in jobs `6295763`
  (30B) and `6295764` (32B), with 600/600 `ok` each round and deterministic
  prediction hashes.
- Experiment C1 batch 1/2/4/8 completed in jobs `6294007`-`6294010` for the
  Development-selected `4b-bf16` profile.
- Final CPU-pinned Experiment C2 completed in jobs `6294203`-`6294205`. The 2-
  and 4-GPU jobs used one node, disjoint image shards, disjoint 16-core affinity
  per rank and one PyTorch inter-op thread.
- Queue, submit, start, end, elapsed, exit code, allocated resources, node and
  `MaxRSS` records are archived in
  `vlm-cxr-benchmark/runs/pilot-matrix-20260923/slurm-accounting.tsv`.
- Runtime summaries preserve model-load time, pipeline/generation time, peak
  allocated/reserved VRAM and process peak RSS. Raw stdout/stderr and environment
  captures remain under `vlm-cxr-benchmark/logs/` and the run tree.

## Still not verified

- A guaranteed remaining allocation expressed directly as A100 GPU-hours. LANTA
  reports the shared balance in SHr; this document does not invent a conversion.
- Compute-node outbound internet behavior for this project.
- Human HPC operator/contact and external owner sign-off.

## Execution boundary

Every compute job used pre-staged model snapshots and the locked environment with
offline/no-sync behavior, so outbound internet was not required. The executed
matrix is Development/Pilot evidence only: the official V3 confirmatory Test split
remains locked. Technical dry-run and topology questions are now answered, but the
BLOCK-04 closure box remains open pending the explicit quota-unit interpretation,
human operator/contact and owner review recorded in the main checklist.
