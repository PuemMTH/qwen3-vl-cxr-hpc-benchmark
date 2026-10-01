# Submit order for vlm-cxr-benchmark

Run only after the pack, canonical V3 data, pinned model snapshots and frozen
environment have been staged and checked on the login node.

```bash
cd /lustrefs/disk/project/lt200203-aimedi/puem/tmp/vlm-cxr-benchmark
PRE_JOB=$(sbatch --parsable slurm/preprocess.sbatch)
MID_JOB=$(sbatch --parsable --dependency=afterok:${PRE_JOB} slurm/intermediate.sbatch)
POST_JOB=$(sbatch --parsable --dependency=afterok:${MID_JOB} slurm/postprocess.sbatch)
printf '%s\n' "preprocess=${PRE_JOB}" "intermediate=${MID_JOB}" "postprocess=${POST_JOB}"
```

These are development smoke jobs. Do not point them at the V3 test split or unlock
`configs/confirmatory.template.json` before BLOCK-01 through BLOCK-04 and the final
readiness gate are closed.
