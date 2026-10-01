# HPCP

Research materials and a reproducible Qwen3-VL chest X-ray benchmark comparing
local NVIDIA RTX 5070 execution with NVIDIA A100 execution on LANTA HPC.
The repository includes Thai and English documentation, presentations, experiment
configuration, inference and evaluation code, and a curated progress evidence bundle.

## Start here

- [Progress presentation and evidence](progress_presentation_2026-09-24/README.md)
- [Progress slides (15-page PDF)](HPC_Progress_A_B_C_2026-09-25_15slides.pdf)
- [Experiment design](VLM_experiment_design.md)
- [Benchmark execution pack](ai-for-research/module-06-vlm/vlm-cxr-benchmark/README.md)
- [Experiment design workbook and readiness checks](ai-for-research/module-06-vlm/README.md)
- [AI-for-Research course materials](bin/README.md)
- [RTX 5070 / A100 comparison report](RTX5070_vs_LANTA_A100.html)

Download HTML reports and open them in a browser to view their full layout.

## Reproduce the dataset

Downloaded datasets, Python environments, model weights, raw run directories,
and local execution outputs are excluded from Git. Presentation assets and
curated evidence remain included. Some historical documents refer to local files
that are intentionally absent from this public repository.

The canonical dataset is the chest X-ray subset of
[Mendeley Data V3](https://doi.org/10.17632/rscbjbr9sj.3). From the repository root:

```bash
uv run --with requests --with pillow --with remotezip \
  python scripts/download_mendeley_v3_cxr.py \
  --output-dir data/mendeley-v3-chest-xray
```

Follow the benchmark pack README for environment setup, dataset staging,
three-stage execution, and tests. SLURM scripts contain the original LANTA
account and paths; adapt these to your own environment before submitting jobs.

## Reading the results

Reports describe the environments and experimental scope at the time of each
run. The progress bundle distinguishes platform speed, batch scaling,
multi-GPU scaling, capacity, and quality evaluation. Its RTX 5070 / A100
comparison uses different software environments, as documented in the bundle.
Development and pilot results do not establish confirmatory Test readiness;
the experiment documentation records the outstanding readiness gates.

Dataset, model, and third-party material licenses remain those of their original
sources. Public availability of this repository does not grant an additional
license for those materials.
