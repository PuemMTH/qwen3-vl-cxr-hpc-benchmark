# Qwen3-VL CXR prompt and inference package

This archive contains the implementation and configuration files needed to
inspect how each chest X-ray image is prompted, parsed and evaluated.

## Frozen prompt

```text
Classify this pediatric AP chest X-ray as NORMAL or PNEUMONIA. Return only one JSON object with exactly one key: {"label":"NORMAL|PNEUMONIA|UNCERTAIN"}. Use UNCERTAIN when the image is insufficient. Do not return Markdown or explanation.
```

## Included components

- `configs/experiments/b-development-4b-bf16-batch1.json`: representative
  Development configuration containing the exact prompt and generation limits.
- `configs/models.lock.json`: pinned Qwen3-VL model IDs and revisions.
- `tools/build_experiment_configs.py`: source used to generate experiment configs.
- `vlm_cxr/preprocess.py`: creates the blind manifest and separates labels.
- `vlm_cxr/inference.py`: image preparation, chat message construction and
  deterministic generation.
- `vlm_cxr/parser.py`: strict one-key JSON parser; invalid outputs become
  `UNCERTAIN`.
- `vlm_cxr/metrics.py`: image metrics and patient-level majority aggregation.
- `vlm_cxr/evaluate.py`: evaluation and confidence-interval orchestration.
- `reports/FULL_EXPERIMENT_REPORT_TH.md`: complete Thai Development/Pilot report.
- `reports/Qwen3_VL_CXR_Full_Experiment_Report_TH.html`: standalone HTML report.
- `FILES.sha256`: SHA-256 checksums of every packaged file except itself.

## Scope

The files document a completed Development/Pilot experiment. The official V3
confirmatory Test split remains locked and is not included in this package.
Dataset images, model weights, patient identifiers and large raw logs are not
included.
