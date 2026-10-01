# Frozen evaluation metrics

## Units

- **Primary:** one row per randomized V3 `patient_id`.
- **Secondary:** one row per image.
- Repeated model execution of an image never increases either sample size.

## Output and failure policy

The only accepted output is a JSON object with exactly one key and one uppercase
value:

```json
{"label":"NORMAL|PNEUMONIA|UNCERTAIN"}
```

Invalid JSON, additional keys, unsupported labels, runtime failures, OOM-skipped
rows and missing predictions become `UNCERTAIN`. They stay in the true-class
denominator and count as incorrect. They are never dropped.

## Patient aggregation

For a patient with `n` images, `NORMAL` or `PNEUMONIA` must receive at least
`floor(n/2)+1` votes. Otherwise the patient prediction is `UNCERTAIN`. Therefore
a tie, all failures, or a non-majority plurality cannot silently become a class.

## Primary metrics

- `recall_c = TP_c / N_true_c`, for `c ∈ {NORMAL, PNEUMONIA}`.
- `balanced_accuracy = (recall_NORMAL + recall_PNEUMONIA) / 2`.
- `accuracy = correct patients / all patients`.
- `precision_c = TP_c / N_predicted_c`; `UNCERTAIN` is not a target class.
- `F1_c = 2 × precision_c × recall_c / (precision_c + recall_c)`.
- `macro_F1 = (F1_NORMAL + F1_PNEUMONIA) / 2`.
- `uncertainty_rate = predicted UNCERTAIN / all patients`.
- `coverage = 1 - uncertainty_rate`.

## Confidence intervals and paired comparison

- Patient-level 95% percentile confidence intervals use a stratified patient
  bootstrap, sampling patients with replacement within each true class.
- Default: 2,000 replicates, seed `20260923`.
- Paired model comparison requires identical patient sets and labels.
- Candidate-minus-comparator confidence intervals use a paired, true-class-
  stratified patient bootstrap, preserving the model pairing within every draw.
- McNemar uses the exact two-sided binomial test on discordant patient correctness.

The proposed success thresholds in the experiment design remain owner/statistician
decisions; this package computes evidence but does not silently choose a winner.

## Systems-performance metrics

Warm-up batches and failed batches are retained in `batches.jsonl` but excluded
from successful-throughput summaries. The code reports these scopes separately:

- model loading: snapshot ready to model and processor ready;
- batch generation latency: synchronized generation time per request/batch;
- amortized generation sec/image: batch generation latency divided by actual
  successful batch size, with p50/p95 across measured batches;
- generation throughput: `3600 * successful images / generation seconds`;
- batch pipeline latency: image read/resize, tensor preparation, generation and
  decode for each batch;
- pipeline throughput: `3600 * successful images / summed batch pipeline seconds`;
- peak allocated/reserved VRAM: maximum PyTorch process counters over measured
  and warm-up batches.

Batch latency is not relabelled as per-image request latency. Model loading is not
included in generation latency. Complete turnaround, queue time, allocated
GPU-hours, speedup and scaling efficiency require scheduler/run-level evidence and
are computed only after matched runs exist.
