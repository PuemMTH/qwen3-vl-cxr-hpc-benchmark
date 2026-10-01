from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import load_json, resolve_pack_path, sha256_file, write_json
from .metrics import (
    PREDICTION_LABELS,
    TRUE_LABELS,
    aggregate_patients,
    classification_metrics,
    paired_mcnemar,
    paired_patient_bootstrap_difference_ci,
    patient_bootstrap_ci,
)


def _read_labels(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "patient_id", "true_label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Label manifest missing fields: {sorted(missing)}")
        rows = list(reader)
    ids = [row["image_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Label manifest must contain unique image IDs")
    if any(row["true_label"] not in TRUE_LABELS for row in rows):
        raise ValueError("Label manifest contains an unsupported true label")
    return rows


def _read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = str(row.get("image_id", ""))
            if not image_id:
                raise ValueError(f"Prediction line {line_number} has no image_id")
            if image_id in result:
                raise ValueError(f"Duplicate prediction for image_id={image_id}")
            prediction = str(row.get("predicted_label", "UNCERTAIN"))
            if prediction not in PREDICTION_LABELS:
                raise ValueError(f"Unsupported prediction label: {prediction}")
            result[image_id] = row
    return result


def join_labels_and_predictions(
    label_rows: list[dict[str, str]], predictions: dict[str, dict[str, Any]]
) -> list[dict[str, object]]:
    label_ids = {row["image_id"] for row in label_rows}
    extras = sorted(set(predictions) - label_ids)
    if extras:
        raise ValueError(f"Predictions contain IDs outside label manifest: {extras[:5]}")
    joined: list[dict[str, object]] = []
    for label in label_rows:
        prediction = predictions.get(label["image_id"])
        if prediction is None:
            prediction = {
                "predicted_label": "UNCERTAIN",
                "parse_ok": False,
                "parse_error": "missing_prediction",
                "status": "missing_prediction",
                "raw_text": "",
            }
        joined.append(
            {
                "image_id": label["image_id"],
                "patient_id": label["patient_id"],
                "true_label": label["true_label"],
                "predicted_label": prediction["predicted_label"],
                "parse_ok": bool(prediction.get("parse_ok", False)),
                "parse_error": prediction.get("parse_error"),
                "status": prediction.get("status", "unknown"),
                "raw_text": prediction.get("raw_text", ""),
            }
        )
    return joined


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(metrics: dict[str, object]) -> str:
    primary = metrics["primary_patient_level"]
    secondary = metrics["secondary_image_level"]
    failures = metrics["failure_accounting"]
    metadata = metrics["prediction_metadata"]
    return f"""# VLM CXR evaluation report

**Backend:** `{metadata['backend']}`  
**Profile:** `{metadata['profile']}`  
**Model revision:** `{metadata['model_revision']}`

## Primary: patient level

| Metric | Value |
|---|---:|
| Patients | {primary['n']} |
| Balanced accuracy | {primary['balanced_accuracy']:.6f} |
| Accuracy | {primary['accuracy']:.6f} |
| Macro-F1 | {primary['macro_f1']:.6f} |
| NORMAL recall | {primary['per_class']['NORMAL']['recall']:.6f} |
| PNEUMONIA recall | {primary['per_class']['PNEUMONIA']['recall']:.6f} |
| Uncertainty rate | {primary['uncertainty_rate']:.6f} |

Patient prediction requires a strict majority of all images from that patient.
Ties, missing outputs, invalid JSON and execution failures become `UNCERTAIN` and
remain in the denominator as incorrect predictions.

## Secondary: image level

| Metric | Value |
|---|---:|
| Images | {secondary['n']} |
| Balanced accuracy | {secondary['balanced_accuracy']:.6f} |
| Accuracy | {secondary['accuracy']:.6f} |
| Macro-F1 | {secondary['macro_f1']:.6f} |
| Uncertainty rate | {secondary['uncertainty_rate']:.6f} |

## Failure accounting

| Status | Count |
|---|---:|
""" + "".join(f"| {name} | {count} |\n" for name, count in failures.items())


def evaluate_predictions(
    labels_path: Path,
    predictions_path: Path,
    output_path: Path,
    config_path: Path,
) -> dict[str, object]:
    config = load_json(config_path)
    evaluation = config["evaluation"]
    label_rows = _read_labels(labels_path)
    predictions = _read_predictions(predictions_path)
    joined = join_labels_and_predictions(label_rows, predictions)
    patient_rows = aggregate_patients(joined)
    image_metrics = classification_metrics(joined)
    patient_metrics = classification_metrics(patient_rows)
    bootstrap = patient_bootstrap_ci(
        patient_rows,
        replicates=int(evaluation["bootstrap_replicates"]),
        seed=int(evaluation["bootstrap_seed"]),
        confidence_level=float(evaluation["confidence_level"]),
    )
    statuses = Counter(str(row["status"]) for row in joined)
    failure_accounting = dict(sorted(statuses.items()))
    failure_accounting["parse_ok_false"] = sum(not bool(row["parse_ok"]) for row in joined)
    metadata_fields = ("backend", "profile", "model_id", "model_revision", "precision")
    prediction_metadata = {}
    for field in metadata_fields:
        values = {str(row.get(field, "unknown")) for row in predictions.values()}
        if len(values) != 1:
            raise ValueError(f"Predictions contain mixed {field} values: {sorted(values)}")
        prediction_metadata[field] = next(iter(values))
    metrics: dict[str, object] = {
        "primary_unit": "patient",
        "patient_aggregation": "strict majority across all patient images; otherwise UNCERTAIN",
        "failure_policy": "UNCERTAIN, parse failure, runtime failure and missing prediction remain in denominator as incorrect",
        "labels_manifest": str(labels_path.resolve()),
        "labels_manifest_sha256": sha256_file(labels_path),
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": sha256_file(predictions_path),
        "config_sha256": sha256_file(config_path),
        "prediction_metadata": prediction_metadata,
        "primary_patient_level": patient_metrics,
        "primary_patient_bootstrap_ci": bootstrap,
        "secondary_image_level": image_metrics,
        "failure_accounting": failure_accounting,
    }

    comparator_value = evaluation.get("comparator_predictions")
    if comparator_value:
        comparator_path = resolve_pack_path(comparator_value)
        comparator_joined = join_labels_and_predictions(
            label_rows, _read_predictions(comparator_path)
        )
        comparator_patients = aggregate_patients(comparator_joined)
        comparator_metrics = classification_metrics(comparator_patients)
        metrics["paired_comparison"] = {
            "comparator_predictions": str(comparator_path),
            "comparator_sha256": sha256_file(comparator_path),
            "candidate_minus_comparator_balanced_accuracy": float(
                patient_metrics["balanced_accuracy"]
            )
            - float(comparator_metrics["balanced_accuracy"]),
            "comparator_patient_metrics": comparator_metrics,
            "paired_bootstrap_candidate_minus_comparator": (
                paired_patient_bootstrap_difference_ci(
                    patient_rows,
                    comparator_patients,
                    replicates=int(evaluation["bootstrap_replicates"]),
                    seed=int(evaluation["bootstrap_seed"]),
                    confidence_level=float(evaluation["confidence_level"]),
                )
            ),
            "mcnemar": paired_mcnemar(patient_rows, comparator_patients),
        }

    output_path.mkdir(parents=True, exist_ok=True)
    _write_csv(output_path / "joined_predictions.csv", joined)
    _write_csv(output_path / "patient_predictions.csv", patient_rows)
    write_json(output_path / "metrics.json", metrics)
    (output_path / "evaluation_report.md").write_text(
        _markdown_report(metrics), encoding="utf-8"
    )
    artifact_paths = [
        output_path / "joined_predictions.csv",
        output_path / "patient_predictions.csv",
        output_path / "metrics.json",
        output_path / "evaluation_report.md",
    ]
    (output_path / "artifact_checksums.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in artifact_paths),
        encoding="utf-8",
    )
    return metrics
