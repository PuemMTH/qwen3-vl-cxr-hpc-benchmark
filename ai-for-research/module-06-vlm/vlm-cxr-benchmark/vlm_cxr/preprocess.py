from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from .common import load_json, sha256_file, write_json


REQUIRED_FIELDS = {
    "image_id",
    "patient_id",
    "normalized_class",
    "source_split",
    "study_role",
    "relative_path",
    "sha256",
}


def _deterministic_stratified_limit(
    rows: list[dict[str, str]], limit: int | None, seed: int
) -> list[dict[str, str]]:
    if limit is None or limit >= len(rows):
        return rows
    labels = ("NORMAL", "PNEUMONIA")
    quotas = {labels[0]: limit // 2 + limit % 2, labels[1]: limit // 2}
    selected: list[dict[str, str]] = []
    for label in labels:
        candidates = [row for row in rows if row["normalized_class"] == label]
        ordered = sorted(
            candidates,
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['image_id']}".encode("utf-8")
            ).hexdigest(),
        )
        if len(ordered) < quotas[label]:
            raise ValueError(f"Not enough {label} rows for requested limit {limit}")
        selected.extend(ordered[: quotas[label]])
    return sorted(selected, key=lambda row: row["image_id"])


def prepare_dataset(
    input_path: Path, output_path: Path, config_path: Path
) -> dict[str, object]:
    config = load_json(config_path)
    dataset_config = config["dataset"]
    manifest_path = input_path if input_path.is_file() else input_path / "manifest.csv"
    dataset_root = manifest_path.parent
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset manifest missing fields: {sorted(missing)}")
        source_rows = list(reader)

    source_split = dataset_config["source_split"]
    study_role = dataset_config["study_role"]
    if source_split == "test" and not dataset_config.get("confirmatory_unlocked", False):
        raise PermissionError(
            "Confirmatory preprocessing is locked. Close readiness gates and set "
            "dataset.confirmatory_unlocked=true in a frozen config."
        )
    rows = [
        row
        for row in source_rows
        if row["source_split"] == source_split and row["study_role"] == study_role
    ]
    if not rows:
        raise ValueError(f"No rows for source_split={source_split}, study_role={study_role}")
    rows = _deterministic_stratified_limit(
        rows, dataset_config.get("limit"), int(dataset_config["selection_seed"])
    )

    seen_images: set[str] = set()
    patient_labels: defaultdict[str, set[str]] = defaultdict(set)
    blind_rows: list[dict[str, str]] = []
    label_rows: list[dict[str, str]] = []
    for row in rows:
        image_id = row["image_id"]
        if not image_id or image_id in seen_images:
            raise ValueError(f"Duplicate or empty image_id: {image_id!r}")
        seen_images.add(image_id)
        image_path = (dataset_root / row["relative_path"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if dataset_config.get("verify_sha256", True):
            actual = sha256_file(image_path)
            if actual != row["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {image_id}: {actual}")
        patient_labels[row["patient_id"]].add(row["normalized_class"])
        blind_rows.append(
            {
                "image_id": image_id,
                "patient_id": row["patient_id"],
                "source_split": row["source_split"],
                "study_role": row["study_role"],
                "image_path": str(image_path),
                "image_sha256": row["sha256"],
            }
        )
        label_rows.append(
            {
                "image_id": image_id,
                "patient_id": row["patient_id"],
                "true_label": row["normalized_class"],
                "source_split": row["source_split"],
                "study_role": row["study_role"],
            }
        )
    cross_class = {patient: values for patient, values in patient_labels.items() if len(values) > 1}
    if cross_class:
        raise ValueError(f"Patients span normalized classes: {list(cross_class)[:5]}")

    output_path.mkdir(parents=True, exist_ok=True)
    blind_path = output_path / "blind_manifest.csv"
    labels_path = output_path / "labels.csv"
    with blind_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(blind_rows[0]))
        writer.writeheader()
        writer.writerows(blind_rows)
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(label_rows[0]))
        writer.writeheader()
        writer.writerows(label_rows)

    image_counts = Counter(row["true_label"] for row in label_rows)
    patient_counts = Counter()
    for patient_id, labels in patient_labels.items():
        patient_counts[next(iter(labels))] += 1
    summary = {
        "dataset_manifest": str(manifest_path.resolve()),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "source_split": source_split,
        "study_role": study_role,
        "images": len(label_rows),
        "patients": len(patient_labels),
        "images_by_class": dict(sorted(image_counts.items())),
        "patients_by_class": dict(sorted(patient_counts.items())),
        "blind_manifest": str(blind_path.resolve()),
        "blind_manifest_sha256": sha256_file(blind_path),
        "labels_manifest": str(labels_path.resolve()),
        "labels_manifest_sha256": sha256_file(labels_path),
        "labels_separated_from_inference_manifest": True,
    }
    write_json(output_path / "summary.json", summary)
    return summary
