#!/usr/bin/env python3
"""Export the pinned HF chest X-ray dataset into split/class folders.

The exporter accepts only label IDs 0 and 1, preserves the encoded image bytes,
verifies every image with Pillow, and writes traceable manifests and checksums.
Filename-derived subject tokens are deliberately not called patient IDs because
the Hugging Face schema does not provide a verified patient_id column.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


LABELS = {0: "NORMAL", 1: "PNEUMONIA"}
SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}
MANIFEST_FIELDS = [
    "dataset_id",
    "dataset_revision",
    "split",
    "label_id",
    "label",
    "source_shard",
    "source_row",
    "original_filename",
    "filename_subject_token",
    "relative_path",
    "sha256",
    "byte_size",
    "width",
    "height",
    "image_format",
    "development_flag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    return parser.parse_args()


def split_from_name(name: str) -> str:
    if name.startswith("train-"):
        return "train"
    if name.startswith("validation-"):
        return "validation"
    if name.startswith("test-"):
        return "test"
    raise ValueError(f"Unrecognized shard name: {name}")


def filename_subject_token(filename: str) -> str:
    """Return a grouping token only; this is not a verified patient ID."""
    stem = Path(filename).stem
    pneumonia = re.match(r"^(person\d+)_", stem, flags=re.IGNORECASE)
    if pneumonia:
        return pneumonia.group(1).lower()
    normal = re.match(r"^((?:NORMAL2-)?IM-\d+)-", stem, flags=re.IGNORECASE)
    if normal:
        return normal.group(1).upper()
    return ""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source_data = args.source / "data"
    shards = sorted(
        source_data.glob("*.parquet"),
        key=lambda path: (SPLIT_ORDER[split_from_name(path.name)], path.name),
    )
    if not shards:
        raise SystemExit(f"No Parquet shards found under {source_data}")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int]] = []
    excluded: list[dict[str, str | int]] = []
    output_paths: set[str] = set()
    hash_paths: defaultdict[str, list[str]] = defaultdict(list)
    subject_splits: defaultdict[str, set[str]] = defaultdict(set)
    source_checksums: list[tuple[str, str]] = []

    for shard in shards:
        split = split_from_name(shard.name)
        source_checksums.append((sha256_file(shard), f"data/{shard.name}"))
        table = pq.read_table(shard, columns=["image", "label"])
        images = table.column("image").to_pylist()
        labels = table.column("label").to_pylist()
        for source_row, (image, raw_label) in enumerate(zip(images, labels)):
            label_id = int(raw_label)
            original_filename = Path(image["path"]).name
            if label_id not in LABELS:
                excluded.append(
                    {
                        "split": split,
                        "source_shard": shard.name,
                        "source_row": source_row,
                        "original_filename": original_filename,
                        "label_id": label_id,
                    }
                )
                continue

            label = LABELS[label_id]
            data = image["bytes"]
            if not isinstance(data, bytes):
                raise TypeError(f"Missing encoded bytes in {shard.name}:{source_row}")
            relative_path = Path(split) / label / original_filename
            relative_text = relative_path.as_posix()
            if relative_text in output_paths:
                raise RuntimeError(f"Duplicate output path: {relative_text}")
            output_paths.add(relative_text)

            with Image.open(io.BytesIO(data)) as decoded:
                decoded.verify()
            with Image.open(io.BytesIO(data)) as decoded:
                width, height = decoded.size
                image_format = decoded.format or "UNKNOWN"

            destination = args.output / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            digest = sha256_bytes(data)
            if sha256_file(destination) != digest:
                raise RuntimeError(f"Checksum mismatch after writing {destination}")

            token = filename_subject_token(original_filename)
            if token:
                subject_splits[token].add(split)
            hash_paths[digest].append(relative_text)
            rows.append(
                {
                    "dataset_id": "hf-vision/chest-xray-pneumonia",
                    "dataset_revision": args.revision,
                    "split": split,
                    "label_id": label_id,
                    "label": label,
                    "source_shard": shard.name,
                    "source_row": source_row,
                    "original_filename": original_filename,
                    "filename_subject_token": token,
                    "relative_path": relative_text,
                    "sha256": digest,
                    "byte_size": len(data),
                    "width": width,
                    "height": height,
                    "image_format": image_format,
                    "development_flag": "unassigned",
                }
            )

    manifest = args.output / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    pilot_rows: list[dict[str, str | int]] = []
    for label in LABELS.values():
        selected = [
            row for row in rows if row["split"] == "test" and row["label"] == label
        ][:10]
        for row in selected:
            copy = dict(row)
            copy["development_flag"] = "pilot_candidate_first_10_per_class"
            pilot_rows.append(copy)
    pilot_manifest = args.output / "pilot_20_candidate_manifest.csv"
    with pilot_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(pilot_rows)

    if excluded:
        excluded_path = args.output / "excluded_non_target_rows.csv"
        with excluded_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(excluded[0]))
            writer.writeheader()
            writer.writerows(excluded)

    counts: Counter[tuple[str, str]] = Counter(
        (str(row["split"]), str(row["label"])) for row in rows
    )
    cross_split_tokens = {
        token: sorted(splits)
        for token, splits in sorted(subject_splits.items())
        if len(splits) > 1
    }
    duplicate_hash_groups = [
        paths for paths in hash_paths.values() if len(paths) > 1
    ]
    summary = {
        "dataset_id": "hf-vision/chest-xray-pneumonia",
        "dataset_revision": args.revision,
        "source_url": "https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia",
        "original_source_doi": "10.17632/rscbjbr9sj.2",
        "license": "CC BY 4.0",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted_labels": LABELS,
        "other_class_rows_excluded": len(excluded),
        "total_exported": len(rows),
        "counts": {
            split: {
                label: counts[(split, label)] for label in LABELS.values()
            }
            for split in SPLIT_ORDER
        },
        "schema_has_patient_id": False,
        "filename_subject_token_note": (
            "Derived from filenames for audit only; not a verified patient_id."
        ),
        "filename_subject_token_count": len(subject_splits),
        "filename_subject_tokens_crossing_splits": cross_split_tokens,
        "duplicate_content_group_count": len(duplicate_hash_groups),
        "duplicate_content_groups": duplicate_hash_groups,
        "pilot_candidate_rule": "first 10 rows per target label in the test split",
        "pilot_candidate_count": len(pilot_rows),
        "source_card_row_total": 5856,
        "source_card_narrative_total": 5863,
        "row_count_discrepancy_note": (
            "Pinned Parquet metadata and split counts total 5,856; the narrative "
            "README sentence says 5,863. Export uses the actual pinned Parquet rows."
        ),
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    checksums = args.output / "source_checksums.sha256"
    checksums.write_text(
        "".join(f"{digest}  {relative}\n" for digest, relative in source_checksums),
        encoding="utf-8",
    )
    artifact_checksums = args.output / "artifact_checksums.sha256"
    artifact_paths = [manifest, pilot_manifest, summary_path, checksums]
    if excluded:
        artifact_paths.append(args.output / "excluded_non_target_rows.csv")
    artifact_checksums.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(args.output).as_posix()}\n"
            for path in artifact_paths
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
