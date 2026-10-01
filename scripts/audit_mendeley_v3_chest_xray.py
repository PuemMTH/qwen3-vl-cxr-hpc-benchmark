#!/usr/bin/env python3
"""Audit Mendeley V3 CXR patient IDs from the official ZIP directory.

This reads the remote ZIP central directory without downloading the 8.4 GB OCT
payload, writes a metadata manifest for the CXR entries, and optionally compares
the archive entries with a local V2/Hugging Face export by exact bytes using
file-size and CRC32 keys.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from remotezip import RemoteZip


DATASET_URL = "https://data.mendeley.com/datasets/rscbjbr9sj/3"
ARCHIVE_URL = (
    "https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/"
    "810b2ce2-11c3-4424-996e-3bef36600907/file_downloaded"
)
ARCHIVE_SHA256 = "acbc42869440eec719361942d0ba560f132742230262f22e78bfb93eb5ea4563"
ARCHIVE_SIZE = 8_441_154_111
NAME_PATTERN = re.compile(
    r"^(?P<disease>NORMAL|BACTERIA|VIRUS)-(?P<patient_id>\d+)-"
    r"(?P<image_number>\d+)\.jpeg$",
    flags=re.IGNORECASE,
)
FIELDS = [
    "dataset_version",
    "archive_sha256",
    "archive_path",
    "split",
    "folder_class",
    "disease_label",
    "normalized_class",
    "patient_id",
    "image_number",
    "file_size",
    "crc32",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-root", type=Path)
    parser.add_argument("--local-manifest", type=Path)
    return parser.parse_args()


def audit_remote() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    with RemoteZip(ARCHIVE_URL) as archive:
        for info in archive.infolist():
            name = info.filename
            if not name.startswith("CellData/chest_xray/") or not name.lower().endswith(
                ".jpeg"
            ):
                continue
            parts = PurePosixPath(name).parts
            if len(parts) != 5:
                raise RuntimeError(f"Unexpected CXR archive path: {name}")
            match = NAME_PATTERN.fullmatch(parts[-1])
            if match is None:
                raise RuntimeError(f"Unexpected V3 CXR filename: {name}")
            disease = match.group("disease").upper()
            normalized = "NORMAL" if disease == "NORMAL" else "PNEUMONIA"
            rows.append(
                {
                    "dataset_version": 3,
                    "archive_sha256": ARCHIVE_SHA256,
                    "archive_path": name,
                    "split": parts[2],
                    "folder_class": parts[3],
                    "disease_label": disease,
                    "normalized_class": normalized,
                    "patient_id": match.group("patient_id"),
                    "image_number": match.group("image_number"),
                    "file_size": info.file_size,
                    "crc32": f"{info.CRC:08x}",
                }
            )
    return rows


def compare_local(
    remote_rows: list[dict[str, str | int]], local_root: Path, local_manifest: Path
) -> dict[str, object]:
    with local_manifest.open(newline="", encoding="utf-8") as handle:
        local_rows = list(csv.DictReader(handle))

    local_by_key: defaultdict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in local_rows:
        path = local_root / row["relative_path"]
        data = path.read_bytes()
        key = (len(data), zlib.crc32(data) & 0xFFFFFFFF)
        local_by_key[key].append(row)

    remote_by_key: defaultdict[tuple[int, int], list[dict[str, str | int]]] = (
        defaultdict(list)
    )
    for row in remote_rows:
        remote_by_key[(int(row["file_size"]), int(str(row["crc32"]), 16))].append(row)

    safe_rows = 0
    safe_by_local_split: Counter[str] = Counter()
    safe_patients_by_local_split: defaultdict[str, set[str]] = defaultdict(set)
    for key in local_by_key.keys() & remote_by_key.keys():
        local_group = local_by_key[key]
        remote_group = remote_by_key[key]
        patient_ids = {str(row["patient_id"]) for row in remote_group}
        if len(local_group) == len(remote_group) and len(patient_ids) == 1:
            patient_id = next(iter(patient_ids))
            safe_rows += len(local_group)
            for row in local_group:
                safe_by_local_split[row["split"]] += 1
                safe_patients_by_local_split[row["split"]].add(patient_id)

    local_unmatched = [
        row
        for key in local_by_key.keys() - remote_by_key.keys()
        for row in local_by_key[key]
    ]
    remote_unmatched = [
        row
        for key in remote_by_key.keys() - local_by_key.keys()
        for row in remote_by_key[key]
    ]
    count_mismatch_keys = [
        key
        for key in local_by_key.keys() & remote_by_key.keys()
        if len(local_by_key[key]) != len(remote_by_key[key])
    ]
    return {
        "local_rows": len(local_rows),
        "remote_v3_rows": len(remote_rows),
        "exact_safe_mapped_local_rows": safe_rows,
        "exact_safe_mapped_percent": round(safe_rows / len(local_rows) * 100, 4),
        "safe_rows_by_local_split": dict(sorted(safe_by_local_split.items())),
        "safe_unique_patients_by_local_split": {
            split: len(values)
            for split, values in sorted(safe_patients_by_local_split.items())
        },
        "safe_patient_overlap_train_validation": len(
            safe_patients_by_local_split["train"]
            & safe_patients_by_local_split["validation"]
        ),
        "safe_patient_overlap_train_test": len(
            safe_patients_by_local_split["train"]
            & safe_patients_by_local_split["test"]
        ),
        "safe_patient_overlap_validation_test": len(
            safe_patients_by_local_split["validation"]
            & safe_patients_by_local_split["test"]
        ),
        "unmatched_local_rows": len(local_unmatched),
        "unmatched_local_distribution": {
            f"{split}/{label}": count
            for (split, label), count in sorted(
                Counter(
                    (row["split"], row["label"]) for row in local_unmatched
                ).items()
            )
        },
        "unmatched_remote_v3_rows": len(remote_unmatched),
        "unmatched_remote_distribution": {
            f"{split}/{label}": count
            for (split, label), count in sorted(
                Counter(
                    (str(row["split"]), str(row["normalized_class"]))
                    for row in remote_unmatched
                ).items()
            )
        },
        "content_key_count_mismatches": len(count_mismatch_keys),
        "mapping_conclusion": (
            "V3 patient IDs must not be copied wholesale into the local V2/HF "
            "manifest because not every local row has an exact one-to-one V3 match."
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = audit_remote()
    manifest_path = args.output_dir / "mendeley_v3_chest_xray_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    by_split = Counter(str(row["split"]) for row in rows)
    by_split_class = Counter(
        (str(row["split"]), str(row["normalized_class"])) for row in rows
    )
    by_disease = Counter(str(row["disease_label"]) for row in rows)
    patients_by_split: defaultdict[str, set[str]] = defaultdict(set)
    patients_by_split_class: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    images_by_patient: Counter[tuple[str, str]] = Counter()
    classes_by_patient: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        split = str(row["split"])
        normalized_class = str(row["normalized_class"])
        patient_id = str(row["patient_id"])
        patients_by_split[split].add(patient_id)
        patients_by_split_class[(split, normalized_class)].add(patient_id)
        images_by_patient[(split, patient_id)] += 1
        classes_by_patient[(split, patient_id)].add(normalized_class)

    summary: dict[str, object] = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_url": DATASET_URL,
        "dataset_version": 3,
        "archive_url": ARCHIVE_URL,
        "archive_sha256_from_official_api": ARCHIVE_SHA256,
        "archive_size_from_official_api": ARCHIVE_SIZE,
        "cxr_images": len(rows),
        "counts_by_split": dict(sorted(by_split.items())),
        "counts_by_split_class": {
            f"{split}/{label}": count
            for (split, label), count in sorted(by_split_class.items())
        },
        "counts_by_disease_filename_label": dict(sorted(by_disease.items())),
        "unique_patients_by_split": {
            split: len(values) for split, values in sorted(patients_by_split.items())
        },
        "unique_patients_by_split_class": {
            f"{split}/{label}": len(values)
            for (split, label), values in sorted(patients_by_split_class.items())
        },
        "unique_patients_total": len(set().union(*patients_by_split.values())),
        "patient_overlap_train_test": len(
            patients_by_split["train"] & patients_by_split["test"]
        ),
        "patients_spanning_normalized_classes": sum(
            len(classes) > 1 for classes in classes_by_patient.values()
        ),
        "patients_with_multiple_images": sum(
            count > 1 for count in images_by_patient.values()
        ),
        "max_images_per_patient": max(images_by_patient.values()),
        "filename_pattern": "DISEASE-randomized_patient_id-image_number.jpeg",
        "archive_downloaded": False,
        "audit_basis": "remote official ZIP central-directory metadata",
    }
    if args.local_root and args.local_manifest:
        summary["local_v2_hf_comparison"] = compare_local(
            rows, args.local_root, args.local_manifest
        )

    summary_path = args.output_dir / "mendeley_v3_chest_xray_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checksums_path = args.output_dir / "artifact_checksums.sha256"
    checksums_path.write_text(
        f"{sha256_file(manifest_path)}  {manifest_path.name}\n"
        f"{sha256_file(summary_path)}  {summary_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
