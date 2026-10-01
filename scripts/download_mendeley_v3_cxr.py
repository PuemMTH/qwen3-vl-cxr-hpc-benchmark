#!/usr/bin/env python3
"""Download and extract only the CXR portion of official Mendeley Data V3.

The 8.44 GB ZhangLabData.zip also contains OCT data. The CXR local-file records
are contiguous at the beginning of the archive, so this script requests only
that byte range, validates every member against the ZIP central directory, and
writes a patient-aware manifest for NORMAL/PNEUMONIA only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import struct
import tempfile
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import requests
from PIL import Image
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
    "dataset_url",
    "archive_sha256",
    "source_archive_path",
    "source_split",
    "study_role",
    "normalized_class",
    "disease_label",
    "patient_id",
    "image_id",
    "image_number",
    "relative_path",
    "bytes",
    "width",
    "height",
    "mode",
    "crc32",
    "sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_members():
    with RemoteZip(ARCHIVE_URL) as archive:
        all_infos = sorted(archive.infolist(), key=lambda item: item.header_offset)
    cxr_infos = [
        info
        for info in all_infos
        if info.filename.startswith("CellData/chest_xray/")
        and info.filename.lower().endswith(".jpeg")
    ]
    if len(cxr_infos) != 5_856:
        raise RuntimeError(f"Expected 5,856 CXR images, found {len(cxr_infos):,}")
    cxr_offsets = {info.header_offset for info in cxr_infos}
    last_cxr_offset = max(cxr_offsets)
    later_offsets = [
        info.header_offset for info in all_infos if info.header_offset > last_cxr_offset
    ]
    if not later_offsets:
        raise RuntimeError("Could not locate the first post-CXR ZIP member")
    range_end = min(later_offsets) - 1
    non_cxr_files_in_range = [
        info.filename
        for info in all_infos
        if info.header_offset <= range_end
        and not info.is_dir()
        and not (
            info.filename.startswith("CellData/chest_xray/")
            and info.filename.lower().endswith(".jpeg")
        )
    ]
    unexpected_files = [
        name for name in non_cxr_files_in_range if PurePosixPath(name).name != ".DS_Store"
    ]
    if unexpected_files:
        raise RuntimeError(
            "The selective range contains unexpected non-CXR files: "
            + repr(unexpected_files[:3])
        )
    return cxr_infos, range_end


def download_prefix(path: Path, range_end: int) -> None:
    expected = range_end + 1
    headers = {"Range": f"bytes=0-{range_end}", "Accept-Encoding": "identity"}
    with requests.get(ARCHIVE_URL, headers=headers, stream=True, timeout=(30, 300)) as response:
        if response.status_code != 206:
            raise RuntimeError(
                f"Server did not honor the byte range (HTTP {response.status_code}); "
                "aborting to avoid downloading the full archive"
            )
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes 0-{range_end}/"):
            raise RuntimeError(f"Unexpected Content-Range: {content_range!r}")
        written = 0
        next_report = 128 * 1024 * 1024
        with path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                if written >= next_report:
                    print(
                        f"downloaded {written / 1_000_000:.1f}/"
                        f"{expected / 1_000_000:.1f} MB",
                        flush=True,
                    )
                    next_report += 128 * 1024 * 1024
    if written != expected:
        raise RuntimeError(f"Range size mismatch: expected {expected}, wrote {written}")


def extract_member(prefix_handle, info, destination: Path) -> bytes:
    prefix_handle.seek(info.header_offset)
    local_header = prefix_handle.read(30)
    if len(local_header) != 30:
        raise RuntimeError(f"Short local header for {info.filename}")
    fields = struct.unpack("<IHHHHHIIIHH", local_header)
    signature, compression, name_length, extra_length = fields[0], fields[3], fields[9], fields[10]
    if signature != 0x04034B50:
        raise RuntimeError(f"Bad local-header signature for {info.filename}")
    stored_name = prefix_handle.read(name_length).decode("utf-8")
    if stored_name != info.filename:
        raise RuntimeError(f"Local/central filename mismatch for {info.filename}")
    prefix_handle.seek(extra_length, 1)
    compressed = prefix_handle.read(info.compress_size)
    if len(compressed) != info.compress_size:
        raise RuntimeError(f"Short compressed payload for {info.filename}")
    if compression == 0:
        data = compressed
    elif compression == 8:
        data = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(f"Unsupported ZIP compression {compression} for {info.filename}")
    if len(data) != info.file_size:
        raise RuntimeError(f"Uncompressed size mismatch for {info.filename}")
    crc = zlib.crc32(data) & 0xFFFFFFFF
    if crc != info.CRC:
        raise RuntimeError(f"CRC32 mismatch for {info.filename}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return data


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=".mendeley-v3-cxr-", dir=output_dir.parent))
    prefix_path = work_dir / "cxr-archive-prefix.bin"
    final_staging = work_dir / "dataset"
    rows: list[dict[str, object]] = []
    try:
        infos, range_end = discover_members()
        print(
            f"requesting official archive bytes 0-{range_end} "
            f"({(range_end + 1) / 1_000_000:.1f} MB)",
            flush=True,
        )
        download_prefix(prefix_path, range_end)
        with prefix_path.open("rb") as prefix_handle:
            for index, info in enumerate(infos, start=1):
                parts = PurePosixPath(info.filename).parts
                if len(parts) != 5 or parts[:2] != ("CellData", "chest_xray"):
                    raise RuntimeError(f"Unexpected CXR path: {info.filename}")
                source_split, folder_class, filename = parts[2], parts[3], parts[4]
                if source_split not in {"train", "test"}:
                    raise RuntimeError(f"Unexpected source split: {source_split}")
                if folder_class not in {"NORMAL", "PNEUMONIA"}:
                    raise RuntimeError(f"Unexpected target class: {folder_class}")
                match = NAME_PATTERN.fullmatch(filename)
                if match is None:
                    raise RuntimeError(f"Unexpected filename: {filename}")
                disease = match.group("disease").upper()
                normalized = "NORMAL" if disease == "NORMAL" else "PNEUMONIA"
                if normalized != folder_class:
                    raise RuntimeError(f"Filename/folder class mismatch: {info.filename}")
                relative_path = Path("images") / source_split / normalized / filename
                destination = final_staging / relative_path
                data = extract_member(prefix_handle, info, destination)
                with Image.open(destination) as image:
                    image.verify()
                with Image.open(destination) as image:
                    width, height, mode = image.width, image.height, image.mode
                rows.append(
                    {
                        "dataset_version": 3,
                        "dataset_url": DATASET_URL,
                        "archive_sha256": ARCHIVE_SHA256,
                        "source_archive_path": info.filename,
                        "source_split": source_split,
                        "study_role": "development" if source_split == "train" else "confirmatory",
                        "normalized_class": normalized,
                        "disease_label": disease,
                        "patient_id": match.group("patient_id"),
                        "image_id": Path(filename).stem,
                        "image_number": match.group("image_number"),
                        "relative_path": relative_path.as_posix(),
                        "bytes": len(data),
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "crc32": f"{info.CRC:08x}",
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
                if index % 500 == 0 or index == len(infos):
                    print(f"validated and extracted {index:,}/{len(infos):,} images", flush=True)

        manifest_path = final_staging / "manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        images_by_split_class = Counter(
            (str(row["source_split"]), str(row["normalized_class"])) for row in rows
        )
        patients_by_split_class: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        patients_by_split: defaultdict[str, set[str]] = defaultdict(set)
        images_by_patient: Counter[tuple[str, str]] = Counter()
        hashes: defaultdict[str, list[str]] = defaultdict(list)
        for row in rows:
            split = str(row["source_split"])
            label = str(row["normalized_class"])
            patient = str(row["patient_id"])
            patients_by_split[split].add(patient)
            patients_by_split_class[(split, label)].add(patient)
            images_by_patient[(split, patient)] += 1
            hashes[str(row["sha256"])].append(str(row["image_id"]))
        duplicate_groups = [values for values in hashes.values() if len(values) > 1]
        summary = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_url": DATASET_URL,
            "dataset_version": 3,
            "archive_url": ARCHIVE_URL,
            "archive_sha256_from_official_api": ARCHIVE_SHA256,
            "archive_size_from_official_api": ARCHIVE_SIZE,
            "selective_range": f"bytes=0-{range_end}",
            "selective_range_bytes": range_end + 1,
            "retained_scope": "CellData/chest_xray only; normalized classes NORMAL and PNEUMONIA",
            "excluded_archive_scope": "all OCT files and non-CXR archive content",
            "images_total": len(rows),
            "images_by_split_class": {
                f"{split}/{label}": count
                for (split, label), count in sorted(images_by_split_class.items())
            },
            "unique_patients_total": len(set().union(*patients_by_split.values())),
            "unique_patients_by_split": {
                split: len(values) for split, values in sorted(patients_by_split.items())
            },
            "unique_patients_by_split_class": {
                f"{split}/{label}": len(values)
                for (split, label), values in sorted(patients_by_split_class.items())
            },
            "patient_overlap_train_test": len(
                patients_by_split["train"] & patients_by_split["test"]
            ),
            "patients_with_multiple_images": sum(
                count > 1 for count in images_by_patient.values()
            ),
            "max_images_per_patient": max(images_by_patient.values()),
            "exact_duplicate_content_groups": len(duplicate_groups),
            "exact_duplicate_content_images": sum(len(group) for group in duplicate_groups),
            "jpeg_validation_failures": 0,
        }
        summary_path = final_staging / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        checksums_path = final_staging / "artifact_checksums.sha256"
        checksums_path.write_text(
            f"{sha256_file(manifest_path)}  {manifest_path.name}\n"
            f"{sha256_file(summary_path)}  {summary_path.name}\n",
            encoding="utf-8",
        )
        prefix_path.unlink()
        final_staging.rename(output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    else:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
