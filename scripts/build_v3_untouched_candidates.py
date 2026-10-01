#!/usr/bin/env python3
"""Exclude every test image belonging to a pilot-candidate patient."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pilot-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite existing output: {args.output}")

    fields, rows = read_rows(args.manifest)
    _, pilot_rows = read_rows(args.pilot_candidates)
    touched_patients = {row["patient_id"] for row in pilot_rows}
    selected = [
        row
        for row in rows
        if row["source_split"] == "test" and row["patient_id"] not in touched_patients
    ]
    counts = Counter(row["normalized_class"] for row in selected)
    patients = {row["patient_id"] for row in selected}
    if len(selected) != 582 or len(patients) != 411:
        raise RuntimeError(
            f"Unexpected untouched candidate size: {len(selected)} images, "
            f"{len(patients)} patients"
        )
    if counts != {"NORMAL": 224, "PNEUMONIA": 358}:
        raise RuntimeError(f"Unexpected class counts: {dict(counts)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)


if __name__ == "__main__":
    main()
