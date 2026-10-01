#!/usr/bin/env python3
"""Build the deterministic 20-image pilot candidate list from canonical V3."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite existing output: {args.output}")

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        source_fields = list(reader.fieldnames or [])
        rows = list(reader)

    selected: list[dict[str, str]] = []
    ranks: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        label = row["normalized_class"]
        if row["source_split"] != "test" or label not in {"NORMAL", "PNEUMONIA"}:
            continue
        if ranks[label] >= 10:
            continue
        ranks[label] += 1
        selected.append(
            {
                **row,
                "pilot_candidate_rank_within_class": str(ranks[label]),
                "selection_rule": "first 10 canonical V3 test-manifest rows per normalized class",
            }
        )
    if dict(ranks) != {"NORMAL": 10, "PNEUMONIA": 10}:
        raise RuntimeError(f"Unexpected selected counts: {dict(ranks)}")
    if len({row["image_id"] for row in selected}) != 20:
        raise RuntimeError("Pilot candidate image IDs are not unique")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = source_fields + ["pilot_candidate_rank_within_class", "selection_rule"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)


if __name__ == "__main__":
    main()
