#!/usr/bin/env python3
"""Create a non-sensitive aggregate audit of embedded JPEG metadata."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import ExifTags, Image


SENSITIVE_TAG_NAMES = {
    "Artist",
    "Copyright",
    "GPSInfo",
    "ImageDescription",
    "Make",
    "Model",
    "OwnerName",
    "UserComment",
    "XPAuthor",
    "XPComment",
    "XPKeywords",
    "XPSubject",
    "XPTitle",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.root.glob("*/*/*.jpeg"))
    exif_tag_counts: Counter[str] = Counter()
    info_key_counts: Counter[str] = Counter()
    files_with_exif = 0
    files_with_sensitive_tag = 0
    sensitive_tag_counts: Counter[str] = Counter()

    for path in files:
        with Image.open(path) as image:
            exif = image.getexif()
            if exif:
                files_with_exif += 1
            current_sensitive: set[str] = set()
            for key in exif:
                name = str(ExifTags.TAGS.get(key, key))
                exif_tag_counts[name] += 1
                if name in SENSITIVE_TAG_NAMES:
                    sensitive_tag_counts[name] += 1
                    current_sensitive.add(name)
            if current_sensitive:
                files_with_sensitive_tag += 1
            for key in image.info:
                info_key_counts[str(key)] += 1

    report = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root),
        "files_scanned": len(files),
        "files_with_exif_container": files_with_exif,
        "exif_tag_counts": dict(sorted(exif_tag_counts.items())),
        "image_info_key_counts": dict(sorted(info_key_counts.items())),
        "sensitive_tag_names_checked": sorted(SENSITIVE_TAG_NAMES),
        "files_with_checked_sensitive_exif_tag": files_with_sensitive_tag,
        "sensitive_exif_tag_counts": dict(sorted(sensitive_tag_counts.items())),
        "limitations": [
            "No pixel-level OCR or burned-in identifier review was performed.",
            "XMP/Photoshop payload values were not interpreted for identifiers.",
            "Absence of selected EXIF tags does not establish de-identification.",
        ],
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
