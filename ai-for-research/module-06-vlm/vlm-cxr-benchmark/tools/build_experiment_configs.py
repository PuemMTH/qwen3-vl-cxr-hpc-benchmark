#!/usr/bin/env python3
"""Generate the registered A/B/C pilot configurations without unlocking Test."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "experiments"
PROMPT = (
    'Classify this pediatric AP chest X-ray as NORMAL or PNEUMONIA. Return only '
    'one JSON object with exactly one key: {"label":"NORMAL|PNEUMONIA|UNCERTAIN"}. '
    'Use UNCERTAIN when the image is insufficient. Do not return Markdown or explanation.'
)


def config(profile: str, limit: int, batch: int, warmup_images: int) -> dict:
    value = {
        "dataset": {
            "source_split": "train",
            "study_role": "development",
            "limit": limit,
            "selection_seed": 42,
            "verify_sha256": True,
            "confirmatory_unlocked": False,
        },
        "inference": {
            "backend": "transformers",
            "model_registry": "configs/models.lock.json",
            "profile": profile,
            "batch_size": batch,
            "max_image_edge": 512,
            "max_new_tokens": 128,
            "warmup_batches": math.ceil(warmup_images / batch),
            "warmup_images_minimum": warmup_images,
            "local_files_only": True,
            "seed": 42,
            "prompt": PROMPT,
        },
        "evaluation": {
            "bootstrap_replicates": 2000,
            "bootstrap_seed": 20260923,
            "confidence_level": 0.95,
        },
        "study_status": "development_pilot_only",
        "confirmatory_test_accessed": False,
    }
    return value


def write(name: str, value: dict) -> None:
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for profile in ("2b-bf16", "4b-bf16", "8b-nf4"):
        write(f"a-{profile}-batch1", config(profile, 1000, 1, 50))
    for profile in ("2b-bf16", "4b-bf16", "8b-nf4", "8b-bf16"):
        write(f"b-development-{profile}-batch1", config(profile, 600, 1, 1))
    for batch in (1, 2, 4, 8):
        write(f"c1-8b-bf16-batch{batch}", config("8b-bf16", 1000, batch, 50))
        write(f"c2-8b-bf16-batch{batch}", config("8b-bf16", 2000, batch, 50))
        write(f"c1-4b-bf16-batch{batch}", config("4b-bf16", 1000, batch, 50))
        write(f"c2-4b-bf16-batch{batch}", config("4b-bf16", 2000, batch, 50))


if __name__ == "__main__":
    main()
