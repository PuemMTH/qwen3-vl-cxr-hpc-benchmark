#!/usr/bin/env python3
"""Build a checked blind manifest and a separate label escrow."""

from __future__ import annotations

from pathlib import Path

from vlm_cxr.preprocess import prepare_dataset


def run(input_path: Path, output_path: Path, config_path: Path | None = None) -> dict[str, object]:
    if config_path is None:
        raise ValueError("preprocess requires --config")
    return prepare_dataset(input_path, output_path, config_path)
