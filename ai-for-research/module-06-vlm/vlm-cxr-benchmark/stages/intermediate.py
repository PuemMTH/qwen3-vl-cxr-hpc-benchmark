#!/usr/bin/env python3
"""Run blind model inference; this stage never reads labels.csv."""

from __future__ import annotations

from pathlib import Path

from vlm_cxr.inference import run_inference


def run(input_path: Path, output_path: Path, config_path: Path | None = None) -> dict[str, object]:
    if config_path is None:
        raise ValueError("intermediate requires --config")
    return run_inference(input_path, output_path, config_path)
