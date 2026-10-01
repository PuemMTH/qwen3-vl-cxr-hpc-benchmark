#!/usr/bin/env python3
"""Join escrowed labels only after inference and compute frozen metrics."""

from __future__ import annotations

from pathlib import Path

from vlm_cxr.evaluate import evaluate_predictions


def run(input_path: Path, output_path: Path, config_path: Path | None = None) -> dict[str, object]:
    if config_path is None:
        raise ValueError("postprocess requires --config")
    predictions = input_path / "predictions.jsonl"
    labels = input_path.parent / "preprocess" / "labels.csv"
    if not predictions.is_file():
        raise FileNotFoundError(predictions)
    if not labels.is_file():
        raise FileNotFoundError(labels)
    return evaluate_predictions(labels, predictions, output_path, config_path)
