from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from vlm_cxr.common import sha256_file
from vlm_cxr.evaluate import evaluate_predictions
from vlm_cxr.inference import run_inference, summarize_performance
from vlm_cxr.preprocess import prepare_dataset


def _fixture_dataset(root: Path) -> Path:
    fields = [
        "image_id",
        "patient_id",
        "normalized_class",
        "source_split",
        "study_role",
        "relative_path",
        "sha256",
    ]
    rows = []
    for index, label in enumerate(("NORMAL", "NORMAL", "PNEUMONIA", "PNEUMONIA")):
        path = root / "images" / "train" / label / f"image-{index}.jpeg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (16, 16), color=index * 40).save(path, format="JPEG")
        rows.append(
            {
                "image_id": f"image-{index}",
                "patient_id": f"patient-{index}",
                "normalized_class": label,
                "source_split": "train",
                "study_role": "development",
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _config(path: Path, source_split: str = "train", unlocked: bool = False) -> Path:
    value = {
        "dataset": {
            "source_split": source_split,
            "study_role": "development" if source_split == "train" else "confirmatory",
            "limit": None,
            "selection_seed": 42,
            "verify_sha256": True,
            "confirmatory_unlocked": unlocked,
        },
        "inference": {
            "backend": "mock",
            "model_registry": "configs/models.lock.json",
            "profile": "2b-bf16",
            "batch_size": 2,
            "max_image_edge": 512,
            "max_new_tokens": 32,
            "warmup_batches": 0,
            "local_files_only": True,
            "seed": 42,
            "prompt": "fixture",
        },
        "evaluation": {
            "bootstrap_replicates": 20,
            "bootstrap_seed": 42,
            "confidence_level": 0.95,
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_mock_pipeline_is_label_blind_and_complete(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = _fixture_dataset(dataset)
    config = _config(tmp_path / "config.json")
    preprocess_dir = tmp_path / "outputs" / "preprocess"
    inference_dir = tmp_path / "outputs" / "intermediate"
    final_dir = tmp_path / "outputs" / "final"

    pre = prepare_dataset(manifest, preprocess_dir, config)
    assert pre["images"] == 4
    header = (preprocess_dir / "blind_manifest.csv").read_text().splitlines()[0]
    assert "true_label" not in header and "normalized_class" not in header

    run = run_inference(preprocess_dir, inference_dir, config)
    assert run["prediction_rows"] == 4
    assert run["performance"]["successful_measured_images"] == 4
    metrics = evaluate_predictions(
        preprocess_dir / "labels.csv",
        inference_dir / "predictions.jsonl",
        final_dir,
        config,
    )
    assert metrics["primary_patient_level"]["n"] == 4
    assert (final_dir / "artifact_checksums.sha256").is_file()


def test_confirmatory_preprocessing_is_locked(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = _fixture_dataset(dataset)
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    for row in rows:
        row["source_split"] = "test"
        row["study_role"] = "confirmatory"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = _config(tmp_path / "locked.json", source_split="test", unlocked=False)
    with pytest.raises(PermissionError, match="Confirmatory preprocessing is locked"):
        prepare_dataset(manifest, tmp_path / "out", config)


def test_performance_scopes_are_not_mixed() -> None:
    batches = [
        {"warmup": True, "status": "success", "batch_size": 2,
         "generation_seconds": 99.0, "pipeline_seconds": 100.0},
        {"warmup": False, "status": "success", "batch_size": 2,
         "generation_seconds": 2.0, "pipeline_seconds": 3.0},
        {"warmup": False, "status": "success", "batch_size": 2,
         "generation_seconds": 4.0, "pipeline_seconds": 5.0},
        {"warmup": False, "status": "oom", "batch_size": 2,
         "pipeline_seconds": 0.0},
    ]
    result = summarize_performance(
        batches,
        {"model_loading_seconds": 7.0, "peak_allocated_bytes": 10,
         "peak_reserved_bytes": 12},
    )
    assert result["successful_measured_images"] == 4
    assert result["generation_seconds_total"] == 6.0
    assert result["generation_throughput_images_per_hour"] == 2400.0
    assert result["amortized_generation_seconds_per_image"]["p50"] == 1.5
    assert result["model_loading_seconds"] == 7.0
