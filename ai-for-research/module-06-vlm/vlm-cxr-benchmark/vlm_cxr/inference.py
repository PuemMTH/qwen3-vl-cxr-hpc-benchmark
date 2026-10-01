from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common import load_json, resolve_pack_path, sha256_file, write_json
from .parser import parse_model_output


BLIND_FIELDS = {
    "image_id",
    "patient_id",
    "source_split",
    "study_role",
    "image_path",
    "image_sha256",
}


def _read_blind_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = BLIND_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Blind manifest missing fields: {sorted(missing)}")
        if "true_label" in (reader.fieldnames or []) or "normalized_class" in (
            reader.fieldnames or []
        ):
            raise ValueError("Inference manifest must not contain ground-truth labels")
        rows = list(reader)
    ids = [row["image_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Blind manifest must contain unique image IDs")
    return rows


def _mock_raw_text(image_id: str, seed: int) -> str:
    value = int(hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest(), 16) % 5
    label = "NORMAL" if value < 2 else "PNEUMONIA" if value < 4 else "UNCERTAIN"
    return json.dumps({"label": label}, separators=(",", ":"))


def _base_prediction(row: dict[str, str]) -> dict[str, object]:
    return {
        "image_id": row["image_id"],
        "patient_id": row["patient_id"],
        "source_split": row["source_split"],
        "study_role": row["study_role"],
        "image_sha256": row["image_sha256"],
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _run_mock(
    rows: list[dict[str, str]], config: dict[str, Any]
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    predictions = []
    started = time.perf_counter()
    for row in rows:
        raw_text = _mock_raw_text(row["image_id"], int(config["seed"]))
        parsed = parse_model_output(raw_text)
        predictions.append(
            {
                **_base_prediction(row),
                "status": "ok",
                "raw_text": raw_text,
                **parsed.to_dict(),
                "batch_index": len(predictions) // int(config["batch_size"]),
                "output_tokens": None,
            }
        )
    elapsed = time.perf_counter() - started
    batches = [
        {
            "batch_index": index,
            "batch_size": len(chunk),
            "generation_seconds": 0.0,
            "pipeline_seconds": elapsed / max(1, (len(rows) + int(config["batch_size"]) - 1) // int(config["batch_size"])),
            "status": "success",
        }
        for index, offset in enumerate(range(0, len(rows), int(config["batch_size"])))
        for chunk in [rows[offset : offset + int(config["batch_size"])]]
    ]
    return predictions, batches, {"backend": "mock", "pipeline_seconds": elapsed}


def summarize_performance(
    batches: list[dict[str, object]], environment: dict[str, object]
) -> dict[str, object]:
    """Summarize measured (non-warm-up) batches without mixing timing scopes."""
    measured = [
        row
        for row in batches
        if not bool(row.get("warmup", False)) and row.get("status") == "success"
    ]
    images = sum(int(row["batch_size"]) for row in measured)
    generation = [float(row["generation_seconds"]) for row in measured]
    pipeline = [float(row["pipeline_seconds"]) for row in measured]
    amortized = [
        float(row["generation_seconds"]) / int(row["batch_size"]) for row in measured
    ]

    def distribution(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        return {
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "minimum": min(values),
            "maximum": max(values),
        }

    generation_total = sum(generation)
    pipeline_total = sum(pipeline)
    return {
        "timing_scope_note": (
            "model loading, batch generation, amortized sec/image and end-to-end "
            "batch pipeline timing are reported separately"
        ),
        "successful_measured_batches": len(measured),
        "successful_measured_images": images,
        "model_loading_seconds": environment.get("model_loading_seconds"),
        "batch_generation_latency_seconds": distribution(generation),
        "amortized_generation_seconds_per_image": distribution(amortized),
        "batch_pipeline_latency_seconds": distribution(pipeline),
        "generation_seconds_total": generation_total,
        "pipeline_seconds_total": pipeline_total,
        "generation_throughput_images_per_hour": (
            3600 * images / generation_total if generation_total else None
        ),
        "pipeline_throughput_images_per_hour": (
            3600 * images / pipeline_total if pipeline_total else None
        ),
        "peak_allocated_bytes": environment.get("peak_allocated_bytes"),
        "peak_reserved_bytes": environment.get("peak_reserved_bytes"),
    }


def _run_transformers(
    rows: list[dict[str, str]], config: dict[str, Any], profile: dict[str, Any]
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from PIL import Image, ImageOps
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the transformers backend")
    process_affinity = sorted(os.sched_getaffinity(0))
    if "cpu_threads" in config:
        torch.set_num_threads(int(config["cpu_threads"]))
        torch.set_num_interop_threads(1)
    elif os.getenv("VLM_CPU_AFFINITY"):
        # Replica jobs set this only after assigning a disjoint CPU shard.
        torch.set_num_interop_threads(1)
    required_gpus = int(profile.get("minimum_gpus", 1))
    available_gpus = torch.cuda.device_count()
    if available_gpus < required_gpus:
        raise RuntimeError(
            f"Profile requires {required_gpus} CUDA GPUs, but only {available_gpus} are visible"
        )
    cuda_devices = list(range(required_gpus))

    def reset_cuda_peaks() -> None:
        for device_index in cuda_devices:
            with torch.cuda.device(device_index):
                torch.cuda.reset_peak_memory_stats()

    def synchronize_cuda() -> None:
        for device_index in cuda_devices:
            with torch.cuda.device(device_index):
                torch.cuda.synchronize()

    def peak_allocated_per_gpu() -> list[int]:
        result = []
        for device_index in cuda_devices:
            with torch.cuda.device(device_index):
                result.append(torch.cuda.max_memory_allocated())
        return result

    def peak_reserved_per_gpu() -> list[int]:
        result = []
        for device_index in cuda_devices:
            with torch.cuda.device(device_index):
                result.append(torch.cuda.max_memory_reserved())
        return result
    torch.manual_seed(int(config["seed"]))
    model_path = snapshot_download(
        profile["model_id"],
        revision=profile["revision"],
        local_files_only=bool(config.get("local_files_only", True)),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"],
    )
    device_map: dict[str, int] | str = {"": 0} if required_gpus == 1 else "balanced"
    load_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": device_map,
        "attn_implementation": "sdpa",
        "local_files_only": True,
    }
    if required_gpus > 1:
        load_kwargs["max_memory"] = {
            index: int(torch.cuda.get_device_properties(index).total_memory * 0.94)
            for index in range(required_gpus)
        }
    if profile["precision"] == "nf4":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    reset_cuda_peaks()
    load_started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs).eval()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    synchronize_cuda()
    loading_seconds = time.perf_counter() - load_started
    load_peak_per_gpu = peak_allocated_per_gpu()
    load_peak = sum(load_peak_per_gpu)

    predictions: list[dict[str, object]] = []
    batches: list[dict[str, object]] = []
    batch_size = int(config["batch_size"])
    max_edge = int(config["max_image_edge"])

    def process_batch(batch_rows: list[dict[str, str]], batch_index: int, emit: bool) -> None:
        images = []
        messages = []
        batch_started = time.perf_counter()
        try:
            for row in batch_rows:
                with Image.open(row["image_path"]) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                    image.thumbnail((max_edge, max_edge))
                    image.load()
                images.append(image)
                messages.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": config["prompt"]},
                            ],
                        }
                    ]
                )
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={"padding": True},
            ).to("cuda:0")
            synchronize_cuda()
            generation_started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=int(config["max_new_tokens"]),
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            synchronize_cuda()
            generation_seconds = time.perf_counter() - generation_started
            new_tokens = generated[:, inputs["input_ids"].shape[1] :]
            decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)
            if emit:
                for index, (row, raw_text) in enumerate(zip(batch_rows, decoded, strict=True)):
                    parsed = parse_model_output(raw_text)
                    predictions.append(
                        {
                            **_base_prediction(row),
                            "status": "ok" if parsed.parse_ok else "parse_failure",
                            "raw_text": raw_text,
                            **parsed.to_dict(),
                            "batch_index": batch_index,
                            "output_tokens": int(new_tokens[index].numel()),
                        }
                    )
            batches.append(
                {
                    "batch_index": batch_index,
                    "batch_size": len(batch_rows),
                    "warmup": not emit,
                    "generation_seconds": generation_seconds,
                    "pipeline_seconds": time.perf_counter() - batch_started,
                    "peak_allocated_bytes": sum(peak_allocated_per_gpu()),
                    "peak_reserved_bytes": sum(peak_reserved_per_gpu()),
                    "peak_allocated_bytes_per_gpu": peak_allocated_per_gpu(),
                    "status": "success",
                }
            )
            del generated, inputs, new_tokens
        finally:
            for image in images:
                image.close()

    warmup = int(config.get("warmup_batches", 0))
    for warmup_index in range(warmup):
        process_batch(rows[:batch_size], -(warmup_index + 1), emit=False)

    pipeline_started = time.perf_counter()
    failed_at: int | None = None
    for batch_index, offset in enumerate(range(0, len(rows), batch_size)):
        batch_rows = rows[offset : offset + batch_size]
        reset_cuda_peaks()
        try:
            process_batch(batch_rows, batch_index, emit=True)
        except Exception as error:
            status = "oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "runtime_error"
            for row in batch_rows:
                predictions.append(
                    {
                        **_base_prediction(row),
                        "status": status,
                        "raw_text": "",
                        "predicted_label": "UNCERTAIN",
                        "parse_ok": False,
                        "parse_error": f"{status}:{error}",
                        "batch_index": batch_index,
                        "output_tokens": 0,
                    }
                )
            batches.append(
                {
                    "batch_index": batch_index,
                    "batch_size": len(batch_rows),
                    "warmup": False,
                    "pipeline_seconds": 0.0,
                    "status": status,
                    "error": str(error),
                }
            )
            failed_at = offset + len(batch_rows)
            break
    if failed_at is not None:
        for row in rows[failed_at:]:
            predictions.append(
                {
                    **_base_prediction(row),
                    "status": "skipped_after_failure",
                    "raw_text": "",
                    "predicted_label": "UNCERTAIN",
                    "parse_ok": False,
                    "parse_error": "skipped_after_failure",
                    "batch_index": None,
                    "output_tokens": 0,
                }
            )
    pipeline_seconds = time.perf_counter() - pipeline_started
    properties = torch.cuda.get_device_properties(0)
    environment = {
        "backend": "transformers",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": properties.name,
        "gpu_count": required_gpus,
        "gpu_total_bytes": properties.total_memory,
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "host": platform.node(),
        "host_cpu_count": os.cpu_count(),
        "process_cpu_affinity": process_affinity,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("accelerate", "bitsandbytes", "huggingface-hub", "pillow")
        },
        "model_loading_seconds": loading_seconds,
        "load_peak_allocated_bytes": load_peak,
        "load_peak_allocated_bytes_per_gpu": load_peak_per_gpu,
        "hf_device_map": getattr(model, "hf_device_map", None),
        "pipeline_seconds": pipeline_seconds,
        "peak_allocated_bytes": max(
            (int(row.get("peak_allocated_bytes", 0)) for row in batches), default=0
        ),
        "peak_reserved_bytes": max(
            (int(row.get("peak_reserved_bytes", 0)) for row in batches), default=0
        ),
    }
    return predictions, batches, environment


def run_inference(
    input_path: Path, output_path: Path, config_path: Path
) -> dict[str, object]:
    config = load_json(config_path)
    inference_config = config["inference"]
    manifest_path = input_path if input_path.is_file() else input_path / "blind_manifest.csv"
    rows = _read_blind_manifest(manifest_path)
    registry_path = resolve_pack_path(inference_config["model_registry"])
    registry = load_json(registry_path)
    profile_name = inference_config["profile"]
    try:
        profile = registry["profiles"][profile_name]
    except KeyError as error:
        raise ValueError(f"Unknown model profile: {profile_name}") from error

    output_path.mkdir(parents=True, exist_ok=True)
    backend = inference_config["backend"]
    if backend == "mock":
        predictions, batches, environment = _run_mock(rows, inference_config)
    elif backend == "transformers":
        predictions, batches, environment = _run_transformers(rows, inference_config, profile)
    else:
        raise ValueError(f"Unknown inference backend: {backend}")
    if len(predictions) != len(rows):
        raise RuntimeError("Inference must preserve one output row per requested image")
    for prediction in predictions:
        prediction.update(
            {
                "backend": backend,
                "profile": profile_name,
                "model_id": profile["model_id"],
                "model_revision": profile["revision"],
                "precision": profile["precision"],
            }
        )

    prediction_path = output_path / "predictions.jsonl"
    batch_path = output_path / "batches.jsonl"
    _write_jsonl(prediction_path, predictions)
    _write_jsonl(batch_path, batches)
    status_counts: dict[str, int] = {}
    for row in predictions:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "backend": backend,
        "profile": profile_name,
        "model": profile,
        "model_registry": str(registry_path),
        "model_registry_sha256": sha256_file(registry_path),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "blind_manifest": str(manifest_path.resolve()),
        "blind_manifest_sha256": sha256_file(manifest_path),
        "requested_images": len(rows),
        "prediction_rows": len(predictions),
        "status_counts": status_counts,
        "predictions_sha256": sha256_file(prediction_path),
        "environment": environment,
        "performance": summarize_performance(batches, environment),
    }
    write_json(output_path / "summary.json", summary)
    return summary
