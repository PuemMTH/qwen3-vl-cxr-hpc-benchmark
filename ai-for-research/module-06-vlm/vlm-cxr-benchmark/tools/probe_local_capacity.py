#!/usr/bin/env python3
"""Attempt a real single-GPU BF16 load and preserve success/OOM evidence."""

from __future__ import annotations

import argparse
import platform
import time
import traceback
from pathlib import Path
import json

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText

from vlm_cxr.common import load_json, resolve_pack_path, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--registry", default="configs/models.lock.json")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--skip-allocator-warmup",
        action="store_true",
        help="Diagnostic: bypass only Transformers speculative allocator warm-up.",
    )
    parser.add_argument(
        "--model-path",
        help="Complete local snapshot directory; otherwise resolve the pinned HF cache snapshot.",
    )
    args = parser.parse_args()

    registry_path = resolve_pack_path(args.registry)
    registry = load_json(registry_path)
    if args.profile not in registry["profiles"]:
        raise SystemExit(f"Unknown profile: {args.profile}")
    profile = registry["profiles"][args.profile]
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for a local capacity probe")

    if args.model_path:
        snapshot = Path(args.model_path).resolve()
    else:
        snapshot = Path(
            snapshot_download(
                profile["model_id"],
                revision=profile["revision"],
                local_files_only=True,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.txt",
                    "*.jinja",
                    "*.model",
                ],
            )
        )
    if not snapshot.is_dir():
        raise SystemExit(f"Model path is not a directory: {snapshot}")
    weights = sorted(snapshot.glob("*.safetensors"))
    with torch.cuda.device(0):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    status = "load_success"
    error_type = None
    error_message = None
    error_traceback = None
    try:
        if args.skip_allocator_warmup:
            transformers.modeling_utils.caching_allocator_warmup = (
                lambda *unused_args, **unused_kwargs: None
            )
        model = AutoModelForImageTextToText.from_pretrained(
            snapshot,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
            local_files_only=True,
        )
        del model
    except Exception as error:  # evidence capture intentionally covers load failures
        error_text = str(error).lower()
        status = (
            "oom"
            if isinstance(error, torch.cuda.OutOfMemoryError)
            or "out of memory" in error_text
            else "load_failure"
        )
        error_type = type(error).__name__
        error_message = str(error)
        error_traceback = traceback.format_exc()
    elapsed = time.perf_counter() - started
    properties = torch.cuda.get_device_properties(0)
    weight_bytes = sum(path.stat().st_size for path in weights)
    expected_weight_bytes = profile.get("weight_bytes")
    config_path = snapshot / "config.json"
    model_config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    result = {
        "profile": args.profile,
        "model": profile,
        "snapshot": str(snapshot.resolve()),
        "snapshot_revision_matches": (
            snapshot.name == profile["revision"] if not args.model_path else None
        ),
        "weight_files": len(weights),
        "weight_bytes": weight_bytes,
        "expected_weight_bytes": expected_weight_bytes,
        "weight_bytes_match_registry": (
            weight_bytes == int(expected_weight_bytes)
            if expected_weight_bytes is not None
            else None
        ),
        "config_present": config_path.is_file(),
        "config_model_type": model_config.get("model_type"),
        "probe_policy": "real BF16 model load, all weights forced to CUDA device 0; no CPU/disk offload",
        "skip_allocator_warmup": args.skip_allocator_warmup,
        "status": status,
        "failure_stage": None if status == "load_success" else "model_load",
        "error_type": error_type,
        "error_message": error_message,
        "traceback": error_traceback,
        "elapsed_seconds": elapsed,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": properties.name,
            "gpu_total_bytes": properties.total_memory,
            "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
    }
    output = resolve_pack_path(args.output)
    write_json(output, result)
    print(output)
    print(f"status={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
