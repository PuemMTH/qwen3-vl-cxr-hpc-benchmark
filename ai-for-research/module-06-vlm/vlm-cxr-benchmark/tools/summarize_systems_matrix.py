#!/usr/bin/env python3
"""Aggregate A/C1/C2 state files without mixing quality and timing samples."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

from vlm_cxr.common import write_json


PROFILES = ("2b-bf16", "4b-bf16", "8b-nf4")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def repeated(path: Path) -> dict:
    state = load(path / "state.json")
    rounds = [row for row in state["rounds"] if row["status"] == "complete"]
    perf = [row["performance"] for row in rounds]
    status_counts = [row.get("prediction_status_counts", {}) for row in rounds]
    result = {
        "state": str((path / "state.json").resolve()),
        "status": state["status"],
        "completed_repeats": len(rounds),
        "generation_throughput_images_per_hour": [
            float(row["generation_throughput_images_per_hour"]) for row in perf
        ],
        "pipeline_throughput_images_per_hour": [
            float(row["pipeline_throughput_images_per_hour"]) for row in perf
        ],
        "generation_seconds_total": [float(row["generation_seconds_total"]) for row in perf],
        "pipeline_seconds_total": [float(row["pipeline_seconds_total"]) for row in perf],
        "p50_amortized_generation_seconds_per_image": [
            float(row["amortized_generation_seconds_per_image"]["p50"]) for row in perf
        ],
        "p95_amortized_generation_seconds_per_image": [
            float(row["amortized_generation_seconds_per_image"]["p95"]) for row in perf
        ],
        "peak_allocated_bytes": [int(row["peak_allocated_bytes"]) for row in perf],
        "peak_reserved_bytes": [int(row["peak_reserved_bytes"]) for row in perf],
        "model_loading_seconds": [float(row["model_loading_seconds"]) for row in perf],
        "prediction_status_counts": status_counts,
    }
    for key in (
        "generation_throughput_images_per_hour",
        "pipeline_throughput_images_per_hour",
        "generation_seconds_total",
        "pipeline_seconds_total",
        "p50_amortized_generation_seconds_per_image",
        "p95_amortized_generation_seconds_per_image",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "model_loading_seconds",
    ):
        result[f"median_{key}"] = median(result[key])
    result["all_predictions_ok"] = all(
        set(counts) <= {"ok"} and counts.get("ok", 0) > 0 for counts in status_counts
    )
    return result


def paired_speedup_ci(local_times: list[float], remote_times: list[float]) -> dict:
    if len(local_times) != len(remote_times) or not local_times:
        return {"status": "unavailable"}
    ratios = np.asarray(local_times) / np.asarray(remote_times)
    rng = np.random.default_rng(20260923)
    draws = []
    for _ in range(2000):
        indices = rng.integers(0, len(ratios), size=len(ratios))
        draws.append(float(np.median(ratios[indices])))
    return {
        "method": "paired-round nonparametric bootstrap of median time ratio",
        "estimate": float(np.median(ratios)),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "round_ratios": ratios.tolist(),
    }


def replica(path: Path) -> dict:
    state = load(path / "state.json")
    rows = []
    for round_row in state["rounds"]:
        if round_row["status"] != "complete":
            continue
        rank_pipeline = [
            float(rank["performance"]["pipeline_seconds_total"])
            for rank in round_row["ranks"]
        ]
        rank_generation = [
            float(rank["performance"]["generation_seconds_total"])
            for rank in round_row["ranks"]
        ]
        rows.append({
            "repeat": round_row["repeat"],
            "inference_pipeline_seconds": max(rank_pipeline),
            "inference_generation_seconds": max(rank_generation),
            "complete_round_wall_seconds": float(round_row["wall_seconds"]),
            "rank_pipeline_seconds": rank_pipeline,
        })
    return {
        "state": str((path / "state.json").resolve()),
        "status": state["status"],
        "gpus": state["requested_gpus"],
        "completed_repeats": len(rows),
        "rounds": rows,
        "median_inference_pipeline_seconds": median(
            [row["inference_pipeline_seconds"] for row in rows]
        ),
        "median_complete_round_wall_seconds": median(
            [row["complete_round_wall_seconds"] for row in rows]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    result: dict = {"study_status": "development_pilot_only", "experiment_a": {}, "c1": {}, "c2": {}}

    for profile in PROFILES:
        local = repeated(root / "local" / "A" / profile)
        lanta = repeated(root / "lanta" / "A" / profile)
        speedup = paired_speedup_ci(local["pipeline_seconds_total"], lanta["pipeline_seconds_total"])
        result["experiment_a"][profile] = {
            "local_rtx_5070": local,
            "lanta_a100": lanta,
            "a100_pipeline_speedup": speedup,
            "speed_gate_pass": bool(
                speedup.get("estimate", 0) >= 1.5 and speedup.get("lower", 0) > 1.0
            ),
        }

    valid_batches = []
    for batch in (1, 2, 4, 8):
        value = repeated(root / "lanta" / "C1-selected-4b-bf16" / f"batch{batch}")
        result["c1"][str(batch)] = value
        if value["status"] == "complete" and value["all_predictions_ok"]:
            valid_batches.append(batch)
    result["c1_selected_batch"] = max(
        valid_batches,
        key=lambda batch: result["c1"][str(batch)]["median_pipeline_throughput_images_per_hour"],
    ) if valid_batches else None

    available_gpu_counts = []
    for gpus in (1, 2, 4):
        path = root / "lanta" / "C2-selected-4b-bf16-cpu-pinned-v2" / f"gpu{gpus}"
        if (path / "state.json").is_file():
            result["c2"][str(gpus)] = replica(path)
            if result["c2"][str(gpus)]["status"] == "complete":
                available_gpu_counts.append(gpus)
    if 1 in available_gpu_counts:
        baseline = result["c2"]["1"]["median_inference_pipeline_seconds"]
        for gpus in available_gpu_counts:
            current = result["c2"][str(gpus)]
            speedup = baseline / current["median_inference_pipeline_seconds"]
            current["speedup"] = speedup
            current["scaling_efficiency_percent"] = 100 * speedup / gpus
            threshold = 80 if gpus == 2 else 70 if gpus == 4 else 100
            current["scaling_gate_pass"] = current["scaling_efficiency_percent"] >= threshold

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    rows = []
    for profile, value in result["experiment_a"].items():
        rows.append({
            "experiment": "A",
            "configuration": profile,
            "local_pipeline_images_per_hour": value["local_rtx_5070"]["median_pipeline_throughput_images_per_hour"],
            "lanta_pipeline_images_per_hour": value["lanta_a100"]["median_pipeline_throughput_images_per_hour"],
            "speedup": value["a100_pipeline_speedup"].get("estimate"),
            "speedup_ci_lower": value["a100_pipeline_speedup"].get("lower"),
            "speedup_ci_upper": value["a100_pipeline_speedup"].get("upper"),
            "gate_pass": value["speed_gate_pass"],
        })
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
