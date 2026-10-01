#!/usr/bin/env python3
"""Select the fastest valid C1 batch and submit 1/2/4-GPU C2 jobs."""

from __future__ import annotations

import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from vlm_cxr.common import write_json


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "pilot-matrix-20260923"


def main() -> None:
    candidates = {}
    model_profile = "4b-bf16"
    c1_root = RUN_ROOT / "lanta" / "C1-selected-4b-bf16"
    for batch in (1, 2, 4, 8):
        state_path = c1_root / f"batch{batch}" / "state.json"
        state = json.loads(state_path.read_text())
        if state["status"] != "complete" or state.get("completed_repeats") != 5:
            raise SystemExit(f"C1 batch {batch} is not complete: {state_path}")
        throughputs = []
        valid = True
        for row in state["rounds"]:
            counts = row.get("prediction_status_counts", {})
            valid &= set(counts) <= {"ok"} and counts.get("ok") == 1000
            throughputs.append(float(row["performance"]["pipeline_throughput_images_per_hour"]))
        candidates[batch] = {
            "valid": valid,
            "round_pipeline_throughput_images_per_hour": throughputs,
            "median_pipeline_throughput_images_per_hour": statistics.median(throughputs),
            "state": str(state_path),
        }
    eligible = [batch for batch, value in candidates.items() if value["valid"]]
    if not eligible:
        raise SystemExit("No C1 batch completed five clean rounds")
    selected = max(
        eligible,
        key=lambda batch: candidates[batch]["median_pipeline_throughput_images_per_hour"],
    )
    selection = {
        "study_status": "development_pilot_only",
        "selection_rule": "highest median pipeline throughput after five rounds, excluding OOM or non-ok outputs",
        "selected_batch": selected,
        "candidates": candidates,
        "selected_utc": datetime.now(timezone.utc).isoformat(),
    }
    selection["model_profile"] = model_profile
    selection["model_selection_evidence"] = str(
        RUN_ROOT / "lanta" / "B-development" / "quality-summary.json"
    )
    write_json(c1_root / "selection.json", selection)

    manifest = (
        "runs/pilot-matrix-20260923/manifests/"
        "scaling-2000-selected-4b-bf16/blind_manifest.csv"
    )
    config = f"configs/experiments/c2-4b-bf16-batch{selected}.json"
    submission_path = RUN_ROOT / "submissions.tsv"
    resources = {1: (16, "124000M"), 2: (32, "248000M"), 4: (64, "496000M")}
    with submission_path.open("a", encoding="utf-8") as submission_log:
        for gpus in (1, 2, 4):
            cpus, memory = resources[gpus]
            output = f"runs/pilot-matrix-20260923/lanta/C2-selected-4b-bf16/gpu{gpus}"
            exports = ",".join(
                [
                    "ALL",
                    f"CONFIG={config}",
                    f"MANIFEST={manifest}",
                    f"OUTPUT_ROOT={output}",
                    "REPEATS=5",
                    f"REPLICA_GPUS={gpus}",
                ]
            )
            command = [
                "sbatch",
                "--parsable",
                "-J",
                f"vlm-c2-g{gpus}",
                f"--gres=gpu:{gpus}",
                f"--cpus-per-task={cpus}",
                f"--mem={memory}",
                f"--export={exports}",
                "slurm/replica-scaling.sbatch",
            ]
            job_id = subprocess.check_output(command, text=True).strip()
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            submission_log.write(f"C2\t{config}\t{job_id}\t{output}\t{timestamp}\n")
            submission_log.flush()
            print(f"C2 gpus={gpus} batch={selected} job={job_id}")


if __name__ == "__main__":
    main()
