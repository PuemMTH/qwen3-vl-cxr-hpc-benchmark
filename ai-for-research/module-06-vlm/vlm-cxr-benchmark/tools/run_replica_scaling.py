#!/usr/bin/env python3
"""Run one model replica per visible GPU on disjoint manifest shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def visible_gpu_tokens() -> list[str]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value:
        return [token.strip() for token in value.split(",") if token.strip()]
    return [str(index) for index in range(32)]


def make_shards(manifest: Path, directory: Path, count: int) -> list[Path]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows or not fields:
        raise ValueError("Blind manifest is empty")
    shards = [[] for _ in range(count)]
    for index, row in enumerate(rows):
        shards[index % count].append(row)
    if any(not shard for shard in shards):
        raise ValueError("GPU count exceeds manifest rows")
    result = []
    for rank, rows_for_rank in enumerate(shards):
        path = directory / f"shard-{rank:02d}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows_for_rank)
        result.append(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--blind-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--run-stage", type=Path, default=Path("run_stage.py"))
    args = parser.parse_args()
    if args.gpus < 1 or args.repeats < 1:
        parser.error("--gpus and --repeats must be positive")
    gpu_tokens = visible_gpu_tokens()
    if len(gpu_tokens) < args.gpus:
        raise SystemExit(f"Requested {args.gpus} replicas but only {gpu_tokens} are visible")
    allowed_cpus = sorted(os.sched_getaffinity(0))
    if len(allowed_cpus) < args.gpus:
        raise SystemExit(
            f"Only {len(allowed_cpus)} CPUs are in the allocation affinity for {args.gpus} replicas"
        )
    cpu_shards = [allowed_cpus[rank::args.gpus] for rank in range(args.gpus)]

    config = args.config.resolve()
    manifest = args.blind_manifest.resolve()
    run_stage = args.run_stage.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "state.json"
    state = {
        "schema_version": 1,
        "status": "running",
        "started_utc": utc_now(),
        "ended_utc": None,
        "host": platform.node(),
        "pid": os.getpid(),
        "config": str(config),
        "config_sha256": sha256(config),
        "blind_manifest": str(manifest),
        "blind_manifest_sha256": sha256(manifest),
        "requested_gpus": args.gpus,
        "visible_gpu_tokens": gpu_tokens[: args.gpus],
        "allocation_cpu_affinity": allowed_cpus,
        "rank_cpu_affinities": cpu_shards,
        "requested_repeats": args.repeats,
        "rounds": [],
    }
    write_json(state_path, state)
    overall_rc = 0

    for repeat in range(1, args.repeats + 1):
        round_dir = output_root / f"round-{repeat:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        shards = make_shards(manifest, round_dir, args.gpus)
        processes = []
        rank_rows = []
        started = time.perf_counter()
        started_utc = utc_now()
        for rank, (gpu, shard) in enumerate(zip(gpu_tokens, shards, strict=True)):
            rank_dir = round_dir / f"rank-{rank:02d}"
            rank_dir.mkdir()
            log_path = rank_dir / "run.log"
            command = [
                sys.executable,
                "-u",
                str(run_stage),
                "intermediate",
                "--input",
                str(shard),
                "--output",
                str(rank_dir / "intermediate"),
                "--config",
                str(config),
                "--save-json",
                str(rank_dir / "stage.json"),
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["OMP_NUM_THREADS"] = str(len(cpu_shards[rank]))
            environment["MKL_NUM_THREADS"] = str(len(cpu_shards[rank]))
            environment["OPENBLAS_NUM_THREADS"] = str(len(cpu_shards[rank]))
            environment["VLM_CPU_AFFINITY"] = ",".join(map(str, cpu_shards[rank]))
            log = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                preexec_fn=lambda cpus=set(cpu_shards[rank]): os.sched_setaffinity(0, cpus),
            )
            processes.append((process, log, rank_dir))
            rank_rows.append({
                "rank": rank,
                "gpu_token": gpu,
                "cpu_affinity": cpu_shards[rank],
                "shard": str(shard),
                "shard_sha256": sha256(shard),
                "rows": sum(1 for _ in shard.open(encoding="utf-8")) - 1,
                "command": command,
                "log": str(log_path),
            })
        for process, log, _ in processes:
            process.wait()
            log.close()
        elapsed = time.perf_counter() - started
        round_status = "complete"
        for row, (process, _, rank_dir) in zip(rank_rows, processes, strict=True):
            row["exit_code"] = process.returncode
            summary_path = rank_dir / "intermediate" / "summary.json"
            row["summary"] = str(summary_path)
            if summary_path.is_file():
                row["summary_sha256"] = sha256(summary_path)
                summary = json.loads(summary_path.read_text())
                row["prediction_status_counts"] = summary.get("status_counts")
                row["performance"] = summary.get("performance")
            if process.returncode != 0:
                round_status = "failed"
                overall_rc = process.returncode
        state["rounds"].append({
            "repeat": repeat,
            "status": round_status,
            "started_utc": started_utc,
            "ended_utc": utc_now(),
            "wall_seconds": elapsed,
            "ranks": rank_rows,
        })
        if round_status != "complete":
            state["status"] = "failed"
            write_json(state_path, state)
            break
        write_json(state_path, state)

    if state["status"] == "running":
        state["status"] = "complete"
    state["ended_utc"] = utc_now()
    state["completed_repeats"] = sum(r["status"] == "complete" for r in state["rounds"])
    write_json(state_path, state)
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
