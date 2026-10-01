#!/usr/bin/env python3
"""Run identical measured inference rounds while preserving logs and state.

Each round is a fresh process. Model loading and warm-up remain recorded but are
excluded from the measured inference totals produced by ``vlm_cxr.inference``.
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--blind-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--run-stage", type=Path, default=Path("run_stage.py"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

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
        "requested_repeats": args.repeats,
        "rounds": [],
    }
    write_json(state_path, state)
    overall_rc = 0

    for repeat in range(1, args.repeats + 1):
        round_dir = output_root / f"round-{repeat:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        log_path = round_dir / "run.log"
        command = [
            sys.executable,
            "-u",
            str(run_stage),
            "intermediate",
            "--input",
            str(manifest),
            "--output",
            str(round_dir / "intermediate"),
            "--config",
            str(config),
            "--save-json",
            str(round_dir / "stage.json"),
        ]
        row = {
            "repeat": repeat,
            "status": "running",
            "started_utc": utc_now(),
            "ended_utc": None,
            "elapsed_seconds": None,
            "command": command,
            "log": str(log_path),
            "exit_code": None,
        }
        state["rounds"].append(row)
        write_json(state_path, state)
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        row["elapsed_seconds"] = time.perf_counter() - started
        row["ended_utc"] = utc_now()
        row["exit_code"] = completed.returncode
        summary_path = round_dir / "intermediate" / "summary.json"
        row["summary"] = str(summary_path)
        if summary_path.is_file():
            row["summary_sha256"] = sha256(summary_path)
            summary = json.loads(summary_path.read_text())
            row["prediction_status_counts"] = summary.get("status_counts")
            row["performance"] = summary.get("performance")
        row["status"] = "complete" if completed.returncode == 0 else "failed"
        if completed.returncode != 0:
            overall_rc = completed.returncode
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
