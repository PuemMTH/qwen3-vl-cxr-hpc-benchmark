#!/usr/bin/env python3
"""Resolve pinned Qwen3-VL snapshots and write a local cache inventory."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from vlm_cxr.common import load_json, resolve_pack_path, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="configs/models.lock.json")
    parser.add_argument("--output", default="artifacts/model_cache_inventory.local.json")
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow network download; otherwise require complete local snapshots.",
    )
    args = parser.parse_args()
    registry_path = resolve_pack_path(args.registry)
    registry = load_json(registry_path)
    requested = set(args.profile or registry["profiles"])
    unknown = requested - set(registry["profiles"])
    if unknown:
        raise SystemExit(f"Unknown profiles: {sorted(unknown)}")

    snapshots: dict[tuple[str, str], Path] = {}
    profile_rows = []
    for name, profile in registry["profiles"].items():
        if name not in requested:
            continue
        key = (profile["model_id"], profile["revision"])
        if key not in snapshots:
            snapshots[key] = Path(
                snapshot_download(
                    profile["model_id"],
                    revision=profile["revision"],
                    local_files_only=not args.download,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.txt",
                        "*.jinja",
                        "*.model",
                    ],
                )
            )
        snapshot = snapshots[key]
        weights = sorted(snapshot.glob("*.safetensors"))
        revision_matches = snapshot.name == profile["revision"]
        config_present = (snapshot / "config.json").is_file()
        preprocessor_present = (snapshot / "preprocessor_config.json").is_file()
        profile_rows.append(
            {
                "profile": name,
                **profile,
                "snapshot_path": str(snapshot.resolve()),
                "snapshot_directory_name": snapshot.name,
                "revision_matches_snapshot": revision_matches,
                "weight_files": len(weights),
                "weight_bytes": sum(path.stat().st_size for path in weights),
                "config_present": config_present,
                "preprocessor_config_present": preprocessor_present,
                "ready": bool(
                    revision_matches and weights and config_present and preprocessor_present
                ),
            }
        )
    if any(not row["ready"] for row in profile_rows):
        raise RuntimeError("At least one pinned model snapshot is incomplete or mismatched")
    output = resolve_pack_path(args.output)
    write_json(
        output,
        {
            "registry": str(registry_path),
            "download_allowed": args.download,
            "all_ready": all(bool(row["ready"]) for row in profile_rows),
            "profiles": profile_rows,
        },
    )
    print(output)


if __name__ == "__main__":
    main()
