#!/usr/bin/env python3
"""Run one pack stage: preprocess, intermediate, or postprocess."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


STAGES = {"preprocess", "intermediate", "postprocess"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a LANTA pack stage.")
    parser.add_argument("stage", choices=sorted(STAGES), help="Stage to run.")
    parser.add_argument("--input", default="data", help="Input path for this stage.")
    parser.add_argument("--output", default=None, help="Output directory for this stage.")
    parser.add_argument("--config", default=None, help="Optional config file.")
    parser.add_argument("--save-json", default=None, help="Optional JSON summary path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output or f"outputs/{args.stage}")
    config_path = Path(args.config) if args.config else None

    if config_path is None:
        raise SystemExit("--config is required for every stage")
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    module = importlib.import_module(f"stages.{args.stage}")
    result = module.run(input_path=input_path, output_path=output_path, config_path=config_path)

    summary = {"stage": args.stage, "input": str(input_path), "output": str(output_path), "result": result}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
