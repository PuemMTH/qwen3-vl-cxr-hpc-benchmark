#!/usr/bin/env python3
"""Evaluate the four B-development profiles and freeze the best-local choice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm_cxr.common import sha256_file, write_json
from vlm_cxr.evaluate import evaluate_predictions


PROFILES = ("2b-bf16", "4b-bf16", "8b-nf4", "8b-bf16")
LOCAL_CANDIDATES = ("2b-bf16", "4b-bf16", "8b-nf4")


def candidate_key(metrics: dict, profile: str) -> tuple:
    primary = metrics["primary_patient_level"]
    # Frozen before reading results: BA, macro-F1, lower uncertainty, then the
    # declared candidate order (smaller/cheaper configuration wins final tie).
    return (
        float(primary["balanced_accuracy"]),
        float(primary["macro_f1"]),
        -float(primary["uncertainty_rate"]),
        -LOCAL_CANDIDATES.index(profile),
    )


def paired_config(source: Path, comparator: Path, destination: Path) -> Path:
    value = json.loads(source.read_text())
    value["evaluation"]["comparator_predictions"] = str(comparator.resolve())
    value["evaluation"]["comparison_status"] = "development_pilot_only"
    write_json(destination, value)
    return destination


def gate(metrics: dict) -> dict:
    comparison = metrics["paired_comparison"]
    candidate = metrics["primary_patient_level"]
    comparator = comparison["comparator_patient_metrics"]
    delta = comparison["candidate_minus_comparator_balanced_accuracy"]
    recall_deltas = {
        label: float(candidate["per_class"][label]["recall"])
        - float(comparator["per_class"][label]["recall"])
        for label in ("NORMAL", "PNEUMONIA")
    }
    checks = {
        "balanced_accuracy_delta_at_least_0_05": delta >= 0.05,
        "mcnemar_p_below_0_05": float(
            comparison["mcnemar"]["exact_two_sided_p_value"]
        ) < 0.05,
        "no_class_recall_drop_below_minus_0_03": min(recall_deltas.values()) >= -0.03,
    }
    return {
        "candidate_minus_comparator_balanced_accuracy": delta,
        "recall_deltas": recall_deltas,
        "mcnemar": comparison["mcnemar"],
        "checks": checks,
        "numeric_checks_all_pass": all(checks.values()),
        "confirmatory_gate_status": "not_assessable_from_development_data",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    args = parser.parse_args()
    root = args.run_root.resolve()
    labels = args.labels.resolve()
    metrics_by_profile = {}

    for profile in PROFILES:
        prediction = root / profile / "round-01" / "intermediate" / "predictions.jsonl"
        config = (args.config_root / f"b-development-{profile}-batch1.json").resolve()
        if not prediction.is_file():
            raise FileNotFoundError(prediction)
        metrics_by_profile[profile] = evaluate_predictions(
            labels, prediction, root / profile / "evaluation", config
        )

    best_local = max(
        LOCAL_CANDIDATES,
        key=lambda profile: candidate_key(metrics_by_profile[profile], profile),
    )
    candidate_prediction = root / "8b-bf16" / "round-01" / "intermediate" / "predictions.jsonl"
    source_config = (args.config_root / "b-development-8b-bf16-batch1.json").resolve()
    comparisons = {}
    for name, comparator_profile in (("vs-8b-nf4", "8b-nf4"), ("vs-best-local", best_local)):
        comparator_prediction = (
            root / comparator_profile / "round-01" / "intermediate" / "predictions.jsonl"
        )
        output = root / "comparisons" / name
        output.mkdir(parents=True, exist_ok=True)
        config = paired_config(source_config, comparator_prediction, output / "config.json")
        metrics = evaluate_predictions(labels, candidate_prediction, output, config)
        comparisons[name] = {
            "comparator_profile": comparator_profile,
            "metrics": str((output / "metrics.json").resolve()),
            "metrics_sha256": sha256_file(output / "metrics.json"),
            "gate": gate(metrics),
        }

    summary = {
        "study_status": "development_pilot_only",
        "confirmatory_test_accessed": False,
        "selection_rule": (
            "max patient balanced accuracy; tie by macro-F1, lower uncertainty, "
            "then declared order 2b-bf16, 4b-bf16, 8b-nf4"
        ),
        "best_local_profile": best_local,
        "profile_metrics": {
            profile: {
                "balanced_accuracy": metrics["primary_patient_level"]["balanced_accuracy"],
                "accuracy": metrics["primary_patient_level"]["accuracy"],
                "macro_f1": metrics["primary_patient_level"]["macro_f1"],
                "uncertainty_rate": metrics["primary_patient_level"]["uncertainty_rate"],
                "metrics": str((root / profile / "evaluation" / "metrics.json").resolve()),
            }
            for profile, metrics in metrics_by_profile.items()
        },
        "comparisons": comparisons,
    }
    write_json(root / "quality-summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
