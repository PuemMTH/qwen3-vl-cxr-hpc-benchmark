from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

import numpy as np
from scipy.stats import binomtest


TRUE_LABELS = ("NORMAL", "PNEUMONIA")
PREDICTION_LABELS = TRUE_LABELS + ("UNCERTAIN",)


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def classification_metrics(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot evaluate an empty row set")
    confusion = {
        true: {pred: 0 for pred in PREDICTION_LABELS} for true in TRUE_LABELS
    }
    for row in rows:
        true, pred = str(row["true_label"]), str(row["predicted_label"])
        if true not in TRUE_LABELS:
            raise ValueError(f"Unexpected true label: {true}")
        if pred not in PREDICTION_LABELS:
            raise ValueError(f"Unexpected predicted label: {pred}")
        confusion[true][pred] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for label in TRUE_LABELS:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[true][label] for true in TRUE_LABELS)
        recall = _safe_div(tp, support)
        precision = _safe_div(tp, predicted)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class[label] = {
            "support": support,
            "predicted": predicted,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    correct = sum(confusion[label][label] for label in TRUE_LABELS)
    uncertain = sum(confusion[label]["UNCERTAIN"] for label in TRUE_LABELS)
    return {
        "n": len(rows),
        "accuracy": correct / len(rows),
        "balanced_accuracy": sum(float(per_class[x]["recall"]) for x in TRUE_LABELS) / 2,
        "macro_f1": sum(float(per_class[x]["f1"]) for x in TRUE_LABELS) / 2,
        "uncertainty_rate": uncertain / len(rows),
        "coverage": 1 - uncertain / len(rows),
        "per_class": per_class,
        "confusion": confusion,
    }


def aggregate_patients(image_rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Use a strict majority across all images; ties/failures yield UNCERTAIN."""
    groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in image_rows:
        groups[str(row["patient_id"])].append(row)
    result: list[dict[str, object]] = []
    for patient_id, rows in sorted(groups.items()):
        true_labels = {str(row["true_label"]) for row in rows}
        if len(true_labels) != 1:
            raise ValueError(f"Patient {patient_id} spans true labels: {sorted(true_labels)}")
        votes = Counter(str(row["predicted_label"]) for row in rows)
        majority = len(rows) // 2 + 1
        winners = [label for label in TRUE_LABELS if votes[label] >= majority]
        prediction = winners[0] if len(winners) == 1 else "UNCERTAIN"
        result.append(
            {
                "patient_id": patient_id,
                "true_label": next(iter(true_labels)),
                "predicted_label": prediction,
                "n_images": len(rows),
                "votes_normal": votes["NORMAL"],
                "votes_pneumonia": votes["PNEUMONIA"],
                "votes_uncertain": votes["UNCERTAIN"],
                "parse_failures": sum(not bool(row.get("parse_ok", False)) for row in rows),
                "runtime_failures": sum(str(row.get("status")) != "ok" for row in rows),
            }
        )
    return result


def _metric_scalars(metrics: dict[str, object]) -> dict[str, float]:
    per_class = metrics["per_class"]
    assert isinstance(per_class, dict)
    return {
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "uncertainty_rate": float(metrics["uncertainty_rate"]),
        "coverage": float(metrics["coverage"]),
        "recall_normal": float(per_class["NORMAL"]["recall"]),
        "recall_pneumonia": float(per_class["PNEUMONIA"]["recall"]),
    }


def patient_bootstrap_ci(
    patient_rows: Iterable[dict[str, object]],
    replicates: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, dict[str, float]]:
    rows = list(patient_rows)
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    strata = {
        label: [row for row in rows if row["true_label"] == label] for label in TRUE_LABELS
    }
    if any(not values for values in strata.values()):
        raise ValueError("Both true classes are required for stratified bootstrap")
    rng = np.random.default_rng(seed)
    samples: defaultdict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        draw: list[dict[str, object]] = []
        for label in TRUE_LABELS:
            values = strata[label]
            indices = rng.integers(0, len(values), size=len(values))
            draw.extend(values[int(index)] for index in indices)
        for name, value in _metric_scalars(classification_metrics(draw)).items():
            samples[name].append(value)
    alpha = (1 - confidence_level) / 2
    point = _metric_scalars(classification_metrics(rows))
    return {
        name: {
            "estimate": point[name],
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1 - alpha)),
        }
        for name, values in samples.items()
    }


def paired_mcnemar(
    candidate: Iterable[dict[str, object]], comparator: Iterable[dict[str, object]]
) -> dict[str, object]:
    candidate_map = {str(row["patient_id"]): row for row in candidate}
    comparator_map = {str(row["patient_id"]): row for row in comparator}
    if candidate_map.keys() != comparator_map.keys():
        raise ValueError("Candidate and comparator patient sets differ")
    candidate_only, comparator_only = 0, 0
    for patient_id in candidate_map:
        left, right = candidate_map[patient_id], comparator_map[patient_id]
        if left["true_label"] != right["true_label"]:
            raise ValueError(f"True-label mismatch for patient {patient_id}")
        left_ok = left["predicted_label"] == left["true_label"]
        right_ok = right["predicted_label"] == right["true_label"]
        candidate_only += bool(left_ok and not right_ok)
        comparator_only += bool(right_ok and not left_ok)
    discordant = candidate_only + comparator_only
    p_value = 1.0 if discordant == 0 else float(
        binomtest(candidate_only, discordant, p=0.5, alternative="two-sided").pvalue
    )
    return {
        "candidate_correct_comparator_wrong": candidate_only,
        "candidate_wrong_comparator_correct": comparator_only,
        "discordant": discordant,
        "exact_two_sided_p_value": p_value,
    }


def paired_patient_bootstrap_difference_ci(
    candidate: Iterable[dict[str, object]],
    comparator: Iterable[dict[str, object]],
    replicates: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Paired, true-class-stratified patient bootstrap for candidate-comparator."""
    candidate_map = {str(row["patient_id"]): row for row in candidate}
    comparator_map = {str(row["patient_id"]): row for row in comparator}
    if candidate_map.keys() != comparator_map.keys():
        raise ValueError("Candidate and comparator patient sets differ")
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    for patient_id in candidate_map:
        if candidate_map[patient_id]["true_label"] != comparator_map[patient_id]["true_label"]:
            raise ValueError(f"True-label mismatch for patient {patient_id}")

    strata = {
        label: [
            patient_id
            for patient_id, row in candidate_map.items()
            if row["true_label"] == label
        ]
        for label in TRUE_LABELS
    }
    if any(not patient_ids for patient_ids in strata.values()):
        raise ValueError("Both true classes are required for stratified bootstrap")

    def differences(
        candidate_rows: list[dict[str, object]],
        comparator_rows: list[dict[str, object]],
    ) -> dict[str, float]:
        left = _metric_scalars(classification_metrics(candidate_rows))
        right = _metric_scalars(classification_metrics(comparator_rows))
        return {name: left[name] - right[name] for name in left}

    point = differences(list(candidate_map.values()), list(comparator_map.values()))
    rng = np.random.default_rng(seed)
    samples: defaultdict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        sampled_ids: list[str] = []
        for label in TRUE_LABELS:
            patient_ids = strata[label]
            indices = rng.integers(0, len(patient_ids), size=len(patient_ids))
            sampled_ids.extend(patient_ids[int(index)] for index in indices)
        draw = differences(
            [candidate_map[patient_id] for patient_id in sampled_ids],
            [comparator_map[patient_id] for patient_id in sampled_ids],
        )
        for name, value in draw.items():
            samples[name].append(value)

    alpha = (1 - confidence_level) / 2
    return {
        name: {
            "estimate": point[name],
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1 - alpha)),
        }
        for name, values in samples.items()
    }
