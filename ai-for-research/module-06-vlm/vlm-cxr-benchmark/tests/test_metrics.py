import pytest

from vlm_cxr.metrics import (
    aggregate_patients,
    classification_metrics,
    paired_mcnemar,
    paired_patient_bootstrap_difference_ci,
    patient_bootstrap_ci,
)


def test_uncertain_remains_in_denominator() -> None:
    rows = [
        {"true_label": "NORMAL", "predicted_label": "NORMAL"},
        {"true_label": "NORMAL", "predicted_label": "UNCERTAIN"},
        {"true_label": "PNEUMONIA", "predicted_label": "PNEUMONIA"},
        {"true_label": "PNEUMONIA", "predicted_label": "UNCERTAIN"},
    ]
    metrics = classification_metrics(rows)
    assert metrics["accuracy"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["uncertainty_rate"] == 0.5
    assert metrics["per_class"]["NORMAL"]["support"] == 2
    assert metrics["per_class"]["PNEUMONIA"]["support"] == 2


def test_patient_aggregation_requires_strict_majority() -> None:
    rows = [
        {"patient_id": "a", "true_label": "NORMAL", "predicted_label": "NORMAL", "parse_ok": True, "status": "ok"},
        {"patient_id": "a", "true_label": "NORMAL", "predicted_label": "UNCERTAIN", "parse_ok": False, "status": "parse_failure"},
        {"patient_id": "b", "true_label": "PNEUMONIA", "predicted_label": "PNEUMONIA", "parse_ok": True, "status": "ok"},
        {"patient_id": "b", "true_label": "PNEUMONIA", "predicted_label": "PNEUMONIA", "parse_ok": True, "status": "ok"},
        {"patient_id": "b", "true_label": "PNEUMONIA", "predicted_label": "UNCERTAIN", "parse_ok": False, "status": "missing_prediction"},
    ]
    patients = aggregate_patients(rows)
    assert {row["patient_id"]: row["predicted_label"] for row in patients} == {
        "a": "UNCERTAIN",
        "b": "PNEUMONIA",
    }


def test_patient_true_label_conflict_fails() -> None:
    with pytest.raises(ValueError, match="spans true labels"):
        aggregate_patients(
            [
                {"patient_id": "x", "true_label": "NORMAL", "predicted_label": "NORMAL"},
                {"patient_id": "x", "true_label": "PNEUMONIA", "predicted_label": "PNEUMONIA"},
            ]
        )


def test_bootstrap_is_deterministic_and_mcnemar_counts_discordance() -> None:
    candidate = [
        {"patient_id": "n1", "true_label": "NORMAL", "predicted_label": "NORMAL"},
        {"patient_id": "n2", "true_label": "NORMAL", "predicted_label": "UNCERTAIN"},
        {"patient_id": "p1", "true_label": "PNEUMONIA", "predicted_label": "PNEUMONIA"},
        {"patient_id": "p2", "true_label": "PNEUMONIA", "predicted_label": "PNEUMONIA"},
    ]
    comparator = [
        {"patient_id": "n1", "true_label": "NORMAL", "predicted_label": "NORMAL"},
        {"patient_id": "n2", "true_label": "NORMAL", "predicted_label": "NORMAL"},
        {"patient_id": "p1", "true_label": "PNEUMONIA", "predicted_label": "UNCERTAIN"},
        {"patient_id": "p2", "true_label": "PNEUMONIA", "predicted_label": "PNEUMONIA"},
    ]
    assert patient_bootstrap_ci(candidate, 100, 42) == patient_bootstrap_ci(candidate, 100, 42)
    result = paired_mcnemar(candidate, comparator)
    assert result["candidate_correct_comparator_wrong"] == 1
    assert result["candidate_wrong_comparator_correct"] == 1
    assert result["discordant"] == 2
    difference = paired_patient_bootstrap_difference_ci(candidate, comparator, 100, 42)
    assert difference == paired_patient_bootstrap_difference_ci(candidate, comparator, 100, 42)
    assert difference["balanced_accuracy"]["estimate"] == 0.0
    assert difference["recall_normal"]["estimate"] == -0.5
    assert difference["recall_pneumonia"]["estimate"] == 0.5
