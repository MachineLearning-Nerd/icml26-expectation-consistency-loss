#!/usr/bin/env python3
"""Independent stdlib checker for Claim 5 route 3."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path


def hard_ece(
    confidence: list[float], prediction: list[int], labels: list[int]
) -> float:
    value = 0.0
    for bin_index in range(15):
        lower = bin_index / 15
        upper = (bin_index + 1) / 15
        selected = [
            index
            for index, item in enumerate(confidence)
            if lower < item <= upper
        ]
        if selected:
            mean_confidence = sum(confidence[index] for index in selected) / len(
                selected
            )
            mean_accuracy = sum(
                prediction[index] == labels[index] for index in selected
            ) / len(selected)
            value += len(selected) / len(labels) * abs(
                mean_confidence - mean_accuracy
            )
    return value


def check(
    predictions_path: Path, results_path: Path, output_path: Path
) -> dict[str, object]:
    raw = predictions_path.read_bytes()
    recorded = json.loads(results_path.read_text())
    if sha256(raw).hexdigest() != recorded["digests"]["predictions_csv"]:
        raise RuntimeError("route-3 prediction CSV digest mismatch")
    columns: dict[str, list[float | int]] = {
        "label": [],
        "baseline_prediction": [],
        "baseline_confidence": [],
        "ecl_prediction": [],
        "ecl_confidence": [],
    }
    with predictions_path.open(newline="", encoding="utf-8") as handle:
        for expected_index, row in enumerate(csv.DictReader(handle)):
            if int(row["index"]) != expected_index:
                raise RuntimeError("route-3 prediction index is noncontiguous")
            columns["label"].append(int(row["label"]))
            columns["baseline_prediction"].append(
                int(row["baseline_prediction"])
            )
            columns["baseline_confidence"].append(
                float(row["baseline_confidence"])
            )
            columns["ecl_prediction"].append(int(row["ecl_prediction"]))
            columns["ecl_confidence"].append(float(row["ecl_confidence"]))
    labels = columns["label"]
    baseline_prediction = columns["baseline_prediction"]
    baseline_confidence = columns["baseline_confidence"]
    ecl_prediction = columns["ecl_prediction"]
    ecl_confidence = columns["ecl_confidence"]
    baseline_ece = hard_ece(
        baseline_confidence, baseline_prediction, labels
    )
    ecl_ece = hard_ece(ecl_confidence, ecl_prediction, labels)
    baseline_accuracy = sum(
        prediction == label
        for prediction, label in zip(baseline_prediction, labels)
    ) / len(labels)
    ecl_accuracy = sum(
        prediction == label
        for prediction, label in zip(ecl_prediction, labels)
    ) / len(labels)
    rotated = labels[1:] + labels[:1]
    rotated_ece = hard_ece(ecl_confidence, ecl_prediction, rotated)
    observed = recorded["observed"]
    comparisons = {
        "baseline_ece_error": abs(
            baseline_ece - float(observed["baseline_ece_fraction"])
        ),
        "ecl_ece_error": abs(
            ecl_ece - float(observed["ecl_ece_fraction"])
        ),
        "baseline_accuracy_error": abs(
            baseline_accuracy - float(observed["baseline_accuracy"])
        ),
        "ecl_accuracy_error": abs(
            ecl_accuracy - float(observed["ecl_accuracy"])
        ),
    }
    diagnostic = recorded["boundary_diagnostic"]
    controls = {
        "row_count_exact": len(labels) == 99289,
        "all_values_finite": all(
            math.isfinite(value)
            for value in baseline_confidence + ecl_confidence
        ),
        "all_confidences_in_unit_interval": all(
            0 <= value <= 1
            for value in baseline_confidence + ecl_confidence
        ),
        "primary_values_match_with_float32_csv_tolerance": all(
            error <= 5e-8 for error in comparisons.values()
        ),
        "label_rotation_changes_ece": abs(rotated_ece - ecl_ece) > 1e-3,
        "target_labels_declared_unused": (
            recorded["protocol"]["target_labels_used_during_training"] is False
        ),
        "literal_boundary_gradient_rejected": (
            diagnostic["literal_gradient_all_finite"] is False
        ),
        "stabilized_boundary_gradient_finite": (
            diagnostic["stable_gradient_all_finite"] is True
            and diagnostic["epsilon"] == 1e-12
        ),
    }
    passed = all(controls.values())
    result = {
        "status": "PASS" if passed else "FAILED",
        "independence": "stdlib CSV parser and explicit 15-bin loops; no primary implementation import",
        "observed": {
            "baseline_ece_fraction": baseline_ece,
            "ecl_ece_fraction": ecl_ece,
            "baseline_accuracy": baseline_accuracy,
            "ecl_accuracy": ecl_accuracy,
            "rotated_label_ece_fraction": rotated_ece,
        },
        "comparisons": comparisons,
        "negative_controls": controls,
        "prediction_csv_sha256": sha256(raw).hexdigest(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("Claim 5 route 3 independent checker failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.predictions, args.results, args.output)


if __name__ == "__main__":
    main()
