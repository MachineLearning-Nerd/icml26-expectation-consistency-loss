#!/usr/bin/env python3
"""Independent stdlib checker for Claim 5 route-1 prediction evidence."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path


def hard_ece(confidences: list[float], predictions: list[int], labels: list[int]) -> float:
    total = len(labels)
    value = 0.0
    for bin_index in range(15):
        lower = bin_index / 15
        upper = (bin_index + 1) / 15
        selected = [
            index
            for index, confidence in enumerate(confidences)
            if lower < confidence <= upper
        ]
        if selected:
            confidence_mean = sum(confidences[index] for index in selected) / len(
                selected
            )
            accuracy_mean = sum(
                predictions[index] == labels[index] for index in selected
            ) / len(selected)
            value += len(selected) / total * abs(confidence_mean - accuracy_mean)
    return value


def check(predictions_path: Path, results_path: Path, output_path: Path) -> dict[str, object]:
    raw = predictions_path.read_bytes()
    recorded = json.loads(results_path.read_text())
    if sha256(raw).hexdigest() != recorded["digests"]["predictions_csv"]:
        raise RuntimeError("prediction CSV digest mismatch")
    labels: list[int] = []
    baseline_predictions: list[int] = []
    baseline_confidences: list[float] = []
    ecl_predictions: list[int] = []
    ecl_confidences: list[float] = []
    with predictions_path.open(newline="", encoding="utf-8") as handle:
        for expected_index, row in enumerate(csv.DictReader(handle)):
            if int(row["index"]) != expected_index:
                raise RuntimeError("prediction rows are not contiguous")
            labels.append(int(row["label"]))
            baseline_predictions.append(int(row["baseline_prediction"]))
            baseline_confidences.append(float(row["baseline_confidence"]))
            ecl_predictions.append(int(row["ecl_prediction"]))
            ecl_confidences.append(float(row["ecl_confidence"]))
    baseline_ece = hard_ece(
        baseline_confidences, baseline_predictions, labels
    )
    ecl_ece = hard_ece(ecl_confidences, ecl_predictions, labels)
    baseline_accuracy = sum(
        prediction == label
        for prediction, label in zip(baseline_predictions, labels)
    ) / len(labels)
    ecl_accuracy = sum(
        prediction == label for prediction, label in zip(ecl_predictions, labels)
    ) / len(labels)
    observed = recorded["observed"]
    comparisons = {
        "baseline_ece_error": abs(
            baseline_ece - float(observed["baseline_ece_fraction"])
        ),
        "ecl_ece_error": abs(ecl_ece - float(observed["ecl_ece_fraction"])),
        "baseline_accuracy_error": abs(
            baseline_accuracy - float(observed["baseline_accuracy"])
        ),
        "ecl_accuracy_error": abs(
            ecl_accuracy - float(observed["ecl_accuracy"])
        ),
    }
    permutation_labels = labels[1:] + labels[:1]
    permuted_ece = hard_ece(
        ecl_confidences, ecl_predictions, permutation_labels
    )
    controls = {
        "row_count_exact": len(labels) == 99289,
        "all_primary_values_match": all(
            error <= 2e-12 for error in comparisons.values()
        ),
        "label_rotation_changes_ecl": abs(permuted_ece - ecl_ece) > 1e-3,
        "all_confidences_in_unit_interval": all(
            0 <= value <= 1
            for value in baseline_confidences + ecl_confidences
        ),
        "all_labels_predictions_valid": all(
            0 <= value <= 9
            for value in labels + baseline_predictions + ecl_predictions
        ),
    }
    passed = all(controls.values())
    result = {
        "status": "PASS" if passed else "FAILED",
        "independence": "stdlib CSV parser and explicit hard-bin loops; no import from primary route",
        "observed": {
            "baseline_ece_fraction": baseline_ece,
            "ecl_ece_fraction": ecl_ece,
            "baseline_accuracy": baseline_accuracy,
            "ecl_accuracy": ecl_accuracy,
            "label_rotation_ecl_fraction": permuted_ece,
        },
        "comparisons": comparisons,
        "negative_controls": controls,
        "prediction_csv_sha256": sha256(raw).hexdigest(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("Claim 5 independent checker failed")
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
