#!/usr/bin/env python3
"""Independent stdlib checker for Claim 5 route 2."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path

def hard_ece(
    confidences: list[float], predictions: list[int], labels: list[int]
) -> float:
    """Recompute the paper's 15-bin ECE without project implementation code."""
    total = len(labels)
    result = 0.0
    for bin_index in range(15):
        lower = bin_index / 15
        upper = (bin_index + 1) / 15
        selected = [
            index
            for index, confidence in enumerate(confidences)
            if lower < confidence <= upper
        ]
        if selected:
            mean_confidence = sum(confidences[index] for index in selected) / len(
                selected
            )
            accuracy = sum(
                predictions[index] == labels[index] for index in selected
            ) / len(selected)
            result += len(selected) / total * abs(mean_confidence - accuracy)
    return result


def check(predictions_path: Path, results_path: Path, output_path: Path) -> dict[str, object]:
    raw = predictions_path.read_bytes()
    recorded = json.loads(results_path.read_text())
    if sha256(raw).hexdigest() != recorded["digests"]["predictions_csv"]:
        raise RuntimeError("route-2 prediction CSV digest mismatch")
    labels = []
    baseline_prediction = []
    baseline_confidence = []
    calibrated_prediction = []
    calibrated_confidence = []
    with predictions_path.open(newline="", encoding="utf-8") as handle:
        for expected_index, row in enumerate(csv.DictReader(handle)):
            if int(row["index"]) != expected_index:
                raise RuntimeError("noncontiguous prediction index")
            labels.append(int(row["label"]))
            baseline_prediction.append(int(row["baseline_prediction"]))
            baseline_confidence.append(float(row["baseline_confidence"]))
            calibrated_prediction.append(int(row["ecl_prediction"]))
            calibrated_confidence.append(float(row["ecl_confidence"]))
    baseline_ece = hard_ece(
        baseline_confidence, baseline_prediction, labels
    )
    calibrated_ece = hard_ece(
        calibrated_confidence, calibrated_prediction, labels
    )
    accuracy = sum(
        prediction == label
        for prediction, label in zip(baseline_prediction, labels)
    ) / len(labels)
    observed = recorded["observed"]
    comparisons = {
        "baseline_ece_error": abs(
            baseline_ece - float(observed["baseline_ece_fraction"])
        ),
        "ecl_ece_error": abs(
            calibrated_ece - float(observed["ecl_ece_fraction"])
        ),
        "accuracy_error": abs(accuracy - float(observed["baseline_accuracy"])),
    }
    rotated = labels[1:] + labels[:1]
    rotated_ece = hard_ece(
        calibrated_confidence, calibrated_prediction, rotated
    )
    controls = {
        "row_count_exact": len(labels) == 99289,
        "all_values_finite": all(
            math.isfinite(value)
            for value in baseline_confidence + calibrated_confidence
        ),
        "all_confidences_in_unit_interval": all(
            0 <= value <= 1
            for value in baseline_confidence + calibrated_confidence
        ),
        "positive_temperature_preserves_every_prediction": (
            baseline_prediction == calibrated_prediction
        ),
        "primary_values_match_with_float32_csv_tolerance": all(
            error <= 5e-8 for error in comparisons.values()
        ),
        "label_rotation_changes_ece": abs(rotated_ece - calibrated_ece) > 1e-3,
    }
    passed = all(controls.values())
    result = {
        "status": "PASS" if passed else "FAILED",
        "independence": "stdlib CSV loop; no primary implementation import",
        "observed": {
            "baseline_ece_fraction": baseline_ece,
            "ecl_ece_fraction": calibrated_ece,
            "accuracy": accuracy,
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
        raise SystemExit("Claim 5 route 2 independent checker failed")
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
