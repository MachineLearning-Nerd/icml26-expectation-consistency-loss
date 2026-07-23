#!/usr/bin/env python3
"""Independent, fail-closed checker for the Figure 2 simulation route."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


METHODS = ("uncalibrated", "soft_ece", "ecl")
PARADIGMS = ("TopLabel", "Classwise", "Canonical")


def scalar_metric(probs: np.ndarray, labels: np.ndarray, paradigm: str, bins: int = 15) -> float:
    if probs.shape != (400, 3) or labels.shape != (400,):
        raise ValueError("unexpected full-scale array shape")
    one_hot = np.eye(3)[labels]
    if paradigm == "Canonical":
        resolution = 1
        while math.comb(resolution + 2, 2) < bins:
            resolution += 1
        grid = []
        for i in range(resolution + 1):
            for j in range(resolution - i + 1):
                grid.append([i, j, resolution - i - j])
        anchors = (np.asarray(grid, dtype=float) + 1 / 3) / (resolution + 1)
        assignment = ((probs[:, None, :] - anchors[None, :, :]) ** 2).sum(2).argmin(1)
        value = 0.0
        for index in range(len(anchors)):
            mask = assignment == index
            if mask.any():
                value += mask.mean() * np.linalg.norm(probs[mask].mean(0) - one_hot[mask].mean(0))
        return float(value)

    boundaries = np.linspace(0.0, 1.0, bins + 1)
    if paradigm == "TopLabel":
        confidence = probs.max(1)
        observed = probs.argmax(1) == labels
        class_pairs = ((confidence, observed.astype(float)),)
    else:
        class_pairs = tuple((probs[:, k], one_hot[:, k]) for k in range(3))
    values = []
    for confidence, observed in class_pairs:
        value = 0.0
        for lower, upper in zip(boundaries[:-1], boundaries[1:]):
            mask = (confidence > lower) & (confidence <= upper)
            # This reproduces the released Figure 2 implementation exactly.
            if mask.sum() > 5:
                value += mask.mean() * abs(observed[mask].mean() - confidence[mask].mean())
        values.append(value)
    return float(np.mean(values))


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(probs.argmax(1) == labels))


def check(payload: dict[str, object]) -> dict[str, object]:
    contract = payload["paper_contract"]
    if contract["samples_per_domain"] != 400 or contract["bins"] != 15:
        raise ValueError("paper-scale contract was weakened")
    if payload["implementation_choices"]["target_labels_used_for_training"] is not False:
        raise ValueError("target labels leaked into training")
    records = payload["records"]
    expected = 5 * len(PARADIGMS)
    if len(records) != expected:
        raise ValueError(f"expected {expected} records")

    rows = []
    grouped: dict[str, dict[str, list[float]]] = {
        p: {m: [] for m in METHODS} for p in PARADIGMS
    }
    accuracy_grouped: dict[str, dict[str, list[float]]] = {
        p: {m: [] for m in METHODS} for p in PARADIGMS
    }
    seen = set()
    for record in records:
        key = (record["seed"], record["paradigm"])
        if key in seen:
            raise ValueError("duplicate seed/paradigm")
        seen.add(key)
        paradigm = record["paradigm"]
        labels = np.asarray(record["target_labels"], dtype=int)
        for method in METHODS:
            probs = np.asarray(record["probabilities"][method], dtype=float)
            if not np.all(np.isfinite(probs)):
                raise ValueError("non-finite probabilities")
            if np.max(np.abs(probs.sum(1) - 1.0)) > 2e-6 or probs.min() < 0:
                raise ValueError("invalid probabilities")
            metric = scalar_metric(probs, labels, paradigm)
            acc = accuracy(probs, labels)
            grouped[paradigm][method].append(metric)
            accuracy_grouped[paradigm][method].append(acc)
            rows.append(
                {
                    "seed": record["seed"],
                    "paradigm": paradigm,
                    "method": method,
                    "calibration_error": metric,
                    "accuracy": acc,
                }
            )

    summaries = {}
    all_aligned = True
    for paradigm in PARADIGMS:
        summaries[paradigm] = {}
        for method in METHODS:
            values = np.asarray(grouped[paradigm][method])
            acc_values = np.asarray(accuracy_grouped[paradigm][method])
            summaries[paradigm][method] = {
                "mean_calibration_error": float(values.mean()),
                "sample_sd_calibration_error": float(values.std(ddof=1)),
                "mean_accuracy": float(acc_values.mean()),
                "sample_sd_accuracy": float(acc_values.std(ddof=1)),
            }
        ecl = np.asarray(grouped[paradigm]["ecl"])
        uncal = np.asarray(grouped[paradigm]["uncalibrated"])
        soft = np.asarray(grouped[paradigm]["soft_ece"])
        improvement = uncal - ecl
        ci_low = float(improvement.mean() - 2.776 * improvement.std(ddof=1) / math.sqrt(len(improvement)))
        summaries[paradigm]["paired_ecl_vs_uncalibrated"] = {
            "mean_improvement": float(improvement.mean()),
            "95pct_t_ci_low": ci_low,
        }
        route_aligned = bool(ecl.mean() < uncal.mean() and ecl.mean() < soft.mean() and ci_low > 0)
        summaries[paradigm]["route_aligned"] = route_aligned
        all_aligned = all_aligned and route_aligned

    means = np.asarray(
        [[row["source_empirical_mean"], row["target_empirical_mean"]] for row in payload["data_audit"]],
        dtype=float,
    )
    observed_shift = means[:, 1, :].mean(0) - means[:, 0, :].mean(0)
    if np.max(np.abs(observed_shift - np.array([2.0, 2.0]))) > 0.5:
        raise ValueError("negative control: generated domains do not exhibit the contracted shift")

    # Internal tamper control: changing one stored probability must change its recomputed metric.
    first = records[0]
    original = np.asarray(first["probabilities"]["uncalibrated"], dtype=float)
    tampered = original.copy()
    tampered[0] = np.array([1.0, 0.0, 0.0])
    original_metric = scalar_metric(original, np.asarray(first["target_labels"]), first["paradigm"])
    tampered_metric = scalar_metric(tampered, np.asarray(first["target_labels"]), first["paradigm"])
    if original_metric == tampered_metric:
        raise ValueError("negative control failed to perturb independently recomputed metric")

    return {
        "status": "PASS",
        "route_assessment": "ALIGNED" if all_aligned else "DIVERGENT_OR_MIXED",
        "verdict_effect": "This route alone cannot verify or falsify the compound Claim 6.",
        "rows": rows,
        "summaries": summaries,
        "negative_controls": {
            "observed_mean_shift": observed_shift.tolist(),
            "tampered_metric_changed": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for paradigm, summary in result["summaries"].items():
        print(
            "CLAIM6_CHECK "
            f"paradigm={paradigm} "
            f"uncal={summary['uncalibrated']['mean_calibration_error']:.12f} "
            f"soft={summary['soft_ece']['mean_calibration_error']:.12f} "
            f"ecl={summary['ecl']['mean_calibration_error']:.12f} "
            f"ci_low={summary['paired_ecl_vs_uncalibrated']['95pct_t_ci_low']:.12f} "
            f"aligned={summary['route_aligned']}"
        )
    print(f"CLAIM6_CHECK_RESULT status=PASS assessment={result['route_assessment']}")


if __name__ == "__main__":
    main()
