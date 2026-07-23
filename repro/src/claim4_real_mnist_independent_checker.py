#!/usr/bin/env python3
"""Independent NumPy checker for the real-MNIST Appendix F certificate."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import comb, log
from pathlib import Path

import numpy as np


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _temperature(bins: int, decay: float) -> float:
    return -1.0 / (log(decay) * bins * bins)


def _scalar_anchors(bins: int) -> np.ndarray:
    return (2 * np.arange(bins, dtype=np.float64) + 1) / (2 * bins)


def _simplex_anchors(requested: int, classes: int) -> np.ndarray:
    resolution = 1
    while comb(resolution + classes - 1, classes - 1) < requested:
        resolution += 1

    def compositions(total: int, dimensions: int):
        if dimensions == 1:
            yield [total]
            return
        for first in range(total + 1):
            for suffix in compositions(total - first, dimensions - 1):
                yield [first, *suffix]

    grid = np.asarray(list(compositions(resolution, classes)), dtype=np.float64)
    return (grid + 1.0 / classes) / (resolution + 1.0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def _assign(scores: np.ndarray, anchors: np.ndarray, soft_temperature: float) -> np.ndarray:
    if scores.ndim == 1:
        distance = (scores[:, None] - anchors[None, :]) ** 2
    else:
        distance = np.sum(
            (scores[:, None, :] - anchors[None, :, :]) ** 2, axis=2
        )
    return _softmax(-distance / soft_temperature)


def _scalar_loss(
    scores: np.ndarray,
    posterior: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    *,
    bins: int,
    decay: float,
    stabilizer: float,
) -> float:
    assignment = _assign(scores, _scalar_anchors(bins), _temperature(bins, decay))
    source_mass = source @ assignment
    target_mass = target @ assignment
    source_mean = (source[:, None] * assignment).T @ posterior / (
        source_mass + stabilizer
    )
    target_mean = (target[:, None] * assignment).T @ posterior / (
        target_mass + stabilizer
    )
    return float(
        np.sum(target_mass / target.sum() * np.abs(source_mean - target_mean))
    )


def losses(data: dict[str, object], calibration_temperature: float) -> dict[str, float]:
    logits = np.asarray(data["logits"], dtype=np.float64)
    posterior = np.asarray(data["posterior"], dtype=np.float64)
    source = np.asarray(data["source_counts"], dtype=np.float64)
    target = np.asarray(data["target_counts"], dtype=np.float64)
    bins = int(data["requested_bins"])
    decay = float(data["decay_factor"])
    stabilizer = float(data["stabilizer"])
    probabilities = _softmax(logits / calibration_temperature)
    prediction = logits.argmax(axis=1)
    top = _scalar_loss(
        probabilities.max(axis=1),
        posterior[np.arange(len(posterior)), prediction],
        source,
        target,
        bins=bins,
        decay=decay,
        stabilizer=stabilizer,
    )
    class_wise = sum(
        _scalar_loss(
            probabilities[:, class_index],
            posterior[:, class_index],
            source,
            target,
            bins=bins,
            decay=decay,
            stabilizer=stabilizer,
        )
        for class_index in range(probabilities.shape[1])
    )
    anchors = _simplex_anchors(bins, probabilities.shape[1])
    assignment = _assign(
        probabilities, anchors, _temperature(len(anchors), decay)
    )
    source_mass = source @ assignment
    target_mass = target @ assignment
    source_mean = ((source[:, None] * assignment).T @ posterior) / (
        source_mass[:, None] + stabilizer
    )
    target_mean = ((target[:, None] * assignment).T @ posterior) / (
        target_mass[:, None] + stabilizer
    )
    canonical = float(
        np.sum(
            target_mass
            / target.sum()
            * np.linalg.norm(source_mean - target_mean, axis=1)
        )
    )
    return {
        "top_label": top,
        "class_wise": float(class_wise),
        "canonical": canonical,
    }


def check(inputs_path: Path, results_path: Path, output_path: Path) -> dict[str, object]:
    inputs_bytes = inputs_path.read_bytes()
    data = json.loads(inputs_bytes)
    recorded = json.loads(results_path.read_text())
    if sha256(inputs_bytes).hexdigest() != recorded["inputs"]["sha256"]:
        raise RuntimeError("input SHA-256 does not match the primary certificate")
    baseline = losses(data, float(data["calibration_temperature"]))
    step = float(data["finite_difference_step"])
    upper = losses(data, float(data["calibration_temperature"]) + step)
    lower = losses(data, float(data["calibration_temperature"]) - step)
    rows = {}
    all_match = True
    for mode in baseline:
        finite_difference = (upper[mode] - lower[mode]) / (2 * step)
        primary_loss = float(recorded["formulations"][mode]["loss"])
        primary_gradient = float(
            recorded["formulations"][mode]["autograd_temperature_gradient"]
        )
        loss_error = abs(baseline[mode] - primary_loss)
        gradient_error = abs(finite_difference - primary_gradient)
        gradient_scale = max(1.0, abs(finite_difference), abs(primary_gradient))
        mode_match = loss_error <= 2e-11 and gradient_error <= 2e-7 * gradient_scale
        rows[mode] = {
            "numpy_loss": baseline[mode],
            "primary_torch_loss": primary_loss,
            "loss_absolute_error": loss_error,
            "centered_finite_difference_gradient": finite_difference,
            "primary_autograd_gradient": primary_gradient,
            "gradient_absolute_error": gradient_error,
            "gradient_relative_scale_error": gradient_error / gradient_scale,
            "match": mode_match,
        }
        all_match &= mode_match
    output = {
        "status": "VERIFIED" if all_match else "FAILED",
        "independence": "NumPy-only implementation; does not import the PyTorch certificate module.",
        "finite_difference_step": step,
        "input_sha256": sha256(inputs_bytes).hexdigest(),
        "formulations": rows,
        "all_three_losses_and_gradients_match": all_match,
    }
    _atomic_json(output_path, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    if not all_match:
        raise SystemExit("independent Claim 4 checker failed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.inputs, args.results, args.output)


if __name__ == "__main__":
    main()
