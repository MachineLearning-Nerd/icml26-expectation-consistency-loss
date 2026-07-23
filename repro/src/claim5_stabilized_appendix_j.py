#!/usr/bin/env python3
"""Claim 5 route 3: numerically stabilized Appendix-J reconstruction."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch
import torch.nn.functional as F

from claim5_lenet_svhn_reconstruction import (
    BATCH_SIZE,
    BASELINE_EPOCHS,
    ECL_EPOCHS,
    ECL_LAMBDA_CE,
    HEAD_EPOCHS,
    LEARNING_RATE,
    LeNet5,
    atomic_json,
    batches,
    ece,
    extract_features,
    load_full_domains,
    predict_from_features,
    seed_everything,
    soft_ece,
    state_digest,
    train_baseline,
    train_ecl,
    write_predictions,
)


SEED = 260521554
SOFT_ECE_EPSILON = 1e-12


def stable_soft_ece(
    logits: torch.Tensor, correct: torch.Tensor, bins: int = 15
) -> torch.Tensor:
    """Appendix-J Soft-ECE with an explicit finite derivative at zero."""
    confidence = torch.softmax(logits, dim=1).max(dim=1).values
    anchors = (
        2 * torch.arange(bins, dtype=confidence.dtype, device=confidence.device) + 1
    ) / (2 * bins)
    temperature = -1.0 / (math.log(0.9) * bins * bins)
    assignment = torch.softmax(
        -((confidence[:, None] - anchors[None, :]) ** 2) / temperature,
        dim=1,
    )
    mass = assignment.sum(dim=0)
    denominator = torch.clamp(mass, min=1e-5)
    mean_confidence = (assignment.T @ confidence) / denominator
    mean_accuracy = (
        assignment.T @ correct.to(confidence.dtype)
    ) / denominator
    squared = torch.sum(
        mass
        / torch.clamp(mass.sum(), min=1e-5)
        * (mean_confidence - mean_accuracy).square()
    )
    return torch.sqrt(squared + SOFT_ECE_EPSILON)


def boundary_diagnostic() -> dict[str, object]:
    """Expose the exact boundary that invalidated route 1."""
    correctness = torch.tensor([False, True])
    literal_logits = torch.zeros(2, 2, requires_grad=True)
    literal_value = soft_ece(literal_logits, correctness)
    literal_gradient = torch.autograd.grad(literal_value, literal_logits)[0]
    stable_logits = torch.zeros(2, 2, requires_grad=True)
    stable_value = stable_soft_ece(stable_logits, correctness)
    stable_gradient = torch.autograd.grad(stable_value, stable_logits)[0]
    result = {
        "fixture": "two tied binary logits with correctness [0, 1]",
        "literal_value": float(literal_value.detach()),
        "literal_gradient_all_finite": bool(
            torch.all(torch.isfinite(literal_gradient))
        ),
        "stable_value": float(stable_value.detach()),
        "stable_gradient_all_finite": bool(
            torch.all(torch.isfinite(stable_gradient))
        ),
        "stable_gradient_max_abs": float(stable_gradient.detach().abs().max()),
        "epsilon": SOFT_ECE_EPSILON,
    }
    if not (
        result["literal_value"] == 0.0
        and not result["literal_gradient_all_finite"]
        and result["stable_gradient_all_finite"]
        and result["stable_value"] > 0
    ):
        raise RuntimeError("Soft-ECE boundary diagnostic did not separate variants")
    return result


def train_stable_correctness_head(
    model: LeNet5,
    features: torch.Tensor,
    labels: torch.Tensor,
    generator: torch.Generator,
) -> list[dict[str, float]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.classifier2.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        model.classifier2.parameters(), lr=LEARNING_RATE
    )
    rows = []
    model.train()
    for epoch in range(HEAD_EPOCHS):
        loss_sum = 0.0
        calibration_sum = 0.0
        head_correct = 0
        seen = 0
        for index in batches(len(features), generator):
            feature = features[index]
            label = labels[index]
            with torch.no_grad():
                prediction = model.classifier(feature).argmax(dim=1)
                correctness = prediction == label
            optimizer.zero_grad(set_to_none=True)
            logits = model.classifier2(feature)
            calibration = stable_soft_ece(logits, correctness)
            loss = F.cross_entropy(logits, correctness.long()) + calibration
            if not torch.isfinite(loss):
                raise RuntimeError("stabilized head objective became non-finite")
            loss.backward()
            if any(
                parameter.grad is not None
                and not torch.all(torch.isfinite(parameter.grad))
                for parameter in model.classifier2.parameters()
            ):
                raise RuntimeError("stabilized head gradient became non-finite")
            optimizer.step()
            loss_sum += float(loss.detach()) * len(index)
            calibration_sum += float(calibration.detach()) * len(index)
            head_correct += int(
                (logits.argmax(dim=1) == correctness).sum()
            )
            seen += len(index)
        row = {
            "epoch": epoch + 1,
            "head_loss": loss_sum / seen,
            "soft_ece": calibration_sum / seen,
            "head_correctness_accuracy": head_correct / seen,
        }
        rows.append(row)
        if (epoch + 1) % 10 == 0:
            print(
                f"CLAIM5_ROUTE3_HEAD epoch={epoch + 1} "
                f"loss={row['head_loss']:.8f} soft_ece={row['soft_ece']:.8f} "
                f"acc={row['head_correctness_accuracy']:.8f}"
            )
    return rows


def run(
    cache_root: Path, predictions_path: Path, output_path: Path
) -> dict[str, object]:
    started = time.perf_counter()
    seed_everything(SEED)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    diagnostic = boundary_diagnostic()
    (
        source_images,
        source_labels,
        target_images,
        target_labels_tensor,
        provenance,
    ) = load_full_domains(cache_root)
    model = LeNet5()
    baseline_curve = train_baseline(
        model,
        source_images,
        source_labels,
        torch.Generator().manual_seed(SEED + 1),
    )
    source_features = extract_features(model, source_images)
    target_features = extract_features(model, target_images)
    del source_images, target_images
    target_labels = target_labels_tensor.numpy()
    baseline_prediction, baseline_confidence = predict_from_features(
        model, target_features
    )
    baseline_model_digest = state_digest(model)
    head_curve = train_stable_correctness_head(
        model,
        source_features,
        source_labels,
        torch.Generator().manual_seed(SEED + 2),
    )
    ecl_curve = train_ecl(
        model,
        source_features,
        source_labels,
        target_features,
        torch.Generator().manual_seed(SEED + 3),
        torch.Generator().manual_seed(SEED + 4),
    )
    ecl_prediction, ecl_confidence = predict_from_features(
        model, target_features
    )
    baseline_ece = ece(
        baseline_confidence, baseline_prediction, target_labels
    )
    ecl_ece = ece(ecl_confidence, ecl_prediction, target_labels)
    baseline_accuracy = float(
        (baseline_prediction == target_labels).mean()
    )
    ecl_accuracy = float((ecl_prediction == target_labels).mean())
    prediction_sha = write_predictions(
        predictions_path,
        target_labels,
        baseline_prediction,
        baseline_confidence,
        ecl_prediction,
        ecl_confidence,
    )
    paper = {
        "uncal_ece_fraction": 0.619,
        "uncal_std_fraction": 0.0616,
        "ecl_ece_fraction": 0.215,
        "ecl_std_fraction": 0.0151,
        "accuracy_delta_fraction": 0.0165,
    }
    compatibility = {
        "uncal_within_paper_mean_plus_minus_2sd": abs(
            baseline_ece - paper["uncal_ece_fraction"]
        )
        <= 2 * paper["uncal_std_fraction"],
        "ecl_within_paper_mean_plus_minus_2sd": abs(
            ecl_ece - paper["ecl_ece_fraction"]
        )
        <= 2 * paper["ecl_std_fraction"],
        "ecl_reduces_ece": ecl_ece < baseline_ece,
    }
    all_curve_values = [
        value
        for curve in (baseline_curve, head_curve, ecl_curve)
        for row in curve
        for key, value in row.items()
        if key != "epoch"
    ]
    result = {
        "status": "SUBSTANTIVE_SINGLE_SEED",
        "claim": "LeNet-5 target-SVHN ECL ECE 21.5% versus uncalibrated 61.9%, ten runs.",
        "route": "Appendix J frozen correctness head plus Algorithm 2, with explicitly stabilized Soft-ECE boundary",
        "seed": SEED,
        "paper_values": paper,
        "observed": {
            "baseline_ece_fraction": baseline_ece,
            "ecl_ece_fraction": ecl_ece,
            "absolute_ece_reduction": baseline_ece - ecl_ece,
            "relative_ece_reduction": (
                (baseline_ece - ecl_ece) / baseline_ece
                if baseline_ece > 0
                else None
            ),
            "baseline_accuracy": baseline_accuracy,
            "ecl_accuracy": ecl_accuracy,
            "accuracy_delta": ecl_accuracy - baseline_accuracy,
        },
        "compatibility_checks": compatibility,
        "boundary_diagnostic": diagnostic,
        "dataset_provenance": provenance,
        "protocol": {
            "architecture": "exact predecessor LeNet-5 topology",
            "source": "MNIST train+test plus USPS train+test",
            "target": "SVHN train+test",
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "baseline_epochs": BASELINE_EPOCHS,
            "correctness_head_epochs": HEAD_EPOCHS,
            "ecl_finetune_epochs": ECL_EPOCHS,
            "ecl_lambda_ce": ECL_LAMBDA_CE,
            "soft_ece_epsilon_inside_sqrt": SOFT_ECE_EPSILON,
            "target_labels_used_during_training": False,
            "backbone_frozen_for_correctness_head_and_ecl_finetune": True,
        },
        "training_curves": {
            "baseline": baseline_curve,
            "correctness_head": head_curve,
            "ecl": ecl_curve,
        },
        "digests": {
            "baseline_model": baseline_model_digest,
            "ecl_model": state_digest(model),
            "predictions_csv": prediction_sha,
        },
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "torch_threads": torch.get_num_threads(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
        },
        "limitations": [
            "One deterministic seed cannot verify the paper's ten-run mean and standard deviation.",
            "The epsilon repair is explicit but not stated in the paper or current official source.",
            "The digit ECL cross-entropy coefficient is unstated; 0.5 comes from the official top-label demonstration.",
            "The current official source contains no executable digit pipeline, seeds, checkpoints, or per-run observations.",
            "Only the LeNet-5 target-SVHN cell is tested.",
        ],
    }
    if not (
        provenance["all_hashes_match"]
        and len(target_labels) == 99289
        and all(math.isfinite(value) for value in all_curve_values)
        and all(
            math.isfinite(value)
            for value in (
                baseline_ece,
                ecl_ece,
                baseline_accuracy,
                ecl_accuracy,
            )
        )
        and prediction_sha
    ):
        raise RuntimeError("Claim 5 route 3 fail-closed invariant failed")
    atomic_json(output_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "paper_values": paper,
                "observed": result["observed"],
                "compatibility_checks": compatibility,
                "boundary_diagnostic": diagnostic,
                "counts": provenance["counts"],
                "all_hashes_match": provenance["all_hashes_match"],
                "digests": result["digests"],
                "wall_seconds": result["runtime"]["wall_seconds"],
                "limitations": result["limitations"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path.home()
        / ".cache"
        / "openresearch"
        / "datasets"
        / "ecl-digit",
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache_root, args.predictions, args.output)


if __name__ == "__main__":
    main()
