#!/usr/bin/env python3
"""Claim 5 route 2: public-predecessor post-hoc interpretation.

Unlike route 1, this route follows Cali_in_Digit.py: train the primary LeNet
and its correctness head jointly, then choose a positive temperature from
1..50 without target labels.  The missing predecessor ECLoss_hd is filled by
the paper's Appendix-F top-label soft-bin formula and this repair is explicit.
"""
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
    LeNet5,
    atomic_json,
    ece,
    extract_features,
    load_full_domains,
    seed_everything,
    state_digest,
    write_predictions,
)


SEED = 260521553
PRIMARY_LR = 0.001
HEAD_LR = 0.01


def train_joint_predecessor(
    model: LeNet5,
    images: torch.Tensor,
    labels: torch.Tensor,
    generator: torch.Generator,
) -> list[dict[str, float]]:
    primary_optimizer = torch.optim.Adam(model.parameters(), lr=PRIMARY_LR)
    head_optimizer = torch.optim.Adam(model.classifier2.parameters(), lr=HEAD_LR)
    rows = []
    model.train()
    for epoch in range(BASELINE_EPOCHS):
        order = torch.randperm(len(images), generator=generator)
        primary_loss_sum = 0.0
        head_loss_sum = 0.0
        primary_correct = 0
        head_correct = 0
        seen = 0
        for start in range(0, len(order), BATCH_SIZE):
            index = order[start : start + BATCH_SIZE]
            x = images[index]
            y = labels[index]
            primary_optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            primary_loss = F.cross_entropy(logits, y)
            primary_loss.backward()
            primary_optimizer.step()

            with torch.no_grad():
                feature = model.features(x)
                prediction = model.classifier(feature).argmax(dim=1)
                correctness = prediction == y
                accuracy = correctness.float().mean()
                # Literal predecessor order: [1-batch accuracy, batch accuracy].
                class_weight = torch.stack([1 - accuracy, accuracy])
            head_optimizer.zero_grad(set_to_none=True)
            head_logits = model.classifier2(feature.detach())
            head_loss = F.cross_entropy(
                head_logits, correctness.long(), weight=class_weight
            )
            if not torch.isfinite(head_loss):
                raise RuntimeError("predecessor weighted head CE became non-finite")
            head_loss.backward()
            head_optimizer.step()

            primary_loss_sum += float(primary_loss.detach()) * len(index)
            head_loss_sum += float(head_loss.detach()) * len(index)
            primary_correct += int((prediction == y).sum())
            head_correct += int(
                (head_logits.argmax(dim=1) == correctness).sum()
            )
            seen += len(index)
        row = {
            "epoch": epoch + 1,
            "primary_cross_entropy": primary_loss_sum / seen,
            "head_cross_entropy": head_loss_sum / seen,
            "source_accuracy": primary_correct / seen,
            "head_correctness_accuracy": head_correct / seen,
        }
        rows.append(row)
        if (epoch + 1) % 10 == 0:
            print(
                f"CLAIM5_ROUTE2 epoch={epoch + 1} ce={row['primary_cross_entropy']:.8f} "
                f"head_ce={row['head_cross_entropy']:.8f} acc={row['source_accuracy']:.8f} "
                f"head_acc={row['head_correctness_accuracy']:.8f}"
            )
    return rows


@torch.no_grad()
def logits_from_features(
    model: LeNet5, features: torch.Tensor, *, head: bool = False
) -> torch.Tensor:
    module = model.classifier2 if head else model.classifier
    return torch.cat(
        [
            module(features[start : start + 2000])
            for start in range(0, len(features), 2000)
        ]
    )


def hard_source_ece(
    confidence: torch.Tensor, prediction: torch.Tensor, labels: torch.Tensor
) -> float:
    result = 0.0
    correctness = prediction == labels
    for bin_index in range(15):
        lower = bin_index / 15
        upper = (bin_index + 1) / 15
        selected = (confidence > lower) & (confidence <= upper)
        if torch.any(selected):
            result += float(selected.float().mean()) * abs(
                float(confidence[selected].mean())
                - float(correctness[selected].float().mean())
            )
    return result


def appendix_f_top_ecl(
    source_confidence: torch.Tensor,
    target_confidence: torch.Tensor,
    source_p_correct: torch.Tensor,
    target_p_correct: torch.Tensor,
) -> float:
    anchors = (2 * torch.arange(15, dtype=source_confidence.dtype) + 1) / 30
    soft_temperature = -1.0 / (math.log(0.9) * 15 * 15)

    def assignment(confidence: torch.Tensor) -> torch.Tensor:
        return torch.softmax(
            -((confidence[:, None] - anchors[None, :]) ** 2)
            / soft_temperature,
            dim=1,
        )

    source_assignment = assignment(source_confidence)
    target_assignment = assignment(target_confidence)
    source_mass = source_assignment.sum(dim=0)
    target_mass = target_assignment.sum(dim=0)
    source_mean = (source_assignment.T @ source_p_correct) / (
        source_mass + 1e-5
    )
    target_mean = (target_assignment.T @ target_p_correct) / (
        target_mass + 1e-5
    )
    return float(
        torch.sum(
            target_mass
            / target_mass.sum()
            * torch.abs(source_mean - target_mean)
        )
    )


@torch.no_grad()
def temperature_search(
    source_logits: torch.Tensor,
    source_labels: torch.Tensor,
    target_logits: torch.Tensor,
    source_head_logits: torch.Tensor,
    target_head_logits: torch.Tensor,
) -> tuple[int, list[dict[str, float]]]:
    source_p_correct = torch.softmax(source_head_logits, dim=1)[:, 1]
    target_p_correct = torch.softmax(target_head_logits, dim=1)[:, 1]
    cumulative_ece = 0.0
    cumulative_ecl = 0.0
    rows = []
    for temperature in range(1, 51):
        source_probability = torch.softmax(
            source_logits / temperature, dim=1
        )
        target_probability = torch.softmax(
            target_logits / temperature, dim=1
        )
        source_prediction = source_probability.argmax(dim=1)
        source_confidence = source_probability.max(dim=1).values
        target_confidence = target_probability.max(dim=1).values
        source_ece = hard_source_ece(
            source_confidence, source_prediction, source_labels
        )
        ecl_value = appendix_f_top_ecl(
            source_confidence,
            target_confidence,
            source_p_correct,
            target_p_correct,
        )
        cumulative_ece += source_ece
        cumulative_ecl += ecl_value
        if source_ece <= 0 or cumulative_ecl <= 0:
            objective = math.inf
        else:
            adaptive_lambda = cumulative_ece / cumulative_ecl
            objective = (
                source_ece
                + adaptive_lambda / (source_ece * source_ece) * ecl_value
            )
        rows.append(
            {
                "temperature": temperature,
                "source_hard_ece": source_ece,
                "appendix_f_top_ecl": ecl_value,
                "cumulative_adaptive_lambda": (
                    cumulative_ece / cumulative_ecl
                    if cumulative_ecl > 0
                    else None
                ),
                "predecessor_objective": objective,
            }
        )
    finite = [row for row in rows if math.isfinite(row["predecessor_objective"])]
    if not finite:
        raise RuntimeError("no finite predecessor temperature objective")
    best = min(finite, key=lambda row: row["predecessor_objective"])
    return int(best["temperature"]), rows


def array_digest(*values: torch.Tensor) -> str:
    digest = __import__("hashlib").sha256()
    for value in values:
        array = np.ascontiguousarray(value.cpu().numpy())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def run(cache_root: Path, predictions_path: Path, output_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    seed_everything(SEED)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    (
        source_images,
        source_labels,
        target_images,
        target_labels_tensor,
        provenance,
    ) = load_full_domains(cache_root)
    model = LeNet5()
    training_curve = train_joint_predecessor(
        model,
        source_images,
        source_labels,
        torch.Generator().manual_seed(SEED + 1),
    )
    source_features = extract_features(model, source_images)
    target_features = extract_features(model, target_images)
    del source_images, target_images
    source_logits = logits_from_features(model, source_features)
    target_logits = logits_from_features(model, target_features)
    source_head = logits_from_features(model, source_features, head=True)
    target_head = logits_from_features(model, target_features, head=True)
    best_temperature, temperature_curve = temperature_search(
        source_logits,
        source_labels,
        target_logits,
        source_head,
        target_head,
    )
    baseline_probability = torch.softmax(target_logits, dim=1)
    calibrated_probability = torch.softmax(
        target_logits / best_temperature, dim=1
    )
    baseline_prediction = baseline_probability.argmax(dim=1).numpy()
    calibrated_prediction = calibrated_probability.argmax(dim=1).numpy()
    baseline_confidence = baseline_probability.max(dim=1).values.numpy()
    calibrated_confidence = calibrated_probability.max(dim=1).values.numpy()
    target_labels = target_labels_tensor.numpy()
    if not np.array_equal(baseline_prediction, calibrated_prediction):
        raise RuntimeError("positive temperature changed argmax predictions")
    baseline_ece = ece(
        baseline_confidence, baseline_prediction, target_labels
    )
    calibrated_ece = ece(
        calibrated_confidence, calibrated_prediction, target_labels
    )
    accuracy = float((baseline_prediction == target_labels).mean())
    prediction_sha = write_predictions(
        predictions_path,
        target_labels,
        baseline_prediction,
        baseline_confidence,
        calibrated_prediction,
        calibrated_confidence,
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
            calibrated_ece - paper["ecl_ece_fraction"]
        )
        <= 2 * paper["ecl_std_fraction"],
        "ecl_reduces_ece": calibrated_ece < baseline_ece,
        "positive_temperature_accuracy_delta_is_exactly_zero": True,
    }
    result = {
        "status": "SUBSTANTIVE_SINGLE_SEED",
        "claim": "LeNet-5 target-SVHN ECL ECE 21.5% versus uncalibrated 61.9%, ten runs.",
        "route": "literal public-predecessor joint-head and post-hoc temperature semantics",
        "seed": SEED,
        "paper_values": paper,
        "observed": {
            "best_temperature": best_temperature,
            "baseline_ece_fraction": baseline_ece,
            "ecl_ece_fraction": calibrated_ece,
            "absolute_ece_reduction": baseline_ece - calibrated_ece,
            "relative_ece_reduction": (
                (baseline_ece - calibrated_ece) / baseline_ece
                if baseline_ece > 0
                else None
            ),
            "baseline_accuracy": accuracy,
            "ecl_accuracy": accuracy,
            "accuracy_delta": 0.0,
        },
        "compatibility_checks": compatibility,
        "dataset_provenance": provenance,
        "protocol": {
            "architecture": "exact predecessor LeNet-5 topology",
            "source": "MNIST train+test plus USPS train+test",
            "target": "SVHN train+test",
            "batch_size": 100,
            "epochs": 100,
            "primary_optimizer": "Adam lr=0.001",
            "head_optimizer": "Adam lr=0.01",
            "head_loss": "predecessor batch-weighted correctness cross-entropy; no Soft-ECE",
            "temperature_grid": list(range(1, 51)),
            "temperature_objective": "literal predecessor cumulative adaptive weighting with Appendix-F soft ECL replacing missing ECLoss_hd",
            "target_labels_used_for_training_or_selection": False,
        },
        "training_curve": training_curve,
        "temperature_curve": temperature_curve,
        "digests": {
            "model": state_digest(model),
            "logits_and_heads": array_digest(
                source_logits, target_logits, source_head, target_head
            ),
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
            "The predecessor imports an absent ECLoss_hd; this route explicitly replaces it with Appendix F top-label soft ECL.",
            "The predecessor's cumulative lambda makes temperature selection path-dependent on evaluating T in ascending order.",
            "Positive temperature scaling cannot change argmax accuracy, conflicting with the paper's nonzero DeltaACC but not by itself falsifying the ECE cell.",
            "Only LeNet-5 is evaluated in this route.",
        ],
    }
    if not (
        provenance["all_hashes_match"]
        and all(
            math.isfinite(value)
            for value in (
                baseline_ece,
                calibrated_ece,
                accuracy,
            )
        )
        and prediction_sha
    ):
        raise RuntimeError("Claim 5 route 2 fail-closed invariant failed")
    atomic_json(output_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "paper_values": paper,
                "observed": result["observed"],
                "compatibility_checks": compatibility,
                "counts": provenance["counts"],
                "all_hashes_match": provenance["all_hashes_match"],
                "digests": result["digests"],
                "temperature_curve": temperature_curve,
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
