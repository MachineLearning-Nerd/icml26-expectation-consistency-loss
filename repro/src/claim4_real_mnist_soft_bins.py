#!/usr/bin/env python3
"""Full-MNIST differentiable ECL compatibility certificate for Claim 4.

This route implements the canonical, class-wise, and top-label soft-binning
losses from Appendix F.  A temperature parameter changes the trained model's
scores, and PyTorch autograd gradients are compared with centered finite
differences.  The posterior head is trained on a disjoint half of MNIST.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torchvision.datasets import MNIST

from claim3_real_mnist_sample_complexity import (
    audit_mnist_cache,
    load_mnist,
    model_holdout_summary,
    train_softmax,
    validate_split_protocol,
    x_only_domain_probabilities,
)
from claim3_soft_sample_complexity import official_temperature, simplex_anchors


STABILIZER = 1e-5
REQUESTED_BINS = 15
DECAY_FACTOR = 0.9


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_mnist_cache(cache_root: Path) -> Path:
    """Download through torchvision when needed and return the raw IDX path."""
    cache_root = Path(cache_root)
    raw_root = cache_root / "MNIST" / "raw"
    if not all((raw_root / name).is_file() for name in (
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    )):
        MNIST(root=str(cache_root), train=True, download=True)
        MNIST(root=str(cache_root), train=False, download=True)
    return raw_root


def scalar_anchors(bins: int = REQUESTED_BINS) -> np.ndarray:
    if bins < 1:
        raise ValueError("bins must be positive")
    return (2.0 * np.arange(bins, dtype=np.float64) + 1.0) / (2.0 * bins)


def _torch_assignments(scores: torch.Tensor, anchors: torch.Tensor, temperature: float) -> torch.Tensor:
    if scores.ndim == 1:
        squared_distance = (scores[:, None] - anchors[None, :]) ** 2
    elif scores.ndim == 2:
        squared_distance = torch.sum(
            (scores[:, None, :] - anchors[None, :, :]) ** 2, dim=2
        )
    else:
        raise ValueError("scores must be a vector or matrix")
    return torch.softmax(-squared_distance / temperature, dim=1)


def _weighted_scalar_loss(
    scores: torch.Tensor,
    posterior: torch.Tensor,
    source_counts: torch.Tensor,
    target_counts: torch.Tensor,
    anchors: torch.Tensor,
    soft_temperature: float,
) -> torch.Tensor:
    source_assignment = _torch_assignments(scores, anchors, soft_temperature)
    target_assignment = source_assignment
    source_mass = torch.sum(source_counts[:, None] * source_assignment, dim=0)
    target_mass = torch.sum(target_counts[:, None] * target_assignment, dim=0)
    source_mean = torch.sum(
        source_counts[:, None] * source_assignment * posterior[:, None], dim=0
    ) / (source_mass + STABILIZER)
    target_mean = torch.sum(
        target_counts[:, None] * target_assignment * posterior[:, None], dim=0
    ) / (target_mass + STABILIZER)
    target_weight = target_mass / torch.sum(target_counts)
    return torch.sum(target_weight * torch.abs(source_mean - target_mean))


def torch_losses(
    logits: np.ndarray,
    posterior: np.ndarray,
    source_counts: np.ndarray,
    target_counts: np.ndarray,
    *,
    calibration_temperature: float,
    detach_scores: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    dtype = torch.float64
    logits_tensor = torch.as_tensor(logits, dtype=dtype)
    posterior_tensor = torch.as_tensor(posterior, dtype=dtype)
    source_tensor = torch.as_tensor(source_counts, dtype=dtype)
    target_tensor = torch.as_tensor(target_counts, dtype=dtype)
    parameter = torch.tensor(float(calibration_temperature), dtype=dtype, requires_grad=True)
    probabilities = torch.softmax(logits_tensor / parameter, dim=1)
    if detach_scores:
        probabilities = probabilities.detach()

    scalar = torch.as_tensor(scalar_anchors(), dtype=dtype)
    scalar_soft_temperature = official_temperature(REQUESTED_BINS, DECAY_FACTOR)
    prediction = torch.argmax(logits_tensor, dim=1)
    confidence = torch.max(probabilities, dim=1).values
    top_posterior = posterior_tensor[
        torch.arange(len(posterior_tensor)), prediction
    ]
    top = _weighted_scalar_loss(
        confidence,
        top_posterior,
        source_tensor,
        target_tensor,
        scalar,
        scalar_soft_temperature,
    )

    classwise = torch.zeros((), dtype=dtype)
    for class_index in range(probabilities.shape[1]):
        classwise = classwise + _weighted_scalar_loss(
            probabilities[:, class_index],
            posterior_tensor[:, class_index],
            source_tensor,
            target_tensor,
            scalar,
            scalar_soft_temperature,
        )

    canonical_anchors = torch.as_tensor(
        simplex_anchors(REQUESTED_BINS, classes=probabilities.shape[1]), dtype=dtype
    )
    canonical_soft_temperature = official_temperature(
        len(canonical_anchors), DECAY_FACTOR
    )
    assignment = _torch_assignments(
        probabilities, canonical_anchors, canonical_soft_temperature
    )
    source_mass = torch.sum(source_tensor[:, None] * assignment, dim=0)
    target_mass = torch.sum(target_tensor[:, None] * assignment, dim=0)
    source_mean = (
        (source_tensor[:, None] * assignment).T @ posterior_tensor
    ) / (source_mass[:, None] + STABILIZER)
    target_mean = (
        (target_tensor[:, None] * assignment).T @ posterior_tensor
    ) / (target_mass[:, None] + STABILIZER)
    canonical = torch.sum(
        target_mass / torch.sum(target_tensor)
        * torch.linalg.vector_norm(source_mean - target_mean, dim=1)
    )
    return {
        "top_label": top,
        "class_wise": classwise,
        "canonical": canonical,
    }, parameter


def _loss_and_gradient(
    logits: np.ndarray,
    posterior: np.ndarray,
    source_counts: np.ndarray,
    target_counts: np.ndarray,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for mode in ("top_label", "class_wise", "canonical"):
        losses, parameter = torch_losses(
            logits,
            posterior,
            source_counts,
            target_counts,
            calibration_temperature=1.0,
        )
        losses[mode].backward()
        gradient = float(parameter.grad)
        output[mode] = {
            "loss": float(losses[mode].detach()),
            "autograd_temperature_gradient": gradient,
        }
    return output


def _digest_arrays(*arrays: np.ndarray) -> str:
    digest = sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def run(cache_root: Path, inputs_path: Path, output_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    raw_root = prepare_mnist_cache(cache_root)
    audit = audit_mnist_cache(raw_root)
    if not audit["all_files_match"]:
        raise RuntimeError("official MNIST IDX hash audit failed")
    train_images, train_labels, test_images, test_labels = load_mnist(raw_root)
    split = validate_split_protocol(
        (0, 30000),
        (30000, 60000),
        evaluation_split="official_test_10000",
        train_count=len(train_images),
    )
    primary = train_softmax(
        train_images[:30000],
        train_labels[:30000],
        l2=0.001,
        max_iterations=100,
        ftol=1e-10,
    )
    posterior_head = train_softmax(
        train_images[30000:],
        train_labels[30000:],
        l2=0.001,
        max_iterations=100,
        ftol=1e-10,
    )
    probabilities = primary.predict_proba(test_images)
    posterior = posterior_head.predict_proba(test_images)
    logits = np.log(np.clip(probabilities, 1e-300, None))
    source_probability, target_probability, shift = x_only_domain_probabilities(
        test_images, strength=1.1, uniform_floor=0.05
    )
    # Appendix F's epsilon is expressed relative to sample counts.  Scaling
    # finite-pool probabilities by N preserves the domains and count semantics.
    source_counts = len(test_images) * source_probability
    target_counts = len(test_images) * target_probability
    metrics = _loss_and_gradient(logits, posterior, source_counts, target_counts)

    detached_losses, detached_parameter = torch_losses(
        logits,
        posterior,
        source_counts,
        target_counts,
        calibration_temperature=1.0,
        detach_scores=True,
    )
    detached_gradient_is_absent = True
    try:
        detached_losses["canonical"].backward()
        detached_gradient_is_absent = detached_parameter.grad is None
    except RuntimeError:
        detached_gradient_is_absent = True

    permutation = np.random.default_rng(260521552).permutation(len(posterior))
    shuffled_metrics = _loss_and_gradient(
        logits, posterior[permutation], source_counts, target_counts
    )
    shuffle_changes = {
        mode: abs(metrics[mode]["loss"] - shuffled_metrics[mode]["loss"])
        for mode in metrics
    }
    top_loss = metrics["top_label"]["loss"]
    semantic_separation = {
        mode: abs(metrics[mode]["loss"] - top_loss)
        for mode in ("class_wise", "canonical")
    }

    inputs = {
        "schema": "claim4-real-mnist-soft-inputs-v1",
        "logits": logits.tolist(),
        "posterior": posterior.tolist(),
        "source_counts": source_counts.tolist(),
        "target_counts": target_counts.tolist(),
        "requested_bins": REQUESTED_BINS,
        "decay_factor": DECAY_FACTOR,
        "stabilizer": STABILIZER,
        "calibration_temperature": 1.0,
        "finite_difference_step": 1e-4,
        "array_sha256": _digest_arrays(
            logits, posterior, source_counts, target_counts
        ),
    }
    _atomic_json(inputs_path, inputs)
    input_sha256 = sha256(inputs_path.read_bytes()).hexdigest()

    all_finite_nonzero = all(
        math.isfinite(value)
        and abs(value) > 1e-8
        for mode in metrics.values()
        for value in mode.values()
    )
    controls = {
        "official_idx_hashes_match": bool(audit["all_files_match"]),
        "training_splits_disjoint": bool(
            split["primary_and_posterior_training_disjoint"]
        ),
        "test_split_held_out": bool(split["all_training_disjoint_from_evaluation"]),
        "domain_selection_uses_labels": bool(shift["uses_labels"]),
        "detached_scores_have_no_temperature_gradient": detached_gradient_is_absent,
        "shuffled_posterior_changes_every_loss": all(
            change > 1e-5 for change in shuffle_changes.values()
        ),
        "classwise_and_canonical_not_collapsed_to_top_label": all(
            difference > 1e-5 for difference in semantic_separation.values()
        ),
        "all_losses_and_gradients_finite_nonzero": all_finite_nonzero,
    }
    if not (
        controls["official_idx_hashes_match"]
        and controls["training_splits_disjoint"]
        and controls["test_split_held_out"]
        and not controls["domain_selection_uses_labels"]
        and controls["detached_scores_have_no_temperature_gradient"]
        and controls["shuffled_posterior_changes_every_loss"]
        and controls["classwise_and_canonical_not_collapsed_to_top_label"]
        and controls["all_losses_and_gradients_finite_nonzero"]
    ):
        raise RuntimeError(f"fail-closed Claim 4 control failed: {controls}")

    result = {
        "status": "AWAITING_INDEPENDENT_CHECKER",
        "claim": "Appendix F supplies differentiable soft-binning implementations for canonical, class-wise, and top-label ECL.",
        "dataset": {
            "name": "MNIST",
            "training_samples": len(train_images),
            "held_out_test_samples": len(test_images),
            "audit": audit,
        },
        "split_protocol": split,
        "model_training": {
            "primary": {**primary.diagnostics, "sha256": primary.digest()},
            "posterior_head": {
                **posterior_head.diagnostics,
                "sha256": posterior_head.digest(),
            },
            "deterministic_initialization": "all-zero multinomial logistic parameters",
        },
        "holdout": {
            "primary": model_holdout_summary(probabilities, test_labels),
            "posterior_head": model_holdout_summary(posterior, test_labels),
        },
        "covariate_shift": {
            **shift,
            "source_and_target_effective_count": len(test_images),
            "shared_finite_pool_conditional": True,
        },
        "formulations": {
            "top_label": {
                "requested_bins": REQUESTED_BINS,
                "actual_bins": REQUESTED_BINS,
                "soft_temperature": official_temperature(
                    REQUESTED_BINS, DECAY_FACTOR
                ),
                **metrics["top_label"],
            },
            "class_wise": {
                "classes": 10,
                "requested_bins_per_class": REQUESTED_BINS,
                "actual_bins_per_class": REQUESTED_BINS,
                "soft_temperature": official_temperature(
                    REQUESTED_BINS, DECAY_FACTOR
                ),
                **metrics["class_wise"],
            },
            "canonical": {
                "classes": 10,
                "requested_bins": REQUESTED_BINS,
                "actual_simplex_anchors": len(
                    simplex_anchors(REQUESTED_BINS, classes=10)
                ),
                "soft_temperature": official_temperature(
                    len(simplex_anchors(REQUESTED_BINS, classes=10)),
                    DECAY_FACTOR,
                ),
                **metrics["canonical"],
            },
        },
        "negative_controls": {
            "detached_score_gradient_absent": detached_gradient_is_absent,
            "posterior_shuffle_absolute_loss_changes": shuffle_changes,
            "semantic_separation_from_top_label": semantic_separation,
            "all_controls_pass": True,
        },
        "inputs": {
            "path": str(inputs_path),
            "sha256": input_sha256,
            "array_sha256": inputs["array_sha256"],
        },
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "limitations": [
            "The classifiers are full-data multinomial logistic models rather than the paper's deep digit architectures.",
            "The posterior head estimates P(Y|X); ground-truth population posteriors are unavailable for natural images.",
            "This verifies execution and differentiation of the three Appendix F formulations on real images; it does not verify Table 2 benchmark magnitudes.",
            "The deterministic finite-pool shift preserves the empirical conditional by construction but does not identify an unobserved real-world population conditional.",
        ],
    }
    _atomic_json(output_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "formulations": result["formulations"],
                "holdout": result["holdout"],
                "negative_controls": result["negative_controls"],
                "inputs_sha256": input_sha256,
                "wall_seconds": result["runtime"]["wall_seconds"],
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
        default=Path.home() / ".cache" / "openresearch" / "datasets" / "ecl-mnist",
    )
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache_root, args.inputs, args.output)


if __name__ == "__main__":
    main()
