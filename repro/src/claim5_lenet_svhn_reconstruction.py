#!/usr/bin/env python3
"""Claim 5 route 1: full-domain LeNet-5 target-SVHN reconstruction.

The route follows the paper's Algorithm 2 interpretation because the current
official release has no digit pipeline and the public predecessor imports a
nonexistent ECL implementation.  It uses every MNIST and USPS train/test image
as source and every SVHN train/test image as unlabeled target during ECL.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import md5, sha256
import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST, SVHN, USPS

from claim3_real_mnist_sample_complexity import EXPECTED_MNIST


BATCH_SIZE = 100
BASELINE_EPOCHS = 100
HEAD_EPOCHS = 100
ECL_EPOCHS = 100
LEARNING_RATE = 0.001
ECL_LAMBDA_CE = 0.5
SEED = 260521552
SVHN_EXPECTED = {
    "train_32x32.mat": "e26dedcc434d2e4c54c9b2d4a06d8373",
    "test_32x32.mat": "eb5a983be6a315427106f1b164d9cef3",
}
USPS_EXPECTED = {
    "usps.bz2": "ec16c51db3855ca6c91edd34d0e9b197",
    "usps.t.bz2": "8ea070ee2aca1ac39742fdd1ef5ed118",
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def file_hashes(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "bytes": len(raw),
        "md5": md5(raw).hexdigest(),
        "sha256": sha256(raw).hexdigest(),
    }


class LeNet5(nn.Module):
    """Exact LeNet topology from predecessor commit 944d492."""

    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2),
        )

    def features(self, images: torch.Tensor) -> torch.Tensor:
        return self.feature(images).reshape(len(images), -1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _resize_float(values: torch.Tensor, size: int = 28, chunk: int = 4096) -> torch.Tensor:
    output = []
    for start in range(0, len(values), chunk):
        current = values[start : start + chunk].to(dtype=torch.float32) / 255.0
        if current.shape[-2:] != (size, size):
            current = F.interpolate(
                current,
                size=(size, size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        output.append(current)
    return torch.cat(output)


def load_full_domains(cache_root: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    mnist_root = cache_root / "MNIST"
    usps_root = cache_root / "USPS"
    svhn_root = cache_root / "SVHN"
    mnist_train = MNIST(root=str(mnist_root), train=True, download=True)
    mnist_test = MNIST(root=str(mnist_root), train=False, download=True)
    usps_train = USPS(root=str(usps_root), train=True, download=True)
    usps_test = USPS(root=str(usps_root), train=False, download=True)
    svhn_train = SVHN(root=str(svhn_root), split="train", download=True)
    svhn_test = SVHN(root=str(svhn_root), split="test", download=True)

    mnist_images = torch.cat([mnist_train.data, mnist_test.data])[:, None]
    mnist_images = mnist_images.repeat(1, 3, 1, 1).to(torch.float32) / 255.0
    mnist_labels = torch.cat(
        [
            torch.as_tensor(mnist_train.targets, dtype=torch.long),
            torch.as_tensor(mnist_test.targets, dtype=torch.long),
        ]
    )
    usps_images = torch.from_numpy(
        np.concatenate([usps_train.data, usps_test.data])
    )[:, None].repeat(1, 3, 1, 1)
    usps_images = _resize_float(usps_images)
    usps_labels = torch.as_tensor(
        list(usps_train.targets) + list(usps_test.targets), dtype=torch.long
    )
    source_images = torch.cat([mnist_images, usps_images])
    source_labels = torch.cat([mnist_labels, usps_labels])

    svhn_images = torch.from_numpy(
        np.concatenate([svhn_train.data, svhn_test.data])
    )
    target_images = _resize_float(svhn_images)
    target_labels = torch.from_numpy(
        np.concatenate([svhn_train.labels, svhn_test.labels])
    ).to(torch.long)

    provenance = {"files": {}, "counts": {}}
    mnist_raw = mnist_root / "MNIST" / "raw"
    for name, expected in EXPECTED_MNIST.items():
        observed = file_hashes(mnist_raw / name)
        observed["expected_sha256"] = expected["sha256"]
        observed["matches"] = observed["sha256"] == expected["sha256"]
        provenance["files"][f"MNIST/{name}"] = observed
    for name, expected_md5 in USPS_EXPECTED.items():
        observed = file_hashes(usps_root / name)
        observed["expected_md5"] = expected_md5
        observed["matches"] = observed["md5"] == expected_md5
        provenance["files"][f"USPS/{name}"] = observed
    for name, expected_md5 in SVHN_EXPECTED.items():
        observed = file_hashes(svhn_root / name)
        observed["expected_md5"] = expected_md5
        observed["matches"] = observed["md5"] == expected_md5
        provenance["files"][f"SVHN/{name}"] = observed
    provenance["counts"] = {
        "mnist_train": len(mnist_train),
        "mnist_test": len(mnist_test),
        "usps_train": len(usps_train),
        "usps_test": len(usps_test),
        "source_total": len(source_images),
        "svhn_train": len(svhn_train),
        "svhn_test": len(svhn_test),
        "target_total": len(target_images),
    }
    provenance["all_hashes_match"] = all(
        row["matches"] for row in provenance["files"].values()
    )
    if not provenance["all_hashes_match"]:
        raise RuntimeError("digit dataset integrity audit failed")
    if source_images.shape != (79298, 3, 28, 28) or target_images.shape != (
        99289,
        3,
        28,
        28,
    ):
        raise RuntimeError("full-domain digit sample counts differ from protocol")
    return source_images, source_labels, target_images, target_labels, provenance


def batches(count: int, generator: torch.Generator):
    order = torch.randperm(count, generator=generator)
    for start in range(0, count, BATCH_SIZE):
        yield order[start : start + BATCH_SIZE]


def train_baseline(
    model: LeNet5, images: torch.Tensor, labels: torch.Tensor, generator: torch.Generator
) -> list[dict[str, float]]:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rows = []
    for epoch in range(BASELINE_EPOCHS):
        loss_sum = 0.0
        correct = 0
        seen = 0
        for index in batches(len(images), generator):
            x = images[index]
            y = labels[index]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(index)
            correct += int((logits.argmax(dim=1) == y).sum())
            seen += len(index)
        row = {
            "epoch": epoch + 1,
            "cross_entropy": loss_sum / seen,
            "source_accuracy": correct / seen,
        }
        rows.append(row)
        if (epoch + 1) % 10 == 0:
            print(f"CLAIM5_BASELINE epoch={epoch + 1} ce={row['cross_entropy']:.8f} acc={row['source_accuracy']:.8f}")
    return rows


@torch.no_grad()
def extract_features(
    model: LeNet5, images: torch.Tensor, batch_size: int = 1000
) -> torch.Tensor:
    model.eval()
    return torch.cat(
        [model.features(images[start : start + batch_size]) for start in range(0, len(images), batch_size)]
    )


def soft_ece(logits: torch.Tensor, correct: torch.Tensor, bins: int = 15) -> torch.Tensor:
    confidence = torch.softmax(logits, dim=1).max(dim=1).values
    anchors = (2 * torch.arange(bins, dtype=confidence.dtype) + 1) / (2 * bins)
    temperature = -1.0 / (math.log(0.9) * bins * bins)
    assignment = torch.softmax(
        -((confidence[:, None] - anchors[None, :]) ** 2) / temperature, dim=1
    )
    mass = assignment.sum(dim=0)
    mean_confidence = (assignment.T @ confidence) / torch.clamp(mass, min=1e-5)
    mean_accuracy = (assignment.T @ correct.to(confidence.dtype)) / torch.clamp(
        mass, min=1e-5
    )
    return torch.sqrt(
        torch.sum(mass / torch.clamp(mass.sum(), min=1e-5) * (mean_confidence - mean_accuracy) ** 2)
    )


def train_correctness_head(
    model: LeNet5,
    features: torch.Tensor,
    labels: torch.Tensor,
    generator: torch.Generator,
) -> list[dict[str, float]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.classifier2.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model.classifier2.parameters(), lr=LEARNING_RATE)
    rows = []
    model.train()
    for epoch in range(HEAD_EPOCHS):
        loss_sum = 0.0
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
            ce = F.cross_entropy(logits, correctness.long())
            calibration = soft_ece(logits, correctness)
            loss = ce + calibration
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(index)
            head_correct += int((logits.argmax(dim=1) == correctness).sum())
            seen += len(index)
        row = {
            "epoch": epoch + 1,
            "head_loss": loss_sum / seen,
            "head_correctness_accuracy": head_correct / seen,
        }
        rows.append(row)
        if (epoch + 1) % 10 == 0:
            print(f"CLAIM5_HEAD epoch={epoch + 1} loss={row['head_loss']:.8f} acc={row['head_correctness_accuracy']:.8f}")
    return rows


class TopLabelMiniBatchECL(nn.Module):
    """Independent transcription of current official Algorithm 2 code."""

    def __init__(self, bins: int = 15, ema_alpha: float = 0.9, prox_steps: int = 5):
        super().__init__()
        self.bins = bins
        self.ema_alpha = ema_alpha
        self.prox_steps = prox_steps
        self.register_buffer("source_cache", torch.zeros(bins))
        self.register_buffer("target_cache", torch.zeros(bins))

    def forward(
        self,
        source_logits: torch.Tensor,
        target_logits: torch.Tensor,
        source_head_logits: torch.Tensor,
        target_head_logits: torch.Tensor,
    ) -> torch.Tensor:
        source_confidence = torch.softmax(source_logits, dim=1).max(dim=1).values
        target_confidence = torch.softmax(target_logits, dim=1).max(dim=1).values
        source_p = torch.softmax(source_head_logits, dim=1)[:, 1]
        target_p = torch.softmax(target_head_logits, dim=1)[:, 1]
        anchors = (
            2 * torch.arange(self.bins, dtype=source_confidence.dtype) + 1
        ) / (2 * self.bins)
        temperature = -1.0 / (math.log(0.9) * self.bins * self.bins)
        source_assignment = torch.softmax(
            -((source_confidence[:, None] - anchors[None, :]) ** 2) / temperature,
            dim=1,
        )
        target_assignment = torch.softmax(
            -((target_confidence[:, None] - anchors[None, :]) ** 2) / temperature,
            dim=1,
        )
        source_mass = source_assignment.sum(dim=0)
        target_mass = target_assignment.sum(dim=0)
        source_sum = (source_assignment * source_p[:, None]).sum(dim=0)
        target_sum = (target_assignment * target_p[:, None]).sum(dim=0)
        source_new = self.source_cache.clone()
        target_new = self.target_cache.clone()
        loss = torch.zeros(())
        tiny = torch.finfo(source_logits.dtype).eps
        for bin_index in range(self.bins):
            ns = source_mass[bin_index]
            nt = target_mass[bin_index]
            if float(ns.detach()) < tiny or float(nt.detach()) < tiny:
                continue
            source_u = self.source_cache[bin_index]
            target_u = self.target_cache[bin_index]
            weight = nt / (target_mass.sum() + tiny)
            for _ in range(self.prox_steps):
                tau_source = weight / (2 * ns)
                value_source = source_sum[bin_index] / ns - target_u
                source_u = target_u + value_source.sign() * torch.clamp(
                    value_source.abs() - tau_source, min=0.0
                )
                tau_target = weight / (2 * nt)
                value_target = target_sum[bin_index] / nt - source_u
                target_u = source_u + value_target.sign() * torch.clamp(
                    value_target.abs() - tau_target, min=0.0
                )
            source_detached = source_u.detach()
            target_detached = target_u.detach()
            source_new[bin_index] = (
                (1 - self.ema_alpha) * self.source_cache[bin_index]
                + self.ema_alpha * source_detached
            )
            target_new[bin_index] = (
                (1 - self.ema_alpha) * self.target_cache[bin_index]
                + self.ema_alpha * target_detached
            )
            loss = loss + torch.sum(
                source_assignment[:, bin_index]
                * (source_detached - source_p).square()
            )
            loss = loss + torch.sum(
                target_assignment[:, bin_index]
                * (target_detached - target_p).square()
            )
        with torch.no_grad():
            self.source_cache.copy_(source_new)
            self.target_cache.copy_(target_new)
        return loss


def train_ecl(
    model: LeNet5,
    source_features: torch.Tensor,
    source_labels: torch.Tensor,
    target_features: torch.Tensor,
    source_generator: torch.Generator,
    target_generator: torch.Generator,
) -> list[dict[str, float]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    ecl = TopLabelMiniBatchECL()
    rows = []
    model.train()
    for epoch in range(ECL_EPOCHS):
        source_order = torch.randperm(len(source_features), generator=source_generator)
        target_order = torch.randperm(len(target_features), generator=target_generator)
        ce_sum = 0.0
        ecl_sum = 0.0
        correct = 0
        seen = 0
        for start in range(0, len(source_order), BATCH_SIZE):
            source_index = source_order[start : start + BATCH_SIZE]
            target_index = target_order[start : start + len(source_index)]
            source_feature = source_features[source_index]
            target_feature = target_features[target_index]
            labels = source_labels[source_index]
            optimizer.zero_grad(set_to_none=True)
            source_logits = model.classifier(source_feature)
            target_logits = model.classifier(target_feature)
            with torch.no_grad():
                source_head = model.classifier2(source_feature)
                target_head = model.classifier2(target_feature)
            ce = F.cross_entropy(source_logits, labels)
            ecl_loss = ecl(source_logits, target_logits, source_head, target_head)
            loss = ECL_LAMBDA_CE * ce + ecl_loss
            loss.backward()
            optimizer.step()
            ce_sum += float(ce.detach()) * len(source_index)
            ecl_sum += float(ecl_loss.detach()) * len(source_index)
            correct += int((source_logits.argmax(dim=1) == labels).sum())
            seen += len(source_index)
        row = {
            "epoch": epoch + 1,
            "source_cross_entropy": ce_sum / seen,
            "minibatch_ecl": ecl_sum / seen,
            "source_accuracy": correct / seen,
        }
        rows.append(row)
        if (epoch + 1) % 10 == 0:
            print(
                f"CLAIM5_ECL epoch={epoch + 1} ce={row['source_cross_entropy']:.8f} "
                f"ecl={row['minibatch_ecl']:.8f} acc={row['source_accuracy']:.8f}"
            )
    return rows


@torch.no_grad()
def predict_from_features(model: LeNet5, features: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits = torch.cat(
        [
            model.classifier(features[start : start + 2000])
            for start in range(0, len(features), 2000)
        ]
    )
    probabilities = torch.softmax(logits, dim=1)
    return (
        probabilities.argmax(dim=1).cpu().numpy(),
        probabilities.max(dim=1).values.cpu().numpy(),
    )


def ece(confidence: np.ndarray, prediction: np.ndarray, labels: np.ndarray) -> float:
    value = 0.0
    for lower, upper in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if np.any(selected):
            value += float(selected.mean()) * abs(
                float(confidence[selected].mean())
                - float((prediction[selected] == labels[selected]).mean())
            )
    return value


def state_digest(model: nn.Module) -> str:
    digest = sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def write_predictions(
    path: Path,
    labels: np.ndarray,
    baseline_prediction: np.ndarray,
    baseline_confidence: np.ndarray,
    ecl_prediction: np.ndarray,
    ecl_confidence: np.ndarray,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "label",
                "baseline_prediction",
                "baseline_confidence",
                "ecl_prediction",
                "ecl_confidence",
            ]
        )
        for row in zip(
            range(len(labels)),
            labels,
            baseline_prediction,
            baseline_confidence,
            ecl_prediction,
            ecl_confidence,
        ):
            writer.writerow(
                [
                    int(row[0]),
                    int(row[1]),
                    int(row[2]),
                    format(float(row[3]), ".17g"),
                    int(row[4]),
                    format(float(row[5]), ".17g"),
                ]
            )
    temporary.replace(path)
    return sha256(path.read_bytes()).hexdigest()


def run(cache_root: Path, predictions_path: Path, output_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    seed_everything(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    (
        source_images,
        source_labels,
        target_images,
        target_labels_tensor,
        provenance,
    ) = load_full_domains(cache_root)
    model = LeNet5()
    generator = torch.Generator().manual_seed(SEED + 1)
    baseline_curve = train_baseline(model, source_images, source_labels, generator)
    source_features = extract_features(model, source_images)
    target_features = extract_features(model, target_images)
    del source_images, target_images
    target_labels = target_labels_tensor.numpy()
    baseline_prediction, baseline_confidence = predict_from_features(
        model, target_features
    )
    baseline_model_digest = state_digest(model)
    head_curve = train_correctness_head(
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
    ecl_prediction, ecl_confidence = predict_from_features(model, target_features)
    ecl_model_digest = state_digest(model)
    baseline_ece = ece(
        baseline_confidence, baseline_prediction, target_labels
    )
    ecl_ece = ece(ecl_confidence, ecl_prediction, target_labels)
    baseline_accuracy = float((baseline_prediction == target_labels).mean())
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
    result = {
        "status": "SUBSTANTIVE_SINGLE_SEED",
        "claim": "LeNet-5 target-SVHN ECL ECE 21.5% versus uncalibrated 61.9%, reported over ten runs.",
        "route": "paper Algorithm 2 in-training interpretation",
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
        "dataset_provenance": provenance,
        "protocol": {
            "architecture": "predecessor LeNet-5 topology",
            "source": "MNIST train+test concatenated with USPS train+test",
            "target": "SVHN train+test",
            "image_format": "3-channel float RGB resized to 28x28",
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "baseline_epochs": BASELINE_EPOCHS,
            "correctness_head_epochs": HEAD_EPOCHS,
            "ecl_finetune_epochs": ECL_EPOCHS,
            "ecl_lambda_ce": ECL_LAMBDA_CE,
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
            "ecl_model": ecl_model_digest,
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
            "The current official source has no digit pipeline; this route reconstructs Algorithm 2 and records that interpretation.",
            "The predecessor's missing ECLoss_hd import prevents literal execution and its post-hoc path conflicts with the paper's nonzero accuracy deltas.",
            "Only LeNet-5 is tested in this route; ResNet20 and DenseNet40 require separate routes.",
            "The paper does not specify the digit ECL cross-entropy coefficient; 0.5 is taken from the official top-label demonstration.",
        ],
    }
    if not (
        provenance["all_hashes_match"]
        and len(target_labels) == 99289
        and np.isfinite(baseline_ece)
        and np.isfinite(ecl_ece)
        and 0 <= baseline_ece <= 1
        and 0 <= ecl_ece <= 1
        and prediction_sha
    ):
        raise RuntimeError("Claim 5 route 1 fail-closed invariant failed")
    atomic_json(output_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "seed": SEED,
                "paper_values": paper,
                "observed": result["observed"],
                "compatibility_checks": compatibility,
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
