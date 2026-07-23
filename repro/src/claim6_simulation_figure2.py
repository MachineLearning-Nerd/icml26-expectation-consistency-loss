#!/usr/bin/env python3
"""Faithful multi-seed reproduction route for the Figure 2 simulation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "upstream"))
from losses import ECLossMiniBatch  # noqa: E402
from networks import SimpleNet  # noqa: E402
from utils import labeling_function  # noqa: E402


SEEDS = (1907, 2903, 3907, 4903, 5903)
PARADIGMS = ("TopLabel", "Classwise", "Canonical")
NUM_SAMPLES = 400
NUM_BINS = 15
EPOCHS = 100
BATCH_SIZE = 200
LEARNING_RATE = 0.001
CLASSIFICATION_WEIGHT = 0.5


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def stable_soft_ece(logits: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
    confidence = torch.softmax(logits, dim=1).max(dim=1).values
    anchors = (torch.arange(NUM_BINS, dtype=confidence.dtype) * 2 + 1) / (2 * NUM_BINS)
    temperature = -1.0 / (math.log(0.9) * NUM_BINS * NUM_BINS)
    weights = torch.softmax(-((confidence[:, None] - anchors[None, :]) ** 2) / temperature, dim=1)
    masses = weights.sum(dim=0).clamp_min(torch.finfo(confidence.dtype).eps)
    bin_conf = (weights * confidence[:, None]).sum(dim=0) / masses
    bin_acc = (weights * correct.float()[:, None]).sum(dim=0) / masses
    squared = (masses / masses.sum() * (bin_conf - bin_acc).square()).sum()
    return torch.sqrt(squared + 1e-12)


def loader(x: torch.Tensor, y: torch.Tensor, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )


def train_classifier(model: nn.Module, x: torch.Tensor, y: torch.Tensor, seed: int, soft: bool) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        for batch_x, batch_y in loader(x, y, seed + epoch):
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = nn.functional.cross_entropy(logits, batch_y)
            if soft:
                loss = loss + stable_soft_ece(logits, logits.argmax(1).eq(batch_y))
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite classifier objective")
            loss.backward()
            optimizer.step()


def train_head(model: nn.Module, x: torch.Tensor, y: torch.Tensor, paradigm: str, seed: int) -> None:
    optimizer = torch.optim.Adam(model.classifier2.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        for batch_x, batch_y in loader(x, y, seed + 1000 + epoch):
            optimizer.zero_grad()
            output = model.forward_classifier2(batch_x)
            if paradigm == "TopLabel":
                with torch.no_grad():
                    correct = model(batch_x).argmax(1).eq(batch_y)
                loss = nn.functional.cross_entropy(output, correct.long())
                loss = loss + stable_soft_ece(output, correct)
            else:
                loss = nn.functional.cross_entropy(output, batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite auxiliary-head objective")
            loss.backward()
            optimizer.step()


def train_ecl(
    model: nn.Module,
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    target_x: torch.Tensor,
    paradigm: str,
    seed: int,
) -> None:
    optimizer = torch.optim.Adam(model.fc2.parameters(), lr=LEARNING_RATE)
    objective = ECLossMiniBatch(paradigm, num_bins=NUM_BINS, num_classes=3)
    dummy_target_y = torch.zeros(len(target_x), dtype=torch.long)
    for epoch in range(EPOCHS):
        source_loader = loader(source_x, source_y, seed + 2000 + epoch)
        target_loader = loader(target_x, dummy_target_y, seed + 3000 + epoch)
        for (batch_sx, batch_sy), (batch_tx, _) in zip(source_loader, target_loader):
            optimizer.zero_grad()
            source_logits = model(batch_sx)
            target_logits = model(batch_tx)
            accuracy_loss = nn.functional.cross_entropy(source_logits, batch_sy)
            ecl_loss = objective(batch_sx, batch_tx, source_logits, target_logits, model)
            loss = CLASSIFICATION_WEIGHT * accuracy_loss + ecl_loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite ECL objective")
            loss.backward()
            optimizer.step()


def evaluate(model: nn.Module, x: torch.Tensor) -> list[list[float]]:
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1)
    return probabilities.double().cpu().numpy().tolist()


def generate(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    rng = np.random.default_rng(seed)
    covariance = np.diag([5.0, 5.0])
    source = rng.multivariate_normal([0.0, 0.0], covariance, NUM_SAMPLES)
    target = rng.multivariate_normal([2.0, 2.0], covariance, NUM_SAMPLES)
    source_y = labeling_function(source)
    target_y = labeling_function(target)
    metadata = {
        "source_empirical_mean": source.mean(0).tolist(),
        "target_empirical_mean": target.mean(0).tolist(),
        "source_label_counts": np.bincount(source_y, minlength=3).tolist(),
        "target_label_counts": np.bincount(target_y, minlength=3).tolist(),
    }
    return (
        torch.tensor(source, dtype=torch.float32),
        torch.tensor(source_y, dtype=torch.long),
        torch.tensor(target, dtype=torch.float32),
        torch.tensor(target_y, dtype=torch.long),
        metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    records: list[dict[str, object]] = []
    data_audit: list[dict[str, object]] = []

    for seed in SEEDS:
        source_x, source_y, target_x, target_y, metadata = generate(seed)
        data_audit.append({"seed": seed, **metadata})
        for paradigm_index, paradigm in enumerate(PARADIGMS):
            model_seed = seed + 10_000 * paradigm_index
            seed_all(model_seed)
            baseline = SimpleNet(dim=100, num_classes=3, calibration_paradigm=paradigm)
            train_classifier(baseline, source_x, source_y, model_seed, soft=False)
            initial_state = deepcopy(baseline.state_dict())

            seed_all(model_seed)
            soft_model = SimpleNet(dim=100, num_classes=3, calibration_paradigm=paradigm)
            train_classifier(soft_model, source_x, source_y, model_seed, soft=True)

            ecl_model = SimpleNet(dim=100, num_classes=3, calibration_paradigm=paradigm)
            ecl_model.load_state_dict(initial_state)
            train_head(ecl_model, source_x, source_y, paradigm, model_seed)
            train_ecl(ecl_model, source_x, source_y, target_x, paradigm, model_seed)

            records.append(
                {
                    "seed": seed,
                    "paradigm": paradigm,
                    "target_labels": target_y.tolist(),
                    "probabilities": {
                        "uncalibrated": evaluate(baseline, target_x),
                        "soft_ece": evaluate(soft_model, target_x),
                        "ecl": evaluate(ecl_model, target_x),
                    },
                }
            )
            print(f"CLAIM6_SIMULATION seed={seed} paradigm={paradigm} status=complete")

    payload = {
        "schema_version": 1,
        "paper_contract": {
            "source_distribution": "N([0,0], diag([5,5]))",
            "target_distribution": "N([2,2], diag([5,5]))",
            "samples_per_domain": NUM_SAMPLES,
            "network": "released SimpleNet, 2->100 ReLU->3 plus auxiliary head",
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "epochs_per_stage": EPOCHS,
            "bins": NUM_BINS,
            "paradigms": list(PARADIGMS),
        },
        "implementation_choices": {
            "seeds": list(SEEDS),
            "batch_size": BATCH_SIZE,
            "classification_weight": CLASSIFICATION_WEIGHT,
            "toplabel_auxiliary_head": "cross entropy plus numerically stabilized released Soft-ECE",
            "other_auxiliary_heads": "source-label cross entropy, matching released notebook",
            "ecl": "released ECLossMiniBatch, including its stateful proximal caches",
            "target_labels_used_for_training": False,
        },
        "data_audit": data_audit,
        "records": records,
        "runtime_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CLAIM6_SIMULATION_RESULT records={len(records)} status=PASS")


if __name__ == "__main__":
    main()
