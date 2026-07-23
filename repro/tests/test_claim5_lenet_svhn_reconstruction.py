from __future__ import annotations

import numpy as np
import pytest
import torch

from claim5_lenet_independent_checker import hard_ece as independent_ece
from claim5_lenet_svhn_reconstruction import (
    LeNet5,
    TopLabelMiniBatchECL,
    ece,
    soft_ece,
)


def test_lenet_predecessor_shapes():
    model = LeNet5()
    images = torch.zeros(7, 3, 28, 28)
    features = model.features(images)
    assert features.shape == (7, 256)
    assert model.classifier(features).shape == (7, 10)
    assert model.classifier2(features).shape == (7, 2)


def test_primary_and_independent_hard_ece_match():
    confidence = np.array([0.2, 0.4, 0.8, 0.95])
    prediction = np.array([0, 1, 2, 3])
    labels = np.array([0, 0, 2, 0])
    primary = ece(confidence, prediction, labels)
    secondary = independent_ece(
        confidence.tolist(), prediction.tolist(), labels.tolist()
    )
    assert primary == pytest.approx(secondary, abs=1e-15)


def test_soft_ece_is_finite_and_differentiable():
    logits = torch.randn(17, 10, generator=torch.Generator().manual_seed(1), requires_grad=True)
    correct = torch.arange(17) % 2 == 0
    value = soft_ece(logits, correct)
    value.backward()
    assert torch.isfinite(value)
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_minibatch_ecl_updates_cache_and_propagates_score_gradient():
    generator = torch.Generator().manual_seed(2)
    source = torch.randn(19, 10, generator=generator, requires_grad=True)
    target = torch.randn(23, 10, generator=generator, requires_grad=True)
    source_head = torch.randn(19, 2, generator=generator)
    target_head = torch.randn(23, 2, generator=generator)
    loss = TopLabelMiniBatchECL()(source, target, source_head, target_head)
    loss.backward()
    assert torch.isfinite(loss)
    assert source.grad is not None and target.grad is not None
    assert torch.all(torch.isfinite(source.grad))
    assert torch.all(torch.isfinite(target.grad))
