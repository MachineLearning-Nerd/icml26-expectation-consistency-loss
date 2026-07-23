#!/usr/bin/env python3
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from claim4_real_mnist_independent_checker import losses as numpy_losses
from claim4_real_mnist_soft_bins import (
    scalar_anchors,
    torch_losses,
)


def _fixture():
    rng = np.random.default_rng(2605)
    logits = rng.normal(size=(31, 4))
    posterior = rng.dirichlet(np.ones(4), size=31)
    source = 31 * rng.dirichlet(np.ones(31))
    target = 31 * rng.dirichlet(np.ones(31))
    return logits, posterior, source, target


def test_scalar_anchors_match_appendix_formula():
    assert scalar_anchors(4).tolist() == pytest.approx([0.125, 0.375, 0.625, 0.875])
    with pytest.raises(ValueError):
        scalar_anchors(0)


def test_three_torch_losses_are_finite_and_differentiable():
    values = _fixture()
    for mode in ("top_label", "class_wise", "canonical"):
        outputs, parameter = torch_losses(*values, calibration_temperature=1.0)
        outputs[mode].backward()
        assert torch.isfinite(outputs[mode])
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad)


def test_independent_numpy_losses_match_torch_on_small_fixture():
    logits, posterior, source, target = _fixture()
    data = {
        "logits": logits.tolist(),
        "posterior": posterior.tolist(),
        "source_counts": source.tolist(),
        "target_counts": target.tolist(),
        "requested_bins": 15,
        "decay_factor": 0.9,
        "stabilizer": 1e-5,
    }
    independent = numpy_losses(data, 1.0)
    primary, _ = torch_losses(
        logits, posterior, source, target, calibration_temperature=1.0
    )
    for mode in independent:
        assert independent[mode] == pytest.approx(
            float(primary[mode].detach()), rel=2e-12, abs=2e-12
        )


def test_detached_scores_remove_temperature_gradient():
    values = _fixture()
    outputs, parameter = torch_losses(
        *values, calibration_temperature=1.0, detach_scores=True
    )
    assert outputs["canonical"].requires_grad is False
    assert parameter.grad is None
