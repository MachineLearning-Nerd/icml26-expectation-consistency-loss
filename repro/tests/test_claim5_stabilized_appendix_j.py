from __future__ import annotations

import torch
import pytest

from claim5_stabilized_appendix_j import (
    SOFT_ECE_EPSILON,
    boundary_diagnostic,
    stable_soft_ece,
)


def test_stabilized_soft_ece_has_finite_random_gradient():
    logits = torch.randn(
        31,
        2,
        generator=torch.Generator().manual_seed(31),
        requires_grad=True,
    )
    correctness = torch.arange(31) % 3 == 0
    loss = stable_soft_ece(logits, correctness)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_boundary_diagnostic_reproduces_literal_failure():
    result = boundary_diagnostic()
    assert result["literal_value"] == 0.0
    assert result["literal_gradient_all_finite"] is False
    assert result["stable_gradient_all_finite"] is True
    assert result["stable_value"] == pytest.approx(
        SOFT_ECE_EPSILON**0.5, rel=1e-7
    )
