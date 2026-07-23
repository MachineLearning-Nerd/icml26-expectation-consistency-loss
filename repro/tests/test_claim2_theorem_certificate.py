from __future__ import annotations

import numpy as np
import pytest

import claim2_theorem_certificate as certificate


def test_theorem_radius_rejects_undefined_source_bin() -> None:
    with pytest.raises(ValueError):
        certificate.theorem_radius(
            np.asarray([0.5, 0.5]),
            np.asarray([10, 10]),
            np.asarray([10, 0]),
            classes=2,
        )


def test_weighted_cauchy_and_constant_obligations() -> None:
    result = certificate.verify_algebra()
    assert result["all_obligations_pass"]
    assert result["max_left_over_right"] <= 1 + 1e-12
    assert result["declared_absolute_constant"] >= result["derived_absolute_constant"]


def test_radius_is_positive_and_finite() -> None:
    radius = certificate.theorem_radius(
        np.asarray([0.25, 0.75]),
        np.asarray([25, 75]),
        np.asarray([50, 50]),
        classes=10,
    )
    assert np.isfinite(radius)
    assert radius > 0


def test_negative_controls_are_fail_closed() -> None:
    controls = certificate.negative_controls()
    assert len(controls) == 4
    assert all(not control["accepted"] for control in controls)
