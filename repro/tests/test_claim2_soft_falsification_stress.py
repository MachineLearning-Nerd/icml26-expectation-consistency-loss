from __future__ import annotations

import numpy as np
import pytest

from claim2_soft_falsification_stress import soft_radius


def test_soft_radius_rejects_zero_source_denominator() -> None:
    with pytest.raises(ValueError):
        soft_radius(
            np.asarray([0.5, 0.5]),
            np.asarray([5.0, 5.0]),
            np.asarray([5.0, 0.0]),
            classes=3,
        )


def test_soft_radius_is_positive() -> None:
    result = soft_radius(
        np.asarray([0.25, 0.75]),
        np.asarray([2.5, 7.5]),
        np.asarray([4.0, 6.0]),
        classes=3,
    )
    assert np.isfinite(result)
    assert result > 0
