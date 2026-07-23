from __future__ import annotations

from math import sqrt

from claim2_soft_theorem_proof import (
    MASS_CONSTANT,
    SELF_NORMALIZED_CONSTANT,
    TOTAL_CONSTANT,
    deterministic_checks,
)


def test_soft_theorem_constant_budget_is_absolute() -> None:
    assert SELF_NORMALIZED_CONSTANT * sqrt(2) + MASS_CONSTANT < TOTAL_CONSTANT


def test_soft_count_deterministic_reductions() -> None:
    result = deterministic_checks()
    assert result["trials"] == 840
    assert result["max_target_count_identity_error"] < 1e-14
    assert result["max_source_weighted_cauchy_ratio"] <= 1 + 1e-12
    assert result["max_target_weighted_cauchy_ratio"] <= 1 + 1e-12
    assert result["all_pass"]
