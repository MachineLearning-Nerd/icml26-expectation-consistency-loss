#!/usr/bin/env python3
"""Fail-closed tests for gFPPTokv9C Claim 3 (Theorem 3.2)."""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from claim3_sample_complexity import (
    VECTOR_CONCENTRATION_FACTOR,
    BinPlan,
    appendix_coordinate_radius,
    displayed_radius,
    proof_audit,
    vector_concentration_radius,
)


def balanced_plan(bins=4, count=100):
    return BinPlan(
        weights=np.full(bins, 1.0 / bins),
        source_counts=np.full(bins, count),
        target_counts=np.full(bins, count),
    )


@pytest.mark.parametrize(
    ("weights", "source", "target"),
    [
        (np.array([[0.5, 0.5]]), np.array([10, 10]), np.array([10, 10])),
        (np.array([0.5, 0.5]), np.array([[10, 10]]), np.array([10, 10])),
        (np.array([0.5, 0.5]), np.array([10, 10]), np.array([[10, 10]])),
        (np.array([0.5, 0.5]), np.array([10]), np.array([10, 10])),
        (np.array([0.5, 0.5]), np.array([10, 10]), np.array([10])),
    ],
)
def test_bin_plan_requires_aligned_one_dimensional_arrays(weights, source, target):
    with pytest.raises(ValueError):
        BinPlan(weights=weights, source_counts=source, target_counts=target)


@pytest.mark.parametrize(
    ("weights", "source", "target"),
    [
        (np.array([-0.1, 1.1]), np.array([10, 10]), np.array([10, 10])),
        (np.array([0.4, 0.4]), np.array([10, 10]), np.array([10, 10])),
        (np.array([0.5, 0.5]), np.array([0, 10]), np.array([10, 10])),
        (np.array([0.5, 0.5]), np.array([10, -1]), np.array([10, 10])),
        (np.array([0.5, 0.5]), np.array([10, 10]), np.array([10, 0])),
    ],
)
def test_bin_plan_rejects_invalid_weights_and_nonpositive_counts(weights, source, target):
    with pytest.raises(ValueError):
        BinPlan(weights=weights, source_counts=source, target_counts=target)


@pytest.mark.parametrize(
    ("bins", "classes", "delta"),
    [
        (3, 3, 0.05),
        (4, 1, 0.05),
        (4, 3, 0.0),
        (4, 3, 1.0),
        (4, 3, -0.1),
        (4, 3, 1.1),
    ],
)
def test_radius_inputs_fail_closed(bins, classes, delta):
    plan = balanced_plan()
    with pytest.raises(ValueError):
        displayed_radius(plan, bins=bins, classes=classes, delta=delta)
    with pytest.raises(ValueError):
        appendix_coordinate_radius(plan, bins=bins, classes=classes, delta=delta)
    with pytest.raises(ValueError):
        vector_concentration_radius(plan, bins=bins, classes=classes, delta=delta)


def test_appendix_coordinate_route_has_sqrt_k_factor():
    bins, classes, delta = 8, 10, 0.05
    plan = balanced_plan(bins=bins, count=200)
    ratio = appendix_coordinate_radius(
        plan, bins=bins, classes=classes, delta=delta
    ) / displayed_radius(plan, bins=bins, classes=classes, delta=delta)
    assert ratio == pytest.approx(np.sqrt(classes))


@pytest.mark.parametrize("classes", [2, 3, 10, 100])
def test_displayed_and_appendix_k_dependence_are_distinct(classes):
    bins, delta = 4, 0.05
    plan = balanced_plan(bins=bins, count=400)
    common = np.sqrt(
        np.log(2 * bins * classes / delta)
        * np.sum(
            plan.weights
            * (1.0 / plan.target_counts + 1.0 / plan.source_counts)
        )
    )
    displayed = displayed_radius(plan, bins=bins, classes=classes, delta=delta)
    appendix = appendix_coordinate_radius(
        plan, bins=bins, classes=classes, delta=delta
    )
    assert displayed == pytest.approx(common)
    assert appendix == pytest.approx(np.sqrt(classes) * common)


def test_vector_route_is_independent_and_names_its_universal_factor():
    bins, classes, delta = 16, 50, 0.01
    plan = balanced_plan(bins=bins, count=800)
    assert isinstance(VECTOR_CONCENTRATION_FACTOR, float)
    assert np.isfinite(VECTOR_CONCENTRATION_FACTOR)
    assert VECTOR_CONCENTRATION_FACTOR > 0
    vector = vector_concentration_radius(
        plan, bins=bins, classes=classes, delta=delta
    )
    displayed = displayed_radius(plan, bins=bins, classes=classes, delta=delta)
    # The independent route uses log(4B/delta) and separate source/target
    # Cauchy bounds; it must be no worse than the named universal factor times
    # Eq. 9 for K>=2, not be implemented as an equality/reuse of Eq. 9.
    assert vector <= VECTOR_CONCENTRATION_FACTOR * displayed
    assert vector > 0


def test_proof_audit_separates_proof_gap_from_theorem_assessment():
    audit = proof_audit(classes_grid=[2, 3, 10, 100])
    assert audit["appendix_derives_displayed_bound"] is False
    assert audit["appendix_proof_assessment"] == "missing_sqrt_K_factor"
    assert "displayed_bound_not_derived_by_appendix_proof" in audit["findings"]
    assert audit["theorem_statement_assessment"] in {
        "supported",
        "not_supported",
        "inconclusive",
    }
    if audit["independent_vector_argument_supports_displayed_order"]:
        assert (
            "displayed_bound_supported_by_independent_vector_argument"
            in audit["findings"]
        )
    else:
        assert audit["theorem_statement_assessment"] != "supported"


def test_unknown_constants_remain_normalized_not_literal_coverage_values():
    audit = proof_audit(classes_grid=[2, 10])
    semantics = audit["radius_semantics"]
    assert semantics["label"] == "normalized_radius"
    assert semantics["absolute_constant_known"] is False
    assert semantics["literal_coverage_claimed"] is False
    assert semantics["paper_constants"] == ["C", "C1", "C2"]


def test_sparse_and_imbalanced_positive_counts_are_supported():
    plan = BinPlan(
        weights=np.array([0.7, 0.2, 0.09, 0.01]),
        source_counts=np.array([1000, 100, 10, 1]),
        target_counts=np.array([500, 50, 5, 2]),
    )
    kwargs = {"bins": 4, "classes": 3, "delta": 0.05}
    values = [
        displayed_radius(plan, **kwargs),
        appendix_coordinate_radius(plan, **kwargs),
        vector_concentration_radius(plan, **kwargs),
    ]
    assert all(np.isfinite(value) and value > 0 for value in values)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
