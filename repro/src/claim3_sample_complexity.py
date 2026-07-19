#!/usr/bin/env python3
"""Claim 3 proof audit for ECL sample complexity (gFPPTokv9C).

All returned radii are normalized to an unspecified absolute concentration
constant.  They compare dependence on B, K, delta, and per-bin counts; they are
not literal numerical coverage guarantees because Theorem 3.2 does not state
its constants C, C1, or C2.

Paper anchors (arXiv:2605.21552v1):
- Theorem 3.2 / Eq. 9: PDF page 5.
- Remark 3.2: PDF page 5.
- Appendix G / Eqs. 30-31: PDF page 15.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Iterable

import numpy as np

# The independent Hilbert-space route first obtains one source and one target
# radius.  Adding them and applying sqrt(a)+sqrt(b) <= sqrt(2(a+b)) costs this
# universal normalized factor.  It is not a paper-supplied literal constant.
VECTOR_CONCENTRATION_FACTOR = float(sqrt(2.0))


@dataclass(frozen=True)
class BinPlan:
    """Per-bin weights and positive source/target effective sample counts."""

    weights: np.ndarray
    source_counts: np.ndarray
    target_counts: np.ndarray

    def __post_init__(self) -> None:
        arrays = []
        for name in ("weights", "source_counts", "target_counts"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
            arrays.append(value)

        weights, source_counts, target_counts = arrays
        if not (len(weights) == len(source_counts) == len(target_counts)):
            raise ValueError("weights and source/target counts must have equal lengths")
        if len(weights) == 0:
            raise ValueError("a bin plan must contain at least one bin")
        if np.any(weights < 0):
            raise ValueError("bin weights must be nonnegative")
        if not np.isclose(np.sum(weights), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("bin weights must sum to one")
        if np.any(source_counts <= 0) or np.any(target_counts <= 0):
            raise ValueError("all source and target counts must be positive")


def _validate_radius_inputs(
    plan: BinPlan, *, bins: int, classes: int, delta: float
) -> None:
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)):
        raise ValueError("bins must be an integer")
    if bins != len(plan.weights):
        raise ValueError("bins must equal the BinPlan length")
    if isinstance(classes, bool) or not isinstance(classes, (int, np.integer)):
        raise ValueError("classes must be an integer")
    if classes < 2:
        raise ValueError("classes must be at least two")
    if not np.isfinite(delta) or not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must lie strictly between zero and one")


def _combined_count_term(plan: BinPlan) -> float:
    return float(
        np.sum(
            plan.weights
            * (1.0 / plan.target_counts + 1.0 / plan.source_counts)
        )
    )


def displayed_radius(
    plan: BinPlan, *, bins: int, classes: int, delta: float
) -> float:
    """Eq. 9 normalized radius, exactly as displayed in Theorem 3.2."""
    _validate_radius_inputs(plan, bins=bins, classes=classes, delta=delta)
    return float(
        sqrt(log(2.0 * bins * classes / delta) * _combined_count_term(plan))
    )


def appendix_coordinate_radius(
    plan: BinPlan, *, bins: int, classes: int, delta: float
) -> float:
    """Literal Appendix-G coordinate-union route from Eq. 31.

    Eq. 31 bounds each vector mean norm with sqrt(K log(2BK/delta)/n).
    Combining those per-bin source/target bounds therefore retains sqrt(K).
    """
    _validate_radius_inputs(plan, bins=bins, classes=classes, delta=delta)
    return float(sqrt(classes) * displayed_radius(
        plan, bins=bins, classes=classes, delta=delta
    ))


def vector_concentration_radius(
    plan: BinPlan, *, bins: int, classes: int, delta: float
) -> float:
    """Independent dimension-free Hilbert-space concentration route.

    Derivation, normalized to the unspecified universal concentration constant:

    1. Every class-probability vector lies in the probability simplex, whose
       Euclidean diameter is at most sqrt(2); hence centered observations are
       bounded Hilbert-space random vectors by an absolute constant.
    2. A dimension-free Hoeffding/Pinelis inequality for bounded Hilbert-space
       means gives ||mean-E mean|| <= C sqrt(log(1/eta)/n), with no factor K.
    3. A union bound over B bins and two domains uses eta=delta/(2B), producing
       log(2B/delta).  We conservatively use log(4B/delta) to allow two-sided
       tail bookkeeping; for K>=2 this is no larger than Eq. 9's log(2BK/delta).
    4. The reverse triangle inequality bounds each difference-of-norms error by
       the source mean error plus the target mean error.
    5. Weighted Cauchy-Schwarz and sum_j w_j=1 give, separately,
       sum_j w_j/sqrt(n_j) <= sqrt(sum_j w_j/n_j).
    6. Adding source and target terms yields the expression below; optionally
       sqrt(a)+sqrt(b) <= sqrt(2(a+b)) upper-bounds it by
       VECTOR_CONCENTRATION_FACTOR times the Eq. 9 normalized radius.

    This function is independently computed from source and target count terms;
    it does not call or multiply displayed_radius().
    """
    _validate_radius_inputs(plan, bins=bins, classes=classes, delta=delta)
    log_term = log(4.0 * bins / delta)
    source_term = float(np.sum(plan.weights / plan.source_counts))
    target_term = float(np.sum(plan.weights / plan.target_counts))
    return float(sqrt(log_term * source_term) + sqrt(log_term * target_term))


def proof_audit(classes_grid: Iterable[int] = (2, 3, 10, 50)) -> dict[str, object]:
    """Classify the printed proof separately from the theorem statement."""
    classes = [int(value) for value in classes_grid]
    if not classes or any(value < 2 for value in classes):
        raise ValueError("classes_grid must contain integers >= 2")

    # The independent vector derivation above uses only simplex boundedness and
    # standard dimension-free Hilbert-space concentration.  It reaches Eq. 9's
    # order up to an unspecified universal factor; it does not repair Appendix G.
    independent_support = True
    findings = ["displayed_bound_not_derived_by_appendix_proof"]
    if independent_support:
        findings.append("displayed_bound_supported_by_independent_vector_argument")

    return {
        "paper_anchors": {
            "theorem_3_2_eq_9": "PDF page 5",
            "remark_3_2": "PDF page 5",
            "appendix_g_eqs_30_31": "PDF page 15",
        },
        "appendix_derives_displayed_bound": False,
        "appendix_to_displayed_ratio": {
            f"K={value}": float(sqrt(value)) for value in classes
        },
        "independent_vector_argument_supports_displayed_order": independent_support,
        "theorem_statement_assessment": "supported" if independent_support else "inconclusive",
        "appendix_proof_assessment": "missing_sqrt_K_factor",
        "findings": findings,
        "radius_semantics": {
            "label": "normalized_radius",
            "absolute_constant_known": False,
            "literal_coverage_claimed": False,
            "paper_constants": ["C", "C1", "C2"],
            "vector_route_normalized_factor": VECTOR_CONCENTRATION_FACTOR,
        },
        "limitations": [
            "The paper does not state numerical values for C, C1, or C2.",
            "The independent vector argument supports the displayed order but is not the proof printed in Appendix G.",
            "Finite simulations cannot prove a universal concentration theorem.",
        ],
    }
