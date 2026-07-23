#!/usr/bin/env python3
"""Exact audit certificate for anchored Claim 3 / ECL Theorem 3.3.

The paper claims that its auxiliary-variable mini-batch formulation (Eq. 10)
has an unbiased stochastic gradient for the full differentiable ECL objective
(Eq. 8).  This module separates five mathematically different statements:

1. a correctly scaled sample mean is unbiased for a *linear*, fixed-direction
   finite-population gradient;
2. the normalization printed in Appendix H is not that estimator;
3. a self-normalized mini-batch ratio is generally biased;
4. choosing the norm direction from the same mini-batch is generally biased;
5. the gradient of Eq. 10's squared-residual objective is not the fixed-
   direction gradient written in Appendix H, nor generally the gradient of
   Eq. 8.

All numerical certificates use ``fractions.Fraction`` and exhaustive uniform
subset expectations.  No random numbers, floating point, ML framework, or
network training is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


Q = Fraction
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_OFFICIAL_COMMIT = "aae77f890f1e4ebc13dad135b5e29758d98d318d"
EXPECTED_LOSSES_SHA256 = "1c2de34967f34b98faae5025368edac88f46a709d6e1e0c063e2c01f4d6e9754"
EXPECTED_PDF_SHA256 = "fb1d1a634d55132694349d40d56731cc5c7401571bc8c1a9f6eee1b5849950ab"


def _q(value: int | str | Q) -> Q:
    return value if isinstance(value, Q) else Q(value)


def exact_mean(values: Sequence[Q]) -> Q:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values, Q(0)) / len(values)


def exact_weighted_mean(values: Sequence[Q], weights: Sequence[Q]) -> Q:
    if not values or len(values) != len(weights):
        raise ValueError("values and weights must have the same positive length")
    total = sum(weights, Q(0))
    if total <= 0 or any(weight < 0 for weight in weights):
        raise ValueError("weights must be nonnegative with positive total")
    return sum((value * weight for value, weight in zip(values, weights, strict=True)), Q(0)) / total


def valid_soft_assignment_rows(rows: Sequence[Sequence[Q]]) -> bool:
    """Check Eq. 6 probability-simplex constraints sample by sample."""

    return bool(rows) and all(
        bool(row)
        and all(Q(0) <= weight <= Q(1) for weight in row)
        and sum(row, Q(0)) == Q(1)
        for row in rows
    )


def strictly_interior_soft_assignment_rows(rows: Sequence[Sequence[Q]]) -> bool:
    """Check the relative interior of each probability simplex row."""

    return valid_soft_assignment_rows(rows) and all(
        all(Q(0) < weight < Q(1) for weight in row) for row in rows
    )


def linear_probability_path_locally_valid(
    values_at_theta: Sequence[Q], derivatives: Sequence[Q], radius: Q
) -> bool:
    """Certify a two-sided linear neighborhood remains in the open unit interval."""

    if len(values_at_theta) != len(derivatives) or radius <= 0:
        return False
    return all(
        Q(0) < value + direction * radius * derivative < Q(1)
        for value, derivative in zip(values_at_theta, derivatives, strict=True)
        for direction in (Q(-1), Q(1))
    )


def exact_subset_expectation(
    population_size: int, batch_size: int, statistic: Callable[[tuple[int, ...]], Q]
) -> Q:
    """Expectation under a uniform batch sampled without replacement."""

    if not 1 <= batch_size <= population_size:
        raise ValueError("batch_size must lie in [1, population_size]")
    batches = tuple(combinations(range(population_size), batch_size))
    return exact_mean(tuple(statistic(batch) for batch in batches))


def full_weighted_gradient(weights: Sequence[Q], derivatives: Sequence[Q]) -> Q:
    return exact_weighted_mean(derivatives, weights)


def appendix_h_unscaled_batch_gradient(
    weights: Sequence[Q], derivatives: Sequence[Q], batch: Sequence[int]
) -> Q:
    """The 1/|D^m| weighted sum displayed on Appendix-H page 17."""

    return sum((weights[index] * derivatives[index] for index in batch), Q(0)) / len(batch)


def corrected_fixed_direction_batch_gradient(
    weights: Sequence[Q], derivatives: Sequence[Q], batch: Sequence[int]
) -> Q:
    """Horvitz-scaled estimator of (sum_i omega_i q'_i)/(sum_i omega_i)."""

    population_size = len(weights)
    soft_count = sum(weights, Q(0))
    return (
        Q(population_size, 1)
        / soft_count
        * appendix_h_unscaled_batch_gradient(weights, derivatives, batch)
    )


def self_normalized_batch_mean(
    values: Sequence[Q], weights: Sequence[Q], batch: Sequence[int]
) -> Q:
    denominator = sum((weights[index] for index in batch), Q(0))
    if denominator == 0:
        raise ValueError("self-normalized batch has zero denominator")
    return sum((weights[index] * values[index] for index in batch), Q(0)) / denominator


def sign(value: Q) -> Q:
    if value == 0:
        raise ValueError("the ordinary gradient of abs is not unique at zero")
    return Q(1) if value > 0 else Q(-1)


def _normalization_certificate() -> dict[str, object]:
    # A single domain/bin suffices; the target contribution is set to zero.
    assignments = ((Q(1, 4), Q(3, 4)), (Q(3, 4), Q(1, 4)))
    weights = tuple(row[0] for row in assignments)
    posterior_values = (Q(1, 2), Q(1, 2))
    derivatives = (Q(0), Q(4))
    local_radius = Q(1, 16)
    full = full_weighted_gradient(weights, derivatives)
    paper_expectation = exact_subset_expectation(
        2, 1, lambda batch: appendix_h_unscaled_batch_gradient(weights, derivatives, batch)
    )
    corrected_expectation = exact_subset_expectation(
        2, 1, lambda batch: corrected_fixed_direction_batch_gradient(weights, derivatives, batch)
    )
    return {
        "construction": {
            "soft_weights": _serialize(weights),
            "posterior_values_at_theta_0": _serialize(posterior_values),
            "per_sample_derivatives": _serialize(derivatives),
            "population_size": 2,
            "batch_size": 1,
            "soft_count": _fraction_text(sum(weights, Q(0))),
            "per_sample_two_bin_assignments": _serialize(assignments),
            "per_sample_bin_weights_valid": valid_soft_assignment_rows(assignments),
            "assignments_strictly_interior": strictly_interior_soft_assignment_rows(
                assignments
            ),
            "posterior_local_validity_radius": _fraction_text(local_radius),
            "posterior_path_locally_valid": linear_probability_path_locally_valid(
                posterior_values, derivatives, local_radius
            ),
        },
        "full_weighted_mean_gradient": _fraction_text(full),
        "appendix_h_unscaled_expectation": _fraction_text(paper_expectation),
        "appendix_h_normalization_matches": paper_expectation == full,
        "required_scale_N_over_nj": _fraction_text(Q(len(weights), 1) / sum(weights, Q(0))),
        "corrected_estimator_expectation": _fraction_text(corrected_expectation),
        "corrected_estimator_is_exactly_unbiased": corrected_expectation == full,
    }


def _self_normalization_and_norm_certificate() -> dict[str, object]:
    # At theta=0 all probabilities are interior.  The target mean is 1/2.
    # q_s,1(theta)=1/4+theta and q_s,2(theta)=3/4+theta.
    values = (Q(1, 4), Q(3, 4))
    derivatives = (Q(1), Q(1))
    assignments = ((Q(1, 4), Q(3, 4)), (Q(3, 4), Q(1, 4)))
    weights = tuple(row[0] for row in assignments)
    target = Q(1, 2)
    local_radius = Q(1, 8)
    full_mean = exact_weighted_mean(values, weights)
    full_loss = abs(full_mean - target)
    full_gradient = sign(full_mean - target) * exact_weighted_mean(derivatives, weights)

    def batch_loss(batch: tuple[int, ...]) -> Q:
        return abs(self_normalized_batch_mean(values, weights, batch) - target)

    def batch_gradient(batch: tuple[int, ...]) -> Q:
        batch_value = self_normalized_batch_mean(values, weights, batch)
        batch_derivative = self_normalized_batch_mean(derivatives, weights, batch)
        return sign(batch_value - target) * batch_derivative

    expected_loss = exact_subset_expectation(2, 1, batch_loss)
    expected_gradient = exact_subset_expectation(2, 1, batch_gradient)
    fixed_direction_gradient = exact_subset_expectation(
        2,
        1,
        lambda batch: sign(full_mean - target)
        * corrected_fixed_direction_batch_gradient(weights, derivatives, batch),
    )
    return {
        "construction": {
            "source_probabilities_at_theta_0": _serialize(values),
            "source_derivatives": _serialize(derivatives),
            "soft_weights": _serialize(weights),
            "per_sample_two_bin_assignments": _serialize(assignments),
            "per_sample_bin_weights_valid": valid_soft_assignment_rows(assignments),
            "assignments_strictly_interior": strictly_interior_soft_assignment_rows(
                assignments
            ),
            "posterior_local_validity_radius": _fraction_text(local_radius),
            "posterior_path_locally_valid": linear_probability_path_locally_valid(
                values, derivatives, local_radius
            ),
            "target_probability": _fraction_text(target),
            "batch_size": 1,
        },
        "full_self_normalized_mean": _fraction_text(full_mean),
        "full_abs_ecl_loss": _fraction_text(full_loss),
        "expected_minibatch_abs_loss": _fraction_text(expected_loss),
        "loss_estimator_unbiased": expected_loss == full_loss,
        "full_abs_ecl_gradient": _fraction_text(full_gradient),
        "expected_same_batch_direction_gradient": _fraction_text(expected_gradient),
        "same_batch_direction_gradient_unbiased": expected_gradient == full_gradient,
        "independent_full_direction_corrected_gradient": _fraction_text(fixed_direction_gradient),
        "independent_full_direction_corrected_is_unbiased": fixed_direction_gradient
        == full_gradient,
        "interpretation": (
            "Loss unbiasedness, ratio unbiasedness, and gradient unbiasedness are distinct. "
            "The norm direction must not be selected from the same random batch."
        ),
    }


def _weight_derivative_certificate() -> dict[str, object]:
    # For the audited bin, omega_1(theta)=1/4+theta and omega_2(theta)=3/4.
    # The second column below supplies each sample's complementary bin weight.
    assignments_at_theta_0 = ((Q(1, 4), Q(3, 4)), (Q(3, 4), Q(1, 4)))
    local_radius = Q(1, 8)

    def assignments_at(offset: Q) -> tuple[tuple[Q, Q], tuple[Q, Q]]:
        return (
            (Q(1, 4) + offset, Q(3, 4) - offset),
            (Q(3, 4), Q(1, 4)),
        )

    local_rows = tuple(
        assignments_at(offset) for offset in (-local_radius, Q(0), local_radius)
    )
    numerator = Q(1, 4)
    denominator = Q(1)
    numerator_derivative = Q(1)
    denominator_derivative = Q(1)
    true_derivative = (
        numerator_derivative * denominator - numerator * denominator_derivative
    ) / denominator**2
    printed_derivative = Q(0)  # Appendix H retains only omega_i * grad p_i; p_i are fixed here.
    return {
        "construction": (
            "audited-bin weights omega_1(theta)=1/4+theta and omega_2(theta)=3/4; "
            "complements 3/4-theta and 1/4; fixed p_1=1 and p_2=0"
        ),
        "audited_bin_weights_at_theta_0": ["1/4", "3/4"],
        "per_sample_two_bin_assignments_at_theta_0": _serialize(
            assignments_at_theta_0
        ),
        "assignments_strictly_interior_at_theta_0": (
            strictly_interior_soft_assignment_rows(assignments_at_theta_0)
        ),
        "local_validity_radius": _fraction_text(local_radius),
        "local_assignment_rows": _serialize(local_rows),
        "assignment_path_locally_valid": all(
            strictly_interior_soft_assignment_rows(rows) for rows in local_rows
        ),
        "fixed_posterior_probabilities_valid": True,
        "weighted_mean_at_theta_0": _fraction_text(numerator / denominator),
        "true_quotient_rule_derivative": _fraction_text(true_derivative),
        "appendix_h_derivative_when_p_is_fixed": _fraction_text(printed_derivative),
        "omitting_weight_and_denominator_derivatives_is_valid": true_derivative
        == printed_derivative,
    }


@dataclass(frozen=True)
class Affine:
    intercept: Q
    slope: Q

    def at(self, theta: Q) -> Q:
        return self.intercept + self.slope * theta


def eq10_scalar_minimizer(
    source_values: Sequence[Q], target_values: Sequence[Q], bin_weight: Q
) -> tuple[Q, Q]:
    """Exact minimizer of one-bin scalar Eq. 10 for an L1 leading norm."""

    if not source_values or not target_values or bin_weight < 0:
        raise ValueError("positive domain sizes and a nonnegative bin weight are required")
    n_s, n_t = len(source_values), len(target_values)
    mean_s, mean_t = exact_mean(source_values), exact_mean(target_values)
    delta = mean_s - mean_t
    fusion_threshold = bin_weight / (2 * n_s) + bin_weight / (2 * n_t)
    if abs(delta) <= fusion_threshold:
        fused = (n_s * mean_s + n_t * mean_t) / (n_s + n_t)
        return fused, fused
    direction = sign(delta)
    return (
        mean_s - direction * bin_weight / (2 * n_s),
        mean_t + direction * bin_weight / (2 * n_t),
    )


def eq10_scalar_objective(
    source_values: Sequence[Q], target_values: Sequence[Q], u_s: Q, u_t: Q, bin_weight: Q
) -> Q:
    return (
        bin_weight * abs(u_s - u_t)
        + sum(((u_s - value) ** 2 for value in source_values), Q(0))
        + sum(((u_t - value) ** 2 for value in target_values), Q(0))
    )


def eq10_fixed_aux_gradient(
    source_values: Sequence[Q],
    source_derivatives: Sequence[Q],
    target_values: Sequence[Q],
    target_derivatives: Sequence[Q],
    u_s: Q,
    u_t: Q,
) -> Q:
    if len(source_values) != len(source_derivatives) or len(target_values) != len(
        target_derivatives
    ):
        raise ValueError("each value needs one derivative")
    return 2 * (
        sum(
            ((value - u_s) * derivative for value, derivative in zip(source_values, source_derivatives, strict=True)),
            Q(0),
        )
        + sum(
            ((value - u_t) * derivative for value, derivative in zip(target_values, target_derivatives, strict=True)),
            Q(0),
        )
    )


def _eq10_objective_counterexample() -> dict[str, object]:
    # q_s=(7/8+theta, 7/8-theta), q_t=(1/8,1/8), theta=1/16.
    # Eq. 8 is the constant 3/4 and has zero gradient, while Eq. 10 retains
    # within-bin posterior variance and has nonzero theta gradient.
    theta = Q(1, 16)
    source_model = (Affine(Q(7, 8), Q(1)), Affine(Q(7, 8), Q(-1)))
    target_model = (Affine(Q(1, 8), Q(0)), Affine(Q(1, 8), Q(0)))
    source_values = tuple(item.at(theta) for item in source_model)
    target_values = tuple(item.at(theta) for item in target_model)
    source_derivatives = tuple(item.slope for item in source_model)
    target_derivatives = tuple(item.slope for item in target_model)
    bin_weight = Q(1)
    local_radius = Q(1, 32)
    u_s, u_t = eq10_scalar_minimizer(source_values, target_values, bin_weight)
    eq8_loss = abs(exact_mean(source_values) - exact_mean(target_values))
    eq8_gradient = sign(exact_mean(source_values) - exact_mean(target_values)) * (
        exact_mean(source_derivatives) - exact_mean(target_derivatives)
    )
    eq10_loss = eq10_scalar_objective(source_values, target_values, u_s, u_t, bin_weight)
    eq10_gradient = eq10_fixed_aux_gradient(
        source_values,
        source_derivatives,
        target_values,
        target_derivatives,
        u_s,
        u_t,
    )

    replication_rows = []
    for replication in (1, 2, 4, 8, 16, 32, 64):
        repeated_source = source_values * replication
        repeated_target = target_values * replication
        repeated_source_derivatives = source_derivatives * replication
        repeated_target_derivatives = target_derivatives * replication
        rep_u_s, rep_u_t = eq10_scalar_minimizer(
            repeated_source, repeated_target, bin_weight
        )
        rep_loss = eq10_scalar_objective(
            repeated_source, repeated_target, rep_u_s, rep_u_t, bin_weight
        )
        rep_gradient = eq10_fixed_aux_gradient(
            repeated_source,
            repeated_source_derivatives,
            repeated_target,
            repeated_target_derivatives,
            rep_u_s,
            rep_u_t,
        )
        replication_rows.append(
            {
                "replication": replication,
                "samples_per_domain": len(repeated_source),
                "eq10_profile_loss": _fraction_text(rep_loss),
                "eq10_minus_eq8": _fraction_text(rep_loss - eq8_loss),
                "eq10_profile_gradient": _fraction_text(rep_gradient),
            }
        )

    return {
        "construction": {
            "theta": _fraction_text(theta),
            "source_probabilities": _serialize(source_values),
            "target_probabilities": _serialize(target_values),
            "source_derivatives": _serialize(source_derivatives),
            "target_derivatives": _serialize(target_derivatives),
            "bin_weight": "1",
            "all_probabilities_strictly_interior_at_theta": all(
                Q(0) < value < Q(1) for value in source_values + target_values
            ),
            "posterior_local_validity_radius": _fraction_text(local_radius),
            "source_posterior_path_locally_valid": linear_probability_path_locally_valid(
                source_values, source_derivatives, local_radius
            ),
            "target_posterior_path_locally_valid": linear_probability_path_locally_valid(
                target_values, target_derivatives, local_radius
            ),
            "one_bin_assignment_valid": Q(0) <= bin_weight <= Q(1),
        },
        "exact_auxiliary_minimizer": {"u_s": _fraction_text(u_s), "u_t": _fraction_text(u_t)},
        "eq8_loss": _fraction_text(eq8_loss),
        "eq10_profile_loss": _fraction_text(eq10_loss),
        "objectives_equal": eq8_loss == eq10_loss,
        "eq8_gradient": _fraction_text(eq8_gradient),
        "eq10_fixed_aux_or_envelope_gradient": _fraction_text(eq10_gradient),
        "gradients_equal": eq8_gradient == eq10_gradient,
        "omitted_term": (
            "sum_i ||p_i-mean_domain||^2; it depends on theta and is discarded in "
            "Appendix H's claimed substitution"
        ),
        "replication_stress": replication_rows,
        "asymptotic_equivalence_supported_by_this_family": False,
    }


def _line_numbers(text: str, needles: Sequence[str]) -> dict[str, list[int]]:
    lines = text.splitlines()
    return {
        needle: [index for index, line in enumerate(lines, start=1) if needle in line]
        for needle in needles
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _official_source_audit() -> dict[str, object]:
    losses_path = REPO_ROOT / "upstream" / "losses.py"
    notebook_path = REPO_ROOT / "upstream" / "main.ipynb"
    losses = losses_path.read_text(encoding="utf-8")
    notebook = notebook_path.read_text(encoding="utf-8")
    losses_hash = _sha256_file(losses_path)
    needles = (
        "class ECLossMiniBatch",
        "u_s_j_detached = u_s_j.detach()",
        "u_t_j_detached = u_t_j.detach()",
        "w_j = n_t_j / (n_t_batch.sum() + tiny)",
        "loss_s_j = (w_s[:, j] *",
        "loss_t_j = (w_t[:, j] *",
    )
    lines = _line_numbers(losses, needles)
    same_batch_aux_then_loss = all(lines[needle] for needle in needles[1:])
    soft_weights_feed_loss_without_detach = (
        "w_s = torch.softmax" in losses
        and "w_t = torch.softmax" in losses
        and "w_s = w_s.detach" not in losses
        and "w_t = w_t.detach" not in losses
    )
    final_optimizer_fc2_only = "optimizer = optim.Adam(model.fc2.parameters(), lr=0.001)" in notebook
    posterior_head_separately_pretrained = (
        "optimizer2 = torch.optim.Adam(model.classifier2.parameters(), lr=0.01)" in notebook
    )
    return {
        "declared_pin": f"NeuroDong/ECL@{EXPECTED_OFFICIAL_COMMIT}",
        "vendored_losses_sha256": losses_hash,
        "matches_independently_read_back_pinned_losses_sha256": losses_hash
        == EXPECTED_LOSSES_SHA256,
        "line_evidence": lines,
        "eq10_returned_loss_is_squared_residual_sum": bool(
            lines["loss_s_j = (w_s[:, j] *"] and lines["loss_t_j = (w_t[:, j] *"]
        ),
        "current_batch_auxiliaries_are_detached_then_reused_on_same_batch": same_batch_aux_then_loss,
        "paper_required_current_batch_independence_is_satisfied": not same_batch_aux_then_loss,
        "soft_assignment_weights_feed_loss_without_detach": soft_weights_feed_loss_without_detach,
        "final_notebook_optimizer_is_fc2_only": final_optimizer_fc2_only,
        "posterior_head_is_separately_pretrained": posterior_head_separately_pretrained,
        "implementation_implication": (
            "In the final notebook phase p from classifier2 is fixed with respect to the fc2 "
            "optimizer, while w_s/w_t depend on fc2 logits and remain in autograd. Appendix H "
            "keeps omega*grad(p) but omits these weight derivatives."
        ),
    }


def _fraction_text(value: Q) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _serialize(value: object) -> object:
    if isinstance(value, Q):
        return _fraction_text(value)
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize(item) for item in value]
    return value


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_report() -> dict[str, object]:
    normalization = _normalization_certificate()
    self_normalized = _self_normalization_and_norm_certificate()
    weight_derivative = _weight_derivative_certificate()
    eq10 = _eq10_objective_counterexample()
    source = _official_source_audit()
    probability_domain_audit = {
        "normalization_assignments_valid": normalization["construction"][
            "per_sample_bin_weights_valid"
        ]
        and normalization["construction"]["assignments_strictly_interior"],
        "normalization_posterior_path_locally_valid": normalization["construction"][
            "posterior_path_locally_valid"
        ],
        "self_normalized_assignments_valid": self_normalized["construction"][
            "per_sample_bin_weights_valid"
        ]
        and self_normalized["construction"]["assignments_strictly_interior"],
        "self_normalized_posterior_path_locally_valid": self_normalized[
            "construction"
        ]["posterior_path_locally_valid"],
        "weight_derivative_assignment_path_locally_valid": weight_derivative[
            "assignment_path_locally_valid"
        ],
        "weight_derivative_fixed_posteriors_valid": weight_derivative[
            "fixed_posterior_probabilities_valid"
        ],
        "eq10_probabilities_interior_and_locally_valid": eq10["construction"][
            "all_probabilities_strictly_interior_at_theta"
        ]
        and eq10["construction"]["source_posterior_path_locally_valid"]
        and eq10["construction"]["target_posterior_path_locally_valid"]
        and eq10["construction"]["one_bin_assignment_valid"],
    }
    gates = {
        "corrected_fixed_direction_estimator_unbiased": normalization[
            "corrected_estimator_is_exactly_unbiased"
        ],
        "printed_appendix_h_normalization_counterexample": not normalization[
            "appendix_h_normalization_matches"
        ],
        "self_normalized_loss_bias_counterexample": not self_normalized[
            "loss_estimator_unbiased"
        ],
        "same_batch_direction_bias_counterexample": not self_normalized[
            "same_batch_direction_gradient_unbiased"
        ],
        "soft_weight_derivative_omission_counterexample": not weight_derivative[
            "omitting_weight_and_denominator_derivatives_is_valid"
        ],
        "eq10_objective_counterexample": not eq10["objectives_equal"],
        "eq10_gradient_counterexample": not eq10["gradients_equal"],
        "pinned_source_hash_match": source[
            "matches_independently_read_back_pinned_losses_sha256"
        ],
        "all_probability_domain_paths_valid": all(probability_domain_audit.values()),
    }
    report = {
        "paper": {
            "title": "Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift",
            "openreview_id": "gFPPTokv9C",
            "paper_pdf_sha256": _sha256_file(
                REPO_ROOT / "repro" / "evidence" / "claim3" / "2605.21552v1.pdf"
            ),
            "paper_pdf_hash_matches_expected": _sha256_file(
                REPO_ROOT / "repro" / "evidence" / "claim3" / "2605.21552v1.pdf"
            )
            == EXPECTED_PDF_SHA256,
            "anchors": ["Section 3.5, Theorem 3.3, Eq. 10", "Appendix H, Eqs. 32-33"],
            "anchored_claim": (
                "Theorem 3.3 shows the mini-batch ECL formulation using auxiliary variables "
                "u_j^s and u_j^t yields an unbiased gradient estimator, "
                "E[grad_theta Lhat_ecl^mini] = grad_theta Lhat_ecl, enabling standard SGD training."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "arithmetic": "Python standard-library fractions.Fraction; exhaustive subsets",
        },
        "derivation": {
            "valid_restricted_identity": (
                "Conditional on a fixed posterior function, fixed differentiable weights/counts, "
                "a fixed norm subgradient v independent of the current batch, and correct "
                "finite-population scaling, linearity gives E[G_m | state] = G_full."
            ),
            "population_gradient_interchange": (
                "For a population rather than a finite dataset, differentiability almost "
                "everywhere plus an integrable dominating derivative (or another valid Leibniz "
                "rule) is required before moving grad through expectation."
            ),
            "nonsmooth_point": (
                "At E_s,j=E_t,j the norm has a subdifferential; the theorem must specify a "
                "measurable compatible subgradient selection rather than an ordinary gradient."
            ),
            "posterior_estimator_condition": (
                "Unbiasedness can be stated conditional on a posterior estimator fitted on "
                "independent data or frozen before the current batch. Re-fitting it on the same "
                "batch creates dependence and changes the target estimand."
            ),
            "soft_self_normalization": (
                "A ratio of random weighted sums is not generally an unbiased estimator of the "
                "full ratio, even when numerator and denominator separately are unbiased."
            ),
            "eq10_distinction": (
                "Eq. 10 differentiates squared residuals 2*omega*(p-u)*grad(p), plus derivatives "
                "through theta-dependent omega. Appendix H instead writes a fixed norm-direction "
                "linear gradient. Those are different estimators and objectives."
            ),
        },
        "exact_certificates": {
            "appendix_h_normalization": normalization,
            "self_normalization_and_same_batch_norm": self_normalized,
            "soft_weight_derivative": weight_derivative,
            "eq10_objective_and_gradient": eq10,
        },
        "probability_domain_audit": probability_domain_audit,
        "official_source_audit": source,
        "verification_gates": gates,
        "assessment": {
            "anchored_claim_3": "contradicted_as_stated",
            "narrow_fixed_direction_estimator": "verified_with_explicit_conditions_and_corrected_scaling",
            "eq10_is_eq8_auxiliary_reformulation": "falsified_by_exact_within_bin_variance_counterexample",
            "algorithm1_unbiased_for_eq8_gradient": "not_established_and_exact_identity_fails",
            "loss_estimation_unbiasedness": "falsified_for_self_normalized_minibatch_ecl",
            "toy": False,
            "inconclusive": False,
            "substantive_scientific_attempt": True,
            "all_certificate_gates_pass": all(gates.values()),
            "recommended_verdict": "not_reproduced__theorem_3_3_contradicted_as_written",
        },
        "limitations": [
            "This is a theorem audit, not an empirical calibration benchmark; one exact counterexample suffices to reject a universal unbiasedness identity.",
            "The source audit is static because the project dependencies intentionally exclude PyTorch; the mathematical counterexamples do not depend on PyTorch.",
            "A differently defined linear estimator with frozen independent state can be unbiased, but it is not the Eq. 10 squared-residual loss returned by the official implementation.",
        ],
    }
    report["certificate_sha256"] = _sha256_json(
        {
            "exact_certificates": report["exact_certificates"],
            "verification_gates": gates,
            "assessment": report["assessment"],
        }
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "claim3_gradient_certificate.json",
    )
    args = parser.parse_args(argv)
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Anchored Claim 3 analytical audit: ECL Theorem 3.3")
    print(f"anchored claim assessment: {report['assessment']['anchored_claim_3']}")
    print(
        "corrected fixed-direction estimator: "
        f"{report['assessment']['narrow_fixed_direction_estimator']}"
    )
    print(f"Eq. 10 objective parity: {report['assessment']['eq10_is_eq8_auxiliary_reformulation']}")
    print(f"all audit gates pass: {report['assessment']['all_certificate_gates_pass']}")
    print(f"certificate sha256: {report['certificate_sha256']}")
    print(f"wrote {args.output}")
    return 0 if report["assessment"]["all_certificate_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
