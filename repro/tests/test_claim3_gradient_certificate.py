"""Exact tests for the anchored Claim 3 / Theorem 3.3 audit."""

from fractions import Fraction

import pytest

import claim3_gradient_certificate as c3


Q = Fraction


def test_corrected_fixed_direction_estimator_is_exactly_unbiased() -> None:
    certificate = c3.build_report()["exact_certificates"]["appendix_h_normalization"]
    assert certificate["full_weighted_mean_gradient"] == "3"
    assert certificate["appendix_h_unscaled_expectation"] == "3/2"
    assert not certificate["appendix_h_normalization_matches"]
    assert certificate["required_scale_N_over_nj"] == "2"
    assert certificate["corrected_estimator_expectation"] == "3"
    assert certificate["corrected_estimator_is_exactly_unbiased"]


def test_certificate_soft_weights_are_valid_per_sample_bin_probabilities() -> None:
    exact = c3.build_report()["exact_certificates"]
    for name in ("appendix_h_normalization", "self_normalization_and_same_batch_norm"):
        construction = exact[name]["construction"]
        weights = tuple(Q(value) for value in construction["soft_weights"])
        rows = tuple(
            tuple(Q(value) for value in row)
            for row in construction["per_sample_two_bin_assignments"]
        )
        assert all(Q(0) <= weight <= Q(1) for weight in weights)
        assert c3.valid_soft_assignment_rows(rows)
        assert c3.strictly_interior_soft_assignment_rows(rows)
        assert construction["per_sample_bin_weights_valid"]
        assert construction["assignments_strictly_interior"]
        assert construction["posterior_path_locally_valid"]
        assert tuple(row[0] for row in rows) == weights


def test_exhaustive_subset_expectation_matches_manual_fraction() -> None:
    values = (Q(1, 3), Q(2, 3), Q(4, 3))
    expectation = c3.exact_subset_expectation(
        3, 2, lambda batch: sum((values[index] for index in batch), Q(0))
    )
    # Every item is present in two of the three size-two subsets.
    assert expectation == Q(14, 9)


def test_self_normalized_loss_and_same_batch_direction_are_biased() -> None:
    certificate = c3.build_report()["exact_certificates"][
        "self_normalization_and_same_batch_norm"
    ]
    assert certificate["full_self_normalized_mean"] == "5/8"
    assert certificate["full_abs_ecl_loss"] == "1/8"
    assert certificate["expected_minibatch_abs_loss"] == "1/4"
    assert not certificate["loss_estimator_unbiased"]
    assert certificate["full_abs_ecl_gradient"] == "1"
    assert certificate["expected_same_batch_direction_gradient"] == "0"
    assert not certificate["same_batch_direction_gradient_unbiased"]
    assert certificate["independent_full_direction_corrected_gradient"] == "1"
    assert certificate["independent_full_direction_corrected_is_unbiased"]


def test_soft_weight_quotient_derivative_is_not_posterior_derivative_only() -> None:
    certificate = c3.build_report()["exact_certificates"]["soft_weight_derivative"]
    assert certificate["audited_bin_weights_at_theta_0"] == ["1/4", "3/4"]
    assert certificate["weighted_mean_at_theta_0"] == "1/4"
    assert certificate["true_quotient_rule_derivative"] == "3/4"
    assert certificate["appendix_h_derivative_when_p_is_fixed"] == "0"
    assert not certificate["omitting_weight_and_denominator_derivatives_is_valid"]
    assert certificate["assignments_strictly_interior_at_theta_0"]
    assert certificate["local_validity_radius"] == "1/8"
    assert certificate["assignment_path_locally_valid"]
    assert certificate["fixed_posterior_probabilities_valid"]
    for rows in certificate["local_assignment_rows"]:
        exact_rows = tuple(tuple(Q(value) for value in row) for row in rows)
        assert c3.strictly_interior_soft_assignment_rows(exact_rows)


def test_eq10_exact_minimizer_and_profile_gradient_disagree_with_eq8() -> None:
    certificate = c3.build_report()["exact_certificates"]["eq10_objective_and_gradient"]
    assert certificate["construction"]["source_probabilities"] == ["15/16", "13/16"]
    assert certificate["construction"]["target_probabilities"] == ["1/8", "1/8"]
    assert certificate["construction"]["all_probabilities_strictly_interior_at_theta"]
    assert certificate["construction"]["source_posterior_path_locally_valid"]
    assert certificate["construction"]["target_posterior_path_locally_valid"]
    assert certificate["construction"]["one_bin_assignment_valid"]
    assert certificate["exact_auxiliary_minimizer"] == {"u_s": "5/8", "u_t": "3/8"}
    assert certificate["eq8_loss"] == "3/4"
    assert certificate["eq10_profile_loss"] == "65/128"
    assert not certificate["objectives_equal"]
    assert certificate["eq8_gradient"] == "0"
    assert certificate["eq10_fixed_aux_or_envelope_gradient"] == "1/4"
    assert not certificate["gradients_equal"]


def test_eq10_variance_gap_does_not_vanish_under_replication() -> None:
    certificate = c3.build_report()["exact_certificates"]["eq10_objective_and_gradient"]
    rows = certificate["replication_stress"]
    assert [row["replication"] for row in rows] == [1, 2, 4, 8, 16, 32, 64]
    assert rows[0]["eq10_minus_eq8"] == "-31/128"
    assert rows[-1]["eq10_minus_eq8"] == "127/256"
    assert rows[-1]["eq10_profile_gradient"] == "16"
    assert not certificate["asymptotic_equivalence_supported_by_this_family"]


def test_scalar_eq10_minimizer_handles_fused_and_separated_regimes() -> None:
    fused = c3.eq10_scalar_minimizer((Q(1, 2),), (Q(2, 5),), Q(1))
    assert fused == (Q(9, 20), Q(9, 20))
    separated = c3.eq10_scalar_minimizer((Q(1), Q(1)), (Q(0), Q(0)), Q(1))
    assert separated == (Q(3, 4), Q(1, 4))


def test_nondifferentiable_abs_point_requires_explicit_subgradient() -> None:
    with pytest.raises(ValueError, match="ordinary gradient"):
        c3.sign(Q(0))


def test_official_source_matches_pin_and_exposes_semantic_mismatch() -> None:
    source = c3.build_report()["official_source_audit"]
    assert source["declared_pin"].endswith(c3.EXPECTED_OFFICIAL_COMMIT)
    assert source["vendored_losses_sha256"] == c3.EXPECTED_LOSSES_SHA256
    assert source["matches_independently_read_back_pinned_losses_sha256"]
    assert source["eq10_returned_loss_is_squared_residual_sum"]
    assert source["current_batch_auxiliaries_are_detached_then_reused_on_same_batch"]
    assert not source["paper_required_current_batch_independence_is_satisfied"]
    assert source["soft_assignment_weights_feed_loss_without_detach"]
    assert source["final_notebook_optimizer_is_fc2_only"]
    assert source["posterior_head_is_separately_pretrained"]


def test_report_verdict_and_all_audit_gates() -> None:
    report = c3.build_report()
    assert report["paper"]["paper_pdf_hash_matches_expected"]
    assert all(report["probability_domain_audit"].values())
    assert all(report["verification_gates"].values())
    assert report["assessment"]["anchored_claim_3"] == "contradicted_as_stated"
    assert report["assessment"]["narrow_fixed_direction_estimator"] == (
        "verified_with_explicit_conditions_and_corrected_scaling"
    )
    assert report["assessment"]["eq10_is_eq8_auxiliary_reformulation"] == (
        "falsified_by_exact_within_bin_variance_counterexample"
    )
    assert report["assessment"]["all_certificate_gates_pass"]
    assert not report["assessment"]["toy"]
    assert not report["assessment"]["inconclusive"]


def test_report_is_deterministic_within_one_environment() -> None:
    first = c3.build_report()
    second = c3.build_report()
    assert first["certificate_sha256"] == second["certificate_sha256"]
    assert first["exact_certificates"] == second["exact_certificates"]
