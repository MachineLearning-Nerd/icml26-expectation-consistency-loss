"""Exact tests for the general Claim 1 analytical certificate."""

from fractions import Fraction

import claim1_general_certificate as c1


Q = Fraction


def test_large_exact_multiclass_certificate_has_no_float_residual() -> None:
    certificate = c1.build_report()["large_exact_multiclass_certificate"]
    assert certificate["dimensions"] == {
        "n_x": 257,
        "n_classes": 11,
        "n_summary_values": 17,
    }
    assert certificate["source_and_target_marginals_differ"]
    assert certificate["source_eq13_exact"]
    assert certificate["target_eq13_exact"]
    assert certificate["cross_domain_gap_equals_ec_exact"]
    assert certificate["exact_eq13_component_checks"] == 374
    assert certificate["exact_cross_domain_component_checks"] == 187


def test_formal_coefficients_prove_both_directions_for_every_q() -> None:
    source = c1.normalize_integer_weights((2, 7, 5, 11, 3, 13))
    target = c1.normalize_integer_weights((17, 3, 19, 2, 23, 5))
    summaries = ("a", "a", "b", "b", "c", "c")
    certificate = c1.formal_iff_coefficient_certificate(source, target, summaries)
    assert certificate["all_formal_coefficient_maps_equal"]
    assert certificate["iff_for_every_common_posterior_q"]
    assert certificate["support"] == {
        "shared": ["a", "b", "c"],
        "source_only": [],
        "target_only": [],
    }


def test_exact_ec_zero_and_nonzero_direction_witnesses() -> None:
    witnesses = c1.build_report()["direction_witnesses"]
    assert witnesses["ec_zero_implies_equal_calibration_curve"]["marginals_differ"]
    assert witnesses["ec_zero_implies_equal_calibration_curve"]["all_exact_gaps_zero"]
    assert witnesses["unequal_calibration_curve_implies_ec_nonzero"]["marginals_differ"]
    assert witnesses["unequal_calibration_curve_implies_ec_nonzero"][
        "expected_binary_class_gap"
    ] == "1/2"


def test_eq13_fails_if_summary_carries_label_information_beyond_x() -> None:
    control = c1.build_report()["assumption_breaking_controls"]["S_not_X_measurable"]
    assert control["P_Y1_given_S1"] == "1"
    assert control["E_P_Y1_given_X_given_S1"] == "1/2"
    assert control["eq13_residual"] == "1/2"


def test_characterization_fails_without_shared_conditional_kernel() -> None:
    control = c1.build_report()["assumption_breaking_controls"]["shared_conditional_broken"]
    assert control["source_q_ec_residual"] == "0"
    assert control["true_calibration_curve_gap"] == "-1/2"
    assert control["characterization_fails"]


def test_ec_alone_does_not_imply_absolute_calibration() -> None:
    control = c1.build_report()["assumption_breaking_controls"][
        "source_calibration_premise_missing"
    ]
    assert control["prediction_S1"] == "3/4"
    assert control["source_P_Y1_given_S"] == "1/4"
    assert control["target_P_Y1_given_S"] == "1/4"
    assert control["ec_holds"]
    assert control["both_domains_absolutely_miscalibrated"]


def test_cross_domain_statement_rejects_disjoint_summary_support() -> None:
    source = (Q(1), Q(0))
    target = (Q(0), Q(1))
    support = c1.common_summary_support(source, target, (0, 1))
    assert support == {"shared": (), "source_only": (0,), "target_only": (1,)}
    certificate = c1.formal_iff_coefficient_certificate(source, target, (0, 1))
    assert certificate["all_formal_coefficient_maps_equal"]
    assert not certificate["iff_for_every_common_posterior_q"]


def test_source_calibration_turns_curve_equality_into_target_calibration() -> None:
    # One shared summary level S_1=3/4 and a calibrated common posterior mean 3/4.
    source = c1.normalize_integer_weights((1, 3))
    target = c1.normalize_integer_weights((3, 1))
    posterior = ((Q(1, 4), Q(3, 4)), (Q(1, 4), Q(3, 4)))
    summaries = (0, 0)
    source_curve = c1.calibration_curve_from_joint(source, posterior, summaries)[0][1]
    target_curve = c1.calibration_curve_from_joint(target, posterior, summaries)[0][1]
    prediction = Q(3, 4)
    assert source_curve == prediction  # explicit source-calibration premise
    assert source_curve == target_curve  # EC / curve transfer
    assert target_curve == prediction  # target calibration follows


def test_report_classifies_literal_and_stronger_claim_separately() -> None:
    assessment = c1.build_report()["assessment"]
    assert assessment["literal_theorem_3_1"] == (
        "verified_with_support_and_version_qualifications"
    )
    assert assessment["absolute_calibration_from_ec_alone"] == (
        "falsified_without_source_calibration_premise"
    )
    assert assessment["toy"] is False
    assert assessment["inconclusive"] is False
    assert assessment["substantive_scientific_attempt"] is True
    assert assessment["all_certificate_gates_pass"] is True
