#!/usr/bin/env python3
"""Fail-closed tests for gFPPTokv9C Claim 3 (Theorem 3.2)."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from claim3_sample_complexity import (
    VECTOR_CONCENTRATION_FACTOR,
    BinPlan,
    _sample_ece_correct_counts,
    _sample_posterior_vectors,
    appendix_coordinate_radius,
    displayed_radius,
    empirical_ecl,
    empirical_histogram_ece,
    exact_binary_tail,
    generate_covariate_shift_case,
    make_plan_family,
    official_code_audit,
    population_ecl,
    population_histogram_ece,
    proof_audit,
    run_experiment,
    target_bin_mass_radius,
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


def test_omitted_hard_bin_mass_term_is_absorbed_by_displayed_order():
    bins, classes, delta = 8, 10, 0.05
    plan = balanced_plan(bins=bins, count=200)
    mass = target_bin_mass_radius(
        plan, bins=bins, classes=classes, delta=delta
    )
    displayed = displayed_radius(plan, bins=bins, classes=classes, delta=delta)
    assert mass > 0
    assert mass <= displayed


def test_proof_audit_separates_proof_gap_from_theorem_assessment():
    audit = proof_audit(classes_grid=[2, 3, 10, 100])
    assert audit["appendix_derives_displayed_bound"] is False
    assert audit["appendix_proof_assessment"] == "missing_sqrt_K_and_bin_mass_terms"
    assert "displayed_bound_not_derived_by_appendix_proof" in audit["findings"]
    assert "appendix_omits_empirical_target_bin_mass_term" in audit["findings"]
    assert audit["hard_bin_eq5_assessment"] == "supported"
    assert audit["soft_eq8_assessment"] == "inconclusive"
    assert audit["theorem_statement_assessment"] == "supported_for_fixed_hard_bin_eq5"
    if audit["independent_vector_argument_supports_displayed_order"]:
        assert (
            "hard_bin_displayed_order_supported_by_independent_bounded_differences_argument"
            in audit["findings"]
        )
    else:
        assert audit["hard_bin_eq5_assessment"] != "supported"


def test_unknown_constants_remain_normalized_not_literal_coverage_values():
    audit = proof_audit(classes_grid=[2, 10])
    semantics = audit["radius_semantics"]
    assert semantics["label"] == "normalized_radius"
    assert semantics["absolute_constant_known"] is False
    assert semantics["literal_coverage_claimed"] is False
    assert semantics["paper_constants"] == ["C", "C1", "C2"]


def test_sparse_and_imbalanced_positive_counts_are_supported():
    target_counts = np.array([500, 50, 5, 2])
    plan = BinPlan(
        weights=target_counts / target_counts.sum(),
        source_counts=np.array([1000, 100, 10, 1]),
        target_counts=target_counts,
    )
    kwargs = {"bins": 4, "classes": 3, "delta": 0.05}
    values = [
        displayed_radius(plan, **kwargs),
        appendix_coordinate_radius(plan, **kwargs),
        vector_concentration_radius(plan, **kwargs),
    ]
    assert all(np.isfinite(value) and value > 0 for value in values)


def test_population_and_empirical_canonical_ecl_match_exact_samples():
    weights = np.array([0.5, 0.5])
    source_means = np.array([[1.0, 0.0], [0.5, 0.5]])
    target_means = np.array([[0.0, 1.0], [0.5, 0.5]])
    plan = BinPlan(weights, np.array([2, 2]), np.array([2, 2]))
    source = [
        np.array([[1.0, 0.0], [1.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    ]
    target = [
        np.array([[0.0, 1.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    ]
    expected = 0.5 * np.sqrt(2.0)
    assert population_ecl(source_means, target_means, weights) == pytest.approx(expected)
    assert empirical_ecl(source, target, plan) == pytest.approx(expected)


def test_matched_histogram_ece_population_and_empirical():
    weights = np.array([0.4, 0.6])
    anchors = np.array([0.25, 0.75])
    probabilities = np.array([0.5, 0.5])
    expected = 0.4 * 0.25 + 0.6 * 0.25
    assert population_histogram_ece(probabilities, anchors, weights) == pytest.approx(expected)
    assert empirical_histogram_ece(
        np.array([2, 2]), np.array([4, 4]), anchors, weights
    ) == pytest.approx(expected)


def test_covariate_shift_generator_preserves_simplex_and_labels_controls():
    plan = balanced_plan(bins=4, count=20)
    valid = generate_covariate_shift_case(plan=plan, classes=3, seed=7)
    invalid = generate_covariate_shift_case(
        plan=plan, classes=3, seed=7, violate_shared_conditional=True
    )
    assert valid["assumptions_valid"] is True
    assert valid["shared_conditional_atoms"] is True
    assert invalid["assumptions_valid"] is False
    assert invalid["shared_conditional_atoms"] is False
    assert np.array_equal(valid["posterior_atoms"], valid["target_posterior_atoms"])
    assert not np.array_equal(invalid["posterior_atoms"], invalid["target_posterior_atoms"])
    for key in ["source_means", "target_means"]:
        assert np.allclose(valid[key].sum(axis=1), 1.0)
        assert np.all(valid[key] >= 0)
    assert np.allclose(valid["confidence_vectors"].sum(axis=1), 1.0)
    assert np.all(valid["confidence_vectors"][:, 0] > 0.5)
    assert np.allclose(
        valid["confidence_vectors"][:, 0], valid["confidence_anchors"]
    )


def test_ecl_sampler_returns_posterior_atoms_not_one_hot_labels():
    atoms = np.array([[[0.2, 0.8], [0.65, 0.35]]])
    samples = _sample_posterior_vectors(
        atoms,
        np.array([0.4]),
        np.array([100]),
        np.random.default_rng(42),
    )[0]
    assert all(any(np.array_equal(row, atom) for atom in atoms[0]) for row in samples)
    assert np.any((samples > 0) & (samples < 1))
    assert not np.all(np.isin(samples, [0.0, 1.0]))


def test_ecl_sampler_mixture_probability_selects_atom_zero():
    atoms = np.array([[[0.2, 0.8], [0.65, 0.35]]])
    all_zero = _sample_posterior_vectors(
        atoms, np.array([1.0]), np.array([4]), np.random.default_rng(1)
    )[0]
    all_one = _sample_posterior_vectors(
        atoms, np.array([0.0]), np.array([4]), np.random.default_rng(1)
    )[0]
    assert np.all(all_zero == atoms[0, 0])
    assert np.all(all_one == atoms[0, 1])


def test_matched_ece_uses_separate_bernoulli_sampler():
    probabilities = np.array([0.2, 0.8])
    counts = np.array([100, 100])
    first = _sample_ece_correct_counts(probabilities, counts, np.random.default_rng(7))
    second = _sample_ece_correct_counts(probabilities, counts, np.random.default_rng(7))
    assert np.array_equal(first, second)
    assert first.shape == (2,)
    assert np.issubdtype(first.dtype, np.integer)
    assert np.all((first >= 0) & (first <= counts))


def test_bin_plan_rejects_weights_not_derived_from_target_counts():
    with pytest.raises(ValueError, match="target_counts"):
        BinPlan(
            weights=np.array([0.5, 0.5]),
            source_counts=np.array([10, 10]),
            target_counts=np.array([9, 1]),
        )


def test_exact_binary_enumeration_is_normalized_and_deterministic():
    result = exact_binary_tail(
        source_probability=0.25,
        target_probability=0.75,
        source_count=8,
        target_count=9,
        threshold=0.2,
    )
    assert result["probability_mass"] == pytest.approx(1.0, abs=1e-12)
    assert 0 <= result["tail_probability"] <= 1
    assert result == exact_binary_tail(
        source_probability=0.25,
        target_probability=0.75,
        source_count=8,
        target_count=9,
        threshold=0.2,
    )


@pytest.mark.parametrize(
    "family",
    ["balanced", "skewed", "sparse_valid", "source_target_imbalanced"],
)
def test_plan_families_are_positive_and_normalized(family):
    plan = make_plan_family(family, bins=8, count=25)
    assert np.all(plan.source_counts > 0)
    assert np.all(plan.target_counts > 0)
    assert plan.weights.sum() == pytest.approx(1.0)
    assert np.allclose(plan.weights, plan.target_counts / plan.target_counts.sum())


def test_official_code_audit_rejects_direct_eq5_loss_parity():
    root = Path(__file__).resolve().parents[2]
    result = official_code_audit(
        root / "repro" / "evidence" / "claim3" / "official_losses.py"
    )
    assert result["required_markers_present"] is True
    assert result["source_pin_matches"] is True
    assert result["per_bin_mean_statistics_match_eq7_components"] is True
    assert result["direct_eq5_loss_parity_supported"] is False
    assert "Eq.10" in result["semantic_assessment"]


def small_config():
    return {
        "bins": [2, 4],
        "classes": [2, 3],
        "deltas": [0.1, 0.05],
        "per_domain_counts": [12, 25, 50],
        "epsilons": [0.1, 0.2],
        "seeds": [0, 1],
        "plan_families": [
            "balanced",
            "skewed",
            "sparse_valid",
            "source_target_imbalanced",
        ],
    }


def test_experiment_coverage_includes_every_axis_and_separates_controls():
    root = Path(__file__).resolve().parents[2]
    config = small_config()
    result = run_experiment(
        config,
        official_source=root / "repro" / "evidence" / "claim3" / "official_losses.py",
    )
    covered = result["coverage"]["covered_values"]
    for key, expected in [
        ("bins", config["bins"]),
        ("classes", config["classes"]),
        ("deltas", config["deltas"]),
        ("counts", config["per_domain_counts"]),
        ("families", sorted(config["plan_families"])),
        ("seeds", config["seeds"]),
    ]:
        assert covered[key] == sorted(expected)
    assert all(row["assumptions_valid"] is True for row in result["rows"])
    assert all(row["weights_match_target_counts"] is True for row in result["rows"])
    assert all(row["ecl_samples_are_posterior_vectors"] is True for row in result["rows"])
    assert all(row["posterior_function_fixed_before_sampling"] is True for row in result["rows"])
    assert all(row["hard_bin_partition_fixed_before_sampling"] is True for row in result["rows"])
    assert all(row["shared_conditional_atoms"] is True for row in result["rows"])
    assert any(control["assumptions_valid"] is False for control in result["controls"])
    assert result["summary"]["literal_coverage_claimed"] is False
    for name in ["ecl", "histogram_ece"]:
        slope = result["executed_sample_scaling"][name]["rmse_log_slope_vs_count"]
        assert np.isfinite(slope)
    mass_slope = result["executed_target_mass_scaling"]["rmse_log_slope_vs_total_count"]
    assert np.isfinite(mass_slope)
    assert result["formula_derived_slopes"]["label"] == (
        "formula_derived_identity_not_executed_estimator_evidence"
    )


def test_production_grid_executed_scaling_matches_root_n_rate():
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "repro" / "configs" / "claim3.json").read_text())
    result = run_experiment(
        config,
        official_source=root / "repro" / "evidence" / "claim3" / "official_losses.py",
    )
    for name in ["ecl", "histogram_ece"]:
        scaling = result["executed_sample_scaling"][name]
        assert -0.7 < scaling["rmse_log_slope_vs_count"] < -0.3
        assert 1.5 < scaling["implied_sample_complexity_exponent_from_rmse"] < 2.5
    assert (
        -0.7
        < result["executed_target_mass_scaling"]["rmse_log_slope_vs_total_count"]
        < -0.3
    )


def test_cli_uses_repo_relative_defaults_and_is_byte_stable(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = root / "repro" / "src" / "run_claim3_sample_complexity.py"
    config = tmp_path / "config.json"
    config.write_text(json.dumps(small_config()))
    json_out = tmp_path / "result.json"
    markdown_out = tmp_path / "result.md"
    command = [
        sys.executable,
        str(script),
        "--config",
        str(config),
        "--json-out",
        str(json_out),
        "--markdown-out",
        str(markdown_out),
    ]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    first_json = json_out.read_bytes()
    first_markdown = markdown_out.read_bytes()
    subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    assert json_out.read_bytes() == first_json
    assert markdown_out.read_bytes() == first_markdown
    payload = json.loads(first_json)
    assert payload["result"]["proof_audit"]["appendix_derives_displayed_bound"] is False
    assert payload["result"]["summary"]["literal_coverage_claimed"] is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
