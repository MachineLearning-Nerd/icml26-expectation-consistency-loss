#!/usr/bin/env python3
"""Fail-closed tests for the separate soft Eq. 8 attempt."""
from __future__ import annotations

import numpy as np
import pytest

from claim3_soft_sample_complexity import (
    SoftConstruction,
    _domain_statistics,
    estimate_expanded_samples,
    estimate_from_counts,
    exact_population,
    make_construction,
    official_temperature,
    simplex_anchors,
    soft_assignments,
)


def test_official_simplex_grid_cardinalities_and_temperature():
    assert [len(simplex_anchors(value)) for value in [3, 6, 10, 15, 21, 28]] == [3, 6, 10, 15, 21, 28]
    assert official_temperature(15) == pytest.approx(-1.0 / (np.log(0.9) * 15**2))


def test_soft_assignments_are_normalized_and_population_masses_positive():
    construction = make_construction(15)
    assert np.all(construction.assignments >= 0)
    assert np.allclose(construction.assignments.sum(axis=1), 1.0)
    population = exact_population(construction)
    assert np.all(population["source_mass"] > 0)
    assert np.all(population["target_mass"] > 0)
    assert population["ecl"] > 0


def test_count_contraction_matches_independent_expanded_path():
    construction = make_construction(10)
    source_counts = np.random.default_rng(41).multinomial(251, construction.source_probabilities)
    target_counts = np.random.default_rng(42).multinomial(263, construction.target_probabilities)
    vectorized = estimate_from_counts(
        construction, source_counts, target_counts, stabilizer=1e-5
    )
    expanded = estimate_expanded_samples(
        construction, source_counts, target_counts, stabilizer=1e-5
    )
    assert vectorized["ecl"] == pytest.approx(expanded["ecl"], abs=1e-14)
    assert vectorized["matched_oracle_ece"] == pytest.approx(
        expanded["matched_oracle_ece"], abs=1e-14
    )


def test_exact_population_is_independent_of_sample_size_and_stabilizer_bias_shrinks():
    construction = make_construction(6)
    population = exact_population(construction)
    small = estimate_from_counts(
        construction,
        100 * construction.source_probabilities,
        100 * construction.target_probabilities,
        stabilizer=1e-5,
    )
    large = estimate_from_counts(
        construction,
        10000 * construction.source_probabilities,
        10000 * construction.target_probabilities,
        stabilizer=1e-5,
    )
    assert abs(large["ecl"] - population["ecl"]) < abs(small["ecl"] - population["ecl"])


def test_zero_population_mass_is_rejected():
    construction = make_construction(6)
    assignments = construction.assignments.copy()
    assignments[:, -1] = 0.0
    assignments /= assignments.sum(axis=1, keepdims=True)
    with pytest.raises(ValueError, match="population soft-bin masses"):
        _domain_statistics(
            construction.source_probabilities, construction.posterior, assignments
        )


def test_tiny_mass_stress_remains_valid_and_positive():
    construction = make_construction(15, temperature_scale=0.2, tiny_mass=True)
    population = exact_population(construction)
    assert np.all(population["source_mass"] > 0)
    assert np.all(population["target_mass"] > 0)
    assert min(np.min(population["source_mass"]), np.min(population["target_mass"])) < 1e-5


def test_constructor_rejects_nonpositive_domain_probability():
    construction = make_construction(3)
    bad = construction.source_probabilities.copy()
    bad[0] = 0.0
    bad /= bad.sum()
    with pytest.raises(ValueError, match="positive"):
        SoftConstruction(
            scores=construction.scores,
            posterior=construction.posterior,
            source_probabilities=bad,
            target_probabilities=construction.target_probabilities,
            anchors=construction.anchors,
            assignments=construction.assignments,
            temperature=construction.temperature,
        )
