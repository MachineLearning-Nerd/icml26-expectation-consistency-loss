#!/usr/bin/env python3
"""Fail-closed tests for final real-MNIST Claim-3 attempt."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from claim3_real_mnist_sample_complexity import (
    estimate_from_indices,
    estimate_loop_crosscheck,
    exact_real_population,
    image_features,
    load_idx_images,
    load_idx_labels,
    make_real_construction,
    validate_split_protocol,
    x_only_domain_probabilities,
)


def test_idx_parsers_fail_closed_and_read_exact_payload(tmp_path):
    images = np.arange(2 * 28 * 28, dtype=np.uint8).reshape(2, 28, 28)
    labels = np.array([3, 7], dtype=np.uint8)
    image_path = tmp_path / "images"
    label_path = tmp_path / "labels"
    image_path.write_bytes(struct.pack(">IIII", 2051, 2, 28, 28) + images.tobytes())
    label_path.write_bytes(struct.pack(">II", 2049, 2) + labels.tobytes())
    assert np.array_equal(load_idx_images(image_path), images)
    assert np.array_equal(load_idx_labels(label_path), labels)
    image_path.write_bytes(image_path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="payload"):
        load_idx_images(image_path)


def test_split_protocol_rejects_overlap_and_non_test_evaluation():
    valid = validate_split_protocol((0, 30), (30, 60), evaluation_split="official_test_10000", train_count=60)
    assert valid["primary_and_posterior_training_disjoint"] is True
    with pytest.raises(ValueError, match="disjoint"):
        validate_split_protocol((0, 40), (30, 60), evaluation_split="official_test_10000", train_count=60)
    with pytest.raises(ValueError, match="official test"):
        validate_split_protocol((0, 30), (30, 60), evaluation_split="train", train_count=60)


def test_x_only_domain_probabilities_are_label_independent_and_positive():
    images = np.random.default_rng(1).integers(0, 256, size=(40, 28, 28), dtype=np.uint8)
    source_a, target_a, diagnostics_a = x_only_domain_probabilities(images, strength=1.1, uniform_floor=0.05)
    source_b, target_b, diagnostics_b = x_only_domain_probabilities(images.copy(), strength=1.1, uniform_floor=0.05)
    assert np.array_equal(source_a, source_b)
    assert np.array_equal(target_a, target_b)
    assert np.all(source_a > 0) and np.all(target_a > 0)
    assert source_a.sum() == pytest.approx(1.0)
    assert target_a.sum() == pytest.approx(1.0)
    assert diagnostics_a["uses_labels"] is False
    assert diagnostics_a == diagnostics_b


def _synthetic_construction(bins=10):
    rng = np.random.default_rng(9)
    scores = rng.dirichlet(np.ones(10), size=80)
    posterior = 0.8 * scores + 0.2 / 10
    labels = np.argmax(posterior, axis=1)
    images = rng.integers(0, 256, size=(80, 28, 28), dtype=np.uint8)
    source, target, _ = x_only_domain_probabilities(images, strength=0.5, uniform_floor=0.1)
    return make_real_construction(scores, posterior, labels, source, target, bins=bins, decay_factor=0.9)


def test_real_soft_population_and_estimators_are_finite():
    construction = _synthetic_construction()
    population = exact_real_population(construction)
    assert population["ecl"] >= 0
    assert population["matched_canonical_ece"] >= 0
    assert np.all(population["source_mass"] > 0)
    assert np.all(population["target_mass"] > 0)
    source = np.random.default_rng(10).choice(80, size=101, p=construction.source_probabilities)
    target = np.random.default_rng(11).choice(80, size=103, p=construction.target_probabilities)
    estimate = estimate_from_indices(construction, source, target, stabilizer=1e-5)
    assert np.isfinite(estimate["ecl"])
    assert np.isfinite(estimate["matched_canonical_ece"])


def test_vectorized_estimator_matches_explicit_bin_loop():
    construction = _synthetic_construction()
    source = np.random.default_rng(12).choice(80, size=97, p=construction.source_probabilities)
    target = np.random.default_rng(13).choice(80, size=109, p=construction.target_probabilities)
    vectorized = estimate_from_indices(construction, source, target, stabilizer=1e-5)
    loop = estimate_loop_crosscheck(construction, source, target, stabilizer=1e-5)
    assert vectorized["ecl"] == pytest.approx(loop["ecl"], abs=1e-14)
    assert vectorized["matched_canonical_ece"] == pytest.approx(
        loop["matched_canonical_ece"], abs=1e-14
    )


def test_feature_extractor_has_preregistered_dimension_and_range_checks():
    images = np.zeros((3, 28, 28), dtype=np.uint8)
    assert image_features(images).shape == (3, 77)
    with pytest.raises(ValueError, match="shape"):
        image_features(np.zeros((3, 27, 28)))

