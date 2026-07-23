#!/usr/bin/env python3
"""Soft self-normalized Eq. 8 sample-complexity experiment.

This is deliberately separate from the fixed-hard-bin Eq. 5 audit.  Scores,
soft assignments, and the exact posterior oracle are fixed before independent
source/target evaluation samples are drawn.  Population quantities are exact
finite-distribution sums, not large Monte Carlo approximations.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb, log, sqrt
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SoftConstruction:
    scores: np.ndarray
    posterior: np.ndarray
    source_probabilities: np.ndarray
    target_probabilities: np.ndarray
    anchors: np.ndarray
    assignments: np.ndarray
    temperature: float

    def __post_init__(self) -> None:
        scores = _simplex(self.scores, "scores")
        posterior = _simplex(self.posterior, "posterior")
        source = _probabilities(self.source_probabilities, "source_probabilities")
        target = _probabilities(self.target_probabilities, "target_probabilities")
        anchors = _simplex(self.anchors, "anchors")
        assignments = np.asarray(self.assignments, dtype=np.float64)
        if scores.shape != posterior.shape:
            raise ValueError("scores and posterior must have equal shapes")
        if source.shape != (len(scores),) or target.shape != (len(scores),):
            raise ValueError("domain probabilities must align with score atoms")
        if anchors.shape[1] != scores.shape[1]:
            raise ValueError("anchor and score class dimensions must match")
        if assignments.shape != (len(scores), len(anchors)):
            raise ValueError("assignments must have shape (atoms, bins)")
        if np.any(assignments < 0) or not np.allclose(
            assignments.sum(axis=1), 1.0, atol=2e-14, rtol=0.0
        ):
            raise ValueError("assignment rows must be nonnegative and sum to one")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        for name, value in (
            ("scores", scores),
            ("posterior", posterior),
            ("source_probabilities", source),
            ("target_probabilities", target),
            ("anchors", anchors),
            ("assignments", assignments),
        ):
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def _simplex(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"{name} must be a two-dimensional simplex array")
    if not np.all(np.isfinite(array)) or np.any(array < -1e-14):
        raise ValueError(f"{name} must contain finite nonnegative values")
    if not np.allclose(array.sum(axis=1), 1.0, atol=2e-14, rtol=0.0):
        raise ValueError(f"{name} rows must sum to one")
    return array


def _probabilities(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{name} must be a positive one-dimensional probability vector")
    if not np.isclose(array.sum(), 1.0, atol=2e-14, rtol=0.0):
        raise ValueError(f"{name} must sum to one")
    return array


def simplex_anchors(requested_bins: int, classes: int = 3) -> np.ndarray:
    """Reproduce the official source's shifted simplex-grid anchor rule."""
    if requested_bins < 1 or classes < 2:
        raise ValueError("requested_bins/classes are out of range")
    resolution = 1
    while comb(resolution + classes - 1, classes - 1) < requested_bins:
        resolution += 1

    def compositions(total: int, dimensions: int) -> list[list[int]]:
        if dimensions == 1:
            return [[total]]
        rows: list[list[int]] = []
        for first in range(total + 1):
            for suffix in compositions(total - first, dimensions - 1):
                rows.append([first, *suffix])
        return rows

    grid = np.asarray(compositions(resolution, classes), dtype=np.float64)
    return (grid + 1.0 / classes) / (resolution + 1.0)


def official_temperature(actual_bins: int, decay_factor: float = 0.9) -> float:
    if actual_bins < 1 or not 0.0 < decay_factor < 1.0:
        raise ValueError("invalid actual_bins or decay_factor")
    return float(-1.0 / (log(decay_factor) * actual_bins * actual_bins))


def soft_assignments(
    scores: np.ndarray, anchors: np.ndarray, temperature: float
) -> np.ndarray:
    score_array = _simplex(scores, "scores")
    anchor_array = _simplex(anchors, "anchors")
    if score_array.shape[1] != anchor_array.shape[1]:
        raise ValueError("score and anchor dimensions differ")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = -np.sum(
        (score_array[:, None, :] - anchor_array[None, :, :]) ** 2, axis=2
    ) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def make_construction(
    requested_bins: int,
    *,
    decay_factor: float = 0.9,
    temperature_scale: float = 1.0,
    tiny_mass: bool = False,
) -> SoftConstruction:
    """Fixed 3-class discrete covariate-shift construction.

    Both domains use the same score and posterior functions on the same 231
    latent atoms.  Only P(X) differs.  Thus P(Y|X) is exactly shared.
    """
    if temperature_scale <= 0:
        raise ValueError("temperature_scale must be positive")
    resolution = 20
    triples = []
    for first in range(resolution + 1):
        for second in range(resolution - first + 1):
            triples.append([first, second, resolution - first - second])
    grid = np.asarray(triples, dtype=np.float64)
    scores = (grid + 0.75) / (resolution + 2.25)
    base = np.asarray([0.15, 0.35, 0.50])
    posterior = 0.72 * scores + 0.28 * base
    phase = np.arange(len(scores), dtype=np.float64)
    source_log = 1.15 * (scores[:, 0] - scores[:, 2]) + 0.18 * np.sin(phase * 0.37)
    target_log = -1.05 * (scores[:, 0] - scores[:, 2]) + 0.16 * np.cos(phase * 0.29)
    if tiny_mass:
        # Still strictly positive on every atom, but deliberately concentrates
        # each domain at a different corner to expose random-denominator risk.
        source_log += 13.0 * scores[:, 0]
        target_log += 13.0 * scores[:, 2]
    source = np.exp(source_log - source_log.max())
    target = np.exp(target_log - target_log.max())
    source /= source.sum()
    target /= target.sum()
    anchors = simplex_anchors(requested_bins, classes=3)
    temperature = official_temperature(len(anchors), decay_factor) * temperature_scale
    assignments = soft_assignments(scores, anchors, temperature)
    return SoftConstruction(
        scores=scores,
        posterior=posterior,
        source_probabilities=source,
        target_probabilities=target,
        anchors=anchors,
        assignments=assignments,
        temperature=temperature,
    )


def _domain_statistics(
    probabilities: np.ndarray,
    values: np.ndarray,
    assignments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mass = probabilities @ assignments
    if np.any(mass <= 0) or not np.all(np.isfinite(mass)):
        raise ValueError("all population soft-bin masses must be positive")
    numerator = (probabilities[:, None] * assignments).T @ values
    return mass, numerator / mass[:, None]


def exact_population(construction: SoftConstruction) -> dict[str, object]:
    """Exact enumeration of the finite latent distribution."""
    source_mass, source_mean = _domain_statistics(
        construction.source_probabilities,
        construction.posterior,
        construction.assignments,
    )
    target_mass, target_mean = _domain_statistics(
        construction.target_probabilities,
        construction.posterior,
        construction.assignments,
    )
    _, target_score_mean = _domain_statistics(
        construction.target_probabilities,
        construction.scores,
        construction.assignments,
    )
    ecl = float(np.sum(target_mass * np.linalg.norm(source_mean - target_mean, axis=1)))
    ece = float(np.sum(target_mass * np.linalg.norm(target_mean - target_score_mean, axis=1)))
    return {
        "ecl": ecl,
        "matched_canonical_ece": ece,
        "source_mass": source_mass,
        "target_mass": target_mass,
        "source_mean": source_mean,
        "target_mean": target_mean,
        "target_score_mean": target_score_mean,
    }


def _sample_label_counts(
    counts: np.ndarray, posterior: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    label_counts = np.zeros_like(posterior, dtype=np.int64)
    for index in np.flatnonzero(counts):
        label_counts[index] = rng.multinomial(int(counts[index]), posterior[index])
    return label_counts


def estimate_from_counts(
    construction: SoftConstruction,
    source_counts: np.ndarray,
    target_counts: np.ndarray,
    *,
    stabilizer: float,
    target_label_counts: np.ndarray | None = None,
) -> dict[str, object]:
    """Vectorized Eq. 8 and matched canonical ECE from atom counts."""
    if stabilizer < 0 or not np.isfinite(stabilizer):
        raise ValueError("stabilizer must be finite and nonnegative")
    source_counts = np.asarray(source_counts, dtype=np.float64)
    target_counts = np.asarray(target_counts, dtype=np.float64)
    atoms = len(construction.scores)
    if source_counts.shape != (atoms,) or target_counts.shape != (atoms,):
        raise ValueError("count vectors must align with atoms")
    if (
        not np.all(np.isfinite(source_counts))
        or not np.all(np.isfinite(target_counts))
        or np.any(source_counts < 0)
        or np.any(target_counts < 0)
    ):
        raise ValueError("counts must be finite and nonnegative")
    if source_counts.sum() <= 0 or target_counts.sum() <= 0:
        raise ValueError("both domains require samples")
    omega = construction.assignments
    source_soft_counts = source_counts @ omega
    target_soft_counts = target_counts @ omega
    source_numerator = (source_counts[:, None] * omega).T @ construction.posterior
    target_numerator = (target_counts[:, None] * omega).T @ construction.posterior
    source_mean = source_numerator / (source_soft_counts[:, None] + stabilizer)
    target_mean = target_numerator / (target_soft_counts[:, None] + stabilizer)
    target_weights = target_soft_counts / target_soft_counts.sum()
    ecl = float(np.sum(target_weights * np.linalg.norm(source_mean - target_mean, axis=1)))

    target_score_numerator = (target_counts[:, None] * omega).T @ construction.scores
    target_score_mean = target_score_numerator / (target_soft_counts[:, None] + stabilizer)
    oracle_ece = float(
        np.sum(target_weights * np.linalg.norm(target_mean - target_score_mean, axis=1))
    )
    label_ece = None
    if target_label_counts is not None:
        labels = np.asarray(target_label_counts, dtype=np.int64)
        if labels.shape != construction.posterior.shape:
            raise ValueError("target_label_counts must have shape (atoms, classes)")
        if np.any(labels < 0) or not np.array_equal(labels.sum(axis=1), target_counts):
            raise ValueError("target label counts must sum to atom target counts")
        label_numerator = omega.T @ labels
        label_mean = label_numerator / (target_soft_counts[:, None] + stabilizer)
        label_ece = float(
            np.sum(target_weights * np.linalg.norm(label_mean - target_score_mean, axis=1))
        )
    return {
        "ecl": ecl,
        "matched_oracle_ece": oracle_ece,
        "matched_label_ece": label_ece,
        "source_soft_counts": source_soft_counts,
        "target_soft_counts": target_soft_counts,
        "target_weights": target_weights,
    }


def estimate_expanded_samples(
    construction: SoftConstruction,
    source_counts: np.ndarray,
    target_counts: np.ndarray,
    *,
    stabilizer: float,
) -> dict[str, float]:
    """Independent raw-sample path that recomputes every soft assignment."""
    source_indices = np.repeat(np.arange(len(source_counts)), source_counts)
    target_indices = np.repeat(np.arange(len(target_counts)), target_counts)
    source_scores = construction.scores[source_indices]
    target_scores = construction.scores[target_indices]
    source_posterior = construction.posterior[source_indices]
    target_posterior = construction.posterior[target_indices]
    source_omega = soft_assignments(
        source_scores, construction.anchors, construction.temperature
    )
    target_omega = soft_assignments(
        target_scores, construction.anchors, construction.temperature
    )
    ns = source_omega.sum(axis=0)
    nt = target_omega.sum(axis=0)
    source_mean = (source_omega.T @ source_posterior) / (ns[:, None] + stabilizer)
    target_mean = (target_omega.T @ target_posterior) / (nt[:, None] + stabilizer)
    score_mean = (target_omega.T @ target_scores) / (nt[:, None] + stabilizer)
    weights = nt / nt.sum()
    return {
        "ecl": float(np.sum(weights * np.linalg.norm(source_mean - target_mean, axis=1))),
        "matched_oracle_ece": float(
            np.sum(weights * np.linalg.norm(target_mean - score_mean, axis=1))
        ),
    }


def _slope(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    x_array = np.asarray(list(x), dtype=np.float64)
    y_array = np.asarray(list(y), dtype=np.float64)
    if np.any(x_array <= 0) or np.any(y_array <= 0):
        raise ValueError("log slopes require positive values")
    slope, intercept = np.polyfit(np.log(x_array), np.log(y_array), 1)
    return float(slope), float(intercept)


def _summarize_axis(
    rows: list[dict[str, object]], axis: str, values: list[int]
) -> dict[str, object]:
    metrics = {
        "ecl": "ecl_error",
        "matched_oracle_ece": "matched_oracle_ece_error",
        "matched_label_ece": "matched_label_ece_error",
    }
    summary: dict[str, object] = {}
    for label, key in metrics.items():
        rmse = []
        q90 = []
        for value in values:
            errors = np.asarray([row[key] for row in rows if row[axis] == value])
            rmse.append(sqrt(float(np.mean(errors * errors))))
            q90.append(float(np.quantile(errors, 0.9)))
        slope, intercept = _slope(values, rmse)
        summary[label] = {
            "rmse": dict(zip(map(str, values), rmse)),
            "q90": dict(zip(map(str, values), q90)),
            f"rmse_log_slope_vs_{axis}": slope,
            "rmse_log_intercept": intercept,
        }
        if axis == "sample_size":
            tail_values = values[-5:]
            tail_rmse = rmse[-5:]
            tail_slope, tail_intercept = _slope(tail_values, tail_rmse)
            summary[label]["implied_epsilon_sample_complexity_exponent"] = float(-1.0 / slope)
            summary[label]["asymptotic_tail_sample_sizes"] = tail_values
            summary[label]["asymptotic_tail_rmse_log_slope"] = tail_slope
            summary[label]["asymptotic_tail_log_intercept"] = tail_intercept
            summary[label]["asymptotic_tail_implied_epsilon_sample_complexity_exponent"] = float(-1.0 / tail_slope)
            summary[label]["asymptotic_tail_n_rmse_squared_log_slope"] = float(1.0 + 2.0 * tail_slope)
            summary[label]["n_times_rmse_squared"] = dict(
                zip(map(str, values), [n * error * error for n, error in zip(values, rmse)])
            )
    return summary


def _regularizer_bias(
    construction: SoftConstruction, sample_size: int, stabilizer: float
) -> float:
    expected_source = sample_size * construction.source_probabilities
    expected_target = sample_size * construction.target_probabilities
    estimate = estimate_from_counts(
        construction,
        expected_source,
        expected_target,
        stabilizer=stabilizer,
    )
    return abs(float(estimate["ecl"]) - float(exact_population(construction)["ecl"]))


def run_soft_experiment(config: dict[str, object]) -> dict[str, object]:
    seeds = [int(seed) for seed in config["seeds"]]
    if len(seeds) != int(config["replicates"]):
        raise ValueError("replicates must match the explicit seed list")
    stabilizer = float(config["stabilizer"])
    decay = float(config["official_decay_factor"])
    baseline_bins = int(config["baseline_bins"])
    sample_sizes = [int(value) for value in config["sample_sizes"]]
    construction = make_construction(baseline_bins, decay_factor=decay)
    population = exact_population(construction)
    sample_rows: list[dict[str, object]] = []
    for n in sample_sizes:
        for seed in seeds:
            source_rng = np.random.default_rng(10_000_000 + 10_000 * n + seed)
            target_rng = np.random.default_rng(20_000_000 + 10_000 * n + seed)
            label_rng = np.random.default_rng(30_000_000 + 10_000 * n + seed)
            source_counts = source_rng.multinomial(n, construction.source_probabilities)
            target_counts = target_rng.multinomial(n, construction.target_probabilities)
            label_counts = _sample_label_counts(target_counts, construction.posterior, label_rng)
            estimate = estimate_from_counts(
                construction,
                source_counts,
                target_counts,
                stabilizer=stabilizer,
                target_label_counts=label_counts,
            )
            sample_rows.append({
                "sample_size": n,
                "seed": seed,
                "source_target_samples_independent": True,
                "scores_assignments_posterior_fixed_before_sampling": True,
                "ecl": estimate["ecl"],
                "ecl_error": abs(float(estimate["ecl"]) - float(population["ecl"])),
                "matched_oracle_ece": estimate["matched_oracle_ece"],
                "matched_oracle_ece_error": abs(float(estimate["matched_oracle_ece"]) - float(population["matched_canonical_ece"])),
                "matched_label_ece": estimate["matched_label_ece"],
                "matched_label_ece_error": abs(float(estimate["matched_label_ece"]) - float(population["matched_canonical_ece"])),
                "minimum_source_soft_count": float(np.min(estimate["source_soft_counts"])),
                "minimum_target_soft_count": float(np.min(estimate["target_soft_counts"])),
                "minimum_target_weight": float(np.min(estimate["target_weights"])),
            })
    sample_summary = _summarize_axis(sample_rows, "sample_size", sample_sizes)

    bins_values = [int(value) for value in config["actual_bins"]]
    bins_n = int(config["bins_axis_sample_size"])
    bins_rows: list[dict[str, object]] = []
    bins_diagnostics = []
    for requested in bins_values:
        current = make_construction(requested, decay_factor=decay)
        actual = len(current.anchors)
        if actual != requested:
            raise ValueError("configured bins must be exact shifted-grid cardinalities")
        current_population = exact_population(current)
        bins_diagnostics.append({
            "bins": actual,
            "temperature": current.temperature,
            "minimum_source_population_mass": float(np.min(current_population["source_mass"])),
            "minimum_target_population_mass": float(np.min(current_population["target_mass"])),
            "population_ecl": current_population["ecl"],
            "population_matched_canonical_ece": current_population["matched_canonical_ece"],
        })
        for seed in seeds:
            source_rng = np.random.default_rng(40_000_000 + 100_000 * actual + seed)
            target_rng = np.random.default_rng(50_000_000 + 100_000 * actual + seed)
            label_rng = np.random.default_rng(60_000_000 + 100_000 * actual + seed)
            source_counts = source_rng.multinomial(bins_n, current.source_probabilities)
            target_counts = target_rng.multinomial(bins_n, current.target_probabilities)
            label_counts = _sample_label_counts(target_counts, current.posterior, label_rng)
            estimate = estimate_from_counts(
                current,
                source_counts,
                target_counts,
                stabilizer=stabilizer,
                target_label_counts=label_counts,
            )
            bins_rows.append({
                "bins": actual,
                "sample_size": bins_n,
                "seed": seed,
                "ecl_error": abs(float(estimate["ecl"]) - float(current_population["ecl"])),
                "matched_oracle_ece_error": abs(float(estimate["matched_oracle_ece"]) - float(current_population["matched_canonical_ece"])),
                "matched_label_ece_error": abs(float(estimate["matched_label_ece"]) - float(current_population["matched_canonical_ece"])),
                "minimum_source_soft_count": float(np.min(estimate["source_soft_counts"])),
                "minimum_target_soft_count": float(np.min(estimate["target_soft_counts"])),
            })
    bins_summary = _summarize_axis(bins_rows, "bins", bins_values)
    for metric in bins_summary.values():
        slope = float(metric["rmse_log_slope_vs_bins"])
        metric["variance_proxy_log_slope_vs_bins"] = 2.0 * slope

    independent_n = 257
    independent_source = np.random.default_rng(70_000_001).multinomial(
        independent_n, construction.source_probabilities
    )
    independent_target = np.random.default_rng(70_000_002).multinomial(
        independent_n, construction.target_probabilities
    )
    vectorized = estimate_from_counts(
        construction, independent_source, independent_target, stabilizer=stabilizer
    )
    expanded = estimate_expanded_samples(
        construction, independent_source, independent_target, stabilizer=stabilizer
    )
    independent = {
        "sample_size": independent_n,
        "method_a": "atom-count contractions",
        "method_b": "expanded raw samples with assignments recomputed",
        "ecl_absolute_difference": abs(float(vectorized["ecl"]) - expanded["ecl"]),
        "matched_oracle_ece_absolute_difference": abs(float(vectorized["matched_oracle_ece"]) - expanded["matched_oracle_ece"]),
    }

    tiny = make_construction(
        baseline_bins,
        decay_factor=decay,
        temperature_scale=float(config["tiny_mass_temperature_scale"]),
        tiny_mass=True,
    )
    tiny_population = exact_population(tiny)
    tiny_sizes = [int(value) for value in config["tiny_mass_sample_sizes"]]
    tiny_seeds = seeds[: int(config["tiny_mass_replicates"])]
    tiny_rows = []
    for n in tiny_sizes:
        for seed in tiny_seeds:
            source_counts = np.random.default_rng(80_000_000 + 10_000 * n + seed).multinomial(
                n, tiny.source_probabilities
            )
            target_counts = np.random.default_rng(90_000_000 + 10_000 * n + seed).multinomial(
                n, tiny.target_probabilities
            )
            estimate = estimate_from_counts(
                tiny, source_counts, target_counts, stabilizer=stabilizer
            )
            tiny_rows.append({
                "sample_size": n,
                "seed": seed,
                "ecl_error": abs(float(estimate["ecl"]) - float(tiny_population["ecl"])),
                "minimum_source_soft_count": float(np.min(estimate["source_soft_counts"])),
                "minimum_target_soft_count": float(np.min(estimate["target_soft_counts"])),
                "bins_below_stabilizer_source": int(np.sum(estimate["source_soft_counts"] < stabilizer)),
                "bins_below_stabilizer_target": int(np.sum(estimate["target_soft_counts"] < stabilizer)),
            })
    tiny_rmse = []
    for n in tiny_sizes:
        errors = np.asarray([row["ecl_error"] for row in tiny_rows if row["sample_size"] == n])
        tiny_rmse.append(sqrt(float(np.mean(errors * errors))))
    tiny_slope, _ = _slope(tiny_sizes, tiny_rmse)

    zero_control_assignments = construction.assignments.copy()
    zero_control_assignments[:, -1] = 0.0
    zero_control_assignments /= zero_control_assignments.sum(axis=1, keepdims=True)
    zero_rejected = False
    zero_error = ""
    try:
        _domain_statistics(
            construction.source_probabilities,
            construction.posterior,
            zero_control_assignments,
        )
    except ValueError as error:
        zero_rejected = True
        zero_error = str(error)

    main_ecl_slope = float(sample_summary["ecl"]["asymptotic_tail_rmse_log_slope"])
    ece_slope = float(sample_summary["matched_label_ece"]["asymptotic_tail_rmse_log_slope"])
    bins_variance_slope = float(bins_summary["ecl"]["variance_proxy_log_slope_vs_bins"])
    assessment = (
        "verified_for_declared_fixed-function_well-conditioned-positive-mass_construction"
        if -0.62 <= main_ecl_slope <= -0.38
        and -0.65 <= ece_slope <= -0.35
        and bins_variance_slope <= 1.25
        and independent["ecl_absolute_difference"] < 1e-12
        else "inconclusive"
    )
    return {
        "assessment": assessment,
        "scope": {
            "estimator": "canonical differentiable soft-binning self-normalized Eq. 8",
            "scores_fixed_before_evaluation": True,
            "assignments_fixed_before_evaluation": True,
            "posterior_oracle_fixed_before_evaluation": True,
            "source_target_evaluation_sets_independent": True,
            "learned_posterior_same_data_case_covered": False,
            "population_reference": "exact enumeration over 231 latent atoms",
            "matched_ece": "canonical soft-binned ECE using both exact-oracle and sampled-label paths",
        },
        "baseline": {
            "requested_bins": baseline_bins,
            "actual_bins": len(construction.anchors),
            "temperature": construction.temperature,
            "stabilizer": stabilizer,
            "population_ecl": population["ecl"],
            "population_matched_canonical_ece": population["matched_canonical_ece"],
            "minimum_source_population_mass": float(np.min(population["source_mass"])),
            "minimum_target_population_mass": float(np.min(population["target_mass"])),
            "regularizer_bias_at_smallest_n": _regularizer_bias(construction, sample_sizes[0], stabilizer),
            "regularizer_bias_at_largest_n": _regularizer_bias(construction, sample_sizes[-1], stabilizer),
        },
        "sample_size_scaling": sample_summary,
        "bins_scaling": bins_summary,
        "bins_diagnostics": bins_diagnostics,
        "independent_calculation": independent,
        "tiny_mass_stress": {
            "assumptions_valid": True,
            "all_population_masses_positive": bool(
                np.all(tiny_population["source_mass"] > 0)
                and np.all(tiny_population["target_mass"] > 0)
            ),
            "temperature": tiny.temperature,
            "minimum_source_population_mass": float(np.min(tiny_population["source_mass"])),
            "minimum_target_population_mass": float(np.min(tiny_population["target_mass"])),
            "rmse": dict(zip(map(str, tiny_sizes), tiny_rmse)),
            "rmse_log_slope_vs_sample_size": tiny_slope,
            "interpretation": "valid but deliberately ill-conditioned; finite-n behavior is reported, not used to overclaim a mass-uniform constant",
            "rows": tiny_rows,
        },
        "zero_mass_control": {
            "assumptions_valid": False,
            "construction": "one soft-assignment column manually masked to zero; impossible for exact Eq. 6 at finite temperature but represents a zero-mass denominator",
            "rejected": zero_rejected,
            "error": zero_error,
        },
        "sample_rows": sample_rows,
        "bins_rows": bins_rows,
        "limitations": [
            "The experiment supports scaling for a fixed exact posterior oracle; it does not cover a learned posterior evaluated on its own training data.",
            "Finite-grid experiments cannot prove a universal theorem or identify the paper's unspecified absolute constant C.",
            "Eq. 8's fixed stabilizer creates a small deterministic O(1/n) bias relative to the unregularized exact population reference.",
            "Positive mass alone does not give useful finite-n constants; the tiny-mass stress is intentionally ill-conditioned and is reported separately.",
            "Actual simplex-grid bin counts use cardinalities 3, 6, 10, 15, 21, and 28 to match the official anchor rule exactly.",
        ],
    }
