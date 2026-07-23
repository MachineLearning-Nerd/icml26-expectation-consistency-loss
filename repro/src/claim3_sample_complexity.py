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
from hashlib import sha256
from math import comb, log, sqrt
from pathlib import Path
from typing import Iterable

import numpy as np

# The independent Hilbert-space route first obtains one source and one target
# radius.  Adding them and applying sqrt(a)+sqrt(b) <= sqrt(2(a+b)) costs this
# universal normalized factor.  It is not a paper-supplied literal constant.
VECTOR_CONCENTRATION_FACTOR = float(sqrt(2.0))
OFFICIAL_LOSSES_SHA256 = (
    "1c2de34967f34b98faae5025368edac88f46a709d6e1e0c063e2c01f4d6e9754"
)
OFFICIAL_CODE_COMMIT = "aae77f890f1e4ebc13dad135b5e29758d98d318d"


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
        target_weights = target_counts / target_counts.sum()
        if not np.allclose(weights, target_weights, atol=1e-12, rtol=0.0):
            raise ValueError("Eq. 5 weights must equal target_counts / target_counts.sum()")


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


def target_bin_mass_radius(
    plan: BinPlan, *, bins: int, classes: int, delta: float
) -> float:
    """Normalized hard-bin target-mass term omitted by Appendix Eq. 30.

    Eq. 4 integrates with population target-bin masses, whereas hard Eq. 5
    uses empirical target proportions.  For a fixed partition and fixed bin
    distances in [0, sqrt(2)], scalar Hoeffding controls their discrepancy at
    this order.  Because w_j=n_tj/N_t and all bins are nonempty, this term is
    no larger than the target-count part of displayed Eq. 9, up to an absolute
    constant; it changes the proof and constants, but not the displayed rate.
    """
    _validate_radius_inputs(plan, bins=bins, classes=classes, delta=delta)
    total_target = float(np.sum(plan.target_counts))
    return float(sqrt(log(2.0 * bins * classes / delta) / total_target))


def proof_audit(classes_grid: Iterable[int] = (2, 3, 10, 50)) -> dict[str, object]:
    """Classify the printed proof separately from the hard-bin rate statement."""
    classes = [int(value) for value in classes_grid]
    if not classes or any(value < 2 for value in classes):
        raise ValueError("classes_grid must contain integers >= 2")

    # A self-contained route for hard Eq. 5 is available: bounded differences
    # plus E||mean-E mean|| <= sqrt(E||mean-E mean||^2) gives a dimension-free
    # O(sqrt(log(1/eta)/n)) vector-mean radius.  Reverse triangle inequality,
    # a union bound over bins/domains, and weighted Cauchy-Schwarz give the
    # conditional-mean part.  Scalar Hoeffding supplies the empirical target
    # bin-mass term omitted in Eq. 30; target_bin_mass_radius() shows it is
    # absorbed by Eq. 9's order when w_j=n_tj/N_t and bins are nonempty.
    hard_bin_support = True
    findings = [
        "displayed_bound_not_derived_by_appendix_proof",
        "appendix_omits_empirical_target_bin_mass_term",
        "hard_bin_displayed_order_supported_by_independent_bounded_differences_argument",
        "soft_self_normalized_eq8_not_established",
    ]

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
        "independent_vector_argument_supports_displayed_order": hard_bin_support,
        "hard_bin_eq5_assessment": "supported",
        "soft_eq8_assessment": "inconclusive",
        "theorem_statement_assessment": "supported_for_fixed_hard_bin_eq5",
        "appendix_proof_assessment": "missing_sqrt_K_and_bin_mass_terms",
        "findings": findings,
        "independent_derivation": {
            "method": "bounded differences plus second-moment mean bound; no coordinate union",
            "conditional_mean_term": "dimension-free and absorbed by displayed Eq. 9 up to an absolute constant",
            "target_bin_mass_term": "scalar Hoeffding; omitted by Appendix Eq. 30 but absorbed by the target-count part of Eq. 9",
            "changes_asymptotic_rate": False,
        },
        "required_conditions": [
            "fixed hard-bin partition",
            "all reported hard bins have positive source and target counts",
            "bounded posterior-vector function fixed independently of the evaluation samples, or analysis conditional on an independently learned function",
            "iid samples within each domain before conditioning on hard-bin counts",
        ],
        "radius_semantics": {
            "label": "normalized_radius",
            "absolute_constant_known": False,
            "literal_coverage_claimed": False,
            "paper_constants": ["C", "C1", "C2"],
            "vector_route_normalized_factor": VECTOR_CONCENTRATION_FACTOR,
        },
        "limitations": [
            "The paper does not state numerical values for C, C1, or C2.",
            "The independent hard-bin argument supports the displayed order but is not the proof printed in Appendix G.",
            "The paper's soft self-normalized Eq. 8 analog is not established by this audit.",
            "Using the same observations to learn P-hat(Y|X) and evaluate Eq. 5 requires stability or sample-splitting analysis absent from the theorem and this audit.",
            "Finite simulations cannot prove a universal concentration theorem.",
        ],
    }


def _validate_simplex_rows(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"{name} must have shape (bins, classes>=2)")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    if not np.allclose(array.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} rows must sum to one")
    return array


def population_ecl(
    source_means: np.ndarray, target_means: np.ndarray, weights: np.ndarray
) -> float:
    """Canonical population ECL from Eq. 4 on a fixed bin partition."""
    source = _validate_simplex_rows(source_means, name="source_means")
    target = _validate_simplex_rows(target_means, name="target_means")
    weights_array = np.asarray(weights, dtype=np.float64)
    if source.shape != target.shape or weights_array.shape != (source.shape[0],):
        raise ValueError("source, target, and weights must have aligned bin dimensions")
    if np.any(weights_array < 0) or not np.isclose(weights_array.sum(), 1.0):
        raise ValueError("weights must be nonnegative and sum to one")
    return float(np.sum(weights_array * np.linalg.norm(source - target, axis=1)))


def empirical_ecl(
    source_vectors: list[np.ndarray],
    target_vectors: list[np.ndarray],
    plan: BinPlan,
) -> float:
    """Canonical empirical ECL from Eq. 5 using per-bin sample means."""
    if not (len(source_vectors) == len(target_vectors) == len(plan.weights)):
        raise ValueError("sample lists must match the BinPlan length")
    source_means = []
    target_means = []
    for index, (source_bin, target_bin) in enumerate(zip(source_vectors, target_vectors)):
        source = _validate_simplex_rows(source_bin, name=f"source_vectors[{index}]")
        target = _validate_simplex_rows(target_bin, name=f"target_vectors[{index}]")
        if source.shape[1] != target.shape[1]:
            raise ValueError("source and target class dimensions must match")
        if len(source) != int(plan.source_counts[index]):
            raise ValueError("source sample count does not match BinPlan")
        if len(target) != int(plan.target_counts[index]):
            raise ValueError("target sample count does not match BinPlan")
        source_means.append(source.mean(axis=0))
        target_means.append(target.mean(axis=0))
    return population_ecl(
        np.asarray(source_means), np.asarray(target_means), plan.weights
    )


def population_histogram_ece(
    correctness_probabilities: np.ndarray,
    confidence_anchors: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Matched scalar histogram ECE population quantity on target bins."""
    probabilities = np.asarray(correctness_probabilities, dtype=np.float64)
    anchors = np.asarray(confidence_anchors, dtype=np.float64)
    weights_array = np.asarray(weights, dtype=np.float64)
    if not (probabilities.shape == anchors.shape == weights_array.shape):
        raise ValueError("ECE inputs must have matching one-dimensional shapes")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("correctness probabilities must lie in [0,1]")
    if np.any((anchors < 0) | (anchors > 1)):
        raise ValueError("confidence anchors must lie in [0,1]")
    if np.any(weights_array < 0) or not np.isclose(weights_array.sum(), 1.0):
        raise ValueError("weights must be nonnegative and sum to one")
    return float(np.sum(weights_array * np.abs(probabilities - anchors)))


def empirical_histogram_ece(
    correct_counts: np.ndarray,
    target_counts: np.ndarray,
    confidence_anchors: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Matched scalar histogram ECE using fixed target-bin weights."""
    correct = np.asarray(correct_counts, dtype=np.float64)
    counts = np.asarray(target_counts, dtype=np.float64)
    if correct.shape != counts.shape or np.any(counts <= 0):
        raise ValueError("correct and target counts must align and counts must be positive")
    if np.any(correct < 0) or np.any(correct > counts):
        raise ValueError("correct counts must lie between zero and target counts")
    return population_histogram_ece(correct / counts, confidence_anchors, weights)


def make_plan_family(family: str, *, bins: int, count: int) -> BinPlan:
    """Construct Eq.-5-consistent balanced/skewed/sparse/imbalanced plans."""
    if bins < 2 or count < 1:
        raise ValueError("bins must be >=2 and count must be positive")
    if family == "balanced":
        source = np.full(bins, count)
        target = np.full(bins, count)
    elif family == "skewed":
        target_profile = np.geomspace(1.0, 0.05, bins)
        target = np.maximum(
            1, np.rint(count * bins * target_profile / target_profile.sum())
        ).astype(int)
        source = np.full(bins, count)
    elif family == "sparse_valid":
        target_profile = np.geomspace(1.0, 0.01, bins)
        target = np.maximum(
            1, np.rint(count * bins * target_profile / target_profile.sum())
        ).astype(int)
        source = np.maximum(1, target[::-1]).astype(int)
    elif family == "source_target_imbalanced":
        target = np.array(
            [count if index % 2 == 0 else max(1, count // 4) for index in range(bins)]
        )
        source = np.array(
            [max(1, count // 3) if index % 2 == 0 else count for index in range(bins)]
        )
    else:
        raise ValueError(f"unknown plan family: {family}")
    weights = target.astype(np.float64) / target.sum()
    return BinPlan(weights=weights, source_counts=source, target_counts=target)


def generate_covariate_shift_case(
    *, plan: BinPlan, classes: int, seed: int, violate_shared_conditional: bool = False
) -> dict[str, object]:
    """Generate bin means realizable by shared conditional atoms.

    Each bin has two latent X atoms with fixed P(Y|X) vectors. Source and target
    use different mixture weights over those same atoms, so P(Y|X) is shared
    while P(X|bin) changes. The negative control redraws target atoms.
    """
    if classes < 2:
        raise ValueError("classes must be >=2")
    rng = np.random.default_rng(seed)
    atoms = rng.dirichlet(np.ones(classes), size=(len(plan.weights), 2))
    target_atoms = (
        rng.dirichlet(np.ones(classes), size=(len(plan.weights), 2))
        if violate_shared_conditional
        else atoms
    )
    source_mix = np.linspace(0.2, 0.45, len(plan.weights))
    target_mix = np.linspace(0.8, 0.55, len(plan.weights))
    source_means = (
        source_mix[:, None] * atoms[:, 0]
        + (1.0 - source_mix[:, None]) * atoms[:, 1]
    )
    target_means = (
        target_mix[:, None] * target_atoms[:, 0]
        + (1.0 - target_mix[:, None]) * target_atoms[:, 1]
    )
    # A realizable conventional top-label confidence schedule: class zero is
    # predicted in every bin and has confidence strictly above 1/2.
    anchors = 0.5 + 0.5 * (
        (np.arange(len(plan.weights), dtype=float) + 0.5) / len(plan.weights)
    )
    confidence_vectors = np.repeat(
        ((1.0 - anchors) / (classes - 1))[:, None],
        classes,
        axis=1,
    )
    confidence_vectors[:, 0] = anchors
    return {
        "posterior_atoms": atoms,
        "target_posterior_atoms": target_atoms,
        "source_atom_mix": source_mix,
        "target_atom_mix": target_mix,
        "source_means": source_means,
        "target_means": target_means,
        "confidence_anchors": anchors,
        "confidence_vectors": confidence_vectors,
        "assumptions_valid": not violate_shared_conditional,
        "shared_conditional_atoms": not violate_shared_conditional,
        "construction": "two shared posterior-vector atoms per fixed confidence bin with domain-specific latent-X mixture weights",
    }


def _sample_posterior_vectors(
    posterior_atoms: np.ndarray,
    atom_mix: np.ndarray,
    counts: np.ndarray,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Draw latent X atoms and return their unchanged P(Y|X) vectors."""
    atoms = np.asarray(posterior_atoms, dtype=np.float64)
    mix = np.asarray(atom_mix, dtype=np.float64)
    if atoms.ndim != 3 or atoms.shape[1] != 2 or mix.shape != (atoms.shape[0],):
        raise ValueError("posterior atoms must be (bins,2,K) with one mixture per bin")
    vectors = []
    for bin_atoms, probability, count_value in zip(atoms, mix, counts):
        count = int(count_value)
        # atom_mix is the probability of atom zero, matching the population
        # means in generate_covariate_shift_case().
        indices = 1 - rng.binomial(1, probability, size=count)
        vectors.append(bin_atoms[indices].copy())
    return vectors


def _sample_ece_correct_counts(
    correctness_probabilities: np.ndarray,
    counts: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Separate Bernoulli-label sampler for matched scalar histogram ECE."""
    probabilities = np.asarray(correctness_probabilities, dtype=np.float64)
    count_array = np.asarray(counts, dtype=np.int64)
    if probabilities.shape != count_array.shape:
        raise ValueError("ECE probabilities and counts must align")
    if np.any((probabilities < 0) | (probabilities > 1)) or np.any(count_array <= 0):
        raise ValueError("ECE probabilities/counts are invalid")
    return rng.binomial(count_array, probabilities)


def exact_binary_tail(
    *, source_probability: float, target_probability: float, source_count: int,
    target_count: int, threshold: float
) -> dict[str, float]:
    """Exhaust exact K=2 one-bin deviation distribution for counts <=12."""
    if not 0 <= source_probability <= 1 or not 0 <= target_probability <= 1:
        raise ValueError("probabilities must lie in [0,1]")
    if not 1 <= source_count <= 12 or not 1 <= target_count <= 12:
        raise ValueError("exact enumeration counts must lie in [1,12]")
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    population = sqrt(2.0) * abs(source_probability - target_probability)
    mass = 0.0
    tail = 0.0
    expected_deviation = 0.0
    for source_success in range(source_count + 1):
        ps = comb(source_count, source_success) * source_probability**source_success * (1-source_probability)**(source_count-source_success)
        for target_success in range(target_count + 1):
            pt = comb(target_count, target_success) * target_probability**target_success * (1-target_probability)**(target_count-target_success)
            probability = ps * pt
            empirical = sqrt(2.0) * abs(
                source_success/source_count - target_success/target_count
            )
            deviation = abs(empirical - population)
            mass += probability
            expected_deviation += probability * deviation
            if deviation > threshold:
                tail += probability
    return {
        "probability_mass": float(mass),
        "tail_probability": float(tail),
        "expected_deviation": float(expected_deviation),
        "population_ecl": float(population),
        "threshold": float(threshold),
    }


def official_code_audit(path: Path) -> dict[str, object]:
    """Audit whether pinned ECLossMiniBatch directly computes Eq. 5/8."""
    data = path.read_bytes()
    text = data.decode("utf-8")
    required = [
        "class ECLossMiniBatch",
        'elif self.calibration_paradigm == "Canonical":',
        "n_s_batch = w_s.sum(dim=0)",
        "n_t_batch = w_t.sum(dim=0)",
        "m_s_batch = (w_s.unsqueeze(2)",
        "m_t_batch = (w_t.unsqueeze(2)",
        "loss_s_j = (w_s[:, j]",
        "loss_t_j = (w_t[:, j]",
        "return loss/num_classes",
    ]
    missing = [marker for marker in required if marker not in text]
    digest = sha256(data).hexdigest()
    return {
        "repository": "NeuroDong/ECL",
        "commit": OFFICIAL_CODE_COMMIT,
        "sha256": digest,
        "expected_sha256": OFFICIAL_LOSSES_SHA256,
        "source_pin_matches": digest == OFFICIAL_LOSSES_SHA256,
        "required_markers_present": not missing,
        "missing_markers": missing,
        "per_bin_mean_statistics_match_eq7_components": not missing,
        "direct_eq5_loss_parity_supported": False,
        "semantic_assessment": (
            "Official ECLossMiniBatch uses Eq.10 auxiliary/proximal trainable loss; "
            "its canonical per-bin n/m statistics correspond to the soft Eq.7/8 "
            "components, but its returned training objective is not the direct "
            "Theorem-3.2 Eq.5/8 empirical estimator."
        ),
    }


def _coverage_settings(config: dict[str, object]) -> list[dict[str, object]]:
    """Deterministic non-Cartesian design covering every configured value."""
    base = {
        "bins": config["bins"][0],
        "classes": config["classes"][0],
        "delta": config["deltas"][0],
        "count": config["per_domain_counts"][0],
        "family": config["plan_families"][0],
    }
    settings = [base]
    for key, config_key in [
        ("bins", "bins"), ("classes", "classes"), ("delta", "deltas"),
        ("count", "per_domain_counts"), ("family", "plan_families")
    ]:
        for value in config[config_key]:
            item = dict(base)
            item[key] = value
            settings.append(item)
    unique = []
    seen = set()
    for item in settings:
        key = tuple(item[name] for name in ("bins", "classes", "delta", "count", "family"))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _log_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(np.log(np.asarray(x, dtype=float)), np.log(np.asarray(y, dtype=float)), 1)[0])


def _executed_sample_scaling(rows: list[dict[str, object]]) -> dict[str, object]:
    """Fit actually observed errors against per-bin count on a fixed design."""
    candidates = [
        row for row in rows
        if row["family"] == "balanced"
        and row["bins"] == 2
        and row["classes"] == 2
        and row["delta"] == 0.1
    ]
    counts = sorted({int(row["count"]) for row in candidates})
    if len(counts) < 3:
        raise ValueError("executed scaling requires at least three sample counts")

    result: dict[str, object] = {
        "design": "fixed B=2, K=2, delta=0.1, balanced hard bins; errors aggregated over executed seeds",
        "per_bin_counts": counts,
        "expected_error_slope": -0.5,
        "expected_sample_complexity_exponent": 2.0,
    }
    for output_name, row_name in [
        ("ecl", "ecl_deviation"),
        ("histogram_ece", "matched_ece_deviation"),
    ]:
        grouped = {
            count: np.asarray(
                [float(row[row_name]) for row in candidates if int(row["count"]) == count]
            )
            for count in counts
        }
        rmse = np.asarray(
            [sqrt(float(np.mean(np.square(grouped[count])))) for count in counts]
        )
        q90 = np.asarray([float(np.quantile(grouped[count], 0.9)) for count in counts])
        rmse_slope = _log_slope(np.asarray(counts), rmse)
        q90_slope = _log_slope(np.asarray(counts), q90)
        result[output_name] = {
            "rmse_by_count": dict(zip(map(str, counts), map(float, rmse))),
            "q90_by_count": dict(zip(map(str, counts), map(float, q90))),
            "rmse_log_slope_vs_count": rmse_slope,
            "q90_log_slope_vs_count": q90_slope,
            "implied_sample_complexity_exponent_from_rmse": float(-1.0 / rmse_slope),
        }
    return result


def _executed_target_mass_scaling(config: dict[str, object]) -> dict[str, object]:
    """Execute the empirical-target-mass term absent from Appendix Eq. 30."""
    bins = 4
    classes = 2
    delta = 0.05
    population_weights = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    # Realizable distances between binary simplex means, kept fixed so this
    # experiment isolates only target-bin mass estimation.
    bin_distances = sqrt(2.0) * np.array([0.0, 0.25, 0.5, 0.75])
    totals = [bins * int(value) for value in config["per_domain_counts"]]
    seeds = [int(value) for value in config["seeds"]]
    population = float(np.dot(population_weights, bin_distances))
    rows = []
    for total in totals:
        for seed in seeds:
            counts = np.random.default_rng(9_000_000 + total * 1000 + seed).multinomial(
                total, population_weights
            )
            empirical_weights = counts / total
            error = abs(float(np.dot(empirical_weights, bin_distances)) - population)
            radius = sqrt(log(2.0 * bins * classes / delta) / total)
            rows.append({
                "total_target_count": total,
                "seed": seed,
                "target_counts": counts.astype(int).tolist(),
                "population_ecl": population,
                "empirical_weight_only_ecl": float(np.dot(empirical_weights, bin_distances)),
                "absolute_error": error,
                "normalized_mass_radius": radius,
            })
    rmse = []
    q90 = []
    for total in totals:
        errors = np.asarray(
            [row["absolute_error"] for row in rows if row["total_target_count"] == total]
        )
        rmse.append(sqrt(float(np.mean(np.square(errors)))))
        q90.append(float(np.quantile(errors, 0.9)))
    return {
        "design": "four fixed hard bins; multinomial target counts; fixed realizable bin distances isolate empirical target-mass error",
        "population_weights": population_weights.tolist(),
        "bin_distances": bin_distances.tolist(),
        "total_target_counts": totals,
        "seeds_per_count": len(seeds),
        "rmse_by_total_count": dict(zip(map(str, totals), rmse)),
        "q90_by_total_count": dict(zip(map(str, totals), q90)),
        "rmse_log_slope_vs_total_count": _log_slope(np.asarray(totals), np.asarray(rmse)),
        "q90_log_slope_vs_total_count": _log_slope(np.asarray(totals), np.asarray(q90)),
        "max_error_over_normalized_mass_radius": max(
            row["absolute_error"] / row["normalized_mass_radius"] for row in rows
        ),
        "rows": rows,
    }


def run_experiment(config: dict[str, object], *, official_source: Path) -> dict[str, object]:
    """Run deterministic coverage design across every configured axis/value."""
    settings = _coverage_settings(config)
    seeds = [int(seed) for seed in config["seeds"]]
    rows = []
    for setting_index, setting in enumerate(settings):
        bins = int(setting["bins"])
        classes = int(setting["classes"])
        delta = float(setting["delta"])
        count = int(setting["count"])
        family = str(setting["family"])
        plan = make_plan_family(family, bins=bins, count=count)
        for seed in seeds:
            case = generate_covariate_shift_case(plan=plan, classes=classes, seed=seed)
            stream_seed = 1_000_000 * setting_index + seed
            source_rng = np.random.default_rng(stream_seed)
            target_rng = np.random.default_rng(stream_seed + 100_000_000)
            ece_rng = np.random.default_rng(stream_seed + 200_000_000)
            source_vectors = _sample_posterior_vectors(
                case["posterior_atoms"],
                case["source_atom_mix"],
                plan.source_counts,
                source_rng,
            )
            target_vectors = _sample_posterior_vectors(
                case["target_posterior_atoms"],
                case["target_atom_mix"],
                plan.target_counts,
                target_rng,
            )
            pop_ecl = population_ecl(case["source_means"], case["target_means"], plan.weights)
            emp_ecl = empirical_ecl(source_vectors, target_vectors, plan)
            target_correct_prob = np.asarray(case["target_means"])[:, 0]
            target_correct_counts = _sample_ece_correct_counts(
                target_correct_prob, plan.target_counts, ece_rng
            )
            pop_ece = population_histogram_ece(
                target_correct_prob, case["confidence_anchors"], plan.weights
            )
            emp_ece = empirical_histogram_ece(
                target_correct_counts, plan.target_counts,
                case["confidence_anchors"], plan.weights
            )
            weights_match = bool(
                np.allclose(
                    plan.weights,
                    plan.target_counts / plan.target_counts.sum(),
                    atol=1e-12,
                    rtol=0.0,
                )
            )
            if not weights_match or not case["shared_conditional_atoms"]:
                raise AssertionError("valid theorem rows must satisfy Eq. 5 weights and shared atoms")
            rows.append({
                "setting_index": setting_index,
                "seed": seed,
                "bins": bins,
                "classes": classes,
                "delta": delta,
                "count": count,
                "family": family,
                "source_counts": plan.source_counts.astype(int).tolist(),
                "target_counts": plan.target_counts.astype(int).tolist(),
                "assumptions_valid": True,
                "weights_match_target_counts": weights_match,
                "ecl_samples_are_posterior_vectors": True,
                "posterior_function_fixed_before_sampling": True,
                "hard_bin_partition_fixed_before_sampling": True,
                "shared_conditional_atoms": bool(case["shared_conditional_atoms"]),
                "population_ecl": pop_ecl,
                "empirical_ecl": emp_ecl,
                "ecl_deviation": abs(emp_ecl-pop_ecl),
                "displayed_normalized_radius": displayed_radius(plan, bins=bins, classes=classes, delta=delta),
                "appendix_coordinate_normalized_radius": appendix_coordinate_radius(plan, bins=bins, classes=classes, delta=delta),
                "vector_normalized_radius": vector_concentration_radius(plan, bins=bins, classes=classes, delta=delta),
                "target_bin_mass_normalized_radius": target_bin_mass_radius(plan, bins=bins, classes=classes, delta=delta),
                "population_matched_ece": pop_ece,
                "empirical_matched_ece": emp_ece,
                "matched_ece_deviation": abs(emp_ece-pop_ece),
            })

    exact_plan = BinPlan(np.array([1.0]), np.array([12]), np.array([12]))
    exact = exact_binary_tail(
        source_probability=0.25, target_probability=0.75,
        source_count=12, target_count=12,
        threshold=displayed_radius(exact_plan, bins=1, classes=2, delta=0.05),
    )
    exact["construction"] = (
        "valid K=2 latent-X mixture over deterministic posterior atoms e1=(1,0) "
        "and e2=(0,1); binomial outcomes count sampled posterior atoms, not labels"
    )

    control_plan = make_plan_family("skewed", bins=4, count=25)
    valid_control_case = generate_covariate_shift_case(
        plan=control_plan, classes=3, seed=991
    )
    violated_case = generate_covariate_shift_case(
        plan=control_plan,
        classes=3,
        seed=991,
        violate_shared_conditional=True,
    )
    correct_weight_ecl = population_ecl(
        valid_control_case["source_means"],
        valid_control_case["target_means"],
        control_plan.weights,
    )
    uniform_weight_ecl = population_ecl(
        valid_control_case["source_means"],
        valid_control_case["target_means"],
        np.full(4, 0.25),
    )
    violated_population_ecl = population_ecl(
        violated_case["source_means"],
        violated_case["target_means"],
        control_plan.weights,
    )

    rejection_results = {}
    for name, source_counts, target_counts in [
        ("source_samples_omitted", np.array([0, 10]), np.array([10, 10])),
        ("zero_count_bin", np.array([10, 10]), np.array([10, 0])),
    ]:
        try:
            BinPlan(np.array([0.5, 0.5]), source_counts, target_counts)
        except ValueError as error:
            rejection_results[name] = {"rejected": True, "error": str(error)}
        else:
            rejection_results[name] = {"rejected": False, "error": None}

    low_plan = make_plan_family("balanced", bins=2, count=2)
    low_case = generate_covariate_shift_case(plan=low_plan, classes=2, seed=992)
    low_source = _sample_posterior_vectors(
        low_case["posterior_atoms"],
        low_case["source_atom_mix"],
        low_plan.source_counts,
        np.random.default_rng(992),
    )
    low_target = _sample_posterior_vectors(
        low_case["target_posterior_atoms"],
        low_case["target_atom_mix"],
        low_plan.target_counts,
        np.random.default_rng(100_000_992),
    )
    low_deviation = abs(
        empirical_ecl(low_source, low_target, low_plan)
        - population_ecl(
            low_case["source_means"], low_case["target_means"], low_plan.weights
        )
    )
    low_radius = displayed_radius(low_plan, bins=2, classes=2, delta=0.05)

    sparse_plan = make_plan_family("sparse_valid", bins=8, count=25)
    controls = [
        {
            "name": "below_threshold_counts",
            "assumptions_valid": True,
            "outcome": "executed two samples per bin; supporting stress test only because C is unknown",
            "ecl_deviation": low_deviation,
            "displayed_normalized_radius": low_radius,
        },
        {
            "name": "source_samples_omitted",
            "assumptions_valid": False,
            "outcome": "executed construction was rejected by BinPlan",
            **rejection_results["source_samples_omitted"],
        },
        {
            "name": "shared_conditional_violated",
            "assumptions_valid": False,
            "outcome": "executed with independent target conditional atoms and excluded from theorem rows",
            "valid_case_population_ecl": correct_weight_ecl,
            "violated_case_population_ecl": violated_population_ecl,
        },
        {
            "name": "zero_count_bin",
            "assumptions_valid": False,
            "outcome": "executed construction was rejected by BinPlan",
            **rejection_results["zero_count_bin"],
        },
        {
            "name": "uniform_weights_substitution",
            "assumptions_valid": False,
            "outcome": "executed and changed the estimand for nonuniform target weights",
            "correct_weight_ecl": correct_weight_ecl,
            "uniform_weight_ecl": uniform_weight_ecl,
            "absolute_change": abs(correct_weight_ecl - uniform_weight_ecl),
        },
        {
            "name": "sparse_positive_counts",
            "assumptions_valid": True,
            "outcome": "executed as a valid positive-count plan and included in theorem rows",
            "minimum_source_count": int(sparse_plan.source_counts.min()),
            "minimum_target_count": int(sparse_plan.target_counts.min()),
        },
    ]

    epsilons = np.asarray(config.get("epsilons", [0.05, 0.1, 0.2, 0.4]), dtype=float)
    b_values = np.asarray(config["bins"], dtype=float)
    k_values = np.asarray(config["classes"], dtype=float)
    delta = 0.05
    epsilon_budget = 2.0 * 8.0 * np.log(2.0*8.0*3.0/delta) / epsilons**2
    b_budget = 2.0 * b_values * np.log(2.0*b_values*3.0/delta) / 0.1**2
    k_budget = 2.0 * 8.0 * np.log(2.0*8.0*k_values/delta) / 0.1**2
    slopes = {
        "epsilon_inverse_square_log_slope": _log_slope(epsilons**-2, epsilon_budget),
        "bins_log_slope_with_log_factor": _log_slope(b_values, b_budget),
        "classes_log_slope_only": _log_slope(k_values, k_budget),
        "label": "formula_derived_identity_not_executed_estimator_evidence",
    }

    executed_scaling = _executed_sample_scaling(rows)
    target_mass_scaling = _executed_target_mass_scaling(config)

    coverage = {
        "settings": settings,
        "setting_count": len(settings),
        "seeds_per_setting": len(seeds),
        "row_count": len(rows),
        "covered_values": {
            "bins": sorted({row["bins"] for row in rows}),
            "classes": sorted({row["classes"] for row in rows}),
            "deltas": sorted({row["delta"] for row in rows}),
            "counts": sorted({row["count"] for row in rows}),
            "families": sorted({row["family"] for row in rows}),
            "seeds": sorted({row["seed"] for row in rows}),
        },
        "design": "deterministic axis-covering non-Cartesian design; every configured value appears and all 40 seeds run for every selected setting",
    }
    max_ecl_ratio = max(row["ecl_deviation"]/row["displayed_normalized_radius"] for row in rows)
    max_ece_ratio = max(row["matched_ece_deviation"]/row["displayed_normalized_radius"] for row in rows)
    return {
        "proof_audit": proof_audit(config["classes"]),
        "coverage": coverage,
        "summary": {
            "max_ecl_deviation_over_displayed_normalized_radius": max_ecl_ratio,
            "max_matched_ece_deviation_over_displayed_normalized_radius": max_ece_ratio,
            "literal_coverage_claimed": False,
            "valid_theorem_rows": len(rows),
        },
        "exact_binary": exact,
        "controls": controls,
        "official_code_audit": official_code_audit(official_source),
        "executed_sample_scaling": executed_scaling,
        "executed_target_mass_scaling": target_mass_scaling,
        "formula_derived_slopes": slopes,
        "rows": rows,
        "limitations": [
            "All radii are normalized because the paper does not state C, C1, or C2.",
            "The deterministic grid supports scaling and implementation checks but cannot prove the universal theorem.",
            "Matched ECE is conventional scalar top-label histogram ECE; canonical ECL is vector-valued and the semantics are kept separate.",
            "The hard-bin Eq. 5 rate is supported; the soft self-normalized Eq. 8 analog is not verified.",
            "The executed rows use a fixed exact posterior oracle. A learned P-hat evaluated on its training observations needs stability or sample splitting not analyzed here.",
        ],
    }
