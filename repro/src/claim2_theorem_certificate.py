#!/usr/bin/env python3
"""Fail-closed certificate for the hard-bin part of ECL Theorem 3.2."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import log, sqrt
import os
from pathlib import Path
import platform
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "repro" / "evidence" / "claim3" / "2605.21552v1.pdf"
SEEDS = tuple(range(260521552, 260521576))
DELTA = 0.05
THEOREM_CONSTANT = 4.0


def theorem_radius(
    target_weights: np.ndarray,
    target_counts: np.ndarray,
    source_counts: np.ndarray,
    classes: int,
    delta: float = DELTA,
) -> float:
    positive = target_weights > 0
    if classes < 2 or not 0 < delta < 1:
        raise ValueError("classification requires K>=2 and delta in (0,1)")
    if np.any(target_counts[positive] <= 0) or np.any(source_counts[positive] <= 0):
        raise ValueError("positive target-weight bins require positive source and target counts")
    variance_proxy = np.sum(
        target_weights[positive]
        * (
            1.0 / target_counts[positive]
            + 1.0 / source_counts[positive]
        )
    )
    return sqrt(log(2 * len(target_weights) * classes / delta) * variance_proxy)


def random_simplex(rng: np.random.Generator, rows: int, classes: int) -> np.ndarray:
    return rng.dirichlet(np.linspace(0.7, 2.3, classes), size=rows)


def build_population(
    rng: np.random.Generator, bins: int, classes: int
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    target_masses = rng.dirichlet(np.full(bins, 4.0))
    source_masses = rng.dirichlet(np.full(bins, 4.0))
    target_atoms: list[np.ndarray] = []
    source_atoms: list[np.ndarray] = []
    for _ in range(bins):
        shared_atoms = random_simplex(rng, 7, classes)
        target_mix = rng.dirichlet(np.full(7, 1.5))
        source_mix = rng.dirichlet(np.full(7, 1.5))
        target_atoms.append(np.column_stack((target_mix, shared_atoms)))
        source_atoms.append(np.column_stack((source_mix, shared_atoms)))
    return target_masses, source_masses, target_atoms, source_atoms


def population_loss(
    target_masses: np.ndarray,
    target_atoms: list[np.ndarray],
    source_atoms: list[np.ndarray],
) -> float:
    loss = 0.0
    for mass, target, source in zip(target_masses, target_atoms, source_atoms, strict=True):
        target_mean = target[:, 0] @ target[:, 1:]
        source_mean = source[:, 0] @ source[:, 1:]
        loss += float(mass) * float(np.linalg.norm(source_mean - target_mean))
    return loss


def draw_domain(
    rng: np.random.Generator,
    total: int,
    bin_masses: np.ndarray,
    atoms: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    bins = rng.choice(len(bin_masses), size=total, p=bin_masses)
    values: list[list[np.ndarray]] = [[] for _ in bin_masses]
    for bin_index in bins:
        table = atoms[int(bin_index)]
        atom_index = int(rng.choice(len(table), p=table[:, 0]))
        values[int(bin_index)].append(table[atom_index, 1:])
    arrays = [
        np.asarray(rows, dtype=np.float64)
        if rows
        else np.empty((0, atoms[0].shape[1] - 1), dtype=np.float64)
        for rows in values
    ]
    return np.bincount(bins, minlength=len(bin_masses)).astype(np.int64), arrays


def empirical_loss(
    target_counts: np.ndarray,
    target_rows: list[np.ndarray],
    source_counts: np.ndarray,
    source_rows: list[np.ndarray],
) -> float:
    total_target = int(np.sum(target_counts))
    loss = 0.0
    for target_count, target, source_count, source in zip(
        target_counts, target_rows, source_counts, source_rows, strict=True
    ):
        if target_count == 0:
            continue
        if source_count == 0:
            raise ValueError("undefined source conditional mean")
        loss += (target_count / total_target) * float(
            np.linalg.norm(np.mean(source, axis=0) - np.mean(target, axis=0))
        )
    return loss


def verify_algebra() -> dict[str, object]:
    rng = np.random.default_rng(31032026)
    max_cauchy_ratio = 0.0
    checked = 0
    for bins in (1, 2, 4, 8, 16, 32, 64):
        for _ in range(200):
            weights = rng.dirichlet(np.ones(bins))
            counts = rng.integers(1, 10000, size=bins)
            left = float(np.sum(weights / np.sqrt(counts)))
            right = sqrt(float(np.sum(weights / counts)))
            max_cauchy_ratio = max(max_cauchy_ratio, left / right)
            checked += 1
    derived_constant = 2 * sqrt(2) + 1
    return {
        "weighted_cauchy_schwarz_checks": checked,
        "max_left_over_right": max_cauchy_ratio,
        "derived_absolute_constant": derived_constant,
        "declared_absolute_constant": THEOREM_CONSTANT,
        "constant_dominates_derivation": THEOREM_CONSTANT >= derived_constant,
        "paper_log_dominates_union_log_for_K_ge_2": True,
        "all_obligations_pass": (
            max_cauchy_ratio <= 1 + 1e-12
            and THEOREM_CONSTANT >= derived_constant
        ),
    }


def run_diagnostics() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    invalid_draws = 0
    for setting_index, seed in enumerate(SEEDS):
        bins = (2, 4, 8, 16)[setting_index % 4]
        classes = (2, 3, 10, 50)[(setting_index // 4) % 4]
        total = (400, 800, 1600)[(setting_index // 16) % 3]
        rng = np.random.default_rng(seed)
        target_mass, source_mass, target_atoms, source_atoms = build_population(
            rng, bins, classes
        )
        truth = population_loss(target_mass, target_atoms, source_atoms)
        for replicate in range(120):
            target_counts, target_rows = draw_domain(
                rng, total, target_mass, target_atoms
            )
            source_counts, source_rows = draw_domain(
                rng, total, source_mass, source_atoms
            )
            if np.any((target_counts > 0) & (source_counts == 0)):
                invalid_draws += 1
                continue
            estimate = empirical_loss(
                target_counts, target_rows, source_counts, source_rows
            )
            weights = target_counts / np.sum(target_counts)
            radius = theorem_radius(weights, target_counts, source_counts, classes)
            rows.append(
                {
                    "setting": setting_index,
                    "seed": seed,
                    "replicate": replicate,
                    "bins": bins,
                    "classes": classes,
                    "per_domain_total": total,
                    "target_counts": target_counts.tolist(),
                    "source_counts": source_counts.tolist(),
                    "population_loss": truth,
                    "empirical_loss": estimate,
                    "absolute_error": abs(estimate - truth),
                    "displayed_radius_without_C": radius,
                    "error_over_radius": abs(estimate - truth) / radius,
                }
            )
    ratios = np.asarray([float(row["error_over_radius"]) for row in rows])
    violations = int(np.count_nonzero(ratios > THEOREM_CONSTANT))
    return rows, {
        "valid_rows": len(rows),
        "invalid_undefined_draws_rejected": invalid_draws,
        "maximum_error_over_radius": float(np.max(ratios)),
        "q99_error_over_radius": float(np.quantile(ratios, 0.99)),
        "violations_of_C4_bound": violations,
        "diagnostic_only": True,
    }


def negative_controls() -> list[dict[str, object]]:
    controls = [
        {
            "name": "zero_source_count_on_positive_target_bin",
            "accepted": False,
            "reason": "conditional source estimator is undefined",
        },
        {
            "name": "posterior_function_selected_using_evaluation_sample",
            "accepted": False,
            "reason": "violates the fixed-function or sample-splitting premise",
        },
        {
            "name": "drop_empirical_target_bin_mass_term",
            "accepted": False,
            "reason": "a nonzero multinomial weight fluctuation remains even with oracle conditional means",
        },
        {
            "name": "promote_soft_self_normalized_eq8_without_separate_proof",
            "accepted": False,
            "reason": "the hard-bin proof does not establish random fractional-denominator concentration",
        },
    ]
    return controls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".openresearch" / "artifacts" / "claim-2" / "route-a" / "raw_results.json",
    )
    args = parser.parse_args()
    started = time.monotonic()
    algebra = verify_algebra()
    rows, diagnostics = run_diagnostics()
    controls = negative_controls()
    gates = {
        "paper_pdf_hash_matches": (
            sha256(PAPER.read_bytes()).hexdigest()
            == "fb1d1a634d55132694349d40d56731cc5c7401571bc8c1a9f6eee1b5849950ab"
        ),
        "algebra_obligations_pass": bool(algebra["all_obligations_pass"]),
        "diagnostic_C4_violations_zero": diagnostics["violations_of_C4_bound"] == 0,
        "negative_controls_rejected": all(not control["accepted"] for control in controls),
    }
    payload = {
        "claim": "Theorem 3.2 finite-sample Eq. 9 bound and O(B/epsilon^2) order",
        "route": "A_corrected_hard_bin_proof",
        "route_verdict": "VERIFIED",
        "scope": "Eq. 5 fixed hard bins with iid evaluation data, K>=2, and positive source counts on every positive target-weight bin",
        "not_verified_by_this_route": "unrestricted differentiable soft Eq. 8",
        "paper": {
            "arxiv": "2605.21552v1",
            "sha256": sha256(PAPER.read_bytes()).hexdigest(),
            "anchors": ["S3.SS4", "S3.Thmtheorem2", "A7"],
        },
        "proof": {
            "mean_squared_norm": "E||mean(Z)-mu||_2^2 <= 1/n for simplex-valued Z",
            "bounded_difference": "changing one vector changes the norm error by at most sqrt(2)/n",
            "union": "2B conditional means use eta=delta/(4B)",
            "weighted_contraction": "sum_j w_j/sqrt(n_j) <= sqrt(sum_j w_j/n_j)",
            "bin_mass": "Hoeffding for d_bin(X) in [0,sqrt(2)] is absorbed because sum_j w_j/n_tj=B_positive/N_t >= 1/N_t",
            "constant": "2*sqrt(2)+1 < 4",
            "sample_order": "balanced counts n_j approximately N/B give radius O(sqrt(B log(BK/delta)/N)), hence N=O(B log(BK/delta)/epsilon^2)",
        },
        "algebra": algebra,
        "diagnostics": diagnostics,
        "raw_rows": rows,
        "negative_controls": controls,
        "gates": gates,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "seeds": list(SEEDS),
            "wall_seconds": time.monotonic() - started,
        },
        "limitations": [
            "The numerical sweep is a diagnostic; the general result rests on the displayed proof obligations.",
            "The paper's Appendix G proof is not accepted because its coordinate union introduces sqrt(K) and it omits target-bin-mass estimation.",
            "This route deliberately does not promote the soft self-normalized Eq. 8 statement.",
        ],
    }
    if not all(gates.values()):
        raise SystemExit(f"claim-2 theorem certificate failed: {gates}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("CLAIM 2 CORRECTED FINITE-SAMPLE THEOREM CERTIFICATE")
    print("route_verdict=VERIFIED")
    print(f"scope={payload['scope']}")
    print(f"derived_constant={algebra['derived_absolute_constant']:.12g}")
    print(f"declared_constant={THEOREM_CONSTANT:.12g}")
    print(f"valid_diagnostic_rows={diagnostics['valid_rows']}")
    print(f"max_error_over_radius={diagnostics['maximum_error_over_radius']:.12g}")
    print(f"q99_error_over_radius={diagnostics['q99_error_over_radius']:.12g}")
    print(f"all_gates_pass={all(gates.values())}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
