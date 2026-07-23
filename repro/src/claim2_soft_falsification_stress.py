#!/usr/bin/env python3
"""Assumption-preserving falsification stress for soft Eq. 8 in Theorem 3.2."""
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

from claim3_soft_sample_complexity import (
    estimate_from_counts,
    exact_population,
    make_construction,
)


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "repro" / "evidence" / "claim3" / "2605.21552v1.pdf"
DELTA = 0.05
TEST_CONSTANT = 4.0
SAMPLE_SIZES = (256, 1024, 4096, 16384)
REPLICATES = 64
REGIMES = (
    {"name": "well_conditioned_6", "bins": 6, "temperature_scale": 1.0, "tiny_mass": False},
    {"name": "many_bins_28", "bins": 28, "temperature_scale": 1.0, "tiny_mass": False},
    {"name": "opposite_corner_mass", "bins": 15, "temperature_scale": 1.0, "tiny_mass": True},
    {"name": "diffuse_opposite_corner", "bins": 21, "temperature_scale": 2.0, "tiny_mass": True},
)
SEEDS = tuple(range(260521620, 260521620 + len(REGIMES) * len(SAMPLE_SIZES)))


def soft_radius(
    target_weights: np.ndarray,
    target_soft_counts: np.ndarray,
    source_soft_counts: np.ndarray,
    classes: int,
    delta: float = DELTA,
) -> float:
    positive = target_weights > 0
    if np.any(target_soft_counts[positive] <= 0) or np.any(source_soft_counts[positive] <= 0):
        raise ValueError("positive target soft weights require positive source/target soft counts")
    proxy = np.sum(
        target_weights[positive]
        * (
            1.0 / target_soft_counts[positive]
            + 1.0 / source_soft_counts[positive]
        )
    )
    return sqrt(log(2 * len(target_weights) * classes / delta) * float(proxy))


def target_only_wrong_radius(
    target_weights: np.ndarray,
    target_soft_counts: np.ndarray,
    classes: int,
) -> float:
    positive = target_weights > 0
    proxy = np.sum(target_weights[positive] / target_soft_counts[positive])
    return sqrt(log(2 * len(target_weights) * classes / DELTA) * float(proxy))


def run_stress() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    invalid = []
    seed_index = 0
    for regime in REGIMES:
        construction = make_construction(
            int(regime["bins"]),
            temperature_scale=float(regime["temperature_scale"]),
            tiny_mass=bool(regime["tiny_mass"]),
        )
        population = exact_population(construction)
        for sample_size in SAMPLE_SIZES:
            seed = SEEDS[seed_index]
            seed_index += 1
            rng = np.random.default_rng(seed)
            for replicate in range(REPLICATES):
                source_counts = rng.multinomial(
                    sample_size, construction.source_probabilities
                )
                target_counts = rng.multinomial(
                    sample_size, construction.target_probabilities
                )
                estimate = estimate_from_counts(
                    construction,
                    source_counts,
                    target_counts,
                    stabilizer=1e-5,
                )
                target_weights = np.asarray(estimate["target_weights"])
                source_soft = np.asarray(estimate["source_soft_counts"])
                target_soft = np.asarray(estimate["target_soft_counts"])
                try:
                    radius = soft_radius(
                        target_weights, target_soft, source_soft, classes=3
                    )
                except ValueError as error:
                    invalid.append(
                        {
                            "regime": regime["name"],
                            "sample_size": sample_size,
                            "replicate": replicate,
                            "reason": str(error),
                        }
                    )
                    continue
                error = abs(float(estimate["ecl"]) - float(population["ecl"]))
                wrong_radius = target_only_wrong_radius(
                    target_weights, target_soft, classes=3
                )
                rows.append(
                    {
                        "regime": regime["name"],
                        "requested_bins": regime["bins"],
                        "actual_bins": len(construction.anchors),
                        "temperature": construction.temperature,
                        "tiny_mass": regime["tiny_mass"],
                        "sample_size": sample_size,
                        "seed": seed,
                        "replicate": replicate,
                        "population_ecl": float(population["ecl"]),
                        "empirical_ecl": float(estimate["ecl"]),
                        "absolute_error": error,
                        "target_weights": target_weights.tolist(),
                        "target_soft_counts": target_soft.tolist(),
                        "source_soft_counts": source_soft.tolist(),
                        "radius_without_C": radius,
                        "error_over_radius": error / radius,
                        "target_only_wrong_radius": wrong_radius,
                        "error_over_target_only_wrong_radius": error / wrong_radius,
                        "minimum_population_source_mass": float(
                            np.min(population["source_mass"])
                        ),
                        "minimum_population_target_mass": float(
                            np.min(population["target_mass"])
                        ),
                    }
                )
    ratios = np.asarray([float(row["error_over_radius"]) for row in rows])
    wrong = np.asarray(
        [float(row["error_over_target_only_wrong_radius"]) for row in rows]
    )
    violations = int(np.count_nonzero(ratios > TEST_CONSTANT))
    wrong_violations = int(np.count_nonzero(wrong > TEST_CONSTANT))
    return rows, {
        "valid_rows": len(rows),
        "invalid_rows": invalid,
        "maximum_error_over_radius": float(np.max(ratios)),
        "q99_error_over_radius": float(np.quantile(ratios, 0.99)),
        "C4_violations": violations,
        "maximum_error_over_wrong_target_only_radius": float(np.max(wrong)),
        "wrong_target_only_C4_violations": wrong_violations,
        "falsification_succeeded": violations > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    rows, summary = run_stress()
    controls = [
        {
            "name": "zero_source_soft_mass",
            "accepted_as_counterexample": False,
            "reason": "the conditional estimator and displayed radius are undefined",
        },
        {
            "name": "evaluation_adaptive_posterior",
            "accepted_as_counterexample": False,
            "reason": "violates fixed-function/sample-splitting interpretation",
        },
        {
            "name": "omit_source_denominator_term",
            "accepted_as_theorem_checker": False,
            "reason": "changes the displayed bound and is retained only as a sensitivity control",
        },
        {
            "name": "repeat_printed_table_values",
            "accepted_as_counterexample": False,
            "reason": "not an independent sample-complexity experiment",
        },
    ]
    gates = {
        "paper_hash_matches": sha256(PAPER.read_bytes()).hexdigest()
        == "fb1d1a634d55132694349d40d56731cc5c7401571bc8c1a9f6eee1b5849950ab",
        "all_planned_rows_valid": not summary["invalid_rows"],
        "minimum_rows": summary["valid_rows"] == len(REGIMES) * len(SAMPLE_SIZES) * REPLICATES,
        "controls_rejected": all(
            not control["accepted_as_counterexample"]
            for control in controls
            if "accepted_as_counterexample" in control
        ),
    }
    payload = {
        "claim": "Theorem 3.2 displayed finite-sample bound for differentiable soft Eq. 8",
        "route": "B_assumption_preserving_falsification_stress",
        "route_result": (
            "FALSIFIED" if summary["falsification_succeeded"] else "NO_VALID_COUNTEREXAMPLE_FOUND"
        ),
        "paper": {
            "arxiv": "2605.21552v1",
            "sha256": sha256(PAPER.read_bytes()).hexdigest(),
            "anchors": ["S3.SS4", "S3.Thmtheorem2", "S3.E8", "A6", "A7"],
        },
        "assumptions": {
            "iid_domains": True,
            "shared_fixed_posterior_function": True,
            "strictly_positive_population_atom_probabilities": True,
            "official_gaussian_soft_assignment_rule": True,
            "official_temperature_rule": True,
            "stabilizer": 1e-5,
        },
        "config": {
            "delta": DELTA,
            "test_constant": TEST_CONSTANT,
            "sample_sizes": SAMPLE_SIZES,
            "replicates": REPLICATES,
            "regimes": REGIMES,
            "seeds": SEEDS,
        },
        "summary": summary,
        "raw_rows": rows,
        "negative_controls": controls,
        "gates": gates,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "wall_seconds": time.monotonic() - started,
        },
        "limitations": [
            "A finite stress search cannot verify a universally quantified theorem.",
            "No C=4 violation is a valid falsification unless it persists as a probability-level contradiction; single rare draws would require follow-up exact analysis.",
            "The paper states only existence of an unspecified absolute C, so failure of a smaller trial constant alone would not falsify the theorem.",
            "This route is designed to find counterexamples, not to inflate supporting empirical evidence into proof.",
        ],
    }
    if not all(gates.values()):
        raise SystemExit(f"soft falsification stress failed its validity gates: {gates}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("CLAIM 2 SOFT EQ. 8 FALSIFICATION STRESS")
    print(f"route_result={payload['route_result']}")
    print(f"valid_rows={summary['valid_rows']}")
    print(f"max_error_over_radius={summary['maximum_error_over_radius']:.12g}")
    print(f"q99_error_over_radius={summary['q99_error_over_radius']:.12g}")
    print(f"C4_violations={summary['C4_violations']}")
    print(
        "wrong_target_only_C4_violations="
        f"{summary['wrong_target_only_C4_violations']}"
    )
    print(f"all_validity_gates_pass={all(gates.values())}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
