#!/usr/bin/env python3
"""Independent checker for the Claim 2 soft-bin proof certificate."""
from __future__ import annotations

import argparse
import json
from math import log, sqrt
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--stress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proof = json.loads(args.proof.read_text(encoding="utf-8"))
    stress = json.loads(args.stress.read_text(encoding="utf-8"))

    declared = float(proof["algebra"]["declared_absolute_constant"])
    recomputed_budget = 8 * sqrt(2) + 4
    rows = stress["raw_rows"]
    ratios = np.asarray([float(row["error_over_radius"]) for row in rows])
    radius_errors = []
    target_identities = []
    for row in rows:
        weights = np.asarray(row["target_weights"], dtype=float)
        target = np.asarray(row["target_soft_counts"], dtype=float)
        source = np.asarray(row["source_soft_counts"], dtype=float)
        classes = 3
        delta = float(stress["config"]["delta"])
        recomputed = sqrt(
            log(2 * len(weights) * classes / delta)
            * float(np.sum(weights * (1 / target + 1 / source)))
        )
        radius_errors.append(abs(recomputed - float(row["radius_without_C"])))
        target_identities.append(
            abs(float(np.sum(weights / target)) - len(weights) / row["sample_size"])
        )

    gates = {
        "primary_gates_pass": all(proof["gates"].values()),
        "verdict_verified": proof["route_verdict"] == "VERIFIED",
        "confidence_high": proof["confidence"] == "HIGH",
        "constant_budget_recomputed": abs(recomputed_budget - proof["algebra"]["combined_budget"]) < 1e-12,
        "absolute_constant_dominates": declared >= recomputed_budget,
        "all_1024_rows_present": len(rows) == 1024,
        "all_radii_recomputed": max(radius_errors) < 1e-12,
        "target_count_identity_all_rows": max(target_identities) < 1e-12,
        "no_C16_diagnostic_violation": int(np.count_nonzero(ratios > declared)) == 0,
        "all_negative_controls_rejected": all(
            control["accepted"] is False for control in proof["negative_controls"]
        ),
    }
    if not all(gates.values()):
        raise SystemExit(f"independent soft theorem gates failed: {gates}")
    output = {
        "status": "PASS",
        "independence": "recomputed from raw stress rows without importing primary proof code",
        "gates": gates,
        "recomputed_combined_constant_budget": recomputed_budget,
        "maximum_radius_recomputation_error": max(radius_errors),
        "maximum_target_count_identity_error": max(target_identities),
        "maximum_diagnostic_error_over_radius": float(ratios.max()),
        "verdict": "VERIFIED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))
    print("CLAIM2_SOFT_THEOREM_CHECK_RESULT status=PASS verdict=VERIFIED")


if __name__ == "__main__":
    main()
