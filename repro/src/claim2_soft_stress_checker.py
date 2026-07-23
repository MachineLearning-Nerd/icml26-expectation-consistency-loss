#!/usr/bin/env python3
"""Independent checker for the Claim 2 soft falsification stress."""
from __future__ import annotations

import argparse
import json
from math import isclose, log, sqrt
from pathlib import Path


def radius(row: dict[str, object], delta: float = 0.05) -> float:
    weights = [float(value) for value in row["target_weights"]]
    target = [float(value) for value in row["target_soft_counts"]]
    source = [float(value) for value in row["source_soft_counts"]]
    proxy = 0.0
    for weight, target_count, source_count in zip(weights, target, source, strict=True):
        if weight == 0:
            continue
        if target_count <= 0 or source_count <= 0:
            raise ValueError("invalid positive-weight denominator")
        proxy += weight * (1 / target_count + 1 / source_count)
    return sqrt(log(2 * int(row["actual_bins"]) * 3 / delta) * proxy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    mismatches = []
    violations = 0
    maximum = 0.0
    for index, row in enumerate(payload["raw_rows"]):
        recomputed = radius(row)
        error = abs(float(row["empirical_ecl"]) - float(row["population_ecl"]))
        ratio = error / recomputed
        maximum = max(maximum, ratio)
        violations += ratio > 4.0
        if not isclose(recomputed, float(row["radius_without_C"]), rel_tol=1e-12, abs_tol=1e-14):
            mismatches.append({"row": index, "field": "radius"})
        if not isclose(error, float(row["absolute_error"]), rel_tol=1e-12, abs_tol=1e-14):
            mismatches.append({"row": index, "field": "error"})
        if not isclose(ratio, float(row["error_over_radius"]), rel_tol=1e-12, abs_tol=1e-14):
            mismatches.append({"row": index, "field": "ratio"})
    result = {
        "status": "PASS",
        "rows": len(payload["raw_rows"]),
        "mismatches": mismatches,
        "C4_violations": violations,
        "maximum_recomputed_ratio": maximum,
        "falsification_succeeded": violations > 0,
        "independence": "stdlib implementation; imports neither numpy nor the experiment module",
    }
    if mismatches:
        raise SystemExit(f"independent soft checker mismatches: {mismatches[:5]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("CLAIM 2 SOFT STRESS INDEPENDENT CHECKER")
    print("status=PASS")
    print(f"rows={result['rows']}")
    print(f"C4_violations={violations}")
    print(f"maximum_recomputed_ratio={maximum:.12g}")
    print(f"falsification_succeeded={result['falsification_succeeded']}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
