#!/usr/bin/env python3
"""Independent stdlib-oriented checker for Claim 2 route-A output."""
from __future__ import annotations

import argparse
import json
from math import isclose, log, sqrt
from pathlib import Path


def recompute_radius(row: dict[str, object], delta: float = 0.05) -> float:
    target = [int(value) for value in row["target_counts"]]
    source = [int(value) for value in row["source_counts"]]
    total = sum(target)
    variance = 0.0
    for target_count, source_count in zip(target, source, strict=True):
        if target_count == 0:
            continue
        if source_count <= 0:
            raise ValueError("undefined source conditional mean")
        weight = target_count / total
        variance += weight * (1 / target_count + 1 / source_count)
    return sqrt(log(2 * int(row["bins"]) * int(row["classes"]) / delta) * variance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    mismatches = []
    maximum_ratio = 0.0
    for index, row in enumerate(payload["raw_rows"]):
        radius = recompute_radius(row)
        expected_error = abs(float(row["empirical_loss"]) - float(row["population_loss"]))
        expected_ratio = expected_error / radius
        maximum_ratio = max(maximum_ratio, expected_ratio)
        if not isclose(radius, float(row["displayed_radius_without_C"]), rel_tol=1e-12, abs_tol=1e-14):
            mismatches.append({"row": index, "field": "radius"})
        if not isclose(expected_error, float(row["absolute_error"]), rel_tol=1e-12, abs_tol=1e-14):
            mismatches.append({"row": index, "field": "absolute_error"})
        if not isclose(expected_ratio, float(row["error_over_radius"]), rel_tol=1e-12, abs_tol=1e-14):
            mismatches.append({"row": index, "field": "ratio"})
    controls = payload["negative_controls"]
    result = {
        "status": "PASS",
        "input_rows": len(payload["raw_rows"]),
        "mismatches": mismatches,
        "maximum_recomputed_ratio": maximum_ratio,
        "C4_violations": sum(
            float(row["error_over_radius"]) > 4.0 for row in payload["raw_rows"]
        ),
        "all_negative_controls_rejected": all(
            not bool(control["accepted"]) for control in controls
        ),
        "independence": "does not import the certificate implementation or numpy",
    }
    if mismatches or result["C4_violations"] or not result["all_negative_controls_rejected"]:
        raise SystemExit(f"independent checker failed: {result}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("CLAIM 2 INDEPENDENT CHECKER")
    print("status=PASS")
    print(f"input_rows={result['input_rows']}")
    print(f"maximum_recomputed_ratio={maximum_ratio:.12g}")
    print("all_negative_controls_rejected=True")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
