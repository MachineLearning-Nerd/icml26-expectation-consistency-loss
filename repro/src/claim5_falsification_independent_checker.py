#!/usr/bin/env python3
"""Independent checker for the mandatory Claim 5 falsification audit."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    center = mean(values)
    return math.sqrt(
        sum((value - center) ** 2 for value in values)
        / (len(values) - 1)
    )


def check(input_path: Path, output_path: Path) -> dict[str, object]:
    recorded = json.loads(input_path.read_text())
    errors = {}
    for name, row in recorded["statistical_realizability"][
        "realizations"
    ].items():
        values = row["values_percent"]
        cell = recorded["paper_cells"][name]
        errors[name] = {
            "count": len(values),
            "mean_error": abs(mean(values) - cell["mean_percent"]),
            "sample_std_error": abs(
                sample_std(values) - cell["std_percent"]
            ),
            "bounded": all(0 <= value <= 100 for value in values),
        }
    controls = {
        "verdict_exactly_blocked": recorded["verdict"] == "BLOCKED",
        "no_counterexample_claimed": (
            recorded["valid_counterexample_found"] is False
        ),
        "all_six_cells_independently_recomputed": len(errors) == 6,
        "all_realizations_exact": all(
            row["count"] == 10
            and row["mean_error"] <= 1e-12
            and row["sample_std_error"] <= 1e-12
            and row["bounded"]
            for row in errors.values()
        ),
        "three_prior_routes_rejected": all(
            row["valid_counterexample"] is False
            for row in recorded["prior_route_adjudication"].values()
        )
        and len(recorded["prior_route_adjudication"]) == 3,
        "external_unblock_requirements_present": len(
            recorded["unblock_requirements"]
        )
        == 3,
    }
    passed = all(controls.values())
    result = {
        "status": "PASS" if passed else "FAILED",
        "independence": "stdlib-only recomputation; no import from primary audit",
        "cell_recomputations": errors,
        "controls": controls,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("Claim 5 falsification checker failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.input, args.output)


if __name__ == "__main__":
    main()
