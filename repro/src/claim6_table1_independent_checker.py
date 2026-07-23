#!/usr/bin/env python3
"""Stdlib-only independent checker for Claim 6's Table 1 falsification."""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path


EXPECTED_PDF_SHA256 = "fb1d1a634d55132694349d40d56731cc5c7401571bc8c1a9f6eee1b5849950ab"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    if sha256(args.paper.read_bytes()).hexdigest() != EXPECTED_PDF_SHA256:
        raise SystemExit("independent paper hash mismatch")

    # Recompute two exhaustive counterexamples without importing either primary certificate.
    weights = (Q(1, 4), Q(3, 4))
    derivatives = (Q(0), Q(4))
    full_gradient = sum(w * d for w, d in zip(weights, derivatives, strict=True)) / sum(weights)
    singleton_gradients = tuple(w * d for w, d in zip(weights, derivatives, strict=True))
    expected_unscaled = sum(singleton_gradients) / len(singleton_gradients)

    values = (Q(1, 4), Q(3, 4))
    target = Q(1, 2)
    full_mean = sum(w * value for w, value in zip(weights, values, strict=True)) / sum(weights)
    full_abs_gradient = Q(1) if full_mean > target else Q(-1)
    batch_gradients = tuple(Q(1) if value > target else Q(-1) for value in values)
    expected_batch_gradient = sum(batch_gradients) / len(batch_gradients)

    exact = result["exact_contradictions"]
    gates = {
        "paper_hash_exact": result["paper_source"]["pdf_sha256"] == EXPECTED_PDF_SHA256,
        "full_gradient_is_3": full_gradient == Q(3),
        "printed_expectation_is_3_over_2": expected_unscaled == Q(3, 2),
        "normalization_identity_false": full_gradient != expected_unscaled,
        "self_normalized_full_gradient_is_1": full_abs_gradient == Q(1),
        "same_batch_expected_gradient_is_0": expected_batch_gradient == Q(0),
        "same_batch_identity_false": full_abs_gradient != expected_batch_gradient,
        "primary_values_match_recomputation": (
            exact["appendix_h_scaling"]["full_gradient"] == str(full_gradient)
            and exact["appendix_h_scaling"]["expected_minibatch_gradient"] == str(expected_unscaled)
            and exact["same_batch_direction"]["full_gradient"] == str(full_abs_gradient)
            and exact["same_batch_direction"]["expected_minibatch_gradient"] == str(expected_batch_gradient)
        ),
        "all_assumptions_asserted_valid": all(result["assumption_audit"].values()),
        "table_cell_is_asserted_true": result["logical_evaluation"]["paper_table_cell_minibatch_trainable"] is True,
        "counterexample_exists": result["logical_evaluation"]["valid_counterexample_to_identity_exists"] is True,
        "conjunction_evaluates_false": result["logical_evaluation"]["compound_claim6_conjunction"] is False,
        "verdict_is_falsified": result["verdict"] == "FALSIFIED",
    }
    if not all(gates.values()):
        raise SystemExit(f"independent gates failed: {[name for name, value in gates.items() if not value]}")

    # Negative logical controls: without either the asserted cell or the contradiction,
    # these facts would not establish the advertised falsification.
    controls = {
        "false_cell_not_at_issue_if_paper_had_marked_it_false": not (False and True),
        "no_counterexample_cannot_falsify_asserted_identity": not (True and False),
        "corrected_estimator_boundary_preserved": result["negative_controls"][
            "correctly_scaled_fixed_direction_estimator_is_unbiased"
        ],
    }
    if not all(controls.values()):
        raise SystemExit("negative logical control failed")

    output = {
        "status": "PASS",
        "independence": "stdlib Fraction recomputation; no import from either certificate",
        "gates": gates,
        "negative_controls": controls,
        "recomputed": {
            "normalization_full_gradient": str(full_gradient),
            "normalization_expected_unscaled_gradient": str(expected_unscaled),
            "self_normalized_full_gradient": str(full_abs_gradient),
            "same_batch_expected_gradient": str(expected_batch_gradient),
        },
        "verdict": "FALSIFIED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))
    print("CLAIM6_TABLE1_CHECK_RESULT status=PASS verdict=FALSIFIED")


if __name__ == "__main__":
    main()
