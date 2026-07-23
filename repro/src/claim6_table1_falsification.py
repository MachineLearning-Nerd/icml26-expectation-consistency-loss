#!/usr/bin/env python3
"""Link the exact Theorem 3.3 counterexamples to Claim 6's Table 1 cell."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PDF_SHA256 = "fb1d1a634d55132694349d40d56731cc5c7401571bc8c1a9f6eee1b5849950ab"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    claim3 = json.loads(args.claim3.read_text(encoding="utf-8"))
    pdf = ROOT / "repro/evidence/claim3/2605.21552v1.pdf"
    pdf_hash = sha256(pdf.read_bytes()).hexdigest()
    if pdf_hash != EXPECTED_PDF_SHA256:
        raise SystemExit("paper PDF hash mismatch")
    if not claim3["assessment"]["all_certificate_gates_pass"]:
        raise SystemExit("inherited exact counterexample certificate did not pass")
    if not all(claim3["probability_domain_audit"].values()):
        raise SystemExit("a counterexample leaves the theorem probability domain")

    normalization = claim3["exact_certificates"]["appendix_h_normalization"]
    same_batch = claim3["exact_certificates"]["self_normalization_and_same_batch_norm"]
    derivative = claim3["exact_certificates"]["soft_weight_derivative"]
    eq10 = claim3["exact_certificates"]["eq10_objective_and_gradient"]
    contradictions = {
        "appendix_h_scaling": {
            "full_gradient": normalization["full_weighted_mean_gradient"],
            "expected_minibatch_gradient": normalization["appendix_h_unscaled_expectation"],
            "equal": normalization["appendix_h_normalization_matches"],
        },
        "same_batch_direction": {
            "full_gradient": same_batch["full_abs_ecl_gradient"],
            "expected_minibatch_gradient": same_batch["expected_same_batch_direction_gradient"],
            "equal": same_batch["same_batch_direction_gradient_unbiased"],
        },
        "soft_weight_derivative": {
            "true_gradient": derivative["true_quotient_rule_derivative"],
            "printed_gradient": derivative["appendix_h_derivative_when_p_is_fixed"],
            "equal": derivative["omitting_weight_and_denominator_derivatives_is_valid"],
        },
        "eq8_vs_profiled_eq10": {
            "eq8_loss": eq10["eq8_loss"],
            "eq10_loss": eq10["eq10_profile_loss"],
            "loss_equal": eq10["objectives_equal"],
            "eq8_gradient": eq10["eq8_gradient"],
            "eq10_gradient": eq10["eq10_fixed_aux_or_envelope_gradient"],
            "gradient_equal": eq10["gradients_equal"],
        },
    }
    if any(
        item.get("equal", item.get("gradient_equal", True))
        for item in contradictions.values()
    ):
        raise SystemExit("expected every selected exact contradiction to be active")

    result = {
        "schema_version": 1,
        "paper_source": {
            "pdf_sha256": pdf_hash,
            "anchors": ["Table 1 (PDF page 3)", "Section 3.5", "Theorem 3.3", "Appendix H"],
        },
        "exact_claim_contract": {
            "table1_assertion": "ECL has all five listed capabilities, including theoretical mini-batch trainability.",
            "paper_operational_definition": (
                "Section 3.5 says mini-batch trainability requires the mini-batch gradient "
                "to be an unbiased estimator of the full-dataset gradient."
            ),
            "quantifier": "the displayed gradient equality over source and target mini-batches",
            "logical_form": "all_five(ECL) implies minibatch_unbiased(ECL)",
        },
        "assumption_audit": {
            "all_probability_domain_paths_valid": True,
            "soft_assignments_are_probability_simplex_rows": True,
            "posterior_paths_are_locally_valid": True,
            "ordinary_gradients_avoid_nonsmooth_zero_points": True,
            "official_source_hash_matches_pin": claim3["verification_gates"]["pinned_source_hash_match"],
        },
        "exact_contradictions": contradictions,
        "logical_evaluation": {
            "paper_table_cell_minibatch_trainable": True,
            "required_unbiased_gradient_identity": True,
            "valid_counterexample_to_identity_exists": True,
            "table1_all_five_conjunction": False,
            "compound_claim6_conjunction": False,
        },
        "negative_controls": {
            "correctly_scaled_fixed_direction_estimator_is_unbiased": normalization[
                "corrected_estimator_is_exactly_unbiased"
            ],
            "counterexample_does_not_reject_every_possible_minibatch_estimator": True,
            "pacs_or_figure2_needed_to_falsify_a_false_conjunction": False,
        },
        "verdict": "FALSIFIED",
        "confidence": "HIGH",
        "verdict_basis": (
            "An assumption-complete exact counterexample falsifies the paper's own operational "
            "definition of one required Table 1 capability; one false conjunct falsifies the "
            "simultaneous-all-five and compound Claim 6 assertions."
        ),
        "limitations": [
            "This does not say the released loop cannot execute or sometimes optimize a useful objective.",
            "It falsifies 'theoretically mini-batch trainable' under the paper's explicit unbiased-gradient definition.",
            "It does not independently reproduce the PACS numbers; they are unnecessary to falsify a conjunction whose Table 1 component is false.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("CLAIM6_TABLE1 exact_definition=unbiased_minibatch_gradient")
    for name, values in contradictions.items():
        print(f"CLAIM6_TABLE1_COUNTEREXAMPLE name={name} values={json.dumps(values, sort_keys=True)}")
    print("CLAIM6_TABLE1_RESULT verdict=FALSIFIED confidence=HIGH")


if __name__ == "__main__":
    main()
