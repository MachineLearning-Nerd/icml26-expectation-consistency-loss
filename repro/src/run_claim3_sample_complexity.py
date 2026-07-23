#!/usr/bin/env python3
"""Run the CPU Claim 3 sample-complexity audit for gFPPTokv9C."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np

from claim3_sample_complexity import run_experiment

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "repro" / "configs" / "claim3.json"
DEFAULT_JSON = ROOT / "outputs" / "claim3_sample_complexity.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "CLAIM3_SAMPLE_COMPLEXITY_AUDIT.md"
DEFAULT_OFFICIAL = ROOT / "repro" / "evidence" / "claim3" / "official_losses.py"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def render_markdown(result: dict[str, object]) -> str:
    proof = result["proof_audit"]
    summary = result["summary"]
    coverage = result["coverage"]
    official = result["official_code_audit"]
    exact = result["exact_binary"]
    executed = result["executed_sample_scaling"]
    target_mass = result["executed_target_mass_scaling"]
    slopes = result["formula_derived_slopes"]
    lines = [
        "# Claim 3 Sample-Complexity Audit",
        "",
        "Paper: *Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift* (`gFPPTokv9C`, arXiv `2605.21552v1`).",
        "",
        "## Local assessment",
        "",
        f"- Hard-bin Eq. 5 statement: `{proof['hard_bin_eq5_assessment']}` by an independent bounded-differences argument.",
        f"- Soft self-normalized Eq. 8 statement: `{proof['soft_eq8_assessment']}`.",
        f"- Printed Appendix-G proof: `{proof['appendix_proof_assessment']}`; its coordinate-wise route is `sqrt(K)` looser than displayed Eq. 9 and Eq. 30 omits empirical target-bin mass error.",
        "- The omitted hard-bin mass term changes the proof and constants, but scalar Hoeffding plus `w_j=n_tj/N_t` absorbs it into Eq. 9's order.",
        "- All numerical radii are normalized to unknown absolute constants; no literal coverage constant is claimed.",
        "",
        "## Independent hard-bin derivation",
        "",
        "For fixed hard bins, write `d_j = ||mu_s,j-mu_t,j||`, population target masses `pi_j`, and empirical target proportions `w_j=n_tj/N_t`. Then",
        "",
        "`|sum_j w_j d_hat_j - sum_j pi_j d_j| <= sum_j w_j |d_hat_j-d_j| + |sum_j (w_j-pi_j)d_j|`.",
        "",
        "- Bounded differences plus the second-moment mean bound gives dimension-free `||mu_hat-mu|| = O(sqrt(log(1/eta)/n))` for simplex-valued vectors; no coordinate union or `sqrt(K)` is needed.",
        "- Reverse triangle inequality, a union bound over bins/domains, and weighted Cauchy-Schwarz yield the conditional-mean part of displayed Eq. 9 up to an absolute constant.",
        "- Since `0 <= d_j <= sqrt(2)`, scalar Hoeffding bounds the second (target-bin-mass) term by `O(sqrt(log(1/delta)/N_t))`. Because `sum_j w_j/n_tj = B/N_t` for positive hard-bin counts, that missing term is absorbed by Eq. 9's displayed order.",
        "- Required scope: fixed hard bins, positive reported bin counts, iid evaluation samples, and a bounded posterior-vector function fixed independently of those samples (or sample splitting).",
        "",
        "## Executed evidence",
        "",
        f"- Valid theorem rows: {summary['valid_theorem_rows']}",
        f"- Axis-covering settings: {coverage['setting_count']} with {coverage['seeds_per_setting']} seeds each",
        f"- Maximum ECL deviation / displayed normalized radius: {summary['max_ecl_deviation_over_displayed_normalized_radius']:.6g}",
        f"- Maximum matched-ECE deviation / displayed normalized radius: {summary['max_matched_ece_deviation_over_displayed_normalized_radius']:.6g}",
        f"- Exact binary probability mass: {exact['probability_mass']:.12f}",
        f"- Exact binary normalized-threshold tail: {exact['tail_probability']:.6g}",
        f"- Executed ECL RMSE-vs-count slope: {executed['ecl']['rmse_log_slope_vs_count']:.6f} (root-n target: -0.5)",
        f"- Executed histogram-ECE RMSE-vs-count slope: {executed['histogram_ece']['rmse_log_slope_vs_count']:.6f} (root-n target: -0.5)",
        f"- Implied ECL sample-complexity exponent: {executed['ecl']['implied_sample_complexity_exponent_from_rmse']:.6f} (target: 2)",
        f"- Implied histogram-ECE sample-complexity exponent: {executed['histogram_ece']['implied_sample_complexity_exponent_from_rmse']:.6f} (target: 2)",
        f"- Executed target-bin-mass RMSE-vs-total-count slope: {target_mass['rmse_log_slope_vs_total_count']:.6f} (root-n target: -0.5)",
        f"- Executed ECL q90-vs-count slope: {executed['ecl']['q90_log_slope_vs_count']:.6f}",
        f"- Executed histogram-ECE q90-vs-count slope: {executed['histogram_ece']['q90_log_slope_vs_count']:.6f}",
        f"- Executed target-bin-mass q90-vs-total-count slope: {target_mass['q90_log_slope_vs_total_count']:.6f}",
        "- The ECL generator uses fixed confidence bins, shared `P(Y|X)` atoms across domains, different source/target latent-X mixtures, and samples atom zero with exactly the declared mixture probability.",
        "- Histogram ECE is executed separately with realizable top-label confidences above 0.5 and Bernoulli correctness observations.",
        "",
        "## Official implementation audit",
        "",
        f"- `{official['semantic_assessment']}`",
        f"- Pinned source matches `{official['repository']}@{official['commit']}` evidence SHA-256: `{official['source_pin_matches']}`",
        f"- Direct Eq. 5 returned-loss parity supported: `{official['direct_eq5_loss_parity_supported']}`",
        "- Therefore official training-code execution would test Eq. 10 optimization, not directly reproduce the Eq. 5 finite-sample estimator bound.",
        "",
        "## Formula-derived identities (not executed evidence)",
        "",
        f"- epsilon^-2 slope: {slopes['epsilon_inverse_square_log_slope']:.6f}",
        f"- B slope including the displayed logarithm: {slopes['bins_log_slope_with_log_factor']:.6f}",
        f"- K dependence slope (logarithmic only): {slopes['classes_log_slope_only']:.6f}",
        "- These values are algebraic evaluations of the displayed formula, not estimator regressions and not proof.",
        "",
        "## Controls",
        "",
    ]
    for control in result["controls"]:
        lines.append(
            f"- `{control['name']}` — assumptions valid: `{control['assumptions_valid']}`; {control['outcome']}"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in result["limitations"])
    lines.extend([
        "",
        "## Reproduce",
        "",
        "```bash",
        ".venv/bin/python -m pytest repro/tests/test_claim3_sample_complexity.py -q",
        ".venv/bin/python repro/src/run_claim3_sample_complexity.py",
        "```",
        "",
        "Artifacts: `outputs/claim3_sample_complexity.json`, `docs/CLAIM3_SAMPLE_COMPLEXITY_AUDIT.md`, and `repro/evidence/claim3/SHA256SUMS`.",
    ])
    lines.append("")
    return "\n".join(lines)


def build_payload(config_path: Path, official_source: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text())
    result = run_experiment(config, official_source=official_source)
    paper_pdf = ROOT / "repro" / "evidence" / "claim3" / "2605.21552v1.pdf"
    verdict = ROOT / "repro" / "evidence" / "claim3" / "official_verdict_record.json"
    return {
        "paper": "gFPPTokv9C",
        "claim_number": 3,
        "claim": "Computing ECL loss has the same sample complexity as Expected Calibration Error (ECE).",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "source_evidence": {
            "paper": {
                "arxiv": "2605.21552v1",
                "path": str(paper_pdf.relative_to(ROOT)),
                "sha256": sha256(paper_pdf.read_bytes()).hexdigest(),
            },
            "official_code": {
                "repository": result["official_code_audit"]["repository"],
                "commit": result["official_code_audit"]["commit"],
                "path": str(official_source.relative_to(ROOT)),
                "sha256": result["official_code_audit"]["sha256"],
            },
            "prior_verdict": {
                "path": str(verdict.relative_to(ROOT)),
                "sha256": sha256(verdict.read_bytes()).hexdigest(),
            },
        },
        "config": config,
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--official-source", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    payload = build_payload(args.config.resolve(), args.official_source.resolve())
    result = payload["result"]
    atomic_write_text(args.json_out.resolve(), json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    atomic_write_text(args.markdown_out.resolve(), render_markdown(result))

    proof = result["proof_audit"]
    coverage = result["coverage"]
    summary = result["summary"]
    exact = result["exact_binary"]
    official = result["official_code_audit"]
    executed = result["executed_sample_scaling"]
    target_mass = result["executed_target_mass_scaling"]
    slopes = result["formula_derived_slopes"]
    print("ECL CLAIM 3 SAMPLE-COMPLEXITY AUDIT")
    print(f"theorem_statement={proof['theorem_statement_assessment']}")
    print(f"appendix_proof={proof['appendix_proof_assessment']}")
    print(f"appendix_derives_displayed={proof['appendix_derives_displayed_bound']}")
    print(f"settings={coverage['setting_count']} seeds_per_setting={coverage['seeds_per_setting']} rows={coverage['row_count']}")
    print(f"covered_bins={coverage['covered_values']['bins']}")
    print(f"covered_classes={coverage['covered_values']['classes']}")
    print(f"covered_deltas={coverage['covered_values']['deltas']}")
    print(f"covered_counts={coverage['covered_values']['counts']}")
    print(f"covered_families={coverage['covered_values']['families']}")
    print(f"max_ecl_deviation_over_normalized_radius={summary['max_ecl_deviation_over_displayed_normalized_radius']:.8g}")
    print(f"max_matched_ece_deviation_over_normalized_radius={summary['max_matched_ece_deviation_over_displayed_normalized_radius']:.8g}")
    print(f"exact_binary_mass={exact['probability_mass']:.12f} exact_tail={exact['tail_probability']:.8g}")
    print(f"official_direct_eq5_parity={official['direct_eq5_loss_parity_supported']}")
    print(f"official_source_pin_matches={official['source_pin_matches']}")
    print(f"executed_ecl_rmse_slope={executed['ecl']['rmse_log_slope_vs_count']:.8g}")
    print(f"executed_ece_rmse_slope={executed['histogram_ece']['rmse_log_slope_vs_count']:.8g}")
    print(f"executed_target_mass_rmse_slope={target_mass['rmse_log_slope_vs_total_count']:.8g}")
    print(f"epsilon_inverse_square_slope={slopes['epsilon_inverse_square_log_slope']:.8g}")
    print(f"bins_slope_with_log={slopes['bins_log_slope_with_log_factor']:.8g}")
    print(f"classes_log_only_slope={slopes['classes_log_slope_only']:.8g}")
    print("literal_coverage_claimed=False")
    try:
        json_label = args.json_out.resolve().relative_to(ROOT)
    except ValueError:
        json_label = args.json_out
    try:
        markdown_label = args.markdown_out.resolve().relative_to(ROOT)
    except ValueError:
        markdown_label = args.markdown_out
    print(f"json={json_label}")
    print(f"markdown={markdown_label}")


if __name__ == "__main__":
    main()
