#!/usr/bin/env python3
"""Run the CPU-only soft Eq. 8 sample-complexity experiment."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
import platform
from pathlib import Path
import tempfile

import numpy as np

from claim3_soft_sample_complexity import run_soft_experiment

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "repro" / "configs" / "claim3_soft.json"
DEFAULT_JSON = ROOT / "outputs" / "claim3_soft_sample_complexity.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "CLAIM3_SOFT_SAMPLE_COMPLEXITY_AUDIT.md"


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
    baseline = result["baseline"]
    sample = result["sample_size_scaling"]
    bins = result["bins_scaling"]
    independent = result["independent_calculation"]
    tiny = result["tiny_mass_stress"]
    lines = [
        "# Claim 3 Soft Eq. 8 Sample-Complexity Audit",
        "",
        "Paper: *Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift* (`gFPPTokv9C`, arXiv `2605.21552v1`).",
        "",
        "## Assessment",
        "",
        f"- `{result['assessment']}`",
        "- This is a second substantive approach: differentiable Gaussian soft assignments and random self-normalized denominators from Eq. 8, not the prior fixed hard-bin Eq. 5 estimator.",
        "- Scores, assignments, and the exact posterior oracle are fixed before evaluation; source and target evaluation samples are independent.",
        "- The learned-posterior/same-data case remains outside scope.",
        "",
        "## Exact construction",
        "",
        "- Population quantities are exact finite sums over 231 shared latent atoms; the domains differ only in `P(X)`.",
        f"- Bins: {baseline['actual_bins']}; official temperature: {baseline['temperature']:.9g}; Eq. 8 stabilizer: {baseline['stabilizer']:.1e}",
        f"- Minimum source/target population soft mass: {baseline['minimum_source_population_mass']:.6g} / {baseline['minimum_target_population_mass']:.6g}",
        f"- Exact population ECL / matched canonical ECE: {baseline['population_ecl']:.9g} / {baseline['population_matched_canonical_ece']:.9g}",
        f"- Stabilizer bias at smallest/largest n: {baseline['regularizer_bias_at_smallest_n']:.3g} / {baseline['regularizer_bias_at_largest_n']:.3g}",
        "",
        "## Executed scaling",
        "",
        f"- Soft Eq. 8 ECL RMSE slope vs n: {sample['ecl']['rmse_log_slope_vs_sample_size']:.6f} over all n and {sample['ecl']['asymptotic_tail_rmse_log_slope']:.6f} over n=512..8192.",
        f"- Soft Eq. 8 tail implied epsilon exponent: {sample['ecl']['asymptotic_tail_implied_epsilon_sample_complexity_exponent']:.6f}; tail slope of `n * RMSE^2`: {sample['ecl']['asymptotic_tail_n_rmse_squared_log_slope']:.6f} (root-n target: 0).",
        f"- Matched label-ECE RMSE slope vs n: {sample['matched_label_ece']['rmse_log_slope_vs_sample_size']:.6f} over all n and {sample['matched_label_ece']['asymptotic_tail_rmse_log_slope']:.6f} over the same tail.",
        f"- Matched oracle-ECE RMSE slope vs n: {sample['matched_oracle_ece']['rmse_log_slope_vs_sample_size']:.6f}",
        f"- ECL variance-proxy slope vs B: {bins['ecl']['variance_proxy_log_slope_vs_bins']:.6f} (claimed sample-order ceiling: 1)",
        f"- Matched label-ECE variance-proxy slope vs B: {bins['matched_label_ece']['variance_proxy_log_slope_vs_bins']:.6f}",
        "- Raw replicate rows and `n * RMSE^2` diagnostics are preserved in the JSON artifact.",
        "",
        "## Independent calculation",
        "",
        f"- Count contraction versus expanded raw samples, ECL absolute difference: {independent['ecl_absolute_difference']:.3g}",
        f"- Count contraction versus expanded raw samples, matched oracle-ECE difference: {independent['matched_oracle_ece_absolute_difference']:.3g}",
        "",
        "## Denominator controls",
        "",
        f"- Tiny-mass stress remains strictly positive: `{tiny['all_population_masses_positive']}`; minimum source/target mass `{tiny['minimum_source_population_mass']:.3g}` / `{tiny['minimum_target_population_mass']:.3g}`.",
        f"- Tiny-mass ECL RMSE slope vs n: {tiny['rmse_log_slope_vs_sample_size']:.6f}. This ill-conditioned finite-n control is not used to claim a mass-uniform constant.",
        f"- Zero-mass masked-column negative control rejected: `{result['zero_mass_control']['rejected']}`.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in result["limitations"])
    lines.extend([
        "",
        "## Reproduce",
        "",
        "```bash",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest repro/tests/test_claim3_soft_sample_complexity.py -q",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python repro/src/run_claim3_soft_sample_complexity.py",
        "```",
        "",
        "Artifact: `outputs/claim3_soft_sample_complexity.json`.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    result = run_soft_experiment(config)
    paper = ROOT / "repro" / "evidence" / "claim3" / "2605.21552v1.pdf"
    payload = {
        "paper": "gFPPTokv9C",
        "claim_number": "live legacy C3 / anchored C2",
        "claim": "Computing ECL loss has the same sample complexity as Expected Calibration Error (ECE).",
        "attempt": 2,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_ceiling": 1,
        },
        "source_evidence": {
            "paper_path": str(paper.relative_to(ROOT)),
            "paper_sha256": sha256(paper.read_bytes()).hexdigest(),
            "paper_anchors": ["Section 3.4", "Equations 5-8", "Theorem 3.2", "Appendix F", "Appendix G"],
            "config_path": str(config_path.relative_to(ROOT)),
            "config_sha256": sha256(config_path.read_bytes()).hexdigest(),
        },
        "config": config,
        "result": result,
    }
    atomic_write_text(args.json_out.resolve(), json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    atomic_write_text(args.markdown_out.resolve(), render_markdown(result))
    sample = result["sample_size_scaling"]
    bins = result["bins_scaling"]
    print("ECL CLAIM 3 SOFT EQ. 8 SAMPLE-COMPLEXITY AUDIT")
    print(f"assessment={result['assessment']}")
    print(f"rows_sample_size={len(result['sample_rows'])}")
    print(f"rows_bins={len(result['bins_rows'])}")
    print(f"ecl_rmse_slope_n={sample['ecl']['rmse_log_slope_vs_sample_size']:.8g}")
    print(f"ecl_tail_rmse_slope_n={sample['ecl']['asymptotic_tail_rmse_log_slope']:.8g}")
    print(f"ecl_epsilon_exponent={sample['ecl']['implied_epsilon_sample_complexity_exponent']:.8g}")
    print(f"ecl_tail_epsilon_exponent={sample['ecl']['asymptotic_tail_implied_epsilon_sample_complexity_exponent']:.8g}")
    print(f"label_ece_rmse_slope_n={sample['matched_label_ece']['rmse_log_slope_vs_sample_size']:.8g}")
    print(f"ecl_variance_proxy_slope_bins={bins['ecl']['variance_proxy_log_slope_vs_bins']:.8g}")
    print(f"tiny_mass_rmse_slope_n={result['tiny_mass_stress']['rmse_log_slope_vs_sample_size']:.8g}")
    print(f"independent_ecl_abs_diff={result['independent_calculation']['ecl_absolute_difference']:.8g}")
    print(f"json={args.json_out.resolve().relative_to(ROOT)}")
    print(f"markdown={args.markdown_out.resolve().relative_to(ROOT)}")


if __name__ == "__main__":
    main()
