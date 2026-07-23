#!/usr/bin/env python3
"""Run final Claim-3 real-MNIST soft Eq. 8 attempt."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import scipy

from claim3_real_mnist_sample_complexity import run_real_mnist_experiment

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "repro" / "configs" / "claim3_real_mnist.json"
DEFAULT_JSON = ROOT / "outputs" / "claim3_real_mnist_sample_complexity.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "CLAIM3_REAL_MNIST_SAMPLE_COMPLEXITY_AUDIT.md"


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
    audit = result["dataset_audit"]
    primary = result["model_training"]["primary"]
    posterior = result["model_training"]["posterior_head"]
    holdout = result["holdout"]
    shift = result["covariate_shift"]
    baseline = result["baseline"]
    sample = result["sample_size_scaling"]
    bins = result["bins_scaling"]
    cross = result["independent_crosscheck"]
    lines = [
        "# Claim 3 Real-MNIST Soft Eq. 8 Sample-Complexity Audit",
        "",
        "Final substantive attempt for live legacy C3 / anchored C2.",
        "",
        "## Assessment",
        "",
        f"- `{result['assessment']}`",
        "- This is real-data, real-trained-model evidence for fixed-B sample order. It is not a repair of Appendix G and not a universal B-dependence proof.",
        "",
        "## Dataset provenance and license boundary",
        "",
        f"- All four cached uncompressed IDX files match preregistered SHA-256, magic, count, and byte-size records: `{audit['all_files_match']}`.",
        "- Cached split: 60,000 training and 10,000 test 28x28 images, matching the MNIST homepage.",
        f"- Cache-local license file present: `{audit['cache_local_license_file_present']}`.",
        f"- External Keras MNIST documentation states `{audit['external_license_statement']}`: {audit['external_license_reference']}",
        f"- Dataset homepage: {audit['dataset_homepage']}",
        "",
        "## Actual training and holdout",
        "",
        "- Primary classifier: multinomial logistic regression trained on official training indices 0..29,999.",
        "- Additional posterior head: independently trained multinomial logistic regression on indices 30,000..59,999.",
        "- Sample-complexity evaluation: official 10,000-image test split, disjoint from both training sets.",
        f"- Primary optimizer iterations/converged/wall seconds: {primary['iterations']} / `{primary['converged']}` / {primary['wall_seconds']:.3f}",
        f"- Posterior-head optimizer iterations/converged/wall seconds: {posterior['iterations']} / `{posterior['converged']}` / {posterior['wall_seconds']:.3f}",
        f"- Primary holdout accuracy / NLL / hard-bin ECE: {holdout['primary']['accuracy']:.6f} / {holdout['primary']['negative_log_likelihood']:.6f} / {holdout['primary']['top_label_ece_15_hard_bins']:.6f}",
        f"- Posterior-head holdout accuracy / NLL: {holdout['posterior_head']['accuracy']:.6f} / {holdout['posterior_head']['negative_log_likelihood']:.6f}",
        "",
        "## X-only covariate shift",
        "",
        f"- Selection function: `{shift['selection_function']}`; labels used: `{shift['uses_labels']}`.",
        f"- Source/target effective pool sizes: {shift['source_effective_sample_size']:.1f} / {shift['target_effective_sample_size']:.1f}.",
        f"- Source/target label-distribution total variation induced by X-only sampling: {shift['source_target_label_distribution_total_variation']:.6f}.",
        f"- Label-permutation control maximum weight change: {shift['label_permutation_control_max_weight_change']:.3g}.",
        "",
        "## Fixed-B sample-size evidence",
        "",
        f"- Baseline B / temperature: {baseline['bins']} / {baseline['temperature']:.9g}.",
        f"- Minimum source/target population soft mass: {baseline['minimum_source_population_mass']:.6g} / {baseline['minimum_target_population_mass']:.6g}.",
        f"- Exact finite-pool population ECL / matched ECE: {baseline['population_ecl']:.9g} / {baseline['population_matched_ece']:.9g}.",
        f"- ECL RMSE slope overall/tail: {sample['ecl']['rmse_log_slope_vs_sample_size']:.6f} / {sample['ecl']['tail_rmse_log_slope']:.6f}; tail epsilon exponent {sample['ecl']['tail_implied_epsilon_exponent']:.6f}.",
        f"- Matched ECE RMSE slope overall/tail: {sample['matched_ece']['rmse_log_slope_vs_sample_size']:.6f} / {sample['matched_ece']['tail_rmse_log_slope']:.6f}; tail epsilon exponent {sample['matched_ece']['tail_implied_epsilon_exponent']:.6f}.",
        f"- Absolute ECL/ECE tail-slope difference: {result['fixed_B_comparability']['absolute_tail_slope_difference']:.6f}.",
        "- Both finite-grid fits are faster than the root-n reference; this supports a no-worse comparable order here, not an asymptotic rate identification.",
        "- Raw per-replicate rows and `n * RMSE^2` diagnostics are preserved in the JSON artifact.",
        "",
        "## Construction-specific B sweep",
        "",
        f"- Executed exact simplex-grid bin counts: {list(map(int, result['bins_scaling']['ecl']['rmse'].keys()))}.",
        f"- ECL / matched-ECE variance-proxy slopes vs B: {bins['ecl']['variance_proxy_log_slope_vs_bins']:.6f} / {bins['matched_ece']['variance_proxy_log_slope_vs_bins']:.6f}.",
        "- These slopes apply only to this MNIST construction and the official B-dependent temperature; they do not prove a universal O(B) theorem.",
        "",
        "## Independent cross-check and controls",
        "",
        f"- Matrix contraction versus explicit per-bin loop ECL difference: {cross['ecl_absolute_difference']:.3g}.",
        f"- Matrix contraction versus explicit per-bin loop matched-ECE difference: {cross['matched_ece_absolute_difference']:.3g}.",
        f"- Fail-closed controls: `{result['fail_closed_controls']}`.",
        f"- Total experiment wall time: {result['wall_seconds']:.3f} seconds with BLAS/OpenMP threads fixed to one.",
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
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest repro/tests/test_claim3_real_mnist_sample_complexity.py -q",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python repro/src/run_claim3_real_mnist_sample_complexity.py",
        "```",
        "",
        "Artifact: `outputs/claim3_real_mnist_sample_complexity.json`.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    data_root = args.data_root.resolve() if args.data_root else (ROOT / str(config["data_root"])).resolve()
    result = run_real_mnist_experiment(config, data_root=data_root)
    payload = {
        "paper": "gFPPTokv9C",
        "claim_number": "live legacy C3 / anchored C2",
        "claim": "Computing ECL loss has the same sample complexity as Expected Calibration Error (ECE).",
        "attempt": 3,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "thread_ceiling": 1,
        },
        "source_evidence": {
            "config_path": str(config_path.relative_to(ROOT)),
            "config_sha256": sha256(config_path.read_bytes()).hexdigest(),
            "data_root_read_only": str(config["data_root"]),
            "data_root_override_supported": True,
        },
        "config": config,
        "result": result,
    }
    atomic_write_text(args.json_out.resolve(), json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    atomic_write_text(args.markdown_out.resolve(), render_markdown(result))
    sample = result["sample_size_scaling"]
    print("ECL CLAIM 3 REAL-MNIST SOFT EQ. 8 ATTEMPT")
    print(f"assessment={result['assessment']}")
    print(f"primary_holdout_accuracy={result['holdout']['primary']['accuracy']:.8g}")
    print(f"posterior_holdout_accuracy={result['holdout']['posterior_head']['accuracy']:.8g}")
    print(f"ecl_tail_rmse_slope_n={sample['ecl']['tail_rmse_log_slope']:.8g}")
    print(f"ece_tail_rmse_slope_n={sample['matched_ece']['tail_rmse_log_slope']:.8g}")
    print(f"sample_rows={len(result['sample_rows'])} bins_rows={len(result['bins_rows'])}")
    print(f"wall_seconds={result['wall_seconds']:.6g}")
    print(f"json={args.json_out.resolve().relative_to(ROOT)}")
    print(f"markdown={args.markdown_out.resolve().relative_to(ROOT)}")


if __name__ == "__main__":
    main()
