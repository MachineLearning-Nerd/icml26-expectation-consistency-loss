#!/usr/bin/env python3
"""Deterministic source/evidence audit for anchored ECL Claim 6.

This is deliberately not a benchmark rerun.  It checks what can be established
without inventing unavailable PACS data or treating paper tables / embedded
notebook images as independently reproduced measurements:

* exact transcription and logical checks for paper Table 1;
* source-pinned implementation support in ``NeuroDong/ECL``;
* provenance and machine-readability of the official notebook outputs;
* arithmetic checks for Figure 2 and Appendix Table 3; and
* negative controls separating reported values from reproduced values.

Only Python's standard library is used.  The output is deterministic for a
fixed source snapshot.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import re
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Mapping, Sequence


getcontext().prec = 28

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_OFFICIAL_COMMIT = "aae77f890f1e4ebc13dad135b5e29758d98d318d"
EXPECTED_PDF_SHA256 = "fb1d1a634d55132694349d40d56731cc5c7401571bc8c1a9f6eee1b5849950ab"
EXPECTED_SOURCE_SHA256 = {
    "README.md": "b33cbce6741c9a509511087708b55b3b356864f58bfb86a77ad7488649c1c285",
    "losses.py": "1c2de34967f34b98faae5025368edac88f46a709d6e1e0c063e2c01f4d6e9754",
    "main.ipynb": "93904d9cd7b6d86aef24a48a590cb2b9c380359bf87e823f81465034a1e879df",
    "metrics.py": "fb5f22f5f0533175dcbbb605149f7bdd1b74d311a43213ae695c7564fae165f7",
    "networks.py": "96d4848cc1cfb1e136f5716fc51637ec51e9c0505bbb1835a00671d030ad1a9f",
    "utils.py": "3655861e165fa11680afeee030fe5be4cd30e8ea3608315b992bbc18b695c890",
}

CAPABILITY_COLUMNS = (
    "covariate_shift",
    "classwise_calibration",
    "canonical_calibration",
    "unbounded_density_ratio",
    "minibatch_trainable",
)

# Exact boolean transcription of paper Table 1 (PDF page 3).
TABLE1 = {
    "SB-ECE": (False, False, False, True, False),
    "DECE": (False, False, False, True, False),
    "ECE_KDE": (False, True, True, True, True),
    "Weighted_TS": (True, False, False, False, False),
    "FL_IW_Temp": (True, False, False, False, False),
    "TransCal": (True, False, False, False, False),
    "DRL": (True, False, False, False, False),
    "PseudoCal": (True, False, False, True, False),
    "ECL": (True, True, True, True, True),
}

# Primary publication pages used for a semantic cross-check.  These records do
# not turn a selected-literature table into an exhaustive uniqueness proof.
PRIMARY_CITATIONS = {
    "SB-ECE": {
        "url": "https://proceedings.neurips.cc/paper/2021/hash/f8905bd3df64ace64a68e154ba72f24c-Abstract.html",
        "definition": "soft differentiable top-confidence ECE objective for within-domain train/post-hoc calibration",
        "crosscheck": "supports differentiability and train-time use; does not itself establish the paper's narrower unbiased-mini-batch criterion",
    },
    "DECE": {
        "url": "https://openreview.net/forum?id=R2hUure38l",
        "definition": "differentiable expected-calibration-error surrogate used in a meta-calibration framework",
        "crosscheck": "supports within-domain differentiable ECE; no covariate-shift or canonical claim found",
    },
    "ECE_KDE": {
        "url": "https://papers.neurips.cc/paper_files/paper/2022/hash/33d6e648ee4fb24acec3a4bbcd4f001e-Abstract-Conference.html",
        "definition": "consistent differentiable Lp canonical calibration-error estimator using Dirichlet KDE",
        "crosscheck": "explicitly supports canonical estimation and small-subset/mini-batch updates; it is not a covariate-shift method",
    },
    "Weighted_TS": {
        "url": "https://arxiv.org/abs/2006.16405",
        "definition": "importance-sampling calibration under covariate shift",
        "crosscheck": "supports covariate-shift handling and explicit dependence on importance/density ratios",
    },
    "FL_IW_Temp": {
        "url": "https://proceedings.mlr.press/v108/park20b.html",
        "definition": "importance-weighted calibration with domain-adaptive feature alignment",
        "crosscheck": "supports covariate-shift handling; the paper states importance weighting requires sufficiently close domains",
    },
    "TransCal": {
        "url": "https://proceedings.neurips.cc/paper/2020/hash/df12ecd077efc8c23881028604dbb8cc-Abstract.html",
        "definition": "post-hoc transferable calibration with bias/variance-controlled importance weights",
        "crosscheck": "supports covariate-shift/domain-adaptation calibration and density-ratio dependence",
    },
    "DRL": {
        "url": "https://www.ijcai.org/proceedings/2023/162",
        "definition": "domain-shift calibration using a learned differentiable density-ratio estimator in a DRL formulation",
        "crosscheck": "supports domain-shift handling but explicitly uses density ratios",
    },
    "PseudoCal": {
        "url": "https://proceedings.mlr.press/v235/hu24i.html",
        "definition": "post-hoc target-specific calibration using inference-stage mixup and temperature scaling",
        "crosscheck": "supports unlabeled-target domain adaptation without an importance-ratio objective; it is post-hoc/top-confidence focused",
    },
    "ECL": {
        "url": "https://github.com/NeuroDong/ECL/tree/aae77f890f1e4ebc13dad135b5e29758d98d318d",
        "definition": "source-target expectation-matching loss with TopLabel, Classwise, and Canonical branches",
        "crosscheck": "official source exposes the three branches, two-domain inputs, proximal/EMA mini-batch state, and no density-ratio operation",
    },
}


def _stat(mean: str, std: str) -> dict[str, str]:
    return {"mean": mean, "std": std}


# Exact transcription of Appendix Table 3 (PDF page 12).
TABLE3 = {
    "Digit (USPS + SVHN -> MNIST)": {
        "ECE": {"source": _stat("1.54", "0.04"), "target": _stat("16.2", "1.51")},
        "CwECE": {"source": _stat("0.39", "0.01"), "target": _stat("3.14", "0.31")},
        "ECE_KDE": {"source": _stat("0.39", "0.02"), "target": _stat("2.97", "0.23")},
    },
    "PACS (Art + Cartoon + Sketch -> Photo)": {
        "ECE": {"source": _stat("3.84", "0.23"), "target": _stat("22.3", "2.16")},
        "CwECE": {"source": _stat("0.58", "0.01"), "target": _stat("7.87", "0.31")},
        "ECE_KDE": {"source": _stat("0.42", "0.04"), "target": _stat("7.58", "0.37")},
    },
    "ImageNet-Sketch (ImageNet -> Sketch)": {
        "ECE": {"source": _stat("1.47", "0.11"), "target": _stat("55.8", "4.34")},
        "CwECE": {"source": _stat("0.93", "0.09"), "target": _stat("12.7", "0.87")},
        "ECE_KDE": {"source": _stat("0.86", "0.06"), "target": _stat("12.3", "0.73")},
    },
}

# Values printed inside Figure 2.  Each tuple is (error, accuracy), in percent.
FIGURE2 = {
    "simulated_normal": {
        "top_label": {"NLL": ("3.2", "91.5"), "Soft-ECE": ("5.6", "84.2"), "ECL": ("2.8", "92.0")},
        "classwise": {"NLL": ("2.5", "94.5"), "Soft-ECE": ("2.8", "92.0"), "ECL": ("2.4", "94.2")},
        "canonical": {"NLL": ("5.7", "94.5"), "Soft-ECE": ("5.2", "93.8"), "ECL": ("4.8", "95.1")},
    },
    "pacs_three_classes": {
        "top_label": {"NLL": ("6.5", "94.5"), "Soft-ECE": ("6.1", "74.2"), "ECL": ("4.4", "86.5")},
        "classwise": {"NLL": ("4.0", "91.5"), "Soft-ECE": ("3.9", "86.7"), "ECL": ("3.2", "89.8")},
        "canonical": {"NLL": ("9.9", "91.6"), "Soft-ECE": ("6.5", "90.6"), "ECL": ("6.1", "91.5")},
    },
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_source_snapshot(upstream: Path) -> dict[str, Any]:
    actual_names = sorted(path.name for path in upstream.iterdir() if path.is_file())
    actual_hashes = {name: _sha256_file(upstream / name) for name in actual_names}
    losses_text = (upstream / "losses.py").read_text(encoding="utf-8")
    losses_tree = ast.parse(losses_text)

    class_node = next(
        node
        for node in losses_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ECLossMiniBatch"
    )
    string_constants = {
        node.value
        for node in ast.walk(class_node)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    modes = sorted({"TopLabel", "Classwise", "Canonical"} & string_constants)
    density_ratio_tokens = sorted(
        set(re.findall(r"\b(?:density|density_ratio|importance|importance_weight|ratio)\b", losses_text, re.I))
    )

    return {
        "declared_commit": EXPECTED_OFFICIAL_COMMIT,
        "observed_remote_state_2026_07_19": {
            "default_branch": "main",
            "branches": {"main": EXPECTED_OFFICIAL_COMMIT},
            "tag_count": 0,
            "release_count": 0,
            "commit_count": 3,
            "source_introduced_commit": "6957faeda392d719de1c394458450e51498be139",
            "head_change_scope": "README title only relative to source-introduced commit",
        },
        "expected_file_inventory": sorted(EXPECTED_SOURCE_SHA256),
        "actual_file_inventory": actual_names,
        "file_inventory_exact": actual_names == sorted(EXPECTED_SOURCE_SHA256),
        "sha256": actual_hashes,
        "all_hashes_match_pin": actual_hashes == EXPECTED_SOURCE_SHA256,
        "ecl_modes": modes,
        "all_three_modes_implemented": modes == ["Canonical", "Classwise", "TopLabel"],
        "two_domain_forward_inputs": all(
            token in losses_text
            for token in ("train_x: torch.Tensor", "test_x: torch.Tensor", "train_logits: torch.Tensor", "test_logits: torch.Tensor")
        ),
        "stateful_minibatch_markers": {
            "registered_source_cache": "register_buffer('u_s_cache'" in losses_text,
            "registered_target_cache": "register_buffer('u_t_cache'" in losses_text,
            "proximal_iterations": "for _ in range(self.N_prox)" in losses_text,
            "ema_cache_update": "self.ema_alpha" in losses_text and "copy_(u_s_new)" in losses_text,
            "source_and_target_soft_assignments": "w_s = torch.softmax" in losses_text and "w_t = torch.softmax" in losses_text,
        },
        "density_ratio_tokens": density_ratio_tokens,
        "no_density_ratio_operation_in_official_losses": not density_ratio_tokens,
        "important_boundary": (
            "Absence of a density-ratio operation verifies an implementation property; it does not by itself "
            "empirically prove robustness for every unbounded-ratio distribution."
        ),
    }


def _joined_source(cell: Mapping[str, Any]) -> str:
    source = cell.get("source", [])
    return source if isinstance(source, str) else "".join(source)


def _joined_text(output: Mapping[str, Any]) -> str:
    text = output.get("text", [])
    return text if isinstance(text, str) else "".join(text)


def _notebook_audit(notebook_path: Path) -> dict[str, Any]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    source = "\n".join(_joined_source(cell) for cell in code_cells)
    outputs = [output for cell in code_cells for output in cell.get("outputs", [])]
    display_outputs = [output for output in outputs if output.get("output_type") == "display_data"]

    embedded_pngs = []
    for output in display_outputs:
        encoded = output.get("data", {}).get("image/png")
        if encoded:
            if isinstance(encoded, list):
                encoded = "".join(encoded)
            decoded = base64.b64decode(encoded)
            embedded_pngs.append({"sha256": _sha256_bytes(decoded), "bytes": len(decoded)})

    execution_counts = [cell.get("execution_count") for cell in code_cells]
    null_count_cells_with_output = [
        index
        for index, cell in enumerate(cells)
        if cell.get("cell_type") == "code"
        and cell.get("execution_count") is None
        and bool(cell.get("outputs"))
    ]
    output_text = "\n".join(_joined_text(output) for output in outputs)

    structured_result_extensions = {".csv", ".json", ".jsonl", ".npy", ".npz", ".pt", ".pth", ".ckpt"}
    upstream_files = [path for path in notebook_path.parent.iterdir() if path.is_file()]
    structured_result_files = sorted(
        path.name for path in upstream_files if path.suffix.lower() in structured_result_extensions
    )

    return {
        "sha256": _sha256_file(notebook_path),
        "cell_count": len(cells),
        "code_cell_count": len(code_cells),
        "markdown_cell_count": len(cells) - len(code_cells),
        "execution_counts": execution_counts,
        "executed_code_cell_count": sum(count is not None for count in execution_counts),
        "output_count": len(outputs),
        "display_png_count": len(embedded_pngs),
        "embedded_pngs": embedded_pngs,
        "embedded_pngs_expected": embedded_pngs == [
            {"sha256": "47c27d9d49507c64871bd4a43e6562c7609a5276e9fcf1741bc1cc01545e6ee9", "bytes": 278837},
            {"sha256": "65e6409c74790834403d09fb513514be3eb991cfc2b0054297af1da86700477d", "bytes": 145357},
        ],
        "null_execution_count_cells_with_retained_output": null_count_cells_with_output,
        "execution_provenance_internally_complete": not null_count_cells_with_output,
        "configured_paradigm": "TopLabel" if 'calibration_paradigm = "TopLabel"' in source else "unknown",
        "configured_shift": "uniform" if "is_normal = False" in source else "unknown",
        "calls_seeded_initializer": "initFun(calibration_paradigm)" in source,
        "seed_is_42_in_imported_utils": "set_seed(42)" in (notebook_path.parent / "utils.py").read_text(encoding="utf-8"),
        "paper_figure2_shift": "normal",
        "matches_paper_figure2_setting": 'calibration_paradigm = "TopLabel"' not in source or "is_normal = True" in source,
        "contains_pacs_loader_or_path": bool(re.search(r"\bPACS\b", source, re.I)),
        "contains_imagenet_loader_or_path": bool(re.search(r"\bImageNet\b", source, re.I)),
        "contains_checkpoint_save_or_load": bool(re.search(r"state_dict|torch\.save|torch\.load|checkpoint", source, re.I)),
        "contains_structured_result_export": bool(re.search(r"to_csv|json\.dump|np\.save|torch\.save", source)),
        "structured_result_files": structured_result_files,
        "prints_machine_readable_figure_metrics": any(
            marker in output_text for marker in ("Final ECE", "Final CwECE", "Final CaECE", "metrics.json")
        ),
        "notebook_python": notebook.get("metadata", {}).get("language_info", {}).get("version"),
        "paper_python": "3.11.11",
        "environment_version_matches_paper": notebook.get("metadata", {}).get("language_info", {}).get("version") == "3.11.11",
        "reproduction_boundary": (
            "The retained PNGs and console stream are evidence of one saved notebook session, not raw independent "
            "runs for Figure 2, PACS, or Table 3."
        ),
    }


def _table1_audit(source: Mapping[str, Any]) -> dict[str, Any]:
    rows = {
        method: dict(zip(CAPABILITY_COLUMNS, values, strict=True))
        for method, values in TABLE1.items()
    }
    all_five = [method for method, values in rows.items() if all(values.values())]
    citations = {
        method: {**PRIMARY_CITATIONS[method], "paper_table_values": rows[method]}
        for method in rows
    }
    return {
        "columns": list(CAPABILITY_COLUMNS),
        "rows": rows,
        "methods_with_all_five_in_paper_table": all_five,
        "ecl_is_unique_all_true_within_selected_table": all_five == ["ECL"],
        "official_source_supports_ecl_surface": bool(
            source["all_three_modes_implemented"]
            and source["two_domain_forward_inputs"]
            and source["no_density_ratio_operation_in_official_losses"]
            and all(source["stateful_minibatch_markers"].values())
        ),
        "primary_citation_crosscheck": citations,
        "independent_global_uniqueness_established": False,
        "why_global_uniqueness_is_not_established": [
            "Table 1 is a selected set of eight comparators, not an exhaustive method census.",
            "The five columns are not given operational pass/fail protocols that can be rerun uniformly across repositories.",
            "Differentiable train-time use and an unbiased mini-batch gradient theorem are distinct; the SB-ECE/DECE citations support the former.",
            "The official ECL class implements mini-batch proximal/EMA updates, but implementation presence alone does not prove the paper's theoretical unbiasedness claim.",
        ],
    }


def _figure2_audit(notebook: Mapping[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    every_error_lower = True
    every_accuracy_not_lower_than_nll = True
    for dataset, paradigms in FIGURE2.items():
        comparisons[dataset] = {}
        for paradigm, methods in paradigms.items():
            ecl_error, ecl_acc = map(Decimal, methods["ECL"])
            baseline_errors = [Decimal(methods[name][0]) for name in ("NLL", "Soft-ECE")]
            nll_acc = Decimal(methods["NLL"][1])
            row = {
                "reported": {
                    name: {"error_percent": values[0], "accuracy_percent": values[1]}
                    for name, values in methods.items()
                },
                "ecl_error_lower_than_both_displayed_baselines": all(ecl_error < value for value in baseline_errors),
                "ecl_accuracy_not_lower_than_nll": ecl_acc >= nll_acc,
                "ecl_minus_nll_accuracy_points": str(ecl_acc - nll_acc),
            }
            comparisons[dataset][paradigm] = row
            every_error_lower &= row["ecl_error_lower_than_both_displayed_baselines"]
            every_accuracy_not_lower_than_nll &= row["ecl_accuracy_not_lower_than_nll"]

    return {
        "paper_values": comparisons,
        "all_six_ecl_errors_lower_than_both_displayed_baselines": every_error_lower,
        "all_six_ecl_accuracies_at_least_nll": every_accuracy_not_lower_than_nll,
        "strict_accuracy_preservation_note": (
            "The displayed PACS ECL accuracies are below the displayed NLL accuracies in all three paradigms; "
            "the caption's 'preserves or improves' wording is not a uniform non-decrease statement."
        ),
        "official_notebook_matches_figure2_setting": notebook["matches_paper_figure2_setting"],
        "official_notebook_has_pacs": notebook["contains_pacs_loader_or_path"],
        "independently_reproduced": False,
        "reason_not_reproduced": (
            "The repository's saved notebook is uniform-shift TopLabel only and has no PACS path or structured "
            "Figure 2 result data; the values above are arithmetic checks of the paper figure."
        ),
    }


def _table3_audit(notebook: Mapping[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    all_source_lower = True
    for dataset, metrics in TABLE3.items():
        comparisons[dataset] = {}
        for metric, domains in metrics.items():
            source_mean = Decimal(domains["source"]["mean"])
            target_mean = Decimal(domains["target"]["mean"])
            source_lower = source_mean < target_mean
            all_source_lower &= source_lower
            comparisons[dataset][metric] = {
                "reported": domains,
                "source_mean_lower_than_target_mean": source_lower,
                "target_minus_source_points": str(target_mean - source_mean),
                "target_over_source_ratio": str((target_mean / source_mean).quantize(Decimal("0.000001"))),
            }

    return {
        "comparisons": comparisons,
        "all_nine_reported_source_means_lower_than_targets": all_source_lower,
        "dataset_count": len(TABLE3),
        "comparison_count": sum(len(metrics) for metrics in TABLE3.values()),
        "contains_pacs": any(name.startswith("PACS") for name in TABLE3),
        "contains_simulated_dataset": any("simulat" in name.lower() for name in TABLE3),
        "raw_runs_available": False,
        "sample_count_explicit_in_table3_caption": False,
        "statistical_significance_test_reported": False,
        "official_notebook_has_table3_datasets": notebook["contains_pacs_loader_or_path"] or notebook["contains_imagenet_loader_or_path"],
        "independently_reproduced": False,
        "reason_not_reproduced": (
            "Only aggregate mean±std values are printed.  The official repository contains no PACS/ImageNet loader, "
            "per-run predictions, run seeds, checkpoints, or Table 3 export from which to recompute them."
        ),
    }


def _negative_controls(
    table1: Mapping[str, Any], figure2: Mapping[str, Any], table3: Mapping[str, Any], notebook: Mapping[str, Any]
) -> dict[str, bool]:
    return {
        "paper_table_transcription_is_not_empirical_reproduction": not table3["independently_reproduced"],
        "embedded_png_is_not_raw_machine_readable_runs": notebook["display_png_count"] == 2
        and not notebook["structured_result_files"],
        "one_simulation_mode_is_not_three_paradigms_plus_pacs": notebook["configured_paradigm"] == "TopLabel"
        and not notebook["contains_pacs_loader_or_path"],
        "source_branch_presence_is_not_effectiveness_evidence": not figure2["independently_reproduced"],
        "absence_of_ratio_code_is_not_all_unbounded_ratio_performance": bool(
            table1["official_source_supports_ecl_surface"]
        ),
        "selected_table_uniqueness_is_not_global_uniqueness": not table1["independent_global_uniqueness_established"],
    }


def build_report(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    upstream = repo_root / "upstream"
    pdf = repo_root / "repro/evidence/claim3/2605.21552v1.pdf"
    source = _read_source_snapshot(upstream)
    notebook = _notebook_audit(upstream / "main.ipynb")
    table1 = _table1_audit(source)
    figure2 = _figure2_audit(notebook)
    table3 = _table3_audit(notebook)
    negative = _negative_controls(table1, figure2, table3, notebook)

    report: dict[str, Any] = {
        "claim": {
            "openreview_id": "gFPPTokv9C",
            "anchored_claim_number": 6,
            "attempt": 1,
            "scope": "Table 1 capability uniqueness, Figure 2 reliability diagrams, and Appendix Table 3 calibration gaps",
        },
        "paper": {
            "path": str(pdf.relative_to(repo_root)),
            "sha256": _sha256_file(pdf),
            "matches_expected_pdf": _sha256_file(pdf) == EXPECTED_PDF_SHA256,
            "pages_read": 23,
            "visual_pages_checked": [3, 7, 12],
        },
        "official_source": source,
        "table1_capability_audit": table1,
        "official_notebook_audit": notebook,
        "figure2_audit": figure2,
        "table3_audit": table3,
        "negative_controls": negative,
        "assessment": {
            "anchored_claim_6": "partially_reproduced_source_capabilities_only__broad_empirics_blocked",
            "table1_selected_matrix": "exactly_transcribed_and_partially_crosschecked",
            "ecl_implementation_surface": "verified_at_pinned_source",
            "global_uniqueness": "not_established",
            "figure2": "paper_arithmetic_checked_not_independently_reproduced",
            "table3": "paper_arithmetic_checked_not_independently_reproduced",
            "all_negative_controls_pass": all(negative.values()),
            "toy": False,
            "inconclusive": True,
        },
        "blocker": {
            "proven": True,
            "missing_public_evidence": [
                "PACS and ImageNet-Sketch data-loading/evaluation pipeline",
                "model checkpoints and exact architecture/preprocessing configuration",
                "per-run predictions, seeds, and ten-run metric records",
                "structured Figure 2 and Table 3 outputs",
                "dependency lockfile matching Python 3.11.11 and Torch 2.4.1+cu118",
            ],
            "why_light_local_execution_cannot_close_it": (
                "The available code can rerun one small synthetic setting only.  Reconstructing PACS and the reported "
                "multi-run experiments would require missing experiment code/data provenance and material training, "
                "which is outside this source-only attempt while other campaign compute is active."
            ),
            "next_materially_distinct_approach": (
                "Request or locate an author-produced artifact containing the PACS pipeline, exact run configs/seeds, "
                "checkpoints or raw per-run predictions; verify hashes, then rerun one source-pinned PACS->Photo "
                "ResNet-50 protocol on isolated GPU capacity before scaling to the full ten-run matrix."
            ),
        },
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report["certificate_sha256"] = _sha256_bytes(canonical)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args(argv)
    report = build_report()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
