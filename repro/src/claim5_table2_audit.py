#!/usr/bin/env python3
"""Deterministic provenance and exact-arithmetic audit of paper Table 2.

This module does not claim to reproduce the digit experiments. It transcribes
the SVHN rows from Table 2 of arXiv:2605.21552v1, computes every ECL-versus-
baseline reduction with ``fractions.Fraction``, and fails closed when the ten
raw runs needed for an independent empirical reproduction are unavailable.

It can additionally verify local readbacks of two public source snapshots:

* the paper-linked NeuroDong/ECL commit ``aae77f8...``; and
* the older public Anonymous-user-code/ECL commit ``944d492...``.

The second repository has the same project description and contains the digit
script, but this audit labels it only as a probable pre-deanonymization
predecessor. It is not treated as the current official source pin.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import sys
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence


Q = Fraction

PAPER_SHA256 = "fb1d1a634d55132694349d40d56731cc5c7401571bc8c1a9f6eee1b5849950ab"
PAPER_ARXIV_ID = "2605.21552v1"
PAPER_TABLE_PAGE = "PDF page 8 (printed page 8)"

CURRENT_OFFICIAL = {
    "repository": "https://github.com/NeuroDong/ECL",
    "commit": "aae77f890f1e4ebc13dad135b5e29758d98d318d",
    "tree": "7572ba68c4ee432cbc7ce866b304b3d84ec1c9b8",
    "files": {
        "README.md": "b33cbce6741c9a509511087708b55b3b356864f58bfb86a77ad7488649c1c285",
        "losses.py": "1c2de34967f34b98faae5025368edac88f46a709d6e1e0c063e2c01f4d6e9754",
        "main.ipynb": "93904d9cd7b6d86aef24a48a590cb2b9c380359bf87e823f81465034a1e879df",
        "metrics.py": "fb5f22f5f0533175dcbbb605149f7bdd1b74d311a43213ae695c7564fae165f7",
        "networks.py": "96d4848cc1cfb1e136f5716fc51637ec51e9c0505bbb1835a00671d030ad1a9f",
        "utils.py": "3655861e165fa11680afeee030fe5be4cd30e8ea3608315b992bbc18b695c890",
    },
}

PUBLIC_PREDECESSOR = {
    "repository": "https://github.com/Anonymous-user-code/ECL",
    "relationship": "probable_pre_deanonymization_predecessor_not_current_official_pin",
    "commit": "944d492b9d542ebbc0d0396fc57a187b2ce6b293",
    "tree": "7fc9969656b5c31e32d853dcc66785e2655b0534",
    "files": {
        "Cali_in_Digit.py": "783ad6a0ca9efa497b86f5aee79807ff3efb3ba851bcad2db81cc2adde916781",
        "README.md": "c40d5ed1b8a1748dd04647a2a9e6bb84cfc7febbebdf0ab18a39d05439fbe520",
        "Simulated_Cali.ipynb": "88ed64b9716bca458b4aef06d6a61d4483b24bcf1e1cdf5d94e743e3f360b314",
        "Utils.py": "b088f12c41f0aa29f860dfbc3d37bcd53f963e1f9e2a72313a75935f5fcdedb7",
        "losses.py": "bee05f6150c5fd4dc8643fadfe9537a45e61e7e0c70715728aa579caccb1c771",
        "metrics.py": "70ae1b6fc22cd75c9be8f244f5188ffa9017a9d81b20bde3942e5a1b3ce0084b",
        "networks.py": "418613dabe1e2fbdb11e2045e4b62259e7eeec25d5f12692c25dd225fc18f3a8",
    },
}

# Exact decimal strings transcribed from Table 2. Values are ECE percentages.
SVHN_TABLE = {
    "LeNet-5": {
        "Uncal": ("61.9", "6.16"),
        "Soft-ECE": ("62.2", "5.50"),
        "DECE": ("60.8", "5.22"),
        "KDE": ("62.5", "5.80"),
        "TS": ("61.3", "5.89"),
        "TransCal": ("63.7", "4.94"),
        "DRL": ("23.7", "1.93"),
        "PseudoCal": ("52.4", "4.55"),
        "ECL": ("21.5", "1.51"),
        "Oracle": ("1.03", "0.02"),
        "DeltaACC": ("+1.65", "0.65"),
    },
    "ResNet20": {
        "Uncal": ("68.2", "6.44"),
        "Soft-ECE": ("67.5", "5.92"),
        "DECE": ("66.9", "6.10"),
        "KDE": ("67.8", "6.25"),
        "TS": ("68.1", "6.13"),
        "TransCal": ("59.4", "4.63"),
        "DRL": ("40.1", "3.77"),
        "PseudoCal": ("48.2", "3.95"),
        "ECL": ("36.8", "2.08"),
        "Oracle": ("0.50", "0.02"),
        "DeltaACC": ("+2.12", "0.88"),
    },
    "DenseNet40": {
        "Uncal": ("80.8", "6.26"),
        "Soft-ECE": ("81.2", "5.88"),
        "DECE": ("79.5", "6.05"),
        "KDE": ("81.1", "6.15"),
        "TS": ("77.2", "6.98"),
        "TransCal": ("72.9", "5.13"),
        "DRL": ("42.0", "3.36"),
        "PseudoCal": ("64.7", "4.72"),
        "ECL": ("38.4", "3.21"),
        "Oracle": ("0.86", "0.03"),
        "DeltaACC": ("-1.15", "0.45"),
    },
}

NON_ORACLE_BASELINES = (
    "Uncal",
    "Soft-ECE",
    "DECE",
    "KDE",
    "TS",
    "TransCal",
    "DRL",
    "PseudoCal",
)

QUOTED_COMPARISONS = (
    ("LeNet-5", "Uncal"),
    ("ResNet20", "PseudoCal"),
    ("DenseNet40", "Uncal"),
)


def _fraction_text(value: Q) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _decimal_text(value: Q, digits: int = 12) -> str:
    sign = "-" if value < 0 else ""
    value = abs(value)
    scale = 10**digits
    scaled = value * scale
    integer = scaled.numerator // scaled.denominator
    return f"{sign}{integer // scale}.{integer % scale:0{digits}d}".rstrip("0").rstrip(".")


def exact_reduction(baseline: str, ecl: str) -> dict[str, str]:
    """Return exact percentage-point and relative reductions."""

    baseline_q, ecl_q = Q(baseline), Q(ecl)
    if baseline_q <= 0:
        raise ValueError("baseline must be positive")
    reduction = baseline_q - ecl_q
    relative = reduction / baseline_q
    return {
        "baseline_ece_percent": baseline,
        "ecl_ece_percent": ecl,
        "percentage_point_reduction_fraction": _fraction_text(reduction),
        "percentage_point_reduction_decimal": _decimal_text(reduction),
        "relative_reduction_fraction": _fraction_text(relative),
        "relative_reduction_percent_decimal": _decimal_text(100 * relative),
        "ecl_to_baseline_ratio_fraction": _fraction_text(ecl_q / baseline_q),
    }


def all_pairwise_reductions() -> list[dict[str, str]]:
    rows = []
    for architecture, values in SVHN_TABLE.items():
        ecl = values["ECL"][0]
        for baseline in NON_ORACLE_BASELINES:
            rows.append(
                {
                    "architecture": architecture,
                    "comparison": f"{baseline}-minus-ECL",
                    **exact_reduction(values[baseline][0], ecl),
                }
            )
    return rows


def _rounding_interval(text: str) -> tuple[Q, Q]:
    unsigned = text.lstrip("+-")
    decimal_places = len(unsigned.split(".", 1)[1]) if "." in unsigned else 0
    half_unit = Q(1, 2 * 10**decimal_places)
    center = Q(text)
    return center - half_unit, center + half_unit


def raw_run_gate(
    architecture: str,
    method: str,
    raw_runs: Sequence[str | int | Q] | None,
    *,
    std_convention: str = "sample",
) -> dict[str, object]:
    """Fail-closed check for ten raw runs behind one printed mean +/- std.

    Squared standard-deviation intervals avoid floating point. The paper does
    not declare whether its standard deviation uses n or n-1; callers must
    name the convention. No convention is assumed when raw data are missing.
    """

    if architecture not in SVHN_TABLE or method not in SVHN_TABLE[architecture]:
        raise KeyError(f"unknown Table 2 cell: {architecture}/{method}")
    expected_mean, expected_std = SVHN_TABLE[architecture][method]
    result: dict[str, object] = {
        "architecture": architecture,
        "method": method,
        "expected_runs": 10,
        "printed_mean": expected_mean,
        "printed_std": expected_std,
        "std_convention": std_convention,
        "accepted_as_independent_reproduction": False,
    }
    if raw_runs is None:
        result.update({"observed_runs": 0, "reason": "raw_per_run_values_missing"})
        return result
    values = tuple(value if isinstance(value, Q) else Q(value) for value in raw_runs)
    result["observed_runs"] = len(values)
    if len(values) != 10:
        result["reason"] = "requires_exactly_ten_raw_runs"
        return result
    if std_convention not in {"sample", "population"}:
        raise ValueError("std_convention must be 'sample' or 'population'")
    mean = sum(values, Q(0)) / len(values)
    divisor = len(values) - 1 if std_convention == "sample" else len(values)
    variance = sum(((value - mean) ** 2 for value in values), Q(0)) / divisor
    mean_low, mean_high = _rounding_interval(expected_mean)
    std_low, std_high = _rounding_interval(expected_std)
    mean_matches = mean_low <= mean < mean_high
    std_matches = std_low**2 <= variance < std_high**2
    result.update(
        {
            "exact_mean_fraction": _fraction_text(mean),
            "exact_variance_fraction": _fraction_text(variance),
            "printed_mean_rounding_matches": mean_matches,
            "printed_std_rounding_matches": std_matches,
            "accepted_as_independent_reproduction": mean_matches and std_matches,
            "reason": (
                "ten_raw_runs_recompute_to_printed_summary"
                if mean_matches and std_matches
                else "ten_values_do_not_recompute_to_printed_summary"
            ),
        }
    )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(root: Path | None, snapshot: Mapping[str, object]) -> dict[str, object]:
    expected_files = dict(snapshot["files"])
    result: dict[str, object] = {
        "repository": snapshot["repository"],
        "commit": snapshot["commit"],
        "tree": snapshot["tree"],
        "expected_files": sorted(expected_files),
    }
    if root is None:
        result.update({"local_readback_checked": False, "all_hashes_match": None})
        return result
    actual = {}
    missing = []
    for relative, expected_hash in expected_files.items():
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        actual_hash = _sha256(path)
        actual[relative] = {
            "sha256": actual_hash,
            "expected_sha256": expected_hash,
            "matches": actual_hash == expected_hash,
        }
    extra = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and str(path.relative_to(root)) not in expected_files
    )
    result.update(
        {
            "local_readback_checked": True,
            "files": actual,
            "missing_files": missing,
            "extra_files": extra,
            "all_hashes_match": not missing
            and not extra
            and all(item["matches"] for item in actual.values()),
        }
    )
    return result


def _notebook_audit(path: Path) -> dict[str, object]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    outputs = [output for cell in notebook["cells"] for output in cell.get("outputs", [])]
    return {
        "cell_count": len(notebook["cells"]),
        "executed_cell_count": sum(cell.get("execution_count") is not None for cell in notebook["cells"]),
        "saved_output_count": len(outputs),
        "kernel_python": notebook.get("metadata", {}).get("language_info", {}).get("version"),
        "contains_only_synthetic_2d_workflow": (
            "np.random.uniform" in sources and "SimpleNet" in sources and "num_classes=3" in sources
        ),
        "contains_digit_dataset_names": any(name in sources for name in ("MNIST", "USPS", "SVHN")),
        "contains_table2_values": any(value in sources for value in ("61.9", "21.5", "36.8", "38.4")),
        "contains_ten_run_driver": "range(10)" in sources,
    }


def current_source_semantics(root: Path | None) -> dict[str, object]:
    if root is None or not (root / "main.ipynb").is_file():
        return {"checked": False}
    notebook = _notebook_audit(root / "main.ipynb")
    all_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in root.iterdir()
        if path.is_file()
    )
    return {
        "checked": True,
        "notebook": notebook,
        "digit_training_script_present": (root / "Cali_in_Digit.py").is_file(),
        "named_digit_architectures_present": all(
            name in all_text for name in ("LeNet", "ResNet20", "DenseNet40")
        ),
        "digit_datasets_present": all(name in all_text for name in ("MNIST", "USPS", "SVHN")),
        "table2_values_present": all(value in all_text for value in ("61.9", "21.5", "36.8", "38.4")),
        "ten_run_driver_present": "range(10)" in all_text,
        "checkpoint_files_present": bool(list(root.rglob("*.pth")) + list(root.rglob("*.pt"))),
        "requirements_or_lock_present": any(
            (root / name).is_file()
            for name in ("requirements.txt", "environment.yml", "pyproject.toml", "uv.lock", "poetry.lock")
        ),
    }


def _densenet_classifier_is_tuple(networks_source: str) -> bool:
    tree = ast.parse(networks_source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "DenseNet40":
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign) or not isinstance(child.value, ast.Tuple):
                continue
            for target in child.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "classifier"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    return True
    return False


def predecessor_source_semantics(root: Path | None) -> dict[str, object]:
    if root is None or not (root / "Cali_in_Digit.py").is_file():
        return {"checked": False}
    digit = (root / "Cali_in_Digit.py").read_text(encoding="utf-8")
    networks = (root / "networks.py").read_text(encoding="utf-8")
    combined = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in root.glob("*.py")
    )
    return {
        "checked": True,
        "digit_training_script_present": True,
        "device_hardcoded_cuda0": 'DEVICE = "cuda:0"' in digit,
        "default_target": "MNIST" if 'target_domain = "MNIST"' in digit else None,
        "default_architecture": "ResNet20" if "model = ResNet20().to(DEVICE)" in digit else None,
        "architectures_present": {
            name: f"class {name}" in networks for name in ("LeNet", "ResNet20", "DenseNet40")
        },
        "datasets_downloaded_at_runtime": {
            name: f"{name}(" in digit and "download=True" in digit
            for name in ("MNIST", "USPS", "SVHN")
        },
        "source_merges_other_two_domains": "ConcatDataset" in digit,
        "dataset_train_and_test_splits_concatenated": all(
            marker in digit for marker in ("mnist_domain", "usps_domain", "svhn_domain")
        ),
        "training_epochs": 100 if "num_epochs = 100" in digit else None,
        "batch_size": 256 if "batch_size = 256" in digit else None,
        "ece_bins": 15 if "num_bins=15" in digit else None,
        "explicit_digit_seed_present": any(
            token in digit for token in ("manual_seed", "np.random.seed", "random.seed")
        ),
        "ten_run_driver_present": "range(10)" in digit,
        "cli_or_config_present": "argparse" in digit or bool(list(root.glob("*.yaml")) + list(root.glob("*.json"))),
        "dependency_manifest_present": any(
            (root / name).is_file()
            for name in ("requirements.txt", "environment.yml", "pyproject.toml", "uv.lock", "poetry.lock")
        ),
        "checkpoint_files_present": bool(list(root.rglob("*.pth")) + list(root.rglob("*.pt"))),
        "weights_directory_present": (root / "weights" / "Digit").is_dir(),
        "saved_digit_outputs_present": any(
            path.suffix.lower() in {".csv", ".json", ".txt", ".log"}
            for path in root.iterdir()
            if path.name != "README.md"
        ),
        "table2_values_present": all(value in combined for value in ("61.9", "21.5", "36.8", "38.4")),
        "dense_classifier_accidentally_tuple": _densenet_classifier_is_tuple(networks),
        "fresh_training_save_parent_missing": (
            'torch.save(model.state_dict(), "weights/Digit/' in digit
            and not (root / "weights" / "Digit").is_dir()
        ),
        "method_markers": {
            name: marker in digit
            for name, marker in {
                "Uncal": "run_Uncal_ECE",
                "TS": "run_ECE_TS",
                "TransCal": "run_TransCal",
                "DRL": "run_DRL",
                "PseudoCal": "run_PseudoCali",
                "ECL": "run_ECL_TS",
                "Oracle": "run_Oracle_TS",
            }.items()
        },
    }


def _table_rows() -> dict[str, object]:
    return {
        architecture: {
            method: {"mean_ece_percent": mean, "std_ece_percent": std}
            for method, (mean, std) in values.items()
            if method != "DeltaACC"
        }
        | {
            "DeltaACC": {
                "mean_percentage_points": values["DeltaACC"][0],
                "std_percentage_points": values["DeltaACC"][1],
            }
        }
        for architecture, values in SVHN_TABLE.items()
    }


def build_report(
    *,
    paper: Path | None = None,
    official_root: Path | None = None,
    predecessor_root: Path | None = None,
) -> dict[str, object]:
    official_verification = verify_snapshot(official_root, CURRENT_OFFICIAL)
    predecessor_verification = verify_snapshot(predecessor_root, PUBLIC_PREDECESSOR)
    paper_hash = _sha256(paper) if paper is not None else None
    reductions = all_pairwise_reductions()
    quoted = [
        {
            "architecture": architecture,
            "baseline": baseline,
            **exact_reduction(
                SVHN_TABLE[architecture][baseline][0],
                SVHN_TABLE[architecture]["ECL"][0],
            ),
        }
        for architecture, baseline in QUOTED_COMPARISONS
    ]
    negative_controls = {
        "summary_only": raw_run_gate("LeNet-5", "ECL", None),
        "nine_values": raw_run_gate("LeNet-5", "ECL", ["21.5"] * 9),
        "ten_repeated_printed_means": raw_run_gate("LeNet-5", "ECL", ["21.5"] * 10),
        "rounded_table_arithmetic": {
            "accepted_as_independent_reproduction": False,
            "reason": "exact arithmetic over printed rounded means verifies transcription-derived reductions only",
        },
    }
    release_inventory = {
        "observed_at_utc": "2026-07-19",
        "current_official": {
            "heads": {"main": CURRENT_OFFICIAL["commit"]},
            "tags": [],
            "release_count": 0,
            "workflow_count": 0,
            "actions_artifact_count": 0,
            "fork_count": 0,
            "commit_count": 3,
        },
        "public_predecessor": {
            "heads": {"main": PUBLIC_PREDECESSOR["commit"]},
            "tags": [],
            "release_count": 0,
            "actions_artifact_count": 0,
            "fork_count": 0,
            "commit_count": 2,
        },
        "hugging_face_search": {
            "exact_title_models": [],
            "exact_title_datasets": [],
            "public_reproduction_spaces_found": ["Vassilbek/ecl-icml2026-reproduction"],
            "table2_raw_runs_or_checkpoints_found": False,
        },
    }
    blockers = [
        "the current paper-linked commit contains no digit experiment script or named digit architectures",
        "neither public source snapshot contains the ten per-run Table 2 observations or a seed schedule",
        "neither snapshot contains released digit checkpoints or saved digit result files",
        "neither snapshot pins a dependency environment or dataset checksums/revisions",
        "the predecessor script has no ten-run driver and requires manual target/architecture edits",
        "the predecessor digit script uses batch size 256, while Appendix J states a uniform batch size of 100",
        "the predecessor DenseNet40 classifier is a one-element tuple and is not callable as written",
        "the predecessor fresh-training path saves below weights/Digit but that directory is absent",
        "the paper does not state whether the reported standard deviation uses n or n-1",
        "the authorized ceiling for this attempt excludes the required multi-architecture ten-run GPU training",
    ]
    return {
        "schema_version": 1,
        "paper": {
            "title": "Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift",
            "openreview_id": "gFPPTokv9C",
            "arxiv_version": PAPER_ARXIV_ID,
            "table_anchor": PAPER_TABLE_PAGE,
            "paper_sha256_expected": PAPER_SHA256,
            "paper_sha256_observed": paper_hash,
            "paper_hash_matches": paper_hash == PAPER_SHA256 if paper_hash is not None else None,
            "reported_summary": "mean and standard deviation derived from ten runs",
            "reported_experiment_environment": {
                "os": "Ubuntu 20.04.3 LTS",
                "python": "3.11.11",
                "torch": "2.4.1+cu118",
                "cpu": "Intel Core i7-10700 3.70GHz",
                "memory_gb": "125.5",
                "gpus": "10 NVIDIA GeForce RTX 3090, 24GB each",
                "batch_size": 100,
                "optimizer": "Adam",
                "learning_rate": "0.001",
                "training_epochs": 100,
                "digit_preprocessing": "3-channel RGB, resized to 28x28",
            },
        },
        "table2_svhn_transcription": _table_rows(),
        "pairwise_reductions": reductions,
        "quoted_comparison_certificates": quoted,
        "negative_controls": negative_controls,
        "source_release_inventory": release_inventory,
        "current_official_source": {
            "snapshot": official_verification,
            "semantics": current_source_semantics(official_root),
        },
        "public_predecessor_source": {
            "snapshot": predecessor_verification,
            "semantics": predecessor_source_semantics(predecessor_root),
        },
        "blockers": blockers,
        "assessment": {
            "table2_transcription": "verified_against_rendered_paper",
            "printed_mean_pairwise_arithmetic": "verified_exactly",
            "table2_empirical_result": "not_independently_reproduced",
            "local_classification": "inconclusive_source_only_audit",
            "recommended_claim_verdict": "inconclusive",
            "evidence_for_falsification": False,
            "substantive_empirical_attempt": False,
            "reason": (
                "printed summaries and source provenance are auditable, but no raw ten-run evidence or "
                "faithful rerun exists under this attempt's compute ceiling"
            ),
        },
        "next_distinct_attempt": {
            "approach": "faithful_reconstruction_from_public_predecessor",
            "plan": [
                "pin a compatible CUDA/PyTorch/torchvision/scipy/sklearn/torchquad environment",
                "repair only the DenseNet40 tuple and missing output-directory defects while recording patches",
                "parameterize target SVHN and all three architectures without changing the scientific pipeline",
                "declare ten deterministic seeds and archive each raw ECE/accuracy observation and checkpoint",
                "independently recompute 15-bin top-label ECE from saved logits",
            ],
            "minimum_authority_needed": "GPU compute and dependency installation authorization",
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "arithmetic": "fractions.Fraction for all reported reductions and raw-run gates",
            "training_performed": False,
            "gpu_used": False,
            "paid_or_remote_compute_used": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path)
    parser.add_argument("--official-root", type=Path)
    parser.add_argument("--predecessor-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(
        paper=args.paper,
        official_root=args.official_root,
        predecessor_root=args.predecessor_root,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
