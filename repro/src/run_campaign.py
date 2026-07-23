#!/usr/bin/env python3
"""Run the fixed, fail-closed cumulative reproduction suite.

Every experiment node invokes this entrypoint through the inherited OpenResearch
run command. Child nodes may extend it, but must preserve the baseline steps.
"""
from __future__ import annotations

from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".openresearch" / "artifacts"
BASELINE = ARTIFACTS / "baseline"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def run_step(name: str, arguments: list[str], environment: dict[str, str]) -> dict[str, object]:
    started = time.monotonic()
    completed = subprocess.run(
        arguments,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.monotonic() - started
    atomic_write(BASELINE / "logs" / f"{name}.log", completed.stdout)
    print(f"\n===== {name} =====")
    print(completed.stdout.rstrip())
    print(f"STEP_RESULT name={name} exit_code={completed.returncode} wall_seconds={elapsed:.6f}")
    if completed.returncode != 0:
        raise SystemExit(f"fail-closed step failed: {name}")
    return {
        "name": name,
        "command": arguments,
        "exit_code": completed.returncode,
        "wall_seconds": elapsed,
        "log": str((BASELINE / "logs" / f"{name}.log").relative_to(ROOT)),
    }


def file_manifest() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for artifact in sorted(ARTIFACTS.rglob("*")):
        if not artifact.is_file() or artifact.name == "manifest.json":
            continue
        payload = artifact.read_bytes()
        rows.append(
            {
                "path": str(artifact.relative_to(ROOT)),
                "bytes": len(payload),
                "sha256": sha256(payload).hexdigest(),
            }
        )
    return rows


def main() -> None:
    started = time.monotonic()
    BASELINE.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[variable] = "1"

    python = sys.executable
    steps = [
        (
            "claim_1_exact_certificate",
            [
                python,
                "repro/src/claim1_general_certificate.py",
                "--output",
                ".openresearch/artifacts/claim-1/raw_certificate.json",
            ],
        ),
        (
            "claim_2_hard_bin_scaling",
            [
                python,
                "repro/src/run_claim3_sample_complexity.py",
                "--json-out",
                ".openresearch/artifacts/claim-2/hard_bin_scaling.json",
                "--markdown-out",
                ".openresearch/artifacts/claim-2/hard_bin_EVAL.md",
            ],
        ),
        (
            "claim_2_soft_bin_scaling",
            [
                python,
                "repro/src/run_claim3_soft_sample_complexity.py",
                "--json-out",
                ".openresearch/artifacts/claim-2/soft_bin_scaling.json",
                "--markdown-out",
                ".openresearch/artifacts/claim-2/soft_bin_EVAL.md",
            ],
        ),
        (
            "claim_3_exact_counterexamples",
            [
                python,
                "repro/src/claim3_gradient_certificate.py",
                "--output",
                ".openresearch/artifacts/claim-3/raw_certificate.json",
            ],
        ),
        (
            "claim_4_exact_compatibility",
            [
                python,
                "repro/src/claim2_compatibility_certificate.py",
                "--output",
                ".openresearch/artifacts/claim-4/raw_certificate.json",
            ],
        ),
        (
            "claim_5_source_audit",
            [
                python,
                "repro/src/claim5_table2_audit.py",
                "--paper",
                "repro/evidence/claim3/2605.21552v1.pdf",
                "--official-root",
                "upstream",
                "--output",
                ".openresearch/artifacts/claim-5/source_audit.json",
            ],
        ),
        (
            "claim_6_source_audit",
            [
                python,
                "repro/src/claim6_capability_audit.py",
                "--output",
                ".openresearch/artifacts/claim-6/source_audit.json",
            ],
        ),
        (
            "claim_2_corrected_theorem",
            [
                python,
                "repro/src/claim2_theorem_certificate.py",
                "--output",
                ".openresearch/artifacts/claim-2/route-a/raw_results.json",
            ],
        ),
        (
            "claim_2_independent_checker",
            [
                python,
                "repro/src/claim2_theorem_independent_checker.py",
                "--input",
                ".openresearch/artifacts/claim-2/route-a/raw_results.json",
                "--output",
                ".openresearch/artifacts/claim-2/route-a/independent_checker.json",
            ],
        ),
        (
            "claim_2_soft_falsification_stress",
            [
                python,
                "repro/src/claim2_soft_falsification_stress.py",
                "--output",
                ".openresearch/artifacts/claim-2/route-b/raw_results.json",
            ],
        ),
        (
            "claim_2_soft_stress_checker",
            [
                python,
                "repro/src/claim2_soft_stress_checker.py",
                "--input",
                ".openresearch/artifacts/claim-2/route-b/raw_results.json",
                "--output",
                ".openresearch/artifacts/claim-2/route-b/independent_checker.json",
            ],
        ),
        (
            "claim_4_real_mnist_soft_bins",
            [
                python,
                "repro/src/claim4_real_mnist_soft_bins.py",
                "--inputs",
                ".openresearch/artifacts/claim-4/route-real-mnist/inputs.json",
                "--output",
                ".openresearch/artifacts/claim-4/route-real-mnist/raw_results.json",
            ],
        ),
        (
            "claim_4_real_mnist_independent_checker",
            [
                python,
                "repro/src/claim4_real_mnist_independent_checker.py",
                "--inputs",
                ".openresearch/artifacts/claim-4/route-real-mnist/inputs.json",
                "--results",
                ".openresearch/artifacts/claim-4/route-real-mnist/raw_results.json",
                "--output",
                ".openresearch/artifacts/claim-4/route-real-mnist/independent_checker.json",
            ],
        ),
        (
            "claim_5_predecessor_posthoc",
            [
                python,
                "repro/src/claim5_predecessor_posthoc.py",
                "--predictions",
                ".openresearch/artifacts/claim-5/route-2-posthoc/predictions.csv",
                "--output",
                ".openresearch/artifacts/claim-5/route-2-posthoc/raw_results.json",
            ],
        ),
        (
            "claim_5_posthoc_independent_checker",
            [
                python,
                "repro/src/claim5_posthoc_independent_checker.py",
                "--predictions",
                ".openresearch/artifacts/claim-5/route-2-posthoc/predictions.csv",
                "--results",
                ".openresearch/artifacts/claim-5/route-2-posthoc/raw_results.json",
                "--output",
                ".openresearch/artifacts/claim-5/route-2-posthoc/independent_checker.json",
            ],
        ),
        (
            "cumulative_pytest",
            [python, "-m", "pytest", "repro/tests", "-q"],
        ),
    ]

    results = [run_step(name, command, environment) for name, command in steps]
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    runtime = {
        "git_sha": git_sha,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "packages": {
            package: version(package)
            for package in (
                "marimo",
                "matplotlib",
                "numpy",
                "pillow",
                "pytest",
                "scipy",
                "torch",
                "torchvision",
            )
        },
        "deterministic_thread_ceiling": 1,
        "wall_seconds": time.monotonic() - started,
    }
    summary = {
        "status": "PASS",
        "scope": "frozen baseline cumulative regression",
        "runtime": runtime,
        "steps": results,
        "limitations": [
            "The frozen baseline subset preserves the prior exact and synthetic checks; upgrades are reported only by their separate child-route artifacts.",
            "The Claim 4 child route downloads hash-audited MNIST into the shared user cache; the dataset is not vendored in this repository.",
            "Later claim experiments must retain every baseline step and use the same locked environment and command.",
        ],
    }
    atomic_write(BASELINE / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    eval_markdown = "\n".join(
        [
            "# Frozen cumulative baseline",
            "",
            "Verdict: **PASS** for regression preservation only.",
            "",
            f"- Git SHA: `{git_sha}`",
            f"- Python: `{runtime['python']}`",
            f"- Platform: `{runtime['platform']}`",
            f"- Wall time: `{runtime['wall_seconds']:.3f}` seconds",
            f"- Steps passed: `{len(results)}/{len(results)}`",
            "",
            "This baseline does not upgrade any below-full-credit claim.",
            "",
        ]
    )
    atomic_write(BASELINE / "EVAL.md", eval_markdown)
    atomic_write(
        ARTIFACTS / "manifest.json",
        json.dumps({"git_sha": git_sha, "files": file_manifest()}, indent=2, sort_keys=True) + "\n",
    )
    print("\n===== FINAL EVAL =====")
    print(eval_markdown.rstrip())
    print(f"CAMPAIGN_RESULT status=PASS steps={len(results)} git_sha={git_sha}")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
