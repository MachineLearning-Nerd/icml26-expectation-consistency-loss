#!/usr/bin/env python3
"""Fail-closed verification of the ECL publication surface."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


EXPECTED_NAME = "icml26-expectation-consistency-loss"
EXPECTED_IDENTITY = "MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>"
EXPECTED_BRANCHES = {
    "main",
    "experiment/frozen-cumulative-baseline",
    "audit/claim-2-corrected-finite-sample",
    "audit/claim-2-soft-bin-concentration",
    "audit/claim-2-and-4-real-mnist",
    "audit/claim-2-falsification-stress",
    "audit/claim-5-lenet-svhn",
    "audit/claim-5-predecessor-posthoc",
    "audit/claim-5-stabilized-appendix",
    "audit/claim-5-falsification",
    "audit/claim-6-simulation",
    "audit/claim-6-table1-falsification",
    "release/candidate-evidence",
}


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def read_json(root: Path, relative: str) -> dict:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    checks: dict[str, bool] = {}
    details: dict[str, str] = {}

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks[name] = bool(condition)
        if detail:
            details[name] = detail

    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    sources = read_json(root, "sources.json")
    evidence = read_json(root, "evidence/claim_summary.json")
    summary = read_json(root, "reports/ecl-covariate-shift/data/summary.json")
    theorem = read_json(root, "repro/evidence/2026-07-24/artifacts/claim-1/raw_certificate.json")
    c3 = read_json(root, "repro/evidence/2026-07-24/artifacts/claim-3/raw_certificate.json")
    c4 = read_json(root, "repro/evidence/2026-07-24/artifacts/claim-4/route-real-mnist/independent_checker.json")
    c5 = read_json(root, "repro/evidence/2026-07-24/artifacts/claim-5/source_audit.json")

    check("project_name", f'name = "{EXPECTED_NAME}"' in pyproject)
    check("paper_identity", sources["paper"]["openreview_id"] == "gFPPTokv9C"
          and sources["paper"]["arxiv_id"] == "2605.21552")
    check("source_hashes", len(sources["paper"]["source_pdf_sha256"]) == 64
          and len(sources["official_implementation"]["file_sha256"]) == 64)
    check("citation_and_thanks", "@article{dong2026expectation" in readme
          and "Thank you to Jinzong Dong" in readme)
    check("required_docs", all(
        (root / path).is_file()
        for path in (
            "README.md",
            "STATUS.md",
            "sources.json",
            "evidence/claim_summary.json",
            "docs/CLAIM_EVIDENCE.md",
            "docs/BRANCH_AUDIT.md",
            "docs/SOURCE_AUDIT.md",
            "docs/PUBLICATION_GATE.md",
            "docs/research_log.md",
            "reports/ecl-covariate-shift/report.md",
        )
    ))
    check("six_claim_summary", len(summary["claims"]) == 6
          and [row["claim"] for row in summary["claims"]] == [1, 2, 3, 4, 5, 6]
          and sum(row["current_points"] for row in summary["claims"]) == 6)
    check("claim_statuses", evidence["claims"]["C1"]["status"] == "VERIFIED_SCOPED"
          and evidence["claims"]["C2"]["status"] == "VERIFIED_SCOPED_WITH_QUALIFICATION"
          and evidence["claims"]["C3"]["status"] == "FALSIFIED_AS_STATED"
          and evidence["claims"]["C4"]["status"] == "VERIFIED_SCOPED"
          and evidence["claims"]["C5"]["status"] == "BLOCKED"
          and evidence["claims"]["C6"]["status"] == "FALSIFIED_SCOPED")
    check("c1_certificate", theorem["assessment"]["all_certificate_gates_pass"] is True
          and theorem["assessment"]["literal_theorem_3_1"] == "verified_with_support_and_version_qualifications")
    check("c3_counterexamples", c3["assessment"]["anchored_claim_3"] == "contradicted_as_stated"
          and c3["verification_gates"]["eq10_gradient_counterexample"] is True
          and c3["verification_gates"]["same_batch_direction_bias_counterexample"] is True)
    check("c4_independent_checker", c4["status"] == "VERIFIED"
          and c4["all_three_losses_and_gradients_match"] is True)
    check("c5_blocker", c5["assessment"]["local_classification"] == "inconclusive_source_only_audit"
          and c5["assessment"]["table2_empirical_result"] == "not_independently_reproduced")

    tracked = git(root, "ls-files").splitlines()
    check("no_stale_state", "logbook.json" not in tracked
          and not any(path == ".trackio" or path.startswith(".trackio/") for path in tracked))
    branch_output = git(root, "branch", "-a")
    final_refs = set()
    for line in branch_output.splitlines():
        ref = line.strip().removeprefix("*").strip()
        if ref.startswith("remotes/origin/"):
            ref = ref.removeprefix("remotes/origin/")
        final_refs.add(ref)
    check("branch_surface", "master" not in branch_output and "orx/" not in branch_output
          and EXPECTED_BRANCHES <= final_refs, str(sorted(final_refs)))
    check("canonical_branch", git(root, "branch", "--show-current") == "main")
    remote = git(root, "remote", "get-url", "origin")
    check("final_remote", EXPECTED_NAME in remote, remote)

    identities = git(root, "log", "--all", "--format=%an <%ae>%n%cn <%ce>").splitlines()
    unexpected = sorted({item for item in identities if item != EXPECTED_IDENTITY})
    check("commit_identity", bool(identities) and not unexpected, ", ".join(unexpected))
    private_path_marker = "/Users/" + "dineshjinjala/"
    content = "\n".join((root / path).read_text(errors="ignore") for path in tracked if (root / path).is_file())
    check("no_private_workspace_path", "dinesh.jinjala@mareana.com" not in git(root, "log", "--all", "--format=%B")
          and private_path_marker not in content)
    report = (root / "reports/ecl-covariate-shift/report.md").read_text(encoding="utf-8")
    check("active_links_clean",
          "https://github.com/MachineLearning-Nerd/icml26-repro-gFPPTokv9C-ecl-calibration-covariate-shift" not in readme
          and "https://github.com/MachineLearning-Nerd/icml26-repro-gFPPTokv9C-ecl-calibration-covariate-shift/tree/" not in report)

    result = {
        "repository": EXPECTED_NAME,
        "paper": "gFPPTokv9C",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "details": details,
    }
    output = root / "outputs/verification.json"
    if not args.no_write:
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
