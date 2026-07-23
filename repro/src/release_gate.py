#!/usr/bin/env python3
"""Fail-closed validation for the reader-facing release candidate."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "ecl-covariate-shift"
SUMMARY_PATH = REPORT_DIR / "data" / "summary.json"
RELEASE_CONTRACT = ROOT / ".openresearch" / "artifacts" / "release" / "claim_contract.json"
FIXED_COMMAND = "uv run --frozen --python 3.12 python repro/src/run_campaign.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    contract = json.loads(RELEASE_CONTRACT.read_text(encoding="utf-8"))
    claims = summary["claims"]

    require(summary["previous_live_judged_score"] == "6/12", "wrong live baseline")
    require(summary["conservative_projected_range"] == "8-10/12", "wrong forecast range")
    require(summary["fixed_command"] == FIXED_COMMAND, "fixed command changed")
    require(len(claims) == 6, "release must cover exactly six claims")
    require([row["claim"] for row in claims] == list(range(1, 7)), "claim ids are incomplete")

    allowed = set(contract["allowed_verdicts"])
    require(allowed == {"VERIFIED", "FALSIFIED", "BLOCKED"}, "invalid verdict vocabulary")
    require(all(row["verdict"] in allowed for row in claims), "nonterminal verdict found")
    require(sum(row["current_points"] for row in claims) == 6, "current points must remain 6")
    require(sum(row["forecast_points"] for row in claims) == 10, "best forecast must be 10")
    require(claims[4]["verdict"] == "BLOCKED", "Claim 5 must remain blocked")

    report = (REPORT_DIR / "report.md").read_text(encoding="utf-8")
    report_lines = [line for line in report.splitlines() if line.strip()]
    require(report_lines[0].startswith("# "), "report needs one H1 title")
    require(report_lines[1].startswith("!["), "strongest evidence must follow the title")
    require(report.count("\n# ") == 0, "report contains more than one H1")
    image_refs = re.findall(r"!\[[^\]]*\]\((images/[^)]+)\)", report)
    require(len(image_refs) == 4, "report must reference four evidence figures")
    for reference in image_refs:
        image = REPORT_DIR / reference
        require(image.is_file(), f"missing report image: {reference}")
        payload = image.read_bytes()
        require(payload.startswith(b"\x89PNG\r\n\x1a\n"), f"not a PNG: {reference}")
        require(len(payload) > 10_000, f"implausibly small figure: {reference}")

    notebook = ROOT / "notebooks" / "ecl_reproduction.py"
    require(notebook.is_file(), "marimo notebook missing")
    notebook_text = notebook.read_text(encoding="utf-8")
    require("previous live score is **6/12**" in notebook_text, "notebook lacks baseline")
    require("BLOCKED" in notebook_text and "FALSIFIED" in notebook_text, "notebook hides outcomes")

    eval_paths = [
        ROOT / ".openresearch" / "artifacts" / "claim-2" / "route-c-soft-proof" / "EVAL.md",
        ROOT / ".openresearch" / "artifacts" / "claim-4" / "route-real-mnist" / "EVAL.md",
        ROOT / ".openresearch" / "artifacts" / "claim-5" / "route-4-falsification" / "EVAL.md",
        ROOT / ".openresearch" / "artifacts" / "claim-6" / "route-2-table1" / "EVAL.md",
    ]
    for path in eval_paths:
        text = path.read_text(encoding="utf-8")
        require("PENDING RUN" not in text, f"stale pending EVAL: {path}")
        require(any(f"**{verdict}**" in text for verdict in allowed), f"no terminal verdict: {path}")

    check = subprocess.run(
        [sys.executable, "-m", "marimo", "check", "--strict", str(notebook)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(check.stdout.rstrip())
    require(check.returncode == 0, "marimo strict validation failed")

    print(
        "RELEASE_GATE status=PASS claims=6 current_points=6 "
        "forecast_points=10 figures=4 notebook=PASS"
    )


if __name__ == "__main__":
    main()
