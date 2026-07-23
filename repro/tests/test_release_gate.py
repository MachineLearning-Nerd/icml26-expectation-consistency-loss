from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_release_summary_has_only_terminal_verdicts() -> None:
    summary = json.loads(
        (ROOT / "reports/ecl-covariate-shift/data/summary.json").read_text(encoding="utf-8")
    )
    assert len(summary["claims"]) == 6
    assert {row["verdict"] for row in summary["claims"]} <= {
        "VERIFIED",
        "FALSIFIED",
        "BLOCKED",
    }
    assert sum(row["current_points"] for row in summary["claims"]) == 6
    assert sum(row["forecast_points"] for row in summary["claims"]) == 10


def test_report_opens_with_evidence_and_all_images_exist() -> None:
    report_dir = ROOT / "reports/ecl-covariate-shift"
    lines = [
        line
        for line in (report_dir / "report.md").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines[0].startswith("# ")
    assert lines[1].startswith("![")
    for name in (
        "headline-exact-counterexamples.png",
        "claim2-proof-budget.png",
        "claim4-independent-agreement.png",
        "claim6-simulation.png",
    ):
        assert (report_dir / "images" / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
