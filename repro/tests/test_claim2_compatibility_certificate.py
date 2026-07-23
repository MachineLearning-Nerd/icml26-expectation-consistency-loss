#!/usr/bin/env python3
"""Regression tests for the exact Claim 2 three-paradigm certificate."""

from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import claim2_compatibility_certificate as certificate


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_fixture_is_strict_common_support_covariate_shift() -> None:
    fixture = certificate.make_fixture(calibrated=False)
    assert len(fixture.atoms) == 12
    assert fixture.source_mass != fixture.target_mass
    assert all(mass > 0 for mass in fixture.source_mass)
    assert all(mass > 0 for mass in fixture.target_mass)
    assert sum(fixture.source_mass, start=Fraction(0)) == 1
    assert sum(fixture.target_mass, start=Fraction(0)) == 1
    # Covariate shift: the posterior is stored once per atom and is therefore
    # shared by source and target; only P(X) changes.
    assert all(sum(atom.posterior, start=Fraction(0)) == 1 for atom in fixture.atoms)


def test_calibrated_fixture_uses_q_equal_to_score() -> None:
    fixture = certificate.make_fixture(calibrated=True)
    assert all(atom.posterior == atom.score for atom in fixture.atoms)


def test_canonical_identity_and_losses_are_exact_rationals() -> None:
    zero = certificate.canonical_certificate(certificate.make_fixture(calibrated=True))
    diagnostic = certificate.canonical_certificate(certificate.make_fixture(calibrated=False))
    assert zero["identity_exact"] is True
    assert zero["ecl_l1_loss"] == Fraction(0)
    assert diagnostic["identity_exact"] is True
    assert diagnostic["ecl_l1_loss"] == Fraction(1, 6)
    assert diagnostic["summary_state_count"] == 6
    assert diagnostic["ecl_residual"] == diagnostic["direct_calibration_gap"]


def test_classwise_executes_every_class_and_exact_scalar_grouping() -> None:
    zero = certificate.classwise_certificate(certificate.make_fixture(calibrated=True))
    diagnostic = certificate.classwise_certificate(certificate.make_fixture(calibrated=False))
    assert zero["classes_executed"] == 3
    assert zero["identity_exact"] is True
    assert zero["ecl_l1_loss"] == Fraction(0)
    assert diagnostic["identity_exact"] is True
    assert diagnostic["ecl_l1_loss"] == Fraction(1, 6)
    assert set(diagnostic["per_class"]) == {"class_0", "class_1", "class_2"}
    for class_result in diagnostic["per_class"].values():
        assert class_result["identity_exact"] is True
        assert class_result["ecl_residual"] == class_result["direct_calibration_gap"]


def test_toplabel_uses_fixed_classifier_correctness_event() -> None:
    fixture = certificate.make_fixture(calibrated=False)
    result = certificate.toplabel_certificate(fixture)
    assert result["identity_exact"] is True
    assert result["ecl_l1_loss"] == Fraction(1, 48)
    assert result["summary_state_count"] == 2
    assert result["predicted_classes_exercised"] == [0, 1, 2]
    for atom in fixture.atoms:
        assert atom.correct_event_posterior == atom.posterior[atom.predicted_class]


def test_toplabel_zero_loss_for_exactly_calibrated_fixture() -> None:
    result = certificate.toplabel_certificate(certificate.make_fixture(calibrated=True))
    assert result["identity_exact"] is True
    assert result["ecl_l1_loss"] == 0


def test_canonical_wrong_grouping_is_numerically_rejected() -> None:
    controls = certificate.negative_controls(certificate.make_fixture(calibrated=False))
    control = controls["canonical_wrong_grouping"]
    assert control["partition_mismatch"] is True
    assert control["correct_state_count"] == 6
    assert control["wrong_state_count"] == 2
    assert control["correct_loss"] == Fraction(1, 12)
    assert control["wrong_loss"] == Fraction(1, 18)
    assert control["rejected"] is True


def test_classwise_wrong_coordinate_is_numerically_rejected() -> None:
    controls = certificate.negative_controls(certificate.make_fixture(calibrated=False))
    control = controls["classwise_wrong_coordinate"]
    assert control["partition_mismatch"] is True
    assert control["correct_loss"] == Fraction(1, 72)
    assert control["wrong_loss"] == Fraction(1, 48)
    assert control["rejected"] is True


def test_toplabel_max_posterior_substitution_is_rejected() -> None:
    controls = certificate.negative_controls(certificate.make_fixture(calibrated=False))
    control = controls["toplabel_wrong_event"]
    counterexample = control["semantic_counterexample"]
    assert counterexample["predicted_class"] == 0
    assert counterexample["correct_event_posterior"] == Fraction(1, 5)
    assert counterexample["incorrect_max_posterior"] == Fraction(7, 10)
    assert control["rejected"] is True


def test_broken_shared_posterior_invalidates_all_three_identities() -> None:
    controls = certificate.negative_controls(certificate.make_fixture(calibrated=False))
    control = controls["broken_shared_posterior"]
    assert control["canonical_identity_rejected"] is True
    assert control["classwise_identity_rejected"] is True
    assert control["toplabel_identity_rejected"] is True
    assert control["rejected"] is True


def test_every_negative_control_is_rejected() -> None:
    controls = certificate.negative_controls(certificate.make_fixture(calibrated=False))
    assert controls["all_rejected"] is True


def test_official_source_sha_and_all_mode_semantics_are_pinned() -> None:
    audit = certificate.audit_official_source(REPOSITORY_ROOT)
    assert audit["commit"] == certificate.OFFICIAL_COMMIT
    assert audit["sha256"] == certificate.OFFICIAL_LOSSES_SHA256
    assert audit["sha256_matches_pin"] is True
    assert audit["all_three_branches_present"] is True
    assert set(audit["semantic_checks"]) == {"TopLabel", "Classwise", "Canonical"}
    assert all(line_number > 0 for line_number in audit["line_numbers"].values())


def test_report_meets_all_predeclared_success_criteria() -> None:
    report = certificate.build_report(REPOSITORY_ROOT)
    assert report["all_success_criteria_pass"] is True
    assert all(report["success_criteria"].values())
    assert report["scope"]["cost_usd"] == 0
    assert report["scope"]["random_seeds"] == "none; deterministic exact arithmetic"


def test_report_contains_no_floating_point_scientific_values() -> None:
    report = certificate.build_report(REPOSITORY_ROOT)

    def walk(value: object) -> None:
        assert not isinstance(value, float)
        if isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                walk(child)

    walk(report)


def test_cli_writes_readable_success_artifact(tmp_path: Path) -> None:
    output = tmp_path / "claim2_certificate.json"
    completed = subprocess.run(
        [
            sys.executable,
            "repro/src/claim2_compatibility_certificate.py",
            "--output",
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["all_success_criteria_pass"] is True
    assert "diagnostic canonical ECL L1 loss: 1/6" in completed.stdout
    assert "negative controls all rejected: True" in completed.stdout
