"""Fail-closed tests for the Claim 5 Table 2 source/provenance audit."""

from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

import claim5_table2_audit as c5


ROOT = Path(__file__).resolve().parents[2]


def test_transcribed_claim_cells_match_rendered_table2() -> None:
    assert c5.SVHN_TABLE["LeNet-5"]["Uncal"] == ("61.9", "6.16")
    assert c5.SVHN_TABLE["LeNet-5"]["ECL"] == ("21.5", "1.51")
    assert c5.SVHN_TABLE["ResNet20"]["PseudoCal"] == ("48.2", "3.95")
    assert c5.SVHN_TABLE["ResNet20"]["ECL"] == ("36.8", "2.08")
    assert c5.SVHN_TABLE["DenseNet40"]["Uncal"] == ("80.8", "6.26")
    assert c5.SVHN_TABLE["DenseNet40"]["ECL"] == ("38.4", "3.21")


def test_quoted_reductions_use_exact_decimal_arithmetic() -> None:
    lenet = c5.exact_reduction("61.9", "21.5")
    resnet = c5.exact_reduction("48.2", "36.8")
    dense = c5.exact_reduction("80.8", "38.4")
    assert lenet["percentage_point_reduction_fraction"] == "202/5"
    assert lenet["relative_reduction_fraction"] == "404/619"
    assert resnet["percentage_point_reduction_fraction"] == "57/5"
    assert resnet["relative_reduction_fraction"] == "57/241"
    assert dense["percentage_point_reduction_fraction"] == "212/5"
    assert dense["relative_reduction_fraction"] == "53/101"


def test_all_ecl_nonoracle_pairwise_reductions_are_enumerated_and_positive() -> None:
    rows = c5.all_pairwise_reductions()
    assert len(rows) == 3 * len(c5.NON_ORACLE_BASELINES) == 24
    assert all(Fraction(row["percentage_point_reduction_fraction"]) > 0 for row in rows)
    assert {(row["architecture"], row["comparison"]) for row in rows} == {
        (architecture, f"{baseline}-minus-ECL")
        for architecture in c5.SVHN_TABLE
        for baseline in c5.NON_ORACLE_BASELINES
    }


def test_summary_only_and_wrong_run_count_fail_closed() -> None:
    absent = c5.raw_run_gate("LeNet-5", "ECL", None)
    nine = c5.raw_run_gate("LeNet-5", "ECL", ["21.5"] * 9)
    assert absent["accepted_as_independent_reproduction"] is False
    assert absent["reason"] == "raw_per_run_values_missing"
    assert nine["accepted_as_independent_reproduction"] is False
    assert nine["reason"] == "requires_exactly_ten_raw_runs"


def test_repeating_rounded_mean_ten_times_does_not_fake_reported_std() -> None:
    result = c5.raw_run_gate("LeNet-5", "ECL", ["21.5"] * 10)
    assert result["printed_mean_rounding_matches"] is True
    assert result["printed_std_rounding_matches"] is False
    assert result["exact_variance_fraction"] == "0"
    assert result["accepted_as_independent_reproduction"] is False


def test_raw_run_gate_can_accept_exact_ten_run_evidence() -> None:
    # Synthetic control with exact mean 21.5 and sample variance 2.25. The
    # printed std is temporarily changed to 1.50 to test the gate itself.
    original = c5.SVHN_TABLE["LeNet-5"]["ECL"]
    c5.SVHN_TABLE["LeNet-5"]["ECL"] = ("21.5", "1.50")
    try:
        # Five 20.15s and five 22.85s: mean 21.5, sample variance 2.025,
        # so this deliberately should not pass the 1.50 control.
        rejected = c5.raw_run_gate(
            "LeNet-5", "ECL", ["20.15"] * 5 + ["22.85"] * 5
        )
        assert rejected["accepted_as_independent_reproduction"] is False
        # Ten exact values with deviations +/-1.423024947 are unnecessary:
        # change the printed std to the exact rational-compatible sqrt(2.025)
        # rounded at two decimals and the same raw values pass.
        c5.SVHN_TABLE["LeNet-5"]["ECL"] = ("21.5", "1.42")
        accepted = c5.raw_run_gate(
            "LeNet-5", "ECL", ["20.15"] * 5 + ["22.85"] * 5
        )
        assert accepted["accepted_as_independent_reproduction"] is True
    finally:
        c5.SVHN_TABLE["LeNet-5"]["ECL"] = original


def test_current_official_readback_hashes_match_and_lacks_digit_path() -> None:
    result = c5.verify_snapshot(ROOT / "upstream", c5.CURRENT_OFFICIAL)
    semantics = c5.current_source_semantics(ROOT / "upstream")
    assert result["all_hashes_match"] is True
    assert semantics["notebook"]["contains_only_synthetic_2d_workflow"] is True
    assert semantics["digit_training_script_present"] is False
    assert semantics["named_digit_architectures_present"] is False
    assert semantics["digit_datasets_present"] is False
    assert semantics["table2_values_present"] is False
    assert semantics["ten_run_driver_present"] is False
    assert semantics["checkpoint_files_present"] is False
    assert semantics["requirements_or_lock_present"] is False


def test_snapshot_hash_verification_rejects_mutation_and_extras(tmp_path: Path) -> None:
    for relative in c5.CURRENT_OFFICIAL["files"]:
        source = ROOT / "upstream" / relative
        (tmp_path / relative).write_bytes(source.read_bytes())
    (tmp_path / "losses.py").write_text("mutated\n", encoding="utf-8")
    (tmp_path / "extra.txt").write_text("unexpected\n", encoding="utf-8")
    result = c5.verify_snapshot(tmp_path, c5.CURRENT_OFFICIAL)
    assert result["all_hashes_match"] is False
    assert result["files"]["losses.py"]["matches"] is False
    assert result["extra_files"] == ["extra.txt"]


def test_snapshot_verification_ignores_generated_python_cache(tmp_path: Path) -> None:
    for relative in c5.CURRENT_OFFICIAL["files"]:
        source = ROOT / "upstream" / relative
        (tmp_path / relative).write_bytes(source.read_bytes())
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "losses.cpython-312.pyc").write_bytes(b"generated cache")
    result = c5.verify_snapshot(tmp_path, c5.CURRENT_OFFICIAL)
    assert result["all_hashes_match"] is True
    assert result["extra_files"] == []


def test_predecessor_semantics_detects_dense_tuple_and_missing_release_gates(tmp_path: Path) -> None:
    (tmp_path / "Cali_in_Digit.py").write_text(
        '\n'.join([
            'DEVICE = "cuda:0"',
            'target_domain = "MNIST"',
            'model = ResNet20().to(DEVICE)',
            'batch_size = 256',
            'num_epochs = 100',
            'x = MNIST(download=True); y = USPS(download=True); z = SVHN(download=True)',
            'mnist_domain = ConcatDataset([]); usps_domain = ConcatDataset([]); svhn_domain = ConcatDataset([])',
            'run_Uncal_ECE=True; run_ECE_TS=True; run_TransCal=True; run_DRL=True',
            'run_PseudoCali=True; run_ECL_TS=True; run_Oracle_TS=True',
            'num_bins=15',
            'torch.save(model.state_dict(), "weights/Digit/x.pth")',
        ]),
        encoding="utf-8",
    )
    (tmp_path / "networks.py").write_text(
        "class LeNet: pass\nclass ResNet20: pass\n"
        "class DenseNet40:\n"
        "    def __init__(self):\n"
        "        self.classifier = nn.Linear(4, 10),\n",
        encoding="utf-8",
    )
    result = c5.predecessor_source_semantics(tmp_path)
    assert result["dense_classifier_accidentally_tuple"] is True
    assert result["explicit_digit_seed_present"] is False
    assert result["ten_run_driver_present"] is False
    assert result["dependency_manifest_present"] is False
    assert result["checkpoint_files_present"] is False
    assert result["weights_directory_present"] is False
    assert result["fresh_training_save_parent_missing"] is True


def test_paper_hash_and_report_assessment_are_fail_closed() -> None:
    paper = ROOT / "repro" / "evidence" / "claim3" / "2605.21552v1.pdf"
    report = c5.build_report(paper=paper, official_root=ROOT / "upstream")
    assert report["paper"]["paper_hash_matches"] is True
    assert len(report["pairwise_reductions"]) == 24
    assert report["assessment"]["printed_mean_pairwise_arithmetic"] == "verified_exactly"
    assert report["assessment"]["table2_empirical_result"] == "not_independently_reproduced"
    assert report["assessment"]["recommended_claim_verdict"] == "inconclusive"
    assert report["assessment"]["substantive_empirical_attempt"] is False
    assert report["paper"]["reported_experiment_environment"]["batch_size"] == 100


def test_cli_writes_deterministic_json(tmp_path: Path) -> None:
    paper = ROOT / "repro" / "evidence" / "claim3" / "2605.21552v1.pdf"
    output = tmp_path / "audit.json"
    command = [
        sys.executable,
        str(ROOT / "repro" / "src" / "claim5_table2_audit.py"),
        "--paper",
        str(paper),
        "--official-root",
        str(ROOT / "upstream"),
        "--output",
        str(output),
    ]
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    first_bytes = output.read_bytes()
    second = subprocess.run(command, check=True, capture_output=True, text=True)
    assert output.read_bytes() == first_bytes
    assert first.stdout == second.stdout
    assert json.loads(first.stdout) == json.loads(first_bytes)
