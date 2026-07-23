"""Tests for the anchored Claim 6 deterministic capability/evidence audit."""

from decimal import Decimal

import claim6_capability_audit as c6


def test_paper_and_official_source_are_exactly_pinned() -> None:
    report = c6.build_report()
    assert report["paper"]["matches_expected_pdf"]
    source = report["official_source"]
    assert source["declared_commit"] == c6.EXPECTED_OFFICIAL_COMMIT
    assert source["file_inventory_exact"]
    assert source["all_hashes_match_pin"]
    assert source["observed_remote_state_2026_07_19"]["commit_count"] == 3
    assert source["observed_remote_state_2026_07_19"]["tag_count"] == 0
    assert source["observed_remote_state_2026_07_19"]["release_count"] == 0


def test_table1_is_exact_and_ecl_is_unique_only_within_selected_rows() -> None:
    table1 = c6.build_report()["table1_capability_audit"]
    assert table1["columns"] == list(c6.CAPABILITY_COLUMNS)
    assert len(table1["rows"]) == 9
    assert table1["methods_with_all_five_in_paper_table"] == ["ECL"]
    assert table1["ecl_is_unique_all_true_within_selected_table"]
    assert not table1["independent_global_uniqueness_established"]


def test_table1_key_negative_and_positive_cells() -> None:
    rows = c6.build_report()["table1_capability_audit"]["rows"]
    assert rows["ECE_KDE"] == {
        "covariate_shift": False,
        "classwise_calibration": True,
        "canonical_calibration": True,
        "unbounded_density_ratio": True,
        "minibatch_trainable": True,
    }
    assert rows["PseudoCal"]["covariate_shift"]
    assert rows["PseudoCal"]["unbounded_density_ratio"]
    assert not rows["PseudoCal"]["minibatch_trainable"]
    assert all(rows["ECL"].values())


def test_official_source_exposes_three_modes_two_domains_and_minibatch_state() -> None:
    report = c6.build_report()
    source = report["official_source"]
    assert source["ecl_modes"] == ["Canonical", "Classwise", "TopLabel"]
    assert source["all_three_modes_implemented"]
    assert source["two_domain_forward_inputs"]
    assert all(source["stateful_minibatch_markers"].values())
    assert source["density_ratio_tokens"] == []
    assert source["no_density_ratio_operation_in_official_losses"]
    assert report["table1_capability_audit"]["official_source_supports_ecl_surface"]


def test_notebook_provenance_is_single_mode_and_not_figure2() -> None:
    notebook = c6.build_report()["official_notebook_audit"]
    assert notebook["cell_count"] == 8
    assert notebook["code_cell_count"] == 8
    assert notebook["markdown_cell_count"] == 0
    assert notebook["configured_paradigm"] == "TopLabel"
    assert notebook["configured_shift"] == "uniform"
    assert notebook["paper_figure2_shift"] == "normal"
    assert not notebook["matches_paper_figure2_setting"]
    assert notebook["calls_seeded_initializer"]
    assert notebook["seed_is_42_in_imported_utils"]


def test_notebook_outputs_are_embedded_images_not_raw_runs() -> None:
    notebook = c6.build_report()["official_notebook_audit"]
    assert notebook["display_png_count"] == 2
    assert notebook["embedded_pngs_expected"]
    assert notebook["structured_result_files"] == []
    assert not notebook["contains_structured_result_export"]
    assert not notebook["prints_machine_readable_figure_metrics"]
    assert not notebook["contains_checkpoint_save_or_load"]
    assert not notebook["contains_pacs_loader_or_path"]
    assert not notebook["contains_imagenet_loader_or_path"]


def test_notebook_execution_and_environment_provenance_are_incomplete() -> None:
    notebook = c6.build_report()["official_notebook_audit"]
    assert notebook["executed_code_cell_count"] == 7
    assert notebook["null_execution_count_cells_with_retained_output"] == [5]
    assert not notebook["execution_provenance_internally_complete"]
    assert notebook["notebook_python"] == "3.12.12"
    assert notebook["paper_python"] == "3.11.11"
    assert not notebook["environment_version_matches_paper"]


def test_figure2_reported_error_arithmetic_and_accuracy_caveat() -> None:
    figure = c6.build_report()["figure2_audit"]
    assert figure["all_six_ecl_errors_lower_than_both_displayed_baselines"]
    assert not figure["all_six_ecl_accuracies_at_least_nll"]
    pacs_top = figure["paper_values"]["pacs_three_classes"]["top_label"]
    assert pacs_top["ecl_minus_nll_accuracy_points"] == "-8.0"
    assert pacs_top["ecl_error_lower_than_both_displayed_baselines"]
    assert not figure["official_notebook_matches_figure2_setting"]
    assert not figure["official_notebook_has_pacs"]
    assert not figure["independently_reproduced"]


def test_table3_all_nine_reported_means_have_positive_gap() -> None:
    table3 = c6.build_report()["table3_audit"]
    assert table3["dataset_count"] == 3
    assert table3["comparison_count"] == 9
    assert table3["all_nine_reported_source_means_lower_than_targets"]
    assert table3["contains_pacs"]
    assert not table3["contains_simulated_dataset"]
    for metrics in table3["comparisons"].values():
        for comparison in metrics.values():
            assert comparison["source_mean_lower_than_target_mean"]
            assert Decimal(comparison["target_minus_source_points"]) > 0
            assert Decimal(comparison["target_over_source_ratio"]) > 1


def test_table3_pacs_arithmetic_is_exact() -> None:
    pacs = c6.build_report()["table3_audit"]["comparisons"][
        "PACS (Art + Cartoon + Sketch -> Photo)"
    ]
    assert pacs["ECE"]["target_minus_source_points"] == "18.46"
    assert pacs["CwECE"]["target_minus_source_points"] == "7.29"
    assert pacs["ECE_KDE"]["target_minus_source_points"] == "7.16"
    assert pacs["ECE"]["target_over_source_ratio"] == "5.807292"


def test_table3_cannot_support_significance_or_independent_reproduction() -> None:
    table3 = c6.build_report()["table3_audit"]
    assert not table3["raw_runs_available"]
    assert not table3["sample_count_explicit_in_table3_caption"]
    assert not table3["statistical_significance_test_reported"]
    assert not table3["official_notebook_has_table3_datasets"]
    assert not table3["independently_reproduced"]


def test_all_negative_controls_reject_overclaiming() -> None:
    report = c6.build_report()
    assert all(report["negative_controls"].values())
    assert report["assessment"]["all_negative_controls_pass"]
    assert report["assessment"]["anchored_claim_6"] == (
        "partially_reproduced_source_capabilities_only__broad_empirics_blocked"
    )
    assert report["assessment"]["inconclusive"]
    assert not report["assessment"]["toy"]


def test_blocker_and_next_approach_are_specific() -> None:
    blocker = c6.build_report()["blocker"]
    assert blocker["proven"]
    assert len(blocker["missing_public_evidence"]) == 5
    assert "PACS->Photo" in blocker["next_materially_distinct_approach"]
    assert "ResNet-50" in blocker["next_materially_distinct_approach"]


def test_report_is_deterministic() -> None:
    first = c6.build_report()
    second = c6.build_report()
    assert first["certificate_sha256"] == second["certificate_sha256"]
    assert first == second
