from __future__ import annotations

import numpy as np
import pytest
import torch

from claim5_predecessor_posthoc import (
    appendix_f_top_ecl,
    hard_source_ece,
    temperature_search,
)


def test_appendix_f_top_ecl_is_finite_and_nonnegative():
    generator = torch.Generator().manual_seed(7)
    source_confidence = torch.rand(29, generator=generator)
    target_confidence = torch.rand(31, generator=generator)
    source_p = torch.rand(29, generator=generator)
    target_p = torch.rand(31, generator=generator)
    value = appendix_f_top_ecl(
        source_confidence, target_confidence, source_p, target_p
    )
    assert np.isfinite(value)
    assert value >= 0


def test_hard_source_ece_matches_known_bin_value():
    confidence = torch.tensor([0.2, 0.4, 0.8, 0.95])
    prediction = torch.tensor([0, 1, 2, 3])
    labels = torch.tensor([0, 0, 2, 0])
    value = hard_source_ece(confidence, prediction, labels)
    expected = (abs(0.2 - 1) + abs(0.4 - 0) + abs(0.8 - 1) + abs(0.95 - 0)) / 4
    assert value == pytest.approx(expected, abs=2e-8)


def test_temperature_search_is_target_label_free_and_preserves_grid():
    generator = torch.Generator().manual_seed(8)
    source_logits = torch.randn(41, 10, generator=generator)
    target_logits = torch.randn(43, 10, generator=generator)
    source_labels = torch.randint(0, 10, (41,), generator=generator)
    source_head = torch.randn(41, 2, generator=generator)
    target_head = torch.randn(43, 2, generator=generator)
    best, rows = temperature_search(
        source_logits,
        source_labels,
        target_logits,
        source_head,
        target_head,
    )
    assert 1 <= best <= 50
    assert [row["temperature"] for row in rows] == list(range(1, 51))
    assert all(np.isfinite(row["appendix_f_top_ecl"]) for row in rows)
