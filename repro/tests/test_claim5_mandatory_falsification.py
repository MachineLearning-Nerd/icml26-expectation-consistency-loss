from __future__ import annotations

import pytest
import statistics

from claim5_mandatory_falsification_audit import ten_run_realization


@pytest.mark.parametrize(
    ("mean", "std"),
    [(61.9, 6.16), (21.5, 1.51), (48.2, 3.95), (38.4, 3.21)],
)
def test_ten_run_realization_matches_sample_summary(mean, std):
    values = ten_run_realization(mean, std)
    assert len(values) == 10
    assert statistics.mean(values) == pytest.approx(mean, abs=1e-12)
    assert statistics.stdev(values) == pytest.approx(std, abs=1e-12)
    assert all(0 <= value <= 100 for value in values)
