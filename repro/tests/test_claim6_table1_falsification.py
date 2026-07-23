from __future__ import annotations

from fractions import Fraction as Q


def test_normalization_counterexample_is_exact() -> None:
    weights = (Q(1, 4), Q(3, 4))
    derivatives = (Q(0), Q(4))
    full = sum(w * d for w, d in zip(weights, derivatives, strict=True)) / sum(weights)
    expected_batch = sum(w * d for w, d in zip(weights, derivatives, strict=True)) / 2
    assert full == 3
    assert expected_batch == Q(3, 2)
    assert full != expected_batch


def test_false_required_cell_falsifies_all_true_conjunction() -> None:
    table_cells = [True, True, True, True, False]
    assert not all(table_cells)
