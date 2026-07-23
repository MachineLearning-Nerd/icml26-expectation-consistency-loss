from __future__ import annotations

import numpy as np

from claim6_simulation_checker import accuracy, scalar_metric


def test_claim6_metrics_are_zero_for_perfect_deterministic_predictions() -> None:
    labels = np.arange(400) % 3
    probs = np.eye(3)[labels]
    assert accuracy(probs, labels) == 1.0
    for paradigm in ("TopLabel", "Classwise", "Canonical"):
        assert scalar_metric(probs, labels, paradigm) == 0.0


def test_claim6_metrics_react_to_uniform_probabilities() -> None:
    labels = np.arange(400) % 3
    probs = np.full((400, 3), 1 / 3)
    assert scalar_metric(probs, labels, "TopLabel") > 0
    assert scalar_metric(probs, labels, "Classwise") > 0
    assert scalar_metric(probs, labels, "Canonical") > 0
