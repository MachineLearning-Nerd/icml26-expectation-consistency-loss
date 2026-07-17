#!/usr/bin/env python3
"""Exact tests for the ECL calibration iff (gFPPTokv9C, Theorem 3.1)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import ecl


def test_iff_identity_ec_equals_gap():
    """EC residual == calibration gap (calibration transfers iff EC holds)."""
    for seed in range(30):
        inst = ecl.make_instance(seed=seed)
        assert np.max(np.abs(ecl.ec_residual(inst) - ecl.calibration_gap(inst))) < 1e-12


def test_iff_both_directions():
    """EC==0  <=>  calibration gap==0, across instances."""
    for seed in range(30):
        inst = ecl.make_instance(seed=seed)
        ec0 = np.max(np.abs(ecl.ec_residual(inst))) < 1e-9
        g0 = np.max(np.abs(ecl.calibration_gap(inst))) < 1e-9
        assert ec0 == g0


def test_total_probability_engine():
    """P(Y=1|S) == E_{X|S}[P(Y=1|X)] on source and target."""
    for seed in range(15):
        inst = ecl.make_instance(seed=seed)
        for px in (inst["ps"], inst["pt"]):
            PY_S = ecl.PY_given_S(px, inst["g"], inst["f"], inst["nS"])
            CX, _ = ecl.cond_X_given_S(px, inst["f"], inst["nS"])
            assert np.max(np.abs(PY_S - CX @ inst["g"])) < 1e-12


def test_negcontrol_breaks_when_not_covariate_shift():
    """Break covariate shift (g_s != g_t): shared-g EC residual != true calibration gap,
    for a clearly-different g_t (robust over a few seeds)."""
    fails = 0
    for seed in range(20):
        rng = np.random.default_rng(seed + 7)
        inst = ecl.make_instance(seed=seed)
        g_t = inst["g"] + 0.5 * (rng.random(inst["nX"]) - 0.5)   # clearly-different P(Y|X) on target
        CXs, _ = ecl.cond_X_given_S(inst["ps"], inst["f"], inst["nS"])
        CXt, _ = ecl.cond_X_given_S(inst["pt"], inst["f"], inst["nS"])
        true_gap = (CXs @ inst["g"]) - (CXt @ g_t)
        ec_shared = (CXs @ inst["g"]) - (CXt @ inst["g"])
        if np.max(np.abs(true_gap - ec_shared)) > 1e-6:
            fails += 1
    assert fails >= 15   # most instances: shared-g EC residual != true gap (iff fails outside covariate shift)


def test_compatibility_classwise():
    """iff holds for an alternative summary S_k (class-wise, Theorem D.2)."""
    rng = np.random.default_rng(5)
    inst = ecl.make_instance(seed=1)
    inst2 = dict(inst); inst2["f"] = rng.integers(0, 4, size=inst["nX"]); inst2["nS"] = 4
    assert np.max(np.abs(ecl.ec_residual(inst2) - ecl.calibration_gap(inst2))) < 1e-12


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
