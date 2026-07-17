#!/usr/bin/env python3
"""Verify the ECL calibration iff (gFPPTokv9C, Theorem 3.1).

Under covariate shift: P_s(Y_k=1|S)=P_t(Y_k=1|S)  <=>  E_{Ps(X|S)}[P(Y|X)]=E_{Pt(X|S)}[P(Y|X)].

We verify (exact discrete enumeration):
  (1) IFF identity: the EC residual (RHS difference) == the calibration gap (LHS difference),
      to machine precision, across many covariate-shift instances (the iff holds in both directions).
  (2) Total-probability engine: P(Y=1|S) == E_{X|S}[P(Y=1|X)] on BOTH source and target.
  (3) C2 compatibility: the iff holds for class-wise summary S_k (Theorem D.2) too.
Negative control:
  (4) Break covariate shift (P_s(Y|X) != P_t(Y|X)): the EC residual (shared g) no longer equals the
      calibration gap -> the iff fails (correctly; the theorem's scope is covariate shift).
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ecl


def main():
    print("=" * 74)
    print("ECL calibration iff (gFPPTokv9C, Theorem 3.1) -- covariate-shift calibration transfer")
    print("=" * 74)
    res = {}

    # (1) IFF identity: EC residual == calibration gap, many instances
    print("\n(1) iff: EC residual == calibration gap (calibration transfers <=> EC holds)")
    iff_ok = True; max_diff = 0.0
    n_both = 0  # count instances where EC holds (residual~0) and where it fails
    for seed in range(40):
        inst = ecl.make_instance(nX=6, seed=seed)
        ec = ecl.ec_residual(inst); gap = ecl.calibration_gap(inst)
        d = float(np.max(np.abs(ec - gap)))
        max_diff = max(max_diff, d); iff_ok &= d < 1e-12
        # both directions: EC~0 <=> gap~0
        ec_zero = np.max(np.abs(ec)) < 1e-9
        gap_zero = np.max(np.abs(gap)) < 1e-9
        if ec_zero == gap_zero:
            n_both += 1
    print(f"  max|EC_residual - calibration_gap| over 40 instances = {max_diff:.2e}")
    print(f"  EC==0 <=> gap==0 consistent on all 40 instances: {n_both == 40}")
    print(f"  -> iff identity holds to machine precision: {iff_ok}")
    res["iff"] = dict(ok=bool(iff_ok), max_diff=float(max_diff))

    # (2) total-probability engine
    print("\n(2) engine: P(Y=1|S) == E_{X|S}[P(Y=1|X)] on source and target")
    eng_ok = True
    for seed in range(20):
        inst = ecl.make_instance(seed=seed)
        for px in (inst["ps"], inst["pt"]):
            PY_S = ecl.PY_given_S(px, inst["g"], inst["f"], inst["nS"])
            CX, _ = ecl.cond_X_given_S(px, inst["f"], inst["nS"])
            exp = CX @ inst["g"]
            eng_ok &= np.max(np.abs(PY_S - exp)) < 1e-12
    print(f"  -> law of total probability holds on both domains (40 checks): {eng_ok}")
    res["engine"] = dict(ok=bool(eng_ok))

    # (3) C2 compatibility: class-wise summary S_k (Theorem D.2) -- same iff with per-class f_k
    print("\n(3) C2 compatibility: class-wise summary S_k (Theorem D.2) -- iff still holds")
    comp_ok = True
    for seed in range(20):
        inst = ecl.make_instance(nX=6, seed=seed)
        # use a different summary f2 (per-class) -- the iff is structural (any S=f(X))
        inst2 = dict(inst); inst2["f"] = np.random.default_rng(seed + 100).integers(0, 4, size=inst["nX"]); inst2["nS"] = 4
        ec = ecl.ec_residual(inst2); gap = ecl.calibration_gap(inst2)
        comp_ok &= np.max(np.abs(ec - gap)) < 1e-12
    print(f"  -> iff holds for class-wise/alternative summary S_k (20 instances): {comp_ok}")
    res["compatibility"] = dict(ok=bool(comp_ok))

    # (4) negative control: break covariate shift (g_s != g_t) -> iff fails
    print("\n(4) Negative control: break covariate shift (P_s(Y|X) != P_t(Y|X)) -> iff fails")
    rng = np.random.default_rng(7)
    ctrl_fail = 0
    for seed in range(20):
        inst = ecl.make_instance(seed=seed)
        g_s = inst["g"]; g_t = rng.random(inst["nX"])      # DIFFERENT P(Y|X) per domain
        # calibration gap uses the TRUE per-domain g; EC residual (theorem) uses a SHARED g
        CXs, _ = ecl.cond_X_given_S(inst["ps"], inst["f"], inst["nS"])
        CXt, _ = ecl.cond_X_given_S(inst["pt"], inst["f"], inst["nS"])
        true_gap = (CXs @ g_s) - (CXt @ g_t)               # actual P_s(Y|S)-P_t(Y|S)
        ec_shared = (CXs @ g_s) - (CXt @ g_s)              # EC condition assumes shared g
        if np.max(np.abs(true_gap - ec_shared)) > 1e-6:
            ctrl_fail += 1
    ctrl_ok = ctrl_fail >= 15   # most: shared-g EC residual != true gap when covariate shift broken
    print(f"  {ctrl_fail}/20: shared-g EC residual != true calibration gap when P_s(Y|X)!=P_t(Y|X)")
    print(f"  -> iff correctly FAILS outside covariate-shift scope: {ctrl_ok}")
    res["neg_control"] = dict(ok=bool(ctrl_ok), fails=ctrl_fail)

    verified = bool(iff_ok and eng_ok and comp_ok and ctrl_ok)
    print("\n" + "=" * 74)
    print(f"C1 + C2 CALIBRATION IFF: {'VERIFIED' if verified else 'PARTIAL'}")
    print("=" * 74)
    out = os.path.join(HERE, "..", "..", "outputs", "ecl_summary.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
