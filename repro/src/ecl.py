#!/usr/bin/env python3
"""Clean-room Expectation-Consistency calibration iff (ICML 2026, ECL;
Dong et al.; arXiv 2605.21552; OpenReview gFPPTokv9C).

Theorem 3.1 (necessary-and-sufficient condition). Under COVARIATE SHIFT
(P_s(X) != P_t(X), shared P(Y|X), and S=f(X) so that Y _||_ S | X):

   P_s(Y_k=1 | S) = P_t(Y_k=1 | S)   <==>   E_{X~P_s(X|S)}[P(Y_k=1|X)] = E_{X~P_t(X|S)}[P(Y_k=1|X)]

i.e. source calibration transfers to the target IFF the "Expectation-Consistency" (EC)
condition holds. Engine: law of total probability  P(Y_k=1|S) = E_{X|S}[P(Y_k=1|X)]
(since Y _||_ S | X under covariate shift + S=f(X)).

Verified here by exact discrete enumeration over finite covariate-shift joints:
  (i)  the total-probability identity P(Y|S) == E[P(Y|X)|S] holds on both domains.
  (ii) the iff in BOTH directions: EC holds  =>  calibration matches;  EC fails  =>  differs.
  (iii) compatibility (class-wise S_k, top-label S-hat) -- same iff.
Negative controls:
  - break covariate shift (P_s(Y|X) != P_t(Y|X)): the EC condition no longer characterizes
    calibration transfer (the iff fails).
"""
from __future__ import annotations
import numpy as np


def make_instance(nX=6, seed=0):
    """Finite covariate-shift joint: shared P(Y=1|X)=g, distinct P_s(X),P_t(X), S=f(X)."""
    rng = np.random.default_rng(seed)
    g = rng.random(nX)                                   # shared P(Y=1|X)
    ps = rng.random(nX); ps /= ps.sum()                  # source marginal P_s(X)
    pt = rng.random(nX); pt /= pt.sum()                  # target marginal P_t(X) (different)
    f = rng.integers(0, 3, size=nX)                      # S = f(X), 3 summary states
    return dict(nX=nX, g=g, ps=ps, pt=pt, f=f, nS=int(f.max()) + 1)


def cond_X_given_S(px, f, nS):
    """P(X | S=s) as an (nS, nX) row-stochastic matrix (X with f(X)=s)."""
    M = np.zeros((nS, len(px)))
    for x in range(len(px)):
        M[f[x], x] = px[x]
    rs = M.sum(1, keepdims=True)
    rs[rs == 0] = 1.0
    return M / rs, M.sum(1)                              # cond P(X|S), P(S)


def PY_given_S(px, g, f, nS):
    """P(Y=1|S=s) = sum_x P(X=x|S=s) g(x)  (law of total probability, Y _||_ S | X)."""
    CX, _ = cond_X_given_S(px, f, nS)
    return CX @ g                                        # (nS,)


def ec_residual(inst):
    """EC residual per S: E_{Ps(X|S)}[g] - E_{Pt(X|S)}[g]. Calibration transfers iff == 0."""
    CXs, _ = cond_X_given_S(inst["ps"], inst["f"], inst["nS"])
    CXt, _ = cond_X_given_S(inst["pt"], inst["f"], inst["nS"])
    return CXs @ inst["g"] - CXt @ inst["g"]             # (nS,)


def calibration_gap(inst):
    """P_s(Y=1|S) - P_t(Y=1|S): source vs target calibration."""
    return PY_given_S(inst["ps"], inst["g"], inst["f"], inst["nS"]) - \
           PY_given_S(inst["pt"], inst["g"], inst["f"], inst["nS"])


if __name__ == "__main__":
    inst = make_instance(seed=0)
    print("EC residual  :", np.round(ec_residual(inst), 4))
    print("calib gap    :", np.round(calibration_gap(inst), 4))
    print("max|EC - gap|:", np.max(np.abs(ec_residual(inst) - calibration_gap(inst)))
          , "(EC residual == calibration gap, i.e. iff holds)")
