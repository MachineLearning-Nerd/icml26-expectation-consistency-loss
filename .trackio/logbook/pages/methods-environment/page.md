# Methods & environment

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_em_01", "created_at": "2026-07-17T19:42:00+00:00", "title": "Setup"}
-->
**Finite covariate-shift joint.** Discrete X (nX=6), shared `P(Y=1|X)=g`, distinct marginals
`P_s(X)≠P_t(X)`, summary `S=f(X)`. Conditional `P(X|S)` by Bayes; `P(Y|S)=Σ_x P(X=x|S)g(x)`.

**Theorem 3.1 iff** = EC residual (RHS difference) equals the calibration gap (LHS difference),
element-wise in S — verified to machine precision across random instances. Two directions:
EC=0 ⟺ calibration matches.

**Environment.** Python 3.12, numpy/scipy, pytest. CPU only. 5/5 tests pass. Official code
`NeuroDong/ECL` (the ECL loss) cross-references; theorem verification is clean-room.

**Scope.** C1 (iff) + C2 (compatibility) exact. C3 (sample-complexity rate) out of scope.
