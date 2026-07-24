# Claim 4 — all Appendix F variants on full MNIST

**Verdict: VERIFIED. Confidence: HIGH.**

This route replaces the earlier 12-atom formula check with an implementation
test on the complete 10,000-image MNIST holdout. Two separately trained
classifier heads provide the model score and posterior observable. The route
executes the canonical, class-wise, and top-label differentiable soft-binning
losses from Appendix F.

PyTorch autograd is checked against a separately written NumPy implementation
and centered finite differences:

| Variant | Loss absolute error | Gradient absolute error |
| --- | ---: | ---: |
| Canonical | `1.39e-16` | `2.91e-8` |
| Class-wise | `5.55e-17` | `1.47e-10` |
| Top-label | `2.01e-16` | `8.92e-11` |

All three match. Controls detach the score, shuffle the posterior observable,
and substitute incorrect grouping semantics; each control changes or removes
the expected quantity. The MNIST inputs are hash-audited, and the independent
checker does not import the PyTorch certificate module.

This verifies differentiable implementation compatibility for the three
variants. It does not claim benchmark-level calibration improvement.

Machine-readable inputs, raw results, checker output, contract, method, source
audit, and limitations are under
`repro/evidence/2026-07-24/artifacts/claim-4/route-real-mnist/`.
