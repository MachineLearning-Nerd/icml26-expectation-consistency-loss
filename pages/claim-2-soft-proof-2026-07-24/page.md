# Claim 2 — corrected soft-bin concentration proof

**Verdict: VERIFIED. Confidence: HIGH.**

The paper claims that the empirical ECL gap has the displayed finite-sample
form in Theorem 3.2 and hence `O(B/epsilon^2)` sample order, up to logarithmic
factors. The contract fixes the posterior and soft-assignment functions before
iid evaluation sampling, requires positive realized denominators and `K >= 2`,
and uses the paper's self-normalized soft Eq. 8 estimator.

Appendix G's printed coordinate-union proof is not valid: it leaves an
unremoved `sqrt(K)` and omits target-bin-mass estimation. The reproduction
therefore supplies a corrected proof rather than certifying the printed
derivation.

A dimension-free Hilbert-space conditional-mean argument costs `8*sqrt(2)`.
Separately controlling target simplex masses costs at most `4`. The combined
budget is

```text
8*sqrt(2) + 4 = 15.313708498985 < 16.
```

Thus the paper's bound holds with an explicit absolute constant `C=16` under
the declared necessary assumptions. This absolute error bound is stronger
than the paper's displayed one-sided inequality. Balanced positive counts
then give `O(B/epsilon^2)` order up to the theorem's logarithmic terms.

The proof obligations include 840 deterministic algebra checks. An additional
1,024-row adversarial soft-bin stress suite had maximum error/radius
`0.087334129`; an independently implemented checker recomputed all radii and
count identities. The stress suite supports the proof but is not substituted
for it.

Evidence is under
`repro/evidence/2026-07-24/artifacts/claim-2/route-c-soft-proof/`, with the
preceding hard-bin proof and falsification stress in sibling directories.
