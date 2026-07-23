# Claim 4 source audit

The audited source is the ar5iv rendering of arXiv 2605.21552 retrieved on
2026-07-23. Its SHA-256 is
`0b5e5f8f9b7af9e0c82e00f289d65e9556df1335924dd25913c8a52dd6868f98`.

Appendix D defines top-label and class-wise expectation consistency. Appendix E
states their corresponding ECL objectives. Appendix F makes the histogram
assignments differentiable:

- Top-label uses scalar anchors `(2j-1)/(2B)`, Gaussian soft membership of the
  maximum score, and the posterior of the predicted class.
- Class-wise applies the scalar construction separately to every score
  coordinate and posterior coordinate, then sums across classes and bins.
- Canonical calibration uses a Gaussian assignment over the complete
  probability vector and a vector norm between source and target conditional
  posterior estimates.
- Each conditional estimate uses the paper's `1e-5` denominator stabilizer.

The exact assertion tested here is implementation compatibility and
differentiability of these three formulas. It is not a quantitative benchmark
performance assertion and has no claim about a particular neural architecture.

The official source tree contains distinct mode branches, but a code audit
alone does not demonstrate that all three differentiable computations execute
on real data or propagate the intended score gradient. This experiment supplies
that missing evidence.
