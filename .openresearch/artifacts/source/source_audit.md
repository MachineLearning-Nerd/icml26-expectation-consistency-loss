# Paper source and exact claim scope

Source: ar5iv HTML for arXiv `2605.21552v1`, retrieved with the explicit
browser User-Agent recorded in `paper_source_manifest.json`.

## Exact quantifiers and assumptions

- Claim 1 (`S3.Thmtheorem1`): for every class `1 <= k <= K`, under covariate
  shift with the shared conditional
  `P(Y_k=1|X)=P_s(Y_k=1|X)=P_t(Y_k=1|X)`, equality of the source and target
  conditional calibration curves given the full score vector `S` holds if and
  only if their conditional posterior expectations given `S` are equal.
- Claim 2 (`S3.Thmtheorem2`): for every `epsilon, delta in (0,1)`, the paper
  considers empirical Eq. 5 or Eq. 8 with `B` bins, target bin weights `w_j`,
  and source/target per-bin counts. It asserts existence of an absolute
  constant `C>0` such that the displayed finite-sample error bound holds with
  probability at least `1-delta`, then states the implied order
  `O(B/epsilon^2)`. Positive denominators, iid evaluation samples, and a
  posterior-vector function fixed independently of evaluation samples are
  required for the estimator to be defined, although the theorem does not
  state them explicitly.
- Claim 3 (`S3.Thmtheorem3`): Eq. 10 is asserted to be asymptotically
  equivalent to Eq. 8 and, in expectation over source and target mini-batches,
  its mini-batch gradient is asserted to equal the full empirical ECL
  gradient. The current full-credit counterexamples test these two asserted
  equalities under their stated domain.
- Claim 4 (`A4`, `A5`, `A6`): the expectation-consistency condition and loss
  are extended to all observed top-label confidence levels and, for every
  class, all observed class-score levels. Appendix F supplies differentiable
  soft-binning formulas for canonical, top-label, and class-wise variants.
- Claim 5 (`S4.T2`): Table 2 reports mean and standard deviation over ten runs.
  For each digit target domain, the other two domains are merged into the
  source. The selected target-SVHN assertions are ECL `21.5 +/- 1.51` versus
  uncalibrated `61.9 +/- 6.16` for LeNet-5, ECL `36.8 +/- 2.08` versus
  PseudoCal `48.2 +/- 3.95` for ResNet20, and ECL `38.4 +/- 3.21` versus
  uncalibrated `80.8 +/- 6.26` for DenseNet40.
- Claim 6 is conjunctive as judged. Table 1 (`S1.T1`) asserts ECL is the only
  listed row with all five capabilities. Figure 2 (`S4.F2`) asserts reduced
  calibration error on both the stated simulated construction and PACS while
  preserving or improving accuracy. Table 3 (`A3.T3`) reports source-target
  calibration gaps for Digit/ResNet20, PACS/ResNet50, and
  ImageNet-Sketch/ViT-L. Evidence for only one component cannot verify the
  combined claim.

## Interpretation policy

Printed table arithmetic, code-path existence, unavailable data, reduced toy
data, and qualitative plot similarity are not independent verification.
Counterexamples count as falsification only if they satisfy every assumption
and contradict the exact quantified statement.
