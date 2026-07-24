# Claim 3 Soft Eq. 8 Sample-Complexity Audit

Paper: *Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift* (`gFPPTokv9C`, arXiv `2605.21552v1`).

## Assessment

- `verified_for_declared_fixed-function_well-conditioned-positive-mass_construction`
- This is a second substantive approach: differentiable Gaussian soft assignments and random self-normalized denominators from Eq. 8, not the prior fixed hard-bin Eq. 5 estimator.
- Scores, assignments, and the exact posterior oracle are fixed before evaluation; source and target evaluation samples are independent.
- The learned-posterior/same-data case remains outside scope.

## Exact construction

- Population quantities are exact finite sums over 231 shared latent atoms; the domains differ only in `P(X)`.
- Bins: 15; official temperature: 0.042183207; Eq. 8 stabilizer: 1.0e-05
- Minimum source/target population soft mass: 0.0195693 / 0.0201563
- Exact population ECL / matched canonical ECE: 0.0363684805 / 0.0940060446
- Stabilizer bias at smallest/largest n: 2.56e-07 / 3.99e-09

## Executed scaling

- Soft Eq. 8 ECL RMSE slope vs n: -0.615476 over all n and -0.537951 over n=512..8192.
- Soft Eq. 8 tail implied epsilon exponent: 1.858905; tail slope of `n * RMSE^2`: -0.075902 (root-n target: 0).
- Matched label-ECE RMSE slope vs n: -0.597819 over all n and -0.600533 over the same tail.
- Matched oracle-ECE RMSE slope vs n: -0.501705
- ECL variance-proxy slope vs B: -1.592781 (claimed sample-order ceiling: 1)
- Matched label-ECE variance-proxy slope vs B: 0.322499
- Raw replicate rows and `n * RMSE^2` diagnostics are preserved in the JSON artifact.

## Independent calculation

- Count contraction versus expanded raw samples, ECL absolute difference: 8.33e-17
- Count contraction versus expanded raw samples, matched oracle-ECE difference: 5.55e-17

## Denominator controls

- Tiny-mass stress remains strictly positive: `True`; minimum source/target mass `8.14e-06` / `8.27e-06`.
- Tiny-mass ECL RMSE slope vs n: -0.150286. This ill-conditioned finite-n control is not used to claim a mass-uniform constant.
- Zero-mass masked-column negative control rejected: `True`.

## Limitations

- The experiment supports scaling for a fixed exact posterior oracle; it does not cover a learned posterior evaluated on its own training data.
- Finite-grid experiments cannot prove a universal theorem or identify the paper's unspecified absolute constant C.
- Eq. 8's fixed stabilizer creates a small deterministic O(1/n) bias relative to the unregularized exact population reference.
- Positive mass alone does not give useful finite-n constants; the tiny-mass stress is intentionally ill-conditioned and is reported separately.
- Actual simplex-grid bin counts use cardinalities 3, 6, 10, 15, 21, and 28 to match the official anchor rule exactly.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest repro/tests/test_claim3_soft_sample_complexity.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python repro/src/run_claim3_soft_sample_complexity.py
```

Artifact: `outputs/claim3_soft_sample_complexity.json`.
