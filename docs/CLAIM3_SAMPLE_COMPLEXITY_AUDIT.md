# Claim 3 Sample-Complexity Audit

Paper: *Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift* (`gFPPTokv9C`, arXiv `2605.21552v1`).

## Local assessment

- Hard-bin Eq. 5 statement: `supported` by an independent bounded-differences argument.
- Soft self-normalized Eq. 8 statement: `inconclusive`.
- Printed Appendix-G proof: `missing_sqrt_K_and_bin_mass_terms`; its coordinate-wise route is `sqrt(K)` looser than displayed Eq. 9 and Eq. 30 omits empirical target-bin mass error.
- The omitted hard-bin mass term changes the proof and constants, but scalar Hoeffding plus `w_j=n_tj/N_t` absorbs it into Eq. 9's order.
- All numerical radii are normalized to unknown absolute constants; no literal coverage constant is claimed.

## Independent hard-bin derivation

For fixed hard bins, write `d_j = ||mu_s,j-mu_t,j||`, population target masses `pi_j`, and empirical target proportions `w_j=n_tj/N_t`. Then

`|sum_j w_j d_hat_j - sum_j pi_j d_j| <= sum_j w_j |d_hat_j-d_j| + |sum_j (w_j-pi_j)d_j|`.

- Bounded differences plus the second-moment mean bound gives dimension-free `||mu_hat-mu|| = O(sqrt(log(1/eta)/n))` for simplex-valued vectors; no coordinate union or `sqrt(K)` is needed.
- Reverse triangle inequality, a union bound over bins/domains, and weighted Cauchy-Schwarz yield the conditional-mean part of displayed Eq. 9 up to an absolute constant.
- Since `0 <= d_j <= sqrt(2)`, scalar Hoeffding bounds the second (target-bin-mass) term by `O(sqrt(log(1/delta)/N_t))`. Because `sum_j w_j/n_tj = B/N_t` for positive hard-bin counts, that missing term is absorbed by Eq. 9's displayed order.
- Required scope: fixed hard bins, positive reported bin counts, iid evaluation samples, and a bounded posterior-vector function fixed independently of those samples (or sample splitting).

## Executed evidence

- Valid theorem rows: 720
- Axis-covering settings: 18 with 40 seeds each
- Maximum ECL deviation / displayed normalized radius: 0.338724
- Maximum matched-ECE deviation / displayed normalized radius: 0.33819
- Exact binary probability mass: 1.000000000000
- Exact binary normalized-threshold tail: 0
- Executed ECL RMSE-vs-count slope: -0.488059 (root-n target: -0.5)
- Executed histogram-ECE RMSE-vs-count slope: -0.510206 (root-n target: -0.5)
- Implied ECL sample-complexity exponent: 2.048932 (target: 2)
- Implied histogram-ECE sample-complexity exponent: 1.959994 (target: 2)
- Executed target-bin-mass RMSE-vs-total-count slope: -0.534695 (root-n target: -0.5)
- Executed ECL q90-vs-count slope: -0.460133
- Executed histogram-ECE q90-vs-count slope: -0.579144
- Executed target-bin-mass q90-vs-total-count slope: -0.537919
- The ECL generator uses fixed confidence bins, shared `P(Y|X)` atoms across domains, different source/target latent-X mixtures, and samples atom zero with exactly the declared mixture probability.
- Histogram ECE is executed separately with realizable top-label confidences above 0.5 and Bernoulli correctness observations.

## Official implementation audit

- `Official ECLossMiniBatch uses Eq.10 auxiliary/proximal trainable loss; its canonical per-bin n/m statistics correspond to the soft Eq.7/8 components, but its returned training objective is not the direct Theorem-3.2 Eq.5/8 empirical estimator.`
- Pinned source matches `NeuroDong/ECL@aae77f890f1e4ebc13dad135b5e29758d98d318d` evidence SHA-256: `True`
- Direct Eq. 5 returned-loss parity supported: `False`
- Therefore official training-code execution would test Eq. 10 optimization, not directly reproduce the Eq. 5 finite-sample estimator bound.

## Formula-derived identities (not executed evidence)

- epsilon^-2 slope: 1.000000
- B slope including the displayed logarithm: 1.147347
- K dependence slope (logarithmic only): 0.125127
- These values are algebraic evaluations of the displayed formula, not estimator regressions and not proof.

## Controls

- `below_threshold_counts` — assumptions valid: `True`; executed two samples per bin; supporting stress test only because C is unknown
- `source_samples_omitted` — assumptions valid: `False`; executed construction was rejected by BinPlan
- `shared_conditional_violated` — assumptions valid: `False`; executed with independent target conditional atoms and excluded from theorem rows
- `zero_count_bin` — assumptions valid: `False`; executed construction was rejected by BinPlan
- `uniform_weights_substitution` — assumptions valid: `False`; executed and changed the estimand for nonuniform target weights
- `sparse_positive_counts` — assumptions valid: `True`; executed as a valid positive-count plan and included in theorem rows

## Limitations

- All radii are normalized because the paper does not state C, C1, or C2.
- The deterministic grid supports scaling and implementation checks but cannot prove the universal theorem.
- Matched ECE is conventional scalar top-label histogram ECE; canonical ECL is vector-valued and the semantics are kept separate.
- The hard-bin Eq. 5 rate is supported; the soft self-normalized Eq. 8 analog is not verified.
- The executed rows use a fixed exact posterior oracle. A learned P-hat evaluated on its training observations needs stability or sample splitting not analyzed here.

## Reproduce

```bash
.venv/bin/python -m pytest repro/tests/test_claim3_sample_complexity.py -q
.venv/bin/python repro/src/run_claim3_sample_complexity.py
```

Artifacts: `outputs/claim3_sample_complexity.json`, `docs/CLAIM3_SAMPLE_COMPLEXITY_AUDIT.md`, and `repro/evidence/claim3/SHA256SUMS`.
