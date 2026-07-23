# Claim 2 — ECL sample complexity

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_gfp_c2_rate_01", "created_at": "2026-07-19T13:34:15+00:00", "title": "Claim and split assessment"}
-->
**Current anchored claim:** Theorem 3.2 gives an `O(B/epsilon^2)` finite-sample rate for
empirical ECL, matching histogram-binning ECE. This is the same result that the legacy
three-claim judge records as Claim 3.

**Assessment:** the fixed-hard-bin Eq. 5 `O(B/epsilon^2)` order is supported. A synthetic soft
Eq. 8 attempt supports root-`n` convergence at fixed `B`, and a final real-MNIST attempt with two
independently trained classifiers supports comparable fixed-`B` empirical order to matched ECE.
The proof printed in Appendix G is not valid as written, the universal Eq. 8 dependence on `B`
is not proved, and no literal numerical coverage constant is claimed.

The source audit is pinned to arXiv `2605.21552v1` (SHA-256
`fb1d1a634d55132694349d40d56731cc5c7401571bc8c1a9f6eee1b5849950ab`) and
`NeuroDong/ECL@aae77f890f1e4ebc13dad135b5e29758d98d318d` (`losses.py` SHA-256
`1c2de34967f34b98faae5025368edac88f46a709d6e1e0c063e2c01f4d6e9754`).

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_gfp_c2_rate_02", "created_at": "2026-07-19T13:34:16+00:00", "title": "Independent hard-bin derivation"}
-->
For fixed hard bins, let `d_j=||mu_s,j-mu_t,j||`, population target masses `pi_j`, and empirical
target weights `w_j=n_tj/N_t`. Then

`|sum_j w_j d_hat_j - sum_j pi_j d_j| <= sum_j w_j |d_hat_j-d_j| + |sum_j (w_j-pi_j)d_j|`.

Bounded differences plus a second-moment mean bound controls simplex-valued vector means at
dimension-free `O(sqrt(log(1/eta)/n))`. Reverse triangle inequality, a bin/domain union bound, and
weighted Cauchy–Schwarz produce Eq. 9's conditional-mean order without `sqrt(K)`. Since
`0<=d_j<=sqrt(2)`, scalar Hoeffding controls the omitted target-mass term; for positive hard-bin
counts `sum_j w_j/n_tj=B/N_t`, so that term is absorbed by Eq. 9 up to an absolute constant.

Appendix G instead applies a coordinate union in Eq. 31, retaining `sqrt(K)`, and Eq. 30 silently
uses the same weights for population Eq. 4 and empirical Eq. 5. Thus the theorem order has an
independent repair, but the printed proof does not establish it.

---
<!-- trackio-cell
{"type": "code", "id": "cell_gfp_c2_rate_03", "created_at": "2026-07-19T13:34:17+00:00", "title": "Executed sample-complexity audit", "command": [".venv/bin/python", "repro/src/run_claim3_sample_complexity.py"], "exit_code": 0, "duration_s": 1.0}
-->
````bash
$ .venv/bin/python repro/src/run_claim3_sample_complexity.py
````

````output
theorem_statement=supported_for_fixed_hard_bin_eq5
appendix_proof=missing_sqrt_K_and_bin_mass_terms
settings=18 seeds_per_setting=40 rows=720
covered_bins=[2, 4, 8, 16, 32]
covered_classes=[2, 3, 10, 50]
covered_counts=[25, 50, 100, 200, 400, 800]
max_ecl_deviation_over_normalized_radius=0.33872407
max_matched_ece_deviation_over_normalized_radius=0.33818962
executed_ecl_rmse_slope=-0.48805923
executed_ece_rmse_slope=-0.51020565
executed_target_mass_rmse_slope=-0.53469468
official_source_pin_matches=True
literal_coverage_claimed=False
````

The RMSE slopes imply sample-complexity exponents `2.048932` (ECL) and `1.959994` (ECE), both near
the predicted `epsilon^-2`. A separate multinomial target-mass experiment gives slope `-0.534695`.
The 720 valid rows span balanced, skewed, sparse-positive, and source/target-imbalanced hard-bin
plans; exact binary enumeration has total probability mass `1.0`.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_gfp_c2_rate_04", "created_at": "2026-07-19T13:34:18+00:00", "title": "Controls, official-code boundary, and limitations"}
-->
Fail-closed controls reject omitted source samples, zero-count bins, changed target weights, and a
broken shared conditional. The audit also fixed a simulator defect: the earlier code declared a
mixture probability for atom zero but sampled atom one; the corrected convention is regression-
tested.

The official `ECLossMiniBatch` uses Eq. 10 auxiliary/proximal training. Its canonical statistics
correspond to soft Eq. 7/8 components, but its returned objective is not a direct evaluation of the
Eq. 5/8 estimator, so official training-code execution is not presented as direct bound evidence.

Limitations of this first attempt: unknown `C,C1,C2`; fixed hard bins; positive counts; iid
evaluation samples; fixed exact posterior oracle. A learned posterior head reused on its own
training data needs stability or sample splitting. The separate soft Eq. 8 attempt below retains
that fixed-function boundary.

---
<!-- trackio-cell
{"type": "code", "id": "cell_gfp_c2_soft_05", "created_at": "2026-07-19T15:23:10+00:00", "title": "Executed soft self-normalized Eq. 8 audit", "command": [".venv/bin/python", "repro/src/run_claim3_soft_sample_complexity.py"], "exit_code": 0, "duration_s": 2.0}
-->
````bash
$ OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python repro/src/run_claim3_soft_sample_complexity.py
````

````output
assessment=verified_for_declared_fixed-function_well-conditioned-positive-mass_construction
rows_sample_size=840
rows_bins=720
ecl_rmse_slope_n=-0.61547563
ecl_tail_rmse_slope_n=-0.53795114
ecl_epsilon_exponent=1.6247597
ecl_tail_epsilon_exponent=1.8589049
label_ece_rmse_slope_n=-0.59781856
ecl_variance_proxy_slope_bins=-1.5927813
tiny_mass_rmse_slope_n=-0.15028563
independent_ecl_abs_diff=8.3266727e-17
json=outputs/claim3_soft_sample_complexity.json
markdown=docs/CLAIM3_SOFT_SAMPLE_COMPLEXITY_AUDIT.md
````

The deterministic output JSON has SHA-256
`1eea7aaae70c2def0478e56214a9544431fa78ed35e5d7e9f1748006385df4a5`; an independent rerun
produced the same digest. Focused tests pass `7/7`, and the complete repository suite passes
`116/116`.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_gfp_c2_soft_06", "created_at": "2026-07-19T15:23:11+00:00", "title": "What the soft-estimator result does and does not establish"}
-->
The experiment implements the canonical soft-binning, self-normalized Eq. 8 estimator rather
than substituting the hard-bin Eq. 5 estimator. Scores, official shifted-simplex anchors, soft
assignments, and an exact posterior oracle are fixed before independent source and target
multinomial evaluation samples are drawn. Population ECL and matched canonical ECE references
are exact sums over `231` latent atoms, not Monte Carlo approximations.

At the baseline `B=15`, every population soft-bin mass is positive and bounded below by
`0.01957` (source) and `0.02016` (target). The Eq. 8 ECL tail RMSE slope is `-0.537951`, with
`n * RMSE^2` slope `-0.075902` and implied epsilon exponent `1.858905`. Matched label-ECE and
oracle-ECE slopes are `-0.597819` and `-0.501705`, respectively. An independently expanded
raw-sample calculation agrees with the atom-count contraction to `8.33e-17`.

The `B={3,6,10,15,21,28}` sweep changes the anchors, official `B`-dependent temperature, and
population reference at each point. Its ECL variance-proxy slope `-1.592781` shows no worsening
on this construction; it is **not** a universal proof of `O(B)` dependence. A valid but
deliberately ill-conditioned tiny-positive-mass stress has slope `-0.150286`, so positive mass
alone does not guarantee useful finite-sample constants. A zero-mass denominator control is
assumption-invalid and rejected. The executed evidence supports comparable fixed-`B` sample order
to ECE; it does not cover a learned posterior evaluated on its own training data or prove the
unrestricted theorem.

---
<!-- trackio-cell
{"type": "code", "id": "cell_gfp_c2_mnist_07", "created_at": "2026-07-19T15:51:58+00:00", "title": "Executed real-MNIST trained-model audit", "command": [".venv/bin/python", "repro/src/run_claim3_real_mnist_sample_complexity.py"], "exit_code": 0, "duration_s": 5.46}
-->
````bash
$ OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python repro/src/run_claim3_real_mnist_sample_complexity.py
````

````output
assessment=real_trained_model_supports_comparable_fixed_B_empirical_sample_order
primary_holdout_accuracy=0.9097
posterior_holdout_accuracy=0.9076
ecl_tail_rmse_slope_n=-0.68093982
ece_tail_rmse_slope_n=-0.74258383
sample_rows=240 bins_rows=120
wall_seconds=4.51029
json=outputs/claim3_real_mnist_sample_complexity.json
markdown=docs/CLAIM3_REAL_MNIST_SAMPLE_COMPLEXITY_AUDIT.md
````

The final JSON has SHA-256
`27530f24ad9799af54f0b040aa2a4c7d587f25a36c39d6ec4b0a349cb8c21b9f`. A separate rerun
reproduced every scientific field exactly; only measured wall-clock fields changed. Focused tests
pass `6/6`, and the complete repository suite passes `122/122`.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_gfp_c2_mnist_08", "created_at": "2026-07-19T15:51:59+00:00", "title": "Real-data protocol, results, controls, and boundary"}
-->
This final attempt uses the public MNIST `60,000/10,000` IDX split rather than a synthetic
population. All four cached files pass preregistered SHA-256, IDX header, count, and byte-size
checks. Dataset provenance: https://www.tensorflow.org/datasets/catalog/mnist. The cache has no
local license file; the external Keras MNIST documentation records CC BY-SA 3.0:
https://keras.io/api/datasets/mnist/.

A primary multinomial softmax classifier is label-trained on training rows `0:30000`; an
independent posterior head is trained on rows `30000:60000`; the official 10,000-image test split
is reserved for estimator evaluation. The models converge after `83/86` L-BFGS iterations and
reach `0.9097/0.9076` holdout accuracy. Source and target probabilities depend only on mean pixel
intensity and horizontal ink center, retain positive support, and induce label-distribution total
variation `0.344391`. A label-permutation control changes no domain weight.

At fixed `B=55`, `40` independent seeds over `n={250,500,1000,2000,4000,8000}` produce `240`
rows. ECL and matched canonical ECE tail RMSE slopes are `-0.680940` and `-0.742584`; their
absolute difference is `0.061644`. Both finite-grid fits are faster than the root-`n` reference,
so this is evidence of comparable no-worse order on this trained-model task, not an asymptotic
rate identification. Matrix and explicit per-bin implementations agree to `4.16e-17` (ECL) and
`2.78e-17` (ECE).

The exact simplex-grid sweep `B={10,55,220}` gives construction-specific variance-proxy exponents
`0.889297` (ECL) and `1.014459` (matched ECE), close to the claimed linear-in-`B` comparison but
not a universal proof because anchors, temperature, and population targets all change. The
baseline minimum soft masses are extremely small (`3.28e-21/2.31e-20`), the posterior head is an
estimate rather than an oracle, and finite-pool sampling with replacement does not identify the
real-world conditional for ambiguous handwriting. This final attempt supplies the judge-requested
real-data trained-model evidence without repairing Appendix G or overstating the theorem.
