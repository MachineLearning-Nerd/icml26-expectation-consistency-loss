# Claim 3 Real-MNIST Soft Eq. 8 Sample-Complexity Audit

Final substantive attempt for live legacy C3 / anchored C2.

## Assessment

- `real_trained_model_supports_comparable_fixed_B_empirical_sample_order`
- This is real-data, real-trained-model evidence for fixed-B sample order. It is not a repair of Appendix G and not a universal B-dependence proof.

## Dataset provenance and license boundary

- All four cached uncompressed IDX files match preregistered SHA-256, magic, count, and byte-size records: `True`.
- Cached split: 60,000 training and 10,000 test 28x28 images, matching the MNIST homepage.
- Cache-local license file present: `False`.
- External Keras MNIST documentation states `CC BY-SA 3.0`: https://keras.io/api/datasets/mnist/
- Dataset homepage: https://www.tensorflow.org/datasets/catalog/mnist

## Actual training and holdout

- Primary classifier: multinomial logistic regression trained on official training indices 0..29,999.
- Additional posterior head: independently trained multinomial logistic regression on indices 30,000..59,999.
- Sample-complexity evaluation: official 10,000-image test split, disjoint from both training sets.
- Primary optimizer iterations/converged/wall seconds: 83 / `True` / 1.307
- Posterior-head optimizer iterations/converged/wall seconds: 86 / `True` / 1.298
- Primary holdout accuracy / NLL / hard-bin ECE: 0.909700 / 0.318889 / 0.035608
- Posterior-head holdout accuracy / NLL: 0.907600 / 0.322568

## X-only covariate shift

- Selection function: `0.75*z(mean_pixel_intensity)+0.25*z(horizontal_center_of_ink)`; labels used: `False`.
- Source/target effective pool sizes: 3497.5 / 6085.0.
- Source/target label-distribution total variation induced by X-only sampling: 0.344391.
- Label-permutation control maximum weight change: 0.

## Fixed-B sample-size evidence

- Baseline B / temperature: 55 / 0.00313759391.
- Minimum source/target population soft mass: 3.27795e-21 / 2.31465e-20.
- Exact finite-pool population ECL / matched ECE: 0.0374254395 / 0.0731699256.
- ECL RMSE slope overall/tail: -0.660699 / -0.680940; tail epsilon exponent 1.468559.
- Matched ECE RMSE slope overall/tail: -0.684736 / -0.742584; tail epsilon exponent 1.346649.
- Absolute ECL/ECE tail-slope difference: 0.061644.
- Both finite-grid fits are faster than the root-n reference; this supports a no-worse comparable order here, not an asymptotic rate identification.
- Raw per-replicate rows and `n * RMSE^2` diagnostics are preserved in the JSON artifact.

## Construction-specific B sweep

- Executed exact simplex-grid bin counts: [10, 55, 220].
- ECL / matched-ECE variance-proxy slopes vs B: 0.889297 / 1.014459.
- These slopes apply only to this MNIST construction and the official B-dependent temperature; they do not prove a universal O(B) theorem.

## Independent cross-check and controls

- Matrix contraction versus explicit per-bin loop ECL difference: 4.16e-17.
- Matrix contraction versus explicit per-bin loop matched-ECE difference: 2.78e-17.
- Fail-closed controls: `{'cache_integrity_passed': True, 'training_evaluation_disjoint': True, 'primary_posterior_training_disjoint': True, 'domain_selection_uses_labels': False, 'label_permutation_changes_domain_weights': False, 'all_baseline_population_masses_positive': True}`.
- Total experiment wall time: 4.510 seconds with BLAS/OpenMP threads fixed to one.

## Limitations

- This is a real-data trained-model experiment, not a reproduction of the paper's large neural architecture or benchmark training pipeline.
- The additional posterior head is a separately trained multinomial logistic model; its probabilities are estimates, not an exact real-world posterior oracle.
- The empirical B sweep changes both the simplex anchor set and the official B-dependent temperature; it is construction-specific and is not a universal O(B) proof.
- The finite evaluation population treats each labeled image as an atom and samples with replacement; this preserves its empirical Y|X but does not identify the population conditional for ambiguous handwriting.
- The cache contains no local license file; CC BY-SA 3.0 is recorded from the external Keras MNIST documentation.
- This is the third and final substantive attempt for the live legacy claim.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest repro/tests/test_claim3_real_mnist_sample_complexity.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python repro/src/run_claim3_real_mnist_sample_complexity.py
```

Artifact: `outputs/claim3_real_mnist_sample_complexity.json`.
