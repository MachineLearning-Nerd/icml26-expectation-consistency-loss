# Claim-to-evidence map

The six claims are assessed independently. A proof, a finite implementation
check, a source audit, and a benchmark result are different evidence types and
are not silently substituted for one another.

## C1 — expectation consistency iff calibration transfer

| Evidence layer | Producer/artifact | What it checks |
| --- | --- | --- |
| Analytical proof | `repro/src/claim1_general_certificate.py` | Tower-property derivation under covariate shift, a shared conditional kernel, `S=f(X)`, and common identified score support. |
| Exact certificate | `repro/evidence/2026-07-24/artifacts/claim-1/raw_certificate.json` | 257 posterior summaries × 17 scores × 11 classes; all formal coefficient maps agree. |
| Negative controls | same artifact | Breaks measurability, shared conditionals, support overlap, and source-calibration premise separately. |

Status: `VERIFIED_SCOPED`. The result transfers a source/target calibration
curve, but expectation consistency alone is not absolute calibration.

## C2 — finite-sample bound and sample order

| Evidence layer | Producer/artifact | What it checks |
| --- | --- | --- |
| Hard-bin route | `repro/src/run_claim3_sample_complexity.py` and `repro/evidence/2026-07-24/artifacts/claim-2/route-a/` | The fixed hard-bin construction and its stated order. |
| Corrected proof | `repro/src/claim2_soft_theorem_proof.py` and `route-c-soft-proof/proof.md` | Hilbert-space conditional means plus simplex masses; `C=16`, budget `15.3137085`. |
| Stress route | `repro/src/claim2_soft_falsification_stress.py` and `route-b/` | Probability-domain adversarial search; it does not promote a finite search to a proof. |
| Independent checker | `route-c-soft-proof/independent_checker.json` | Recomputes all radii/count identities and 1,024 soft-bin diagnostics. |

Status: `VERIFIED_SCOPED_WITH_QUALIFICATION`. The corrected theorem requires
fixed posterior/assignment functions, iid evaluation samples, positive
realized denominators, and `K >= 2`. The printed Appendix G derivation is not
certified because it omits target-bin-mass estimation and leaves a `sqrt(K)`
factor.

## C3 — unbiased mini-batch gradient

| Evidence layer | Producer/artifact | What it checks |
| --- | --- | --- |
| Exact witnesses | `repro/src/claim3_gradient_certificate.py` | Four rational counterexamples, all strictly inside the probability domain. |
| Independent checks | `repro/tests/test_claim3_gradient_certificate.py` and C3 raw certificate | Normalization, same-batch norm direction, soft-weight derivative, and Eq. 8/Eq. 10 objective/gradient differences. |
| Corrected boundary | C3 certificate and `docs/CLAIM3_GRADIENT_AUDIT.md` | A fixed-direction estimator can be unbiased with independent frozen state and correct scaling. |

Status: `FALSIFIED_AS_STATED`. The exact witnesses give full versus claimed
quantities `3 vs 3/2`, `1 vs 0`, `3/4 vs 0`, and `0 vs 1/4`; the Eq. 8/Eq. 10
losses are `3/4` and `65/128`.

## C4 — differentiable calibration variants

| Evidence layer | Producer/artifact | What it checks |
| --- | --- | --- |
| Primary implementation | `repro/src/claim4_real_mnist_soft_bins.py` and `claim4_real_mnist_independent_checker.py` | Canonical, class-wise, and top-label soft-bin losses on the complete 10,000-image MNIST holdout. |
| Independent implementation | `repro/evidence/2026-07-24/artifacts/claim-4/route-real-mnist/independent_checker.json` | NumPy plus centered finite differences, independent of the PyTorch certificate module. |
| Controls | route `controls` and `limitations.md` | Wrong grouping, detached score, shuffled observable, and wrong observable semantics. |

Status: `VERIFIED_SCOPED`. Maximum gradient discrepancies are `2.91e-8`,
`1.47e-10`, and `8.92e-11`; this is formula/gradient compatibility, not a
benchmark-improvement claim.

## C5 — Table 2

| Evidence layer | Producer/artifact | What it checks |
| --- | --- | --- |
| Source arithmetic | `repro/src/claim5_table2_audit.py` and `claim-5/source_audit.json` | Printed mean/standard-deviation pairs and official source inventory. |
| Three execution routes | `claim-5/route-1-lenet`, `route-2-posthoc`, `route-3-stabilized` | Materially different attempts, each with declared deviations. |
| Falsification route | `repro/src/claim5_mandatory_falsification_audit.py` and `route-4-falsification` | Tests whether any route is an assumption-complete contradiction; it is not. |

Status: `BLOCKED`. The ten-run raw predictions/checkpoints, exact seed schedule,
complete three-architecture pipeline, and missing preprocessing/loss details
are not public. The printed numbers are not promoted to independent empirical
evidence.

## C6 — compound capability/empirical claim

| Evidence layer | Producer/artifact | What it checks |
| --- | --- | --- |
| Simulation | `repro/src/claim6_simulation_figure2.py` and `claim-6/route-1-simulation/` | Five seeds under the paper-text simulation protocol; outcome is divergent/mixed. |
| Logical conjunction | `repro/src/claim6_table1_falsification.py` and `claim-6/route-2-table1/` | Table 1's mini-batch-trainability cell is false through the C3 witnesses. |
| Source audit | `docs/CLAIM6_CAPABILITY_AUDIT.md` and `claim-6/source_audit.json` | Official implementation capabilities and paper arithmetic, without inventing PACS/ImageNet data. |

Status: `FALSIFIED_SCOPED`. One false required cell falsifies the all-five
Table 1 conjunction. The separate Figure 2/Table 3 empirical components remain
source-bound or unavailable and are not described as independently reproduced.

## Status vocabulary

- `VERIFIED_SCOPED`: the exact local proof or implementation obligation passes under stated assumptions.
- `VERIFIED_SCOPED_WITH_QUALIFICATION`: a corrected or narrower theorem passes, while the printed derivation or stronger reading does not.
- `FALSIFIED_AS_STATED`: a valid counterexample contradicts the exact quantified claim.
- `BLOCKED`: missing artifacts prevent an assumption-complete attempt.
- `FALSIFIED_SCOPED`: a compound claim is falsified through one required conjunct without claiming all empirical subcomponents were rerun.
