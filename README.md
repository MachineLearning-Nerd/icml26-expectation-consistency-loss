# Expectation Consistency Loss under Covariate Shift

Scoped clean-room audit of **Expectation Consistency Loss: Rethink Confidence
Calibration under Covariate Shift** for the ICML 2026 reproduction collection.

This repository preserves the complete six-claim campaign, including exact
proofs, counterexamples, real-MNIST formula checks, source-only audits, and the
illustrated report. It separates a paper claim from a benchmark result: no
new official score is claimed, and missing author artifacts are recorded rather
than inferred.

## Paper identity

| Field | Value |
| --- | --- |
| Paper | *Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift* |
| Authors | Jinzong Dong, Zhaohui Jiang, Bo Yang |
| arXiv | [2605.21552](https://arxiv.org/abs/2605.21552) |
| OpenReview | [gFPPTokv9C](https://openreview.net/forum?id=gFPPTokv9C) |
| Audited paper source | arXiv v1; SHA-256 `fb1d1a634d55132694349d40d56731cc5c7401571bc8c1a9f6eee1b5849950ab` |
| Official implementation pin | `NeuroDong/ECL@aae77f890f1e4ebc13dad135b5e29758d98d318d` |
| Repository role | Claim-by-claim theory, implementation, provenance, and benchmark-boundary audit |

## Claim status

| Claim | Status | What the evidence establishes |
| --- | --- | --- |
| C1 — expectation consistency iff calibration-curve transfer | `VERIFIED_SCOPED` | Exact conditional-expectation proof and a 561-component rational certificate pass under shared-kernel, measurability, and common-support qualifications. |
| C2 — Eq. 9 finite-sample bound and `O(B/epsilon²)` order | `VERIFIED_SCOPED_WITH_QUALIFICATION` | A corrected soft-bin proof gives absolute constant `C=16` under fixed-function, iid, positive-denominator assumptions; the printed Appendix G derivation is not certified. |
| C3 — Eq. 10 unbiased mini-batch gradient | `FALSIFIED_AS_STATED` | Exact probability-domain witnesses break normalization, same-batch direction, omitted weight derivatives, and Eq. 8/Eq. 10 equivalence. A narrower corrected estimator remains valid. |
| C4 — differentiable canonical, class-wise, and top-label variants | `VERIFIED_SCOPED` | Independent NumPy and PyTorch/finite-difference implementations agree on all three losses and gradients on a full 10,000-image MNIST holdout. |
| C5 — ten-run MNIST/USPS→SVHN Table 2 values | `BLOCKED` | Printed arithmetic is auditable, but raw predictions, checkpoints, exact seeds, and a faithful three-architecture pipeline are unavailable. |
| C6 — five Table 1 capabilities plus Figure 2/Table 3 evidence | `FALSIFIED_SCOPED` | The Table 1 conjunction is falsified by the C3 counterexample; the separate five-seed simulation diverges and is not used as the logical falsification. |

The status vocabulary is deliberately narrow. `VERIFIED_SCOPED` does not mean
that every empirical table in the paper was reproduced; `BLOCKED` means the
available artifacts cannot support a fair attempt; and `FALSIFIED_AS_STATED`
means the exact quantified statement has a valid counterexample.

## How each claim is produced

### C1 — calibration transfer

[`repro/src/claim1_general_certificate.py`](repro/src/claim1_general_certificate.py)
derives the tower-property identity and checks it with exact rational
arithmetic. The output covers `257` exact posterior summaries, `17` score
values, `11` classes, formal coefficient maps for every common posterior, and
assumption-breaking controls. The main artifact is under
[`repro/evidence/2026-07-24/artifacts/claim-1/`](repro/evidence/2026-07-24/artifacts/claim-1/);
the detailed interpretation is [`docs/CLAIM1_GENERAL_PROOF_AUDIT.md`](docs/CLAIM1_GENERAL_PROOF_AUDIT.md).

The proof transfers a source calibration curve to the target only on common
identified score support and does not claim absolute calibration from
expectation consistency alone.

### C2 — finite-sample sample complexity

Three routes make the production path explicit:

1. [`repro/src/run_claim3_sample_complexity.py`](repro/src/run_claim3_sample_complexity.py)
   checks the hard-bin construction.
2. [`repro/src/claim2_soft_theorem_proof.py`](repro/src/claim2_soft_theorem_proof.py)
   proves the corrected soft-bin bound with budget
   `8*sqrt(2) + 4 = 15.3137085 < 16`.
3. [`repro/src/claim2_soft_falsification_stress.py`](repro/src/claim2_soft_falsification_stress.py)
   searches the allowed probability domain; its independent checker does not
   label a finite search as a proof.

The proof and 1,024-row stress artifacts live under
[`repro/evidence/2026-07-24/artifacts/claim-2/`](repro/evidence/2026-07-24/artifacts/claim-2/).
The conclusion requires fixed posterior/assignment functions, iid evaluation
samples, positive realized denominators, and `K >= 2`; it does not certify the
paper's printed Appendix G derivation as written.

### C3 — mini-batch gradient theorem

[`repro/src/claim3_gradient_certificate.py`](repro/src/claim3_gradient_certificate.py)
constructs exact `Fraction` witnesses strictly inside the probability domain.
The four independent failures are:

| Witness | Full quantity | Claimed quantity |
| --- | ---: | ---: |
| Appendix H normalization | `3` | `3/2` |
| Same-batch norm direction | `1` | `0` |
| Omitted soft-weight derivative | `3/4` | `0` |
| Eq. 8 versus profiled Eq. 10 gradient | `0` | `1/4` |

The corresponding losses are `3/4` and `65/128`. The artifact also identifies
the narrower fixed-direction estimator that is unbiased when its state is
frozen independently and correctly scaled. See
[`docs/CLAIM3_GRADIENT_AUDIT.md`](docs/CLAIM3_GRADIENT_AUDIT.md).

### C4 — differentiable calibration variants

[`repro/src/claim4_real_mnist_independent_checker.py`](repro/src/claim4_real_mnist_independent_checker.py)
checks canonical, class-wise, and top-label soft-binning on a complete 10,000
image MNIST holdout. Independent NumPy/finite-difference gradients agree with
the primary implementation to `2.91e-8`, `1.47e-10`, and `8.92e-11`; loss
errors are at machine precision. The route and controls are documented in
[`pages/claim-4-real-mnist-2026-07-24/page.md`](pages/claim-4-real-mnist-2026-07-24/page.md).

This verifies formula/gradient compatibility, not an improvement in benchmark
calibration.

### C5 — Table 2 empirical benchmark

[`repro/src/claim5_table2_audit.py`](repro/src/claim5_table2_audit.py) and
the four route pages audit the official source, predecessor source, printed
Table 2 arithmetic, and three materially different full-data attempts. The
literal route reaches a nonfinite Soft-ECE path; repaired routes cover one
LeNet seed; the mandatory fourth route shows that the printed mean/standard
deviation pairs are arithmetically realizable. None reproduces the unavailable
ten-run, three-architecture generating protocol, so the correct status remains
`BLOCKED`.

### C6 — capability conjunction and empirical support

[`repro/src/claim6_table1_falsification.py`](repro/src/claim6_table1_falsification.py)
uses the C3 witnesses to falsify the Table 1 “theoretically mini-batch
trainable” cell. This is sufficient to falsify the all-five conjunction without
pretending to reproduce PACS or ImageNet-Sketch. The separate paper-text
simulation is produced by
[`repro/src/claim6_simulation_figure2.py`](repro/src/claim6_simulation_figure2.py)
and is reported as divergent/mixed rather than upgraded to a universal claim.

## Repository and branch map

The full original branch history is preserved under clean purpose-based names
in [`docs/BRANCH_AUDIT.md`](docs/BRANCH_AUDIT.md). `main` is the integrated
publication surface; `audit/*` branches isolate claim attempts; `experiment/*`
holds the frozen cumulative baseline; and `release/*` holds the release-candidate
assembly. There are no final `orx/*` branches.

The illustrated report is [`reports/ecl-covariate-shift/report.md`](reports/ecl-covariate-shift/report.md),
the source/claim contract is under `.openresearch/artifacts/`, and the raw
campaign evidence is under `repro/evidence/2026-07-24/artifacts/`.

## Reproduce and verify

```bash
uv sync --frozen
uv run pytest -q repro/tests
uv run python repro/src/release_gate.py
uv run --frozen --python 3.12 python repro/src/run_campaign.py
uv run python repro/src/verify_results.py
uv run python repro/src/publication_gate.py --skip-producers
```

The cumulative historical run completed `122` tests on local CPU. The long
campaign command is not required to read the committed evidence; its exact
environment, inputs, and limits are recorded in the report and source audit.

## Citation

```bibtex
@article{dong2026expectation,
  title   = {Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift},
  author  = {Dong, Jinzong and Jiang, Zhaohui and Yang, Bo},
  journal = {arXiv preprint arXiv:2605.21552},
  year    = {2026},
  doi     = {10.48550/arXiv.2605.21552},
  url     = {https://arxiv.org/abs/2605.21552}
}
```

## Thank you

Thank you to Jinzong Dong, Zhaohui Jiang, and Bo Yang for developing a clear
framework for calibration under covariate shift and for stating the claims
precisely enough to support independent theorem checks and constructive
counterexamples. This repository is an independent reproduction/audit effort
and is not affiliated with the authors.

## Provenance and limits

The paper source, official-code pin, public Space history, artifact hashes, and
missing benchmark inputs are recorded in [`sources.json`](sources.json) and
[`docs/SOURCE_AUDIT.md`](docs/SOURCE_AUDIT.md). The fail-closed publication
check is specified in [`docs/PUBLICATION_GATE.md`](docs/PUBLICATION_GATE.md).
Historical judge scores and forecasts remain provenance only; no score change
is asserted by this GitHub repository.
