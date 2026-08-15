# Status — ECL under covariate shift (`gFPPTokv9C`)

Last audited: 2026-08-15. Repository target: `icml26-expectation-consistency-loss`.

## Current scoped verdict

| Claim | Status | Evidence boundary |
| --- | --- | --- |
| C1 calibration-transfer condition | `VERIFIED_SCOPED` | Exact theorem/certificate with common-support and shared-kernel qualifications. |
| C2 finite-sample order | `VERIFIED_SCOPED_WITH_QUALIFICATION` | Corrected soft-bin proof under explicit fixed-function assumptions; printed proof has a missing `sqrt(K)`/mass term. |
| C3 mini-batch gradient | `FALSIFIED_AS_STATED` | Exact rational probability-domain counterexamples; corrected restricted estimator is separately identified. |
| C4 differentiable variants | `VERIFIED_SCOPED` | Full-MNIST implementation/gradient agreement for canonical, class-wise, and top-label forms. |
| C5 Table 2 | `BLOCKED` | Source arithmetic and multiple routes audited; raw ten-run artifacts and faithful pipeline unavailable. |
| C6 compound capability/empirical claim | `FALSIFIED_SCOPED` | Table 1 conjunction fails through C3; Figure 2/Table 3 are not represented as independently reproduced. |

## Campaign state

- Historical cumulative campaign: 22/22 steps and 122 tests passed.
- Historical official score records and forecasts are preserved as provenance;
  this repository claims no new judge score.
- The final public surface is a clean `main` branch plus purpose-based audit
  branches. The prior `orx/*` names are mapped in
  [`docs/BRANCH_AUDIT.md`](docs/BRANCH_AUDIT.md).
- The release report and notebook remain available, but no missing official
  checkpoints, raw predictions, exact seeds, or benchmark assets are implied.

## Gate meaning

`SCOPED_PASS` means the repository identity, evidence paths, claim statuses,
source provenance, branch naming, and stale-state hygiene checks pass. It does
not mean that all six paper claims are verified or that an official score has
changed.
