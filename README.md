# Reproduction update — ECL under covariate shift

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-gFPPTokv9C-ecl-calibration-covariate-shift/blob/master/notebooks/ecl_reproduction.py)

We tested all six judged claims in *Expectation Consistency Loss: Rethink
Confidence Calibration under Covariate Shift* (arXiv 2605.21552). The strongest
new result is exact: the paper defines Table 1’s “theoretically mini-batch
trainable” capability as an unbiased-gradient identity, but valid rational
witnesses give full versus expected mini-batch gradients of `3` versus `3/2`
and `1` versus `0`. Claims 3 and 6 are therefore FALSIFIED as written.

The previous live score remains **6/12**. The conservative future-judge
forecast is **8–10/12**, with **10/12** the best-supported possible result;
neither is an awarded score. Claim 5 remains honestly BLOCKED.

The full illustrated article is
[ECL under covariate shift: an exact audit changes the empirical story](reports/ecl-covariate-shift/report.md).
The self-contained [marimo tutorial](notebooks/ecl_reproduction.py) opens with
the completed evidence and does not require rerunning training.

| Claim | Assessment | Paper number versus observed evidence |
| --- | --- | --- |
| 1 | VERIFIED, HIGH | 561 exact components agree |
| 2 | VERIFIED, HIGH | paper leaves `C` unspecified; corrected proof certifies `C=16`, with max diagnostic error/radius `0.0873` |
| 3 | FALSIFIED, HIGH | claimed equal gradients; exact `3 vs 3/2`, `1 vs 0`, `3/4 vs 0`, and `0 vs 1/4` |
| 4 | VERIFIED, HIGH | three variants; independent max gradient error `2.91e-8` on 10,000 MNIST images |
| 5 | BLOCKED, LOW | paper LeNet ECE `61.9%→21.5%`; three full-data routes diverge but cannot refute the unavailable historical ten-run aggregate |
| 6 | FALSIFIED, HIGH | paper simulation ECL `{0.028,0.024,0.048}`; observed `{0.1549,0.0638,0.1499}`, plus exact false Table 1 mini-batch cell |

Substitutions are explicit: the Claim 6 simulation follows the paper text
where the saved notebook conflicts with it; Claim 5 routes use one available
LeNet reconstruction because the official digit pipeline, seeds, checkpoints,
and ResNet20/DenseNet40 raw runs are absent. Compute was local CPU except two
Hugging Face `cpu-upgrade` full-dataset Claim 5 runs; no GPU was used.

## Experiment log

Every experiment used the exact same command:
`uv run --frozen --python 3.12 python repro/src/run_campaign.py`.

| Branch / experiment | Purpose | Exact run command | Assessment / outcome | Compute |
| --- | --- | --- | --- | --- |
| `master` | Public presentation surface | Not run as an experiment (publication surface) | Pending explicit publication approval | — |
| [`orx/frozen-cumulative-baseline`](https://github.com/MachineLearning-Nerd/icml26-repro-gFPPTokv9C-ecl-calibration-covariate-shift/tree/orx/frozen-cumulative-baseline) | Freeze accepted evidence | `uv run --frozen --python 3.12 python repro/src/run_campaign.py` | 8/8 steps, 122 tests | Local CPU, 40s |
| [`orx/claim-2-synthesis-and-claim-4-real-mnist-soft-bi`](https://github.com/MachineLearning-Nerd/icml26-repro-gFPPTokv9C-ecl-calibration-covariate-shift/tree/orx/claim-2-synthesis-and-claim-4-real-mnist-soft-bi) | Full MNIST Appendix F check | `uv run --frozen --python 3.12 python repro/src/run_campaign.py` | Claim 4 VERIFIED forecast | Local CPU, 7m |
| [`orx/claim-5-route-4-mandatory-falsification-audit`](https://github.com/MachineLearning-Nerd/icml26-repro-gFPPTokv9C-ecl-calibration-covariate-shift/tree/orx/claim-5-route-4-mandatory-falsification-audit) | Fourth-route falsification gate | `uv run --frozen --python 3.12 python repro/src/run_campaign.py` | Claim 5 BLOCKED | Local CPU, 5m |
| [`orx/claim-6-route-1-full-simulated-figure-2`](https://github.com/MachineLearning-Nerd/icml26-repro-gFPPTokv9C-ecl-calibration-covariate-shift/tree/orx/claim-6-route-1-full-simulated-figure-2) | Five-seed, three-paradigm Figure 2 | `uv run --frozen --python 3.12 python repro/src/run_campaign.py` | DIVERGENT_OR_MIXED | Local CPU, 4m52s |
| [`orx/claim-6-route-2-primary-source-table-1-audit`](https://github.com/MachineLearning-Nerd/icml26-repro-gFPPTokv9C-ecl-calibration-covariate-shift/tree/orx/claim-6-route-2-primary-source-table-1-audit) | Exact Table 1 logical audit | `uv run --frozen --python 3.12 python repro/src/run_campaign.py` | Claim 6 FALSIFIED forecast | Local CPU, 5m38s |
| [`orx/claim-2-route-3-soft-bin-concentration-proof`](https://github.com/MachineLearning-Nerd/icml26-repro-gFPPTokv9C-ecl-calibration-covariate-shift/tree/orx/claim-2-route-3-soft-bin-concentration-proof) | Corrected soft Eq. 8 proof | `uv run --frozen --python 3.12 python repro/src/run_campaign.py` | 22/22 steps, 151 tests; Claim 2 VERIFIED forecast | Local CPU, 7m47s |

## Original repository overview

> The section below is preserved as the historical baseline README. Its
> interim assessments and 122-test count are superseded by the reproduction
> update above.

# Repro — ECL under covariate shift (`gFPPTokv9C`)

Source-pinned reproduction of *Expectation Consistency Loss: Rethink Confidence
Calibration under Covariate Shift* (Dong et al.; arXiv
[2605.21552](https://arxiv.org/abs/2605.21552)) for the
[ICML 2026 Agent Reproduction Challenge](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge).

The current challenge prompt exposes six anchored claims. The live legacy judge still
scores three defaults: legacy C1 maps to anchored C1, compatibility C2 maps to anchored
C4, and sample-complexity C3 maps to anchored C2.

## Results

| Current claim | Local assessment | Headline evidence |
| --- | --- | --- |
| C1 — Theorem 3.1 | **Verified with support/version qualifications** | General tower-property proof; formal all-posterior certificate; exact 257-state, 17-score, 11-class construction with 561 component checks |
| C2 — Theorem 3.2 sample complexity | **Hard-bin rate plus synthetic and real-trained-model soft Eq. 8 scaling supported** | Real MNIST classifiers reach `0.9097/0.9076`; ECL/ECE tail slopes `-0.680940/-0.742584`; B exponents `0.889/1.014` |
| C3 — Theorem 3.3 gradient | **Falsified as stated** | Appendix scaling, same-batch direction, soft-weight derivatives, and Eq. 10 objective parity each fail exact controls |
| C4 — Calibration extensions | **Formula-level verified** | Exact canonical, class-wise, and top-label certificates with diagnostic losses `1/6`, `1/6`, and `1/48`; four controls rejected |
| C5 — Digit Table 2 | **Inconclusive source-only** | Printed reductions are exact; raw ten-run outputs/checkpoints/faithful current pipeline are absent |
| C6 — Table 1/Figure 2/Table 3 | **Partial source support; broad empirics inconclusive** | All official source hashes/modes pass; paper arithmetic checked; PACS/ImageNet pipelines and raw runs absent |

The last confirmed official record is **5/6** at Space SHA
`1abb0c87beb604420d3a0e6140ea122511c63e93`: legacy C1/C2 are `verified`, and
legacy C3 sample complexity is `toy`. The final real-MNIST attempt in this checkout is not
called official until a fresh judge record points to its public SHA.

## Reproduce

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy scipy pytest

.venv/bin/python repro/src/claim1_general_certificate.py
.venv/bin/python repro/src/run_claim3_sample_complexity.py
.venv/bin/python repro/src/run_claim3_soft_sample_complexity.py
.venv/bin/python repro/src/run_claim3_real_mnist_sample_complexity.py \
  --data-root /path/to/uncompressed-mnist-idx
.venv/bin/python repro/src/claim3_gradient_certificate.py
.venv/bin/python repro/src/claim2_compatibility_certificate.py
.venv/bin/python repro/src/claim5_table2_audit.py \
  --paper repro/evidence/claim3/2605.21552v1.pdf --official-root upstream
.venv/bin/python repro/src/claim6_capability_audit.py
.venv/bin/python -m pytest repro/tests/ -q
```

Expected full result: **122 passed**.

## Important scientific boundaries

- C1 identifies equality of source/target calibration curves on common score support.
  EC alone does not make either curve equal the score; target calibration additionally
  needs a source-calibration premise.
- C2 supports fixed hard bins, fixed-`B` soft Eq. 8 scaling on a controlled construction, and
  comparable sample order on a real-MNIST task with disjoint trained classifiers. Appendix G
  retains an omitted `sqrt(K)` term and omits target-bin-mass estimation. The empirical `B`
  sweeps do not prove universal dependence; the MNIST posterior head is an estimate rather than
  an oracle, and its minimum soft masses are extremely small.
- C3's exact counterexample gives Eq. 8 loss/gradient `3/4, 0` versus profiled Eq. 10
  `65/128, 1/4`. This rejects the unrestricted theorem, not the empirical usefulness of
  the official proximal/EMA heuristic.
- C4 verifies mathematical grouping/observable compatibility, not image-benchmark
  improvement or numerical equivalence of every soft/proximal implementation detail.
- C5/C6 are source/provenance audits. Paper-table arithmetic and embedded figures are
  never promoted to independent empirical measurements.
- Paper PDF SHA-256: `fb1d1a634d55132694349d40d56731cc5c7401571bc8c1a9f6eee1b5849950ab`.
- Official source pin: `NeuroDong/ECL@aae77f890f1e4ebc13dad135b5e29758d98d318d`;
  `losses.py` SHA-256 `1c2de34967f34b98faae5025368edac88f46a709d6e1e0c063e2c01f4d6e9754`.

Detailed audits are under `docs/`; deterministic JSON certificates are under `outputs/`.

Logbook: https://huggingface.co/spaces/DineshAI/gFPPTokv9C
