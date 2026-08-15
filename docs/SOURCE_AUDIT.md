# Source and provenance audit

## Paper source

- **Title:** *Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift*
- **Authors:** Jinzong Dong, Zhaohui Jiang, Bo Yang
- **arXiv:** [2605.21552](https://arxiv.org/abs/2605.21552)
- **OpenReview:** [gFPPTokv9C](https://openreview.net/forum?id=gFPPTokv9C)
- **Section-level HTML:** [ar5iv](https://ar5iv.labs.arxiv.org/html/2605.21552)
- **Audited PDF SHA-256:** `fb1d1a634d55132694349d40d56731cc5c7401571bc8c1a9f6eee1b5849950ab`
- **Retrieval manifest:** `repro/evidence/2026-07-24/artifacts/source/paper_source_manifest.json`

The claim quantifiers and assumptions are transcribed in
`repro/evidence/2026-07-24/artifacts/source/source_audit.md`. The source audit
distinguishes the theorem statements from the paper's printed proof and from
the empirical claims.

## Official implementation

- Repository: `NeuroDong/ECL`
- Revision: `aae77f890f1e4ebc13dad135b5e29758d98d318d`
- Audited file: `losses.py`
- File SHA-256: `1c2de34967f34b98faae5025368edac88f46a709d6e1e0c063e2c01f4d6e9754`

The implementation pin supports the source-path audit and formula inspection.
It does not supply the raw ten-run Table 2 predictions, checkpoints, exact
seeds, or a complete faithful reproduction pipeline for every benchmark.

## Public campaign artifacts

- Historical Space: `DineshAI/gFPPTokv9C`
- Historical artifact bucket: `DineshAI/gFPPTokv9C-artifacts`
- Source report and notebook: `reports/ecl-covariate-shift/` and
  `notebooks/ecl_reproduction.py`
- Machine-readable campaign artifacts:
  `repro/evidence/2026-07-24/artifacts/`

These are campaign artifacts and provenance mirrors, not author-endorsed score
records. Historical judge revisions and scores are retained in the existing
publication report; this repository does not assert a new score.

## Clean-room boundary

- Exact rational proofs and counterexamples are implemented independently in
  `repro/src/` and checked by separate test/checker paths where stated.
- The real-MNIST C4 route uses a separate NumPy checker and hash-audited inputs.
- C5 remains source-only because a different seed or repaired pipeline cannot
  contradict the historical ten-run aggregate.
- Stale Trackio/logbook state is removed from the final publication surface;
  durable pages and evidence artifacts remain under `pages/`, `docs/`, and
  `repro/evidence/`.
