# Research log

## 2026-08-15 — publication-surface cleanup

- Confirmed the paper identity against arXiv `2605.21552` and OpenReview
  `gFPPTokv9C`.
- Classified the six claims separately: two scoped verifications, one scoped
  corrected theorem, one exact falsification, one blocked benchmark, and one
  compound falsification through a required conjunct.
- Preserved the full `orx/*` history under purpose-based `audit/*`,
  `experiment/*`, and `release/*` names.
- Kept the report, notebook, source contract, raw claim evidence, and exact
  counterexamples; removed stale generated logbook state from the canonical
  surface.
- Added claim/source/branch/publication-gate documentation, a citation, and a
  thank-you note to the authors.

## Interpretation rule

Printed paper tables, code-path existence, and historical score records are
provenance. They become independent evidence only when a committed producer or
checker recomputes the relevant quantity under the paper's stated assumptions.
