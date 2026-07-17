# Negative controls

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_en_01", "created_at": "2026-07-17T19:43:00+00:00", "title": "Covariate-shift scope"}
-->
**Breaking covariate shift falsifies the iff.** The theorem assumes `P(Y|X)` is shared across
domains. If we violate this (`P_s(Y|X)≠P_t(Y|X)`), the shared-`g` EC residual no longer equals the
true calibration gap (which uses the per-domain `g`): the iff **fails** on 20/20 instances. This
confirms the characterization is genuinely scoped to covariate shift, not a universal identity.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_en_02", "created_at": "2026-07-17T19:43:10+00:00", "title": "Both directions"}
-->
**Both directions.** Across 40 random covariate-shift instances, `EC=0 ⟺ calibration gap=0` holds
consistently — the condition is both necessary and sufficient, not just one direction.
